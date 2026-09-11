//! Request-key dedup and the busy / capacity admission guards, against the
//! real driver.
//!
//! Request-key idempotency (the `submit → wait → submit` half):
//! - a repeated same-key submit accepts once and returns the original
//!   receipt (dedup is checked *before* the busy guard, so a retry of a lost
//!   receipt resolves while the work still runs);
//! - same key with different args is [`SessionError::Conflict`];
//! - a rejected submit does not consume the key — a later valid submit with
//!   that key is accepted.
//!
//! The one-active-work contract:
//! - a gated gateway holds the work `Running`, so a competing submit is
//!   [`SessionError::Busy`] naming the active [`WorkRef`] — no queue;
//! - `retained_work_capacity` is explicit and, once the slot frees but the
//!   registry is full, a new submit is [`SessionError::CapacityExceeded`];
//! - a rejected submit never disturbs the existing work, which stays
//!   observable.
//!
//! Shared fixtures come from `tests/common`; the gated gateway (a model call
//! that blocks until the test releases it, or the session is cancelled) is
//! local to this target.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, ContentPart, ConversationId, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelOutput, ModelRequest, TextPayload,
};
use causa_runtime::{
    ConversationState, FinishedKind, Session, SessionConfig, SessionError, SubmitRequest,
    TurnRunOptions, WaitEnd, WorkState,
};
use common::{RecordingGateway, endturn_output, runner_with};
use tokio::sync::Semaphore;

// ---- local fixtures -----------------------------------------------------------

/// A gateway whose one model call parks until the test releases it, so the
/// session stays in `Running` for as long as the assertions need. `release`
/// opens the gate permanently: subsequent calls return immediately.
struct GatedGateway {
    text: String,
    calls: AtomicUsize,
    /// One permit per `invoke` entry — lets the test know the work has
    /// actually reached the model, not merely been accepted.
    entered: Semaphore,
    /// One permit per `release()`, consumed by the parked `invoke`.
    release: Semaphore,
    open: AtomicBool,
}

impl GatedGateway {
    fn new(text: &str) -> Arc<Self> {
        Arc::new(Self {
            text: text.to_string(),
            calls: AtomicUsize::new(0),
            entered: Semaphore::new(0),
            release: Semaphore::new(0),
            open: AtomicBool::new(false),
        })
    }

    /// How many model calls have been entered — the "accepted once" evidence.
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Wait until the worker has entered the gated model call. Bounded so a
    /// regression fails loudly instead of hanging the suite.
    async fn wait_entered(&self) {
        tokio::time::timeout(Duration::from_secs(5), self.entered.acquire())
            .await
            .expect("the gated gateway is entered within 5s")
            .expect("the entry semaphore stays open")
            .forget();
    }

    /// Open the gate: unblock the parked call and let every later call pass.
    fn release(&self) {
        self.open.store(true, Ordering::SeqCst);
        self.release.add_permits(1);
    }
}

#[async_trait]
impl ModelGateway for GatedGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        if self.open.load(Ordering::SeqCst) {
            return Ok(endturn_output(&self.text));
        }
        tokio::select! {
            permit = self.release.acquire() => {
                permit.expect("the release semaphore stays open").forget();
                Ok(endturn_output(&self.text))
            }
            // Owner drop cancels the work; release the parked call so the
            // worker cannot outlive the runtime.
            _ = ctrl.cancellation_token().cancelled() => Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                "gated gateway released by cancellation",
            )),
        }
    }
}

/// One text submission part.
fn text(s: &str) -> ContentPart {
    ContentPart::Text(TextPayload::new(s))
}

/// A session over `conversation` bounded by `config`, plus its handle.
fn session_with(
    conversation: &str,
    gateway: Arc<dyn ModelGateway>,
    config: SessionConfig,
) -> (Session, causa_runtime::SessionHandle) {
    let session = Session::new(
        ConversationState::new(ConversationId(conversation.into())),
        Arc::new(runner_with(gateway, vec![])),
        TurnRunOptions::default(),
        config,
    )
    .expect("an idle state is accepted");
    let handle = session.handle();
    (session, handle)
}

fn request(key: &str, parts: Vec<ContentPart>) -> SubmitRequest {
    SubmitRequest {
        request_key: key.into(),
        parts,
    }
}

// ---- request-key idempotency -----------------------------------------------------------------------

///repeated same-key submits accept once, and both retries resolve to the
/// original receipt even while the slot is busy — dedup is checked *before*
/// the busy guard. The single model call pins accept-once.
///
/// This proves accept-once + dedup-before-busy, **not** a data race:
/// `SessionHandle::submit` has no await point, and the dedup lookup and the
/// `by_key` insert share one `Mutex` acquisition, so accept-once is structural
/// (a single lock hold) rather than something two tasks race for. The
/// `tokio::join!` below only interleaves at the task boundary; it cannot
/// create a torn read the lock forbids.
#[tokio::test]
async fn a3_repeated_same_key_submits_accept_once_and_dedup_before_busy() {
    let gateway = GatedGateway::new("only-once");
    let (_session, handle) = session_with("a3-once", gateway.clone(), SessionConfig::default());
    let parts = vec![text("hello")];

    let original = handle
        .submit(request("k", parts.clone()))
        .await
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    let (retry_a, retry_b) = tokio::join!(
        handle.submit(request("k", parts.clone())),
        handle.submit(request("k", parts.clone())),
    );
    assert_eq!(
        retry_a.expect("the retry resolves to the original receipt"),
        original
    );
    assert_eq!(
        retry_b.expect("the second retry resolves to the original receipt"),
        original
    );

    gateway.release();
    let done = handle
        .wait(&original.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    assert_eq!(
        gateway.calls(),
        1,
        "accepting once means exactly one model call"
    );
}

///the same key with different parts is `Conflict`, both while the work
/// runs (dedup precedes the busy guard) and after it finishes. The rejected
/// request leaves the active work untouched.
#[tokio::test]
async fn a3_same_key_different_args_conflicts() {
    let gateway = GatedGateway::new("done");
    let (_session, handle) = session_with("a3-conflict", gateway.clone(), SessionConfig::default());
    let parts = vec![text("one")];

    let original = handle
        .submit(request("key", parts.clone()))
        .await
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;
    let running = handle.observe(&original.work).await.unwrap();

    let conflict = handle
        .submit(request("key", vec![text("two")]))
        .await
        .expect_err("same key, different parts is a conflict");
    assert_eq!(conflict, SessionError::Conflict);

    let after = handle.observe(&original.work).await.unwrap();
    assert_eq!(
        after.revision, running.revision,
        "a rejected submit must not touch the active work"
    );

    gateway.release();
    let done = handle
        .wait(&original.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);

    // The conflict is not a transient busy artifact: it survives the work
    // reaching a terminal state, and the original key+args still dedups.
    assert_eq!(
        handle
            .submit(request("key", vec![text("two")]))
            .await
            .expect_err("the argument mismatch persists"),
        SessionError::Conflict
    );
    assert_eq!(
        handle
            .submit(request("key", parts))
            .await
            .expect("the original key + args still resolve"),
        original
    );
    assert_eq!(
        gateway.calls(),
        1,
        "neither the conflict nor the dedup may run the model again"
    );
}

///a `Busy` rejection does not consume the request key — the very same key
/// is accepted once the slot frees, and the new work gets a fresh identity.
#[tokio::test]
async fn a3_busy_rejection_does_not_consume_the_request_key() {
    let gateway = GatedGateway::new("first");
    let (_session, handle) = session_with("a3-key", gateway.clone(), SessionConfig::default());

    let first = handle
        .submit(request("first", vec![text("a")]))
        .await
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    let retry_parts = vec![text("b")];
    let busy = handle
        .submit(request("retry", retry_parts.clone()))
        .await
        .expect_err("a competing submit is busy");
    assert_eq!(
        busy,
        SessionError::Busy {
            active: first.work.clone()
        }
    );

    gateway.release();
    assert_eq!(
        handle
            .wait(&first.work, Duration::from_secs(5))
            .await
            .unwrap()
            .observation
            .state,
        WorkState::Finished
    );

    // The rejected key was never consumed, so it submits cleanly now.
    let accepted = handle
        .submit(request("retry", retry_parts))
        .await
        .expect("the previously rejected key is still free");
    assert_ne!(
        accepted.work, first.work,
        "a later work never receives an earlier work's identity"
    );
    let done = handle
        .wait(&accepted.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    assert_eq!(gateway.calls(), 2);
}

///an invalid submit (empty parts) is rejected without consuming the key
/// and without leaving an active slot behind.
#[tokio::test]
async fn a3_invalid_input_does_not_consume_the_request_key() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let (_session, handle) = session_with("a3-invalid", gateway.clone(), SessionConfig::default());

    let invalid = handle
        .submit(request("ik", vec![]))
        .await
        .expect_err("empty parts are rejected");
    assert!(
        matches!(invalid, SessionError::InvalidInput(_)),
        "expected InvalidInput, got {invalid:?}"
    );

    let accepted = handle
        .submit(request("ik", vec![text("hello")]))
        .await
        .expect("the previously rejected key is still free");
    let done = handle
        .wait(&accepted.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    match done.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "done");
        }
        other => panic!("expected a completed work, got {other:?}"),
    }
    assert_eq!(gateway.recorded().len(), 1);
}

// ---- the one-active-work contract -----------------------------------------------------------------------

///a gated gateway keeps the work `Running`; a competing submit is `Busy`
/// naming that work, and the rejected submit leaves the existing work's
/// observation untouched.
#[tokio::test]
async fn a4_running_work_is_busy_and_stays_observable() {
    let gateway = GatedGateway::new("final");
    let (_session, handle) = session_with("a4-busy", gateway.clone(), SessionConfig::default());

    let active = handle
        .submit(request("w1", vec![text("one")]))
        .await
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    let observed = handle.observe(&active.work).await.unwrap();
    assert_eq!(observed.work, active.work);
    assert_eq!(observed.state, WorkState::Running);
    assert!(
        observed.revision >= 1,
        "Running is a published state change"
    );
    assert!(observed.finished.is_none());
    assert!(observed.fault.is_none());

    let busy = handle
        .submit(request("w2", vec![text("two")]))
        .await
        .expect_err("one active work per conversation, no queue");
    assert_eq!(
        busy,
        SessionError::Busy {
            active: active.work.clone()
        }
    );

    let after = handle.observe(&active.work).await.unwrap();
    assert_eq!(after.state, WorkState::Running);
    assert_eq!(
        after.revision, observed.revision,
        "the busy rejection must not mutate the active work"
    );

    gateway.release();
    let done = handle
        .wait(&active.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    assert!(matches!(
        done.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));
    assert_eq!(gateway.calls(), 1);
}

///`retained_work_capacity` is explicit. While the single retained slot is
/// busy the busy guard wins (still no queue); once the slot frees but the
/// registry is full, a new submit is `CapacityExceeded` and the retained work
/// stays observable.
#[tokio::test]
async fn a4_tiny_capacity_exceeds_only_when_idle_and_keeps_retained_observable() {
    let gateway = GatedGateway::new("first");
    let config = SessionConfig {
        retained_work_capacity: 1,
        work_deadline: None,
    };
    let (_session, handle) = session_with("a4-cap", gateway.clone(), config);

    let retained = handle
        .submit(request("c1", vec![text("one")]))
        .await
        .expect("the first work fits the capacity");
    gateway.wait_entered().await;

    let busy = handle
        .submit(request("c2", vec![text("two")]))
        .await
        .expect_err("the busy guard is checked before capacity");
    assert_eq!(
        busy,
        SessionError::Busy {
            active: retained.work.clone()
        }
    );

    gateway.release();
    let done = handle
        .wait(&retained.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);

    let over = handle
        .submit(request("c3", vec![text("three")]))
        .await
        .expect_err("the registry is at capacity");
    assert_eq!(over, SessionError::CapacityExceeded);

    // Capacity exhaustion does not hide the retained result.
    let observed = handle.observe(&retained.work).await.unwrap();
    assert_eq!(observed.state, WorkState::Finished);
    assert!(matches!(
        observed.finished,
        Some(FinishedKind::Completed { .. })
    ));
    assert!(observed.fault.is_none());
    assert_eq!(gateway.calls(), 1);
}
