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
//! Shared fixtures come from `tests/common`, including the gated gateway
//! (a model call that parks until the test releases it; the non-cancelling
//! mode ignores the session token, so every parked call needs its release).

mod common;

use std::sync::Arc;
use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, ModelGateway, TextPayload};
use causa_runtime::{
    ConversationState, FinishedKind, Session, SessionConfig, SessionError, SubmitRequest,
    TurnRunOptions, WaitEnd, WorkState,
};
use common::{GatedGateway, RecordingGateway, endturn_output, runner_with};

// ---- local fixtures -----------------------------------------------------------

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
/// `SessionHandle::submit` is synchronous, and the dedup lookup and the
/// `submit_keys` insert share one `Mutex` acquisition, so accept-once is structural
/// (a single lock hold) rather than something two tasks race for. The
/// `tokio::join!` below only interleaves at the task boundary; it cannot
/// create a torn read the lock forbids.
#[tokio::test]
async fn a3_repeated_same_key_submits_accept_once_and_dedup_before_busy() {
    let gateway = GatedGateway::new("only-once", false);
    let (_session, handle) = session_with("a3-once", gateway.clone(), SessionConfig::default());
    let parts = vec![text("hello")];

    let original = handle
        .submit(request("k", parts.clone()))
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    let (retry_a, retry_b) = tokio::join!(
        async { handle.submit(request("k", parts.clone())) },
        async { handle.submit(request("k", parts.clone())) },
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
    let gateway = GatedGateway::new("done", false);
    let (_session, handle) = session_with("a3-conflict", gateway.clone(), SessionConfig::default());
    let parts = vec![text("one")];

    let original = handle
        .submit(request("key", parts.clone()))
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;
    let running = handle.observe(&original.work).unwrap();

    let conflict = handle
        .submit(request("key", vec![text("two")]))
        .expect_err("same key, different parts is a conflict");
    assert_eq!(conflict, SessionError::Conflict);

    let after = handle.observe(&original.work).unwrap();
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
            .expect_err("the argument mismatch persists"),
        SessionError::Conflict
    );
    assert_eq!(
        handle
            .submit(request("key", parts))
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
    let gateway = GatedGateway::new("first", false);
    let (_session, handle) = session_with("a3-key", gateway.clone(), SessionConfig::default());

    let first = handle
        .submit(request("first", vec![text("a")]))
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    let retry_parts = vec![text("b")];
    let busy = handle
        .submit(request("retry", retry_parts.clone()))
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

    // The rejected key was never consumed, so it submits cleanly now. The
    // one-shot gate needs a fresh release for the second work's model call.
    gateway.release();
    let accepted = handle
        .submit(request("retry", retry_parts))
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
        .expect_err("empty parts are rejected");
    assert!(
        matches!(invalid, SessionError::InvalidInput(_)),
        "expected InvalidInput, got {invalid:?}"
    );

    let accepted = handle
        .submit(request("ik", vec![text("hello")]))
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
    let gateway = GatedGateway::new("final", false);
    let (_session, handle) = session_with("a4-busy", gateway.clone(), SessionConfig::default());

    let active = handle
        .submit(request("w1", vec![text("one")]))
        .expect("an idle session accepts the work");
    gateway.wait_entered().await;

    let observed = handle.observe(&active.work).unwrap();
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
        .expect_err("one active work per conversation, no queue");
    assert_eq!(
        busy,
        SessionError::Busy {
            active: active.work.clone()
        }
    );

    let after = handle.observe(&active.work).unwrap();
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
    let gateway = GatedGateway::new("first", false);
    let config = SessionConfig {
        retained_work_capacity: 1,
        work_deadline: None,
    };
    let (_session, handle) = session_with("a4-cap", gateway.clone(), config);

    let retained = handle
        .submit(request("c1", vec![text("one")]))
        .expect("the first work fits the capacity");
    gateway.wait_entered().await;

    let busy = handle
        .submit(request("c2", vec![text("two")]))
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
        .expect_err("the registry is at capacity");
    assert_eq!(over, SessionError::CapacityExceeded);

    // Capacity exhaustion does not hide the retained result.
    let observed = handle.observe(&retained.work).unwrap();
    assert_eq!(observed.state, WorkState::Finished);
    assert!(matches!(
        observed.finished,
        Some(FinishedKind::Completed { .. })
    ));
    assert!(observed.fault.is_none());
    assert_eq!(gateway.calls(), 1);
}
