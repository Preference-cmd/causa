//! Construction and work identity for `causa_runtime::session`.
//!
//! Construction. `Session::new` builds a session from a fresh
//! `ConversationState::new` *and* from `ConversationState::from_history` over
//! validated completed entries, both leaving the session idle and neither
//! calling the model or a tool; a state that already holds an active turn is
//! rejected and every by-value input comes back. No global manager exists in
//! this path — the constructor takes the harness's assembled runner directly.
//!
//! Work identity across works. `submit → wait → submit` yields two distinct
//! `WorkRef`s / `TurnId`s, and the earlier finished work stays observable — its
//! published snapshot is not displaced when the session moves on to a newer
//! "current" work.
//!
//! The plain closed loop and the request-key dedup are pinned by
//! `session_smoke.rs`; these tests add the from-history construction and the
//! two-work retention story. Fixtures come from `common`; the one extra local
//! gateway gates the second model call so a newer work is provably `Running`
//! while the older result is re-read.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, ContentPart, ConversationId, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelOutput, ModelRequest, TextPayload, TurnId,
};
use causa_runtime::{
    ConversationState, FinishedKind, HistoryEntry, SealedResult, Session, SessionConfig,
    SessionError, SessionHandle, SubmitRequest, TurnRunOptions, WorkRef, WorkState,
};
use common::{RecordingGateway, commit_sealed, endturn_output, options_with_limits, runner_with};

/// Submit one text part under `key`.
async fn submit_text(handle: &SessionHandle, key: &str, text: &str) -> causa_runtime::WorkReceipt {
    handle
        .submit(SubmitRequest {
            request_key: key.into(),
            parts: vec![ContentPart::Text(TextPayload::new(text))],
        })
        .await
        .expect("an idle session accepts the work")
}

/// Block until `work` is published `Running`, bounded so a module regression
/// surfaces as a failure rather than a hang.
async fn await_running(handle: &SessionHandle, work: &WorkRef) {
    for _ in 0..2_000 {
        let observation = handle.observe(work).await.expect("work is known");
        if observation.state == WorkState::Running {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("work never reached Running");
}

// ---- local fixture: a gateway whose second call parks until released --------

/// The first model call returns `"first"` immediately; every later call parks
/// on a zero-permit semaphore until [`TwoPhaseGateway::release`]. That keeps a
/// newer work observably `Running` while an earlier finished work is re-read,
/// so the "old result survives the current-work switch" story is exercised with
/// newer work genuinely in flight — not merely after it too has finished.
struct TwoPhaseGateway {
    calls: AtomicUsize,
    gate: tokio::sync::Semaphore,
    recorded: Mutex<Vec<ModelRequest>>,
}

impl TwoPhaseGateway {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            gate: tokio::sync::Semaphore::new(0),
            recorded: Mutex::new(Vec::new()),
        })
    }

    /// Let the parked call proceed.
    fn release(&self) {
        self.gate.add_permits(1);
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn recorded_len(&self) -> usize {
        self.recorded.lock().unwrap().len()
    }
}

#[async_trait]
impl ModelGateway for TwoPhaseGateway {
    async fn invoke(
        &self,
        req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.recorded.lock().unwrap().push(req.clone());
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            Ok(endturn_output("first"))
        } else {
            let permit = self.gate.acquire().await.map_err(|_| {
                ModelInvokeError::new(ModelInvokeErrorKind::Transient, "gate closed")
            })?;
            permit.forget();
            Ok(endturn_output("second"))
        }
    }
}

// ---- construction from an empty state ----------------------------------

#[tokio::test]
async fn a1_new_from_empty_state_is_idle_without_model_call() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("unused"))]);
    let runner = Arc::new(runner_with(gateway.clone(), vec![]));

    // Empty state -> Ok. The contract makes Ok equivalent to "idle": an
    // active/sealed state is rejected below.
    let session = Session::new(
        ConversationState::new(ConversationId("empty".into())),
        runner,
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("a fresh empty state is an idle session base");

    // Construction never touches the model.
    assert!(
        gateway.recorded().is_empty(),
        "new must not call the model from an empty state"
    );

    // The session is bound to that conversation and is immediately usable.
    let handle = session.handle();
    assert_eq!(handle.id().0, "empty");
}

// ---- construction from validated completed history ----------------------

#[tokio::test]
async fn a1_new_from_validated_history_is_idle_without_model_call() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("unused"))]);

    // Build two committed turns, then replay their validated entries.
    let mut seeded = ConversationState::new(ConversationId("hist".into()));
    commit_sealed(&mut seeded, "t1", SealedResult::Completed);
    commit_sealed(&mut seeded, "t2", SealedResult::Completed);
    let entries: Vec<HistoryEntry> = seeded.history().to_vec();
    assert_eq!(entries.len(), 2);

    let replayed = ConversationState::from_history(ConversationId("hist".into()), entries)
        .expect("validated completed history replays");
    assert_eq!(replayed.history_len(), 2);

    let session = Session::new(
        replayed,
        Arc::new(runner_with(gateway.clone(), vec![])),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("a replayed completed history is an idle session base");

    assert!(
        gateway.recorded().is_empty(),
        "new must not call the model when reconstructing from history"
    );
    assert_eq!(session.handle().id().0, "hist");
}

// ---- a state with an active turn is rejected, inputs returned by value --

#[tokio::test]
async fn a1_new_rejects_active_turn_and_returns_the_by_value_inputs() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("unused"))]);

    let mut state = ConversationState::new(ConversationId("busy".into()));
    state
        .begin_turn(TurnId::new("live-turn"))
        .expect("an empty state admits a turn");
    state
        .active_turn_mut()
        .expect("just begun")
        .append_input(TextPayload::new("in"), "user")
        .expect("append");

    let runner = Arc::new(runner_with(gateway.clone(), vec![]));
    let runner_kept = Arc::clone(&runner);
    let options = options_with_limits(3, 5);
    let config = SessionConfig {
        retained_work_capacity: 7,
        work_deadline: Some(Duration::from_secs(2)),
    };

    let rejection = Session::new(state, runner, options, config.clone())
        .expect_err("a state with an active turn is not a valid session base");

    // Rejected as invalid input, not silently dropped.
    assert!(
        matches!(rejection.error, SessionError::InvalidInput(_)),
        "expected InvalidInput, got {:?}",
        rejection.error
    );

    // Every by-value input comes back, not only the state, so a rejected
    // construction costs the caller nothing.
    assert!(
        Arc::ptr_eq(&rejection.runner, &runner_kept),
        "the assembled runner round-trips by value"
    );
    assert_eq!(
        rejection.options.policy.limits.max_model_rounds, 3,
        "the assembled options round-trip"
    );
    assert_eq!(rejection.options.policy.limits.max_tool_calls, 5);
    assert_eq!(
        rejection.config, config,
        "the session config round-trips by value"
    );

    // The state is handed back by value, still holding the live turn's facts.
    let returned = rejection.state;
    assert_eq!(returned.conversation_id().0, "busy");
    assert_eq!(returned.history_len(), 0);
    assert_eq!(
        returned
            .active_turn()
            .expect("the live turn survives")
            .turn_id()
            .0,
        "live-turn"
    );
    assert!(
        gateway.recorded().is_empty(),
        "a rejected construction must not call the model"
    );
}

// ---- two works, distinct identity, the older result retained ------------

#[tokio::test]
async fn a2_two_works_get_distinct_refs_and_the_first_stays_observable() {
    let gateway = TwoPhaseGateway::new();
    let runner = Arc::new(runner_with(gateway.clone(), vec![]));
    let session = Session::new(
        ConversationState::new(ConversationId("two-works".into())),
        runner,
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("an empty state is an idle session base");
    let handle = session.handle();

    // First work runs to completion.
    let first = submit_text(&handle, "first", "one").await;
    let first_wait = handle
        .wait(&first.work, Duration::from_secs(5))
        .await
        .expect("the first work is observable");
    assert_eq!(first_wait.observation.state, WorkState::Finished);
    let first_revision = first_wait.observation.revision;
    match &first_wait.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "first")
        }
        other => panic!("expected the first work to complete, got {other:?}"),
    }

    // Second work: identity is fresh, and it parks in Running.
    let second = submit_text(&handle, "second", "two").await;
    assert_ne!(
        first.work, second.work,
        "each accepted work gets its own WorkRef"
    );
    assert_ne!(
        first.work.turn_id, second.work.turn_id,
        "a later work never reuses an earlier work's TurnId"
    );
    assert_eq!(first.work.conversation_id, second.work.conversation_id);

    await_running(&handle, &second.work).await;
    // The gate guarantees the newer work reached the model and is still live.
    for _ in 0..2_000 {
        if gateway.calls() >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(
        gateway.calls(),
        2,
        "the second work must have begun its model call"
    );

    // While the second work is Running, the first work's result is unchanged:
    // switching the session's "current" work does not displace the old one.
    let first_again = handle
        .observe(&first.work)
        .await
        .expect("the older work stays readable");
    assert_eq!(first_again.state, WorkState::Finished);
    assert_eq!(
        first_again.revision, first_revision,
        "re-reading a retained work must not mutate its snapshot"
    );
    match &first_again.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "first")
        }
        other => panic!("expected the first work's completion to be retained, got {other:?}"),
    }

    // Release the second work; both results now coexist under one session.
    gateway.release();
    let second_wait = handle
        .wait(&second.work, Duration::from_secs(5))
        .await
        .expect("the second work is observable");
    assert_eq!(second_wait.observation.state, WorkState::Finished);
    match &second_wait.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "second")
        }
        other => panic!("expected the second work to complete, got {other:?}"),
    }
    assert_eq!(gateway.recorded_len(), 2, "one model call per work");

    // The first work is still exactly where it was left.
    assert_eq!(
        handle
            .observe(&first.work)
            .await
            .expect("retained after the second finishes")
            .state,
        WorkState::Finished
    );
}
