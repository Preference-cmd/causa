//! Multi-handle observation, finite wait, and the terminal shapes —
//! Completed / Interrupted / Paused / Faulted / owner-drop — and what each
//! does to history and to acceptance.
//!
//! Reuses the shared `common` fixtures (`RecordingGateway`, `runner_with`,
//! `endturn_output`) and adds only two local extras: a gated gateway that stays
//! inside the model call until its attempt token fires, and a panicking
//! gateway. Nothing here touches `src/`.
//!
//! Observation and wait — several handles observe one work; a finite wait
//! returns `ReachedState` or `TimedOut` with a snapshot consistent with
//! `observe`; a timeout (or a dropped wait future) never changes the work's
//! published state.
//!
//! Terminal shapes — Completed commits and grows history; Interrupted carries
//! the real aborted facts plus its cause, stays out of the completed-only
//! history, and the next submit gets a **new** `TurnId`; a panicking gateway is
//! observable as `Faulted` and the session then refuses new work; dropping the
//! `Session` (a cloned handle survives) closes submission (`Closed`) **and**
//! fires the running work's token, so the worker ends `Interrupted`. The same
//! file also covers `Paused` (a fresh-key submit is `Busy` naming the paused
//! work, and `wait` reports `ReachedState`), the accept-time `work_deadline`
//! wired into `RunControl` (a too-slow round ends
//! `Interrupted { TurnDeadlineExceeded }`), and foreign / never-accepted
//! `WorkRef`s resolving to `NotFound`.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, ContentPart, ConversationId, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelOutput, ModelRequest, TextPayload, ToolCallPayload, TurnId,
};
use causa_runtime::{
    BatchDecision, ConversationState, FinishedKind, Session, SessionConfig, SessionError,
    SubmitRequest, TurnInteraction, TurnInterruption, TurnRunOptions, WaitEnd, WorkObservation,
    WorkRef, WorkState,
};
use common::{EchoTool, RecordingGateway, endturn_output, runner_with, tooluse_output};

// ---- local fixtures --------------------------------------------------------

/// A gateway that enters `invoke`, announces it, and then blocks inside the
/// model call until the attempt's cancellation token fires — at which point it
/// reports `Cancelled`, which the driver maps to
/// `TurnInterruption::ExplicitCancellation`. This is the fixture that makes a
/// work observably `Running` for as long as the test likes.
struct GatedGateway {
    entered: AtomicUsize,
    notify: tokio::sync::Notify,
}

impl GatedGateway {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            entered: AtomicUsize::new(0),
            notify: tokio::sync::Notify::new(),
        })
    }

    /// Wait until the runner is inside a model call — at that point the work
    /// has been published `Running` and is deep inside `invoke`.
    async fn wait_entered(&self) {
        while self.entered.load(Ordering::SeqCst) == 0 {
            self.notify.notified().await;
        }
    }
}

#[async_trait]
impl ModelGateway for GatedGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        // `notify_one` (not `notify_waiters`) so a wakeup that lands before the
        // test registers is stored as a permit rather than lost.
        self.notify.notify_one();
        ctrl.cancellation_token().cancelled().await;
        Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Cancelled,
            "gated gateway: cancelled by the session",
        ))
    }
}

/// A gateway that panics inside `invoke` — the worker's `catch_unwind` turns
/// this into an observable `Faulted`, never a permanently `Running` work.
struct PanickingGateway;

#[async_trait]
impl ModelGateway for PanickingGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        panic!("boom: panicking gateway");
    }
}

/// A `TurnInteraction` that pauses on the first tool-use batch — the host
/// approval gate. The turn suspends with its active context left open, so the
/// driver returns `TurnResult::Paused` and the session publishes
/// [`WorkState::Paused`].
struct PausingInteraction;

#[async_trait]
impl TurnInteraction for PausingInteraction {
    async fn decide_batch(&self, _calls: &[ToolCallPayload]) -> BatchDecision {
        BatchDecision::Pause { deadline: None }
    }
}

/// A gateway that oversleeps the work deadline and then returns one tool-use
/// round. The round forces the driver back to its loop top, where
/// `RunControl::should_stop` sees the passed deadline and the turn ends
/// `Interrupted { TurnDeadlineExceeded }`. (An end-turn output would instead
/// short-circuit to `Completed` without ever re-checking the loop top, so the
/// batch is what makes the deadline observable.)
struct SlowGateway;

#[async_trait]
impl ModelGateway for SlowGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        tokio::time::sleep(Duration::from_millis(250)).await;
        Ok(tooluse_output("late", "echo", serde_json::json!({})))
    }
}

// ---- helpers ---------------------------------------------------------------

fn req(key: &str, text: &str) -> SubmitRequest {
    SubmitRequest {
        request_key: key.into(),
        parts: vec![ContentPart::Text(TextPayload::new(text))],
    }
}

fn session_with(id: &str, gateway: Arc<dyn ModelGateway>) -> Session {
    Session::new(
        ConversationState::new(ConversationId(id.into())),
        Arc::new(runner_with(gateway, vec![])),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("an empty state is accepted")
}

/// Field-wise equality of two observations. `WorkObservation` deliberately has
/// no `PartialEq` (its `finished` payload is not comparable), so the test
/// compares the comparable parts and the presence of the optional payloads.
fn assert_same_observation(a: &WorkObservation, b: &WorkObservation) {
    assert_eq!(a.work, b.work, "same work identity");
    assert_eq!(a.revision, b.revision, "same published revision");
    assert_eq!(a.state, b.state, "same management state");
    assert_eq!(
        a.finished.is_some(),
        b.finished.is_some(),
        "same finished-payload presence"
    );
    assert_eq!(a.fault, b.fault, "same fault reason");
}

/// Every frame the gateway was handed, rendered via `Debug` so the committed
/// history and the active turn's facts can be inspected as one blob. The gate
/// is only reached through `RecordingGateway`, whose `frame` is the merged
/// conversation frame.
fn frames_debug(gateway: &RecordingGateway) -> Vec<String> {
    gateway
        .recorded()
        .iter()
        .map(|request| format!("{:?}", request.frame))
        .collect()
}

// ---- observation and finite wait --------------------------------------------------------------------

#[tokio::test]
async fn multiple_handles_observe_one_work_and_finite_wait_never_mutates_it() {
    let gateway = GatedGateway::new();
    let session = session_with("obs", gateway.clone());
    let h1 = session.handle();
    let h2 = h1.clone();

    let receipt = h1
        .submit(req("k1", "hold"))
        .await
        .expect("an idle session accepts the work");
    let work = receipt.work.clone();
    assert_eq!(receipt.accepted_revision, 0, "acceptance is revision 0");

    // Hold the work inside the model call, so `Running` is the steady state.
    gateway.wait_entered().await;

    // Two independent handles see exactly the same published work.
    let o1 = h1.observe(&work).await.expect("h1 observes its own work");
    let o2 = h2.observe(&work).await.expect("h2 observes the same work");
    assert_eq!(o1.state, WorkState::Running);
    assert_same_observation(&o1, &o2);

    // A finite wait that times out returns a snapshot consistent with observe.
    let timed = h1
        .wait(&work, Duration::from_millis(120))
        .await
        .expect("the ref belongs to this session");
    assert_eq!(timed.end, WaitEnd::TimedOut);
    assert_same_observation(&timed.observation, &o1);

    // The timeout did not change the work's state or revision.
    let after_timeout = h2.observe(&work).await.expect("still observable");
    assert_same_observation(&after_timeout, &o1);

    // Releasing an in-flight wait future — the outer `timeout` drops the inner
    // wait before it can complete — likewise does not change the work.
    let released = tokio::time::timeout(
        Duration::from_millis(40),
        h2.wait(&work, Duration::from_secs(30)),
    )
    .await;
    assert!(
        released.is_err(),
        "the outer timeout must preempt the inner long wait"
    );
    assert_same_observation(&h1.observe(&work).await.unwrap(), &o1);

    // Dropping the owner fires the running work's stop token; the surviving
    // handle then observes the terminal state — `ReachedState`, not `TimedOut`.
    drop(session);
    let reached = h2
        .wait(&work, Duration::from_secs(5))
        .await
        .expect("a retained work is readable through a surviving handle");
    assert_eq!(reached.end, WaitEnd::ReachedState);
    assert_eq!(reached.observation.state, WorkState::Finished);
    match reached.observation.finished {
        Some(FinishedKind::Interrupted { cause, .. }) => {
            assert_eq!(cause, TurnInterruption::ExplicitCancellation);
        }
        other => panic!("expected an interrupted work, got {other:?}"),
    }
}

// ---- terminal shapes --------------------------------------------------------------------

#[tokio::test]
async fn completed_commits_into_history_and_the_next_work_sees_it() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(endturn_output("first")),
        Ok(endturn_output("second")),
    ]);
    let session = session_with("hist", gateway.clone());
    let handle = session.handle();

    let r1 = handle.submit(req("a", "alpha-input")).await.unwrap();
    let w1 = handle.wait(&r1.work, Duration::from_secs(5)).await.unwrap();
    assert_eq!(w1.end, WaitEnd::ReachedState);
    assert_eq!(w1.observation.state, WorkState::Finished);
    match &w1.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "first");
        }
        other => panic!("expected a completed work, got {other:?}"),
    }

    // A completed turn leaves the session idle, so the next submit is
    // accepted and — never reusing an identity — gets a fresh `TurnId`.
    let r2 = handle.submit(req("b", "beta-input")).await.unwrap();
    assert_ne!(
        r1.work.turn_id, r2.work.turn_id,
        "a later work must not reuse an earlier work's TurnId"
    );
    let w2 = handle.wait(&r2.work, Duration::from_secs(5)).await.unwrap();
    assert!(matches!(
        w2.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));

    // History grew: the second work's model frame carries the committed first
    // turn alongside its own active input.
    let frames = frames_debug(&gateway);
    assert_eq!(frames.len(), 2, "one model call per completed work");
    assert!(
        frames[0].contains("alpha-input"),
        "the first work's own input reached the model: {frames:?}"
    );
    assert!(
        frames[1].contains("alpha-input"),
        "the committed first turn must appear in the next work's frame: {frames:?}"
    );
    assert!(
        frames[1].contains("beta-input"),
        "the second work's active input is in its frame: {frames:?}"
    );

    // The earlier completed work stays observable after the later one ran.
    let still = handle.observe(&r1.work).await.unwrap();
    assert_eq!(still.state, WorkState::Finished);
    assert!(matches!(
        still.finished,
        Some(FinishedKind::Completed { .. })
    ));
}

#[tokio::test]
async fn interrupted_keeps_its_cause_stays_out_of_history_and_allocates_a_new_turn_id() {
    // The first model call fails non-retryably → the turn is `Interrupted`
    // with a retained cause; the second work completes normally.
    let gateway = RecordingGateway::scripted(vec![
        Err(ModelInvokeErrorKind::Permanent),
        Ok(endturn_output("second")),
    ]);
    let session = session_with("abort", gateway.clone());
    let handle = session.handle();

    let r1 = handle.submit(req("a", "aborted-input")).await.unwrap();
    let w1 = handle.wait(&r1.work, Duration::from_secs(5)).await.unwrap();
    assert_eq!(w1.end, WaitEnd::ReachedState);
    assert_eq!(w1.observation.state, WorkState::Finished);
    match &w1.observation.finished {
        Some(FinishedKind::Interrupted { cause, facts }) => {
            assert!(matches!(
                cause,
                TurnInterruption::RetryExhausted {
                    last_kind: ModelInvokeErrorKind::Permanent,
                    ..
                }
            ));
            // The real aborted facts are carried publicly — the aborted
            // turn's identity and its committed input survive, outside the
            // completed-only history.
            assert_eq!(
                facts.turn_id, r1.work.turn_id,
                "the retained facts belong to the interrupted work's turn"
            );
            assert!(
                !facts.blocks.as_slice().is_empty(),
                "the aborted turn's real facts must not be empty"
            );
        }
        other => panic!("expected an interrupted work, got {other:?}"),
    }
    assert!(
        w1.observation.fault.is_none(),
        "an interruption is not a fault"
    );

    // The aborted work left the slot idle: the next submit is accepted and
    // gets a NEW `TurnId` — the interrupted identity is never replayed.
    let r2 = handle.submit(req("b", "second-input")).await.unwrap();
    assert_ne!(
        r1.work.turn_id, r2.work.turn_id,
        "the next work must receive a new TurnId, never the interrupted one"
    );
    let w2 = handle.wait(&r2.work, Duration::from_secs(5)).await.unwrap();
    assert!(matches!(
        w2.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));

    // "Stays out of completed history": the next work's merged frame carries
    // only its own input, never the aborted turn's facts.
    let frames = frames_debug(&gateway);
    assert_eq!(frames.len(), 2);
    assert!(
        frames[0].contains("aborted-input"),
        "the aborted turn's facts were real and reached the model: {frames:?}"
    );
    assert!(
        !frames[1].contains("aborted-input"),
        "the interrupted turn must not enter completed history: {frames:?}"
    );
    assert!(
        frames[1].contains("second-input"),
        "the next work's own input is in its frame: {frames:?}"
    );

    // The interrupted result stays observable — no automatic retry of the work.
    let still = handle.observe(&r1.work).await.unwrap();
    assert_eq!(still.state, WorkState::Finished);
    match &still.finished {
        Some(FinishedKind::Interrupted { cause, facts }) => {
            assert!(matches!(cause, TurnInterruption::RetryExhausted { .. }));
            assert!(
                !facts.blocks.as_slice().is_empty(),
                "the retained interruption keeps real facts across reads"
            );
        }
        other => panic!("expected the retained interruption, got {other:?}"),
    }
}

#[tokio::test]
async fn panicking_gateway_is_faulted_and_the_session_refuses_new_work() {
    let session = session_with("fault", Arc::new(PanickingGateway));
    let handle = session.handle();

    let r = handle.submit(req("a", "boom-input")).await.unwrap();
    let w = handle
        .wait(&r.work, Duration::from_secs(5))
        .await
        .expect("a faulted work is observable, never a permanent Running");
    assert_eq!(w.end, WaitEnd::ReachedState);
    assert_eq!(w.observation.state, WorkState::Faulted);
    let fault = w
        .observation
        .fault
        .clone()
        .expect("a fault publishes its reason");
    assert!(
        fault.contains("panicked"),
        "the fault reason names the panic: {fault}"
    );
    assert!(
        w.observation.finished.is_none(),
        "a fault must not fake a finished kind"
    );

    // `observe` reads back the same fault.
    let obs = handle.observe(&r.work).await.unwrap();
    assert_eq!(obs.state, WorkState::Faulted);
    assert_eq!(obs.fault.as_deref(), Some(fault.as_str()));

    // The session now refuses new work, carrying the retained reason.
    match handle.submit(req("b", "later")).await {
        Err(SessionError::Faulted { reason }) => assert_eq!(reason, fault),
        other => panic!("expected SessionError::Faulted, got {other:?}"),
    }
}

#[tokio::test]
async fn dropping_the_owner_closes_submission_and_cancels_the_running_work() {
    let gateway = GatedGateway::new();
    let session = session_with("stop", gateway.clone());
    let handle = session.handle();

    let r = handle.submit(req("a", "hold")).await.unwrap();
    gateway.wait_entered().await;
    assert_eq!(
        handle.observe(&r.work).await.unwrap().state,
        WorkState::Running,
        "the work is parked inside the model call"
    );

    // Owner drop: acceptance stops at once, and the running work's token fires.
    drop(session);
    match handle.submit(req("b", "later")).await {
        Err(SessionError::Closed) => {}
        other => panic!("expected SessionError::Closed after the owner dropped, got {other:?}"),
    }

    // The cancelled worker ends `Interrupted`, readable through the surviving
    // handle — drop signals stop, it does not lose the terminal publish.
    let w = handle
        .wait(&r.work, Duration::from_secs(5))
        .await
        .expect("a retained work is readable after the owner dropped");
    assert_eq!(w.end, WaitEnd::ReachedState);
    assert_eq!(w.observation.state, WorkState::Finished);
    match w.observation.finished {
        Some(FinishedKind::Interrupted { cause, facts }) => {
            assert_eq!(cause, TurnInterruption::ExplicitCancellation);
            assert!(
                !facts.blocks.as_slice().is_empty(),
                "a cancelled work still carries its real partial facts"
            );
        }
        other => panic!("expected an interrupted work after cancellation, got {other:?}"),
    }
}

// ---- Paused is reachable, observable, and keeps the conversation busy ---

#[tokio::test]
async fn paused_is_observable_and_keeps_the_conversation_busy() {
    let gateway = RecordingGateway::scripted(vec![Ok(tooluse_output(
        "approval?",
        "echo",
        serde_json::json!({}),
    ))]);
    let options = TurnRunOptions {
        interaction: Arc::new(PausingInteraction),
        ..Default::default()
    };
    let session = Session::new(
        ConversationState::new(ConversationId("paused".into())),
        Arc::new(runner_with(gateway.clone(), vec![Arc::new(EchoTool)])),
        options,
        SessionConfig::default(),
    )
    .expect("an empty state is an idle session base");
    let handle = session.handle();

    let receipt = handle
        .submit(req("p", "approve me"))
        .await
        .expect("an idle session accepts the work");
    let waited = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(
        waited.end,
        WaitEnd::ReachedState,
        "Paused is a returnable state"
    );
    assert_eq!(waited.observation.state, WorkState::Paused);
    assert!(
        waited.observation.finished.is_none(),
        "Paused retains the complete outcome, not a FinishedKind"
    );
    assert!(waited.observation.fault.is_none(), "a pause is not a fault");

    // The paused work still owns the conversation: a fresh-key submit is Busy
    // naming exactly the paused ref — no queue, no steering, no implicit
    // approval.
    let busy = handle
        .submit(req("q", "later"))
        .await
        .expect_err("a paused work keeps the conversation busy");
    assert_eq!(
        busy,
        SessionError::Busy {
            active: receipt.work.clone()
        }
    );

    // The pause persists across a second wait, and no second model call ran.
    let again = handle
        .wait(&receipt.work, Duration::from_millis(200))
        .await
        .expect("the paused work stays observable");
    assert_eq!(again.end, WaitEnd::ReachedState);
    assert_eq!(again.observation.state, WorkState::Paused);
    assert_eq!(again.observation.work, receipt.work);
    assert_eq!(
        gateway.recorded().len(),
        1,
        "a pause must not run the model again"
    );
}

// ---- the accept-time work_deadline is wired into RunControl ----

#[tokio::test]
async fn work_deadline_interrupts_a_slow_model_round() {
    let config = SessionConfig {
        retained_work_capacity: 256,
        work_deadline: Some(Duration::from_millis(50)),
    };
    let session = Session::new(
        ConversationState::new(ConversationId("deadline".into())),
        Arc::new(runner_with(Arc::new(SlowGateway), vec![Arc::new(EchoTool)])),
        TurnRunOptions::default(),
        config,
    )
    .expect("an empty state is an idle session base");
    let handle = session.handle();

    let receipt = handle
        .submit(req("d", "slow"))
        .await
        .expect("an idle session accepts the work");
    let waited = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(waited.end, WaitEnd::ReachedState);
    assert_eq!(waited.observation.state, WorkState::Finished);
    match waited.observation.finished {
        Some(FinishedKind::Interrupted { cause, facts }) => {
            assert_eq!(
                cause,
                TurnInterruption::TurnDeadlineExceeded,
                "the accept-time deadline is the specific interruption cause"
            );
            assert!(
                !facts.blocks.as_slice().is_empty(),
                "the slow round's committed facts are retained"
            );
        }
        other => panic!("expected a deadline interruption, got {other:?}"),
    }
    assert!(
        waited.observation.fault.is_none(),
        "a deadline is not a fault"
    );
}

// ---- foreign / never-accepted refs resolve to NotFound -------------------

#[tokio::test]
async fn unknown_and_foreign_refs_are_not_found() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let session = session_with("nf", gateway.clone());
    let handle = session.handle();

    // (a) Same conversation id, a turn id that was never accepted.
    let never = WorkRef {
        conversation_id: ConversationId("nf".into()),
        turn_id: TurnId::new("never-accepted"),
    };
    assert!(
        matches!(handle.observe(&never).await, Err(SessionError::NotFound(ref w)) if *w == never),
        "an unknown ref is NotFound"
    );
    assert!(
        matches!(handle.wait(&never, Duration::from_millis(10)).await, Err(SessionError::NotFound(ref w)) if *w == never),
        "waiting an unknown ref is NotFound, not a timeout"
    );

    // (b) A ref whose conversation_id differs from the session's — even one
    // reusing a real work's turn id — is never routed to the real work.
    let real = handle
        .submit(req("k", "hi"))
        .await
        .expect("an idle session accepts the work");
    let foreign = WorkRef {
        conversation_id: ConversationId("other".into()),
        turn_id: real.work.turn_id.clone(),
    };
    assert!(
        matches!(handle.observe(&foreign).await, Err(SessionError::NotFound(ref w)) if *w == foreign),
        "a foreign conversation ref is NotFound, never routed"
    );
    assert!(
        matches!(handle.wait(&foreign, Duration::from_millis(10)).await, Err(SessionError::NotFound(ref w)) if *w == foreign),
        "waiting a foreign ref is NotFound, never routed"
    );

    // The genuine ref is unaffected and still resolves.
    let observed = handle
        .observe(&real.work)
        .await
        .expect("the genuine ref stays valid");
    assert_eq!(observed.work, real.work);
    assert_ne!(observed.work, foreign);
}
