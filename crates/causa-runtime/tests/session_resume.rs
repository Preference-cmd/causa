//! Resume acceptance for a paused session work: revision and request
//! validation, continuation (not restart), request-key dedup, foreign refs,
//! and the deadline that includes pause time.
//!
//! Reuses the shared `common` fixtures and adds one local fixture: a counting
//! `echo` tool (so "the tool ran once" is observable). Every test runs
//! offline.

mod common;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use causa_kernel::{
    CallControl, ContextFrame, ConversationId, FrameScope, TextPayload, Tool, ToolCallContext,
    ToolDefinition, ToolOutput, ToolResultPayload, ToolResultStatus, Truncation, TurnId,
};
use causa_runtime::{
    ConversationState, FinishedKind, Session, SessionConfig, SessionError, SessionHandle,
    TurnInterruption, TurnRunOptions, WaitEnd, WorkRef, WorkState,
};
use common::{
    EchoTool, RecordingGateway, SlowGateway, approve, awaiting_echo, endturn_output,
    pausing_session, runner_with, session_req, submit_to_pause, tooluse_output,
};

// ---- local fixtures --------------------------------------------------------

/// The shared `echo` tool plus an execution counter, so a test can prove the
/// tool ran exactly once across a pause and its resume.
struct CountingEcho {
    executions: Arc<AtomicUsize>,
}

impl CountingEcho {
    fn new() -> (Arc<Self>, Arc<AtomicUsize>) {
        let executions = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(Self {
                executions: Arc::clone(&executions),
            }),
            executions,
        )
    }
}

#[async_trait]
impl Tool for CountingEcho {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echo".into(),
            parameters: serde_json::json!({"type":"object"}),
        }
    }

    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
        self.executions.fetch_add(1, Ordering::SeqCst);
        ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput {
                content: serde_json::json!({"echo": ctx.arguments}),
                truncation: Truncation::None,
                meta: None,
                artifact: None,
            },
            media: Vec::new(),
        }
    }
}

// ---- helpers ---------------------------------------------------------------

/// The committed-history length a merged conversation frame reports: the
/// distinct turns among the frame's blocks that are not its active turn.
/// Merged frames carry history blocks first (`merged_frame`), so this is
/// exactly `ConversationState::history_len` as the model saw it.
fn history_len_of(frame: &ContextFrame) -> usize {
    let active = match &frame.scope {
        FrameScope::Conversation { active_turn_id, .. } => active_turn_id,
        FrameScope::Turn { turn_id, .. } => turn_id,
    };
    let mut turns = HashSet::new();
    for block in &frame.model_context.blocks {
        if &block.id.turn_id != active {
            turns.insert(block.id.turn_id.clone());
        }
    }
    turns.len()
}

/// Run one throwaway probe work over an idle session and read the committed
/// history length from the probe's merged frame.
async fn history_len_seen_by_probe(
    handle: &SessionHandle,
    gateway: &RecordingGateway,
    key: &str,
) -> usize {
    let probe = handle
        .submit(session_req(key, "probe"))
        .expect("a finished resume leaves the session idle");
    let done = handle
        .wait(&probe.work, Duration::from_secs(5))
        .await
        .expect("the probe is observable");
    assert_eq!(done.observation.state, WorkState::Finished);
    let frames = gateway.frames();
    history_len_of(frames.last().expect("the probe made a model call"))
}

// ---- validation happens before any execution ------------------------------

/// Rewrites runtime-generated arguments before the approval pause; the host
/// must use the observed prepared calls rather than the model's draft.
struct RewriteApproval;

#[async_trait]
impl causa_runtime::ToolUseHook for RewriteApproval {
    async fn apply(
        &self,
        mut calls: Vec<causa_kernel::ToolCallPayload>,
        ctx: &causa_runtime::HookCtx<'_>,
    ) -> causa_runtime::HookOutcome {
        for call in &mut calls {
            call.arguments = serde_json::json!({
                "hook_round": ctx.round_id.0,
                "draft": call.arguments,
            });
        }
        causa_runtime::HookOutcome::passthrough(calls)
    }
}

#[tokio::test]
async fn observed_pause_drives_dynamic_approval_and_keeps_revision_isolation() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output(
            "first",
            "echo",
            serde_json::json!({"path":"a.rs"}),
        )),
        Ok(tooluse_output(
            "second",
            "echo",
            serde_json::json!({"path":"b.rs"}),
        )),
        Ok(endturn_output("done")),
    ]);
    let (echo, executions) = CountingEcho::new();
    let runner = causa_runtime::TurnRunner::with_hook(
        gateway.clone(),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![echo])),
        Arc::new(RewriteApproval),
    );
    let session = Session::new(
        ConversationState::new(ConversationId("observed-approval".into())),
        Arc::new(runner),
        TurnRunOptions {
            interaction: Arc::new(common::PausingInteraction),
            ..Default::default()
        },
        SessionConfig::default(),
    )
    .unwrap();
    let handle = session.handle();
    let (receipt, first) = submit_to_pause(&handle, "submit").await;
    let Some(causa_runtime::PausePoint::AwaitingApproval { mut prepared, .. }) = first.paused
    else {
        panic!("the observation must describe the approval pause");
    };
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert_eq!(prepared.awaiting[0].arguments["hook_round"], 0);
    let first_calls = prepared.awaiting.clone();
    prepared.awaiting[0].arguments = serde_json::json!({"tampered":true});
    let Some(causa_runtime::PausePoint::AwaitingApproval {
        prepared: retained, ..
    }) = handle.observe(&receipt.work).unwrap().paused
    else {
        panic!("the pause is still observable");
    };
    assert_eq!(
        retained.awaiting, first_calls,
        "observation copies cannot edit the pause"
    );
    handle
        .resume(
            &receipt.work,
            first.revision,
            "approve-first".into(),
            approve(retained.awaiting),
        )
        .unwrap();
    let second = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .unwrap()
        .observation;
    assert!(second.revision > first.revision);
    let Some(causa_runtime::PausePoint::AwaitingApproval { prepared, .. }) = second.paused else {
        panic!("the next batch has its own pause");
    };
    assert_ne!(prepared.awaiting[0].call_id, first_calls[0].call_id);
    assert_eq!(prepared.awaiting[0].arguments["hook_round"], 1);
    let decision = approve(prepared.awaiting);
    assert!(matches!(
        handle.resume(
            &receipt.work,
            first.revision,
            "approve-second".into(),
            decision.clone()
        ),
        Err(SessionError::StaleRevision { .. })
    ));
    handle
        .resume(
            &receipt.work,
            second.revision,
            "approve-second".into(),
            decision,
        )
        .unwrap();
    let done = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .unwrap()
        .observation;
    assert!(matches!(
        done.finished,
        Some(FinishedKind::Completed { .. })
    ));
    assert!(done.paused.is_none());
    assert_eq!(executions.load(Ordering::SeqCst), 2);
    assert_eq!(gateway.recorded().len(), 3);
    session.shutdown().await;
}

#[tokio::test]
async fn resume_with_wrong_revision_is_rejected_untouched() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let session = pausing_session(
        "stale",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    let wrong = paused.revision + 7;
    let err = handle
        .resume(
            &receipt.work,
            wrong,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect_err("a revision that is not the paused revision is stale");
    match err {
        SessionError::StaleRevision {
            work,
            expected,
            actual,
        } => {
            assert_eq!(work, receipt.work);
            assert_eq!(expected, wrong);
            assert_eq!(actual, paused.revision);
        }
        other => panic!("expected StaleRevision, got {other:?}"),
    }

    // The rejected resume executed nothing: the work is still Paused and the
    // model was called exactly once (the original round that paused).
    let still = handle.observe(&receipt.work).unwrap();
    assert_eq!(still.state, WorkState::Paused);
    assert_eq!(
        gateway.recorded().len(),
        1,
        "a stale-revision resume must not reach the model"
    );
}

#[tokio::test]
async fn resume_with_uncovering_decision_is_rejected() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let session = pausing_session(
        "uncover",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    // The right revision, but a decision that omits the awaiting call.
    let err = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(Vec::new()),
        )
        .expect_err("a decision that does not cover the awaiting call is rejected");
    assert!(
        matches!(err, SessionError::InvalidResume(_)),
        "expected InvalidResume, got {err:?}"
    );

    let still = handle.observe(&receipt.work).unwrap();
    assert_eq!(still.state, WorkState::Paused);
    assert_eq!(
        gateway.recorded().len(),
        1,
        "an invalid resume must not reach the model"
    );
}

// ---- a valid resume continues the same turn -------------------------------

#[tokio::test]
async fn resume_continues_the_paused_turn() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
        Ok(endturn_output("probe")),
    ]);
    let session = pausing_session(
        "continue",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    let resumed = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("the paused revision and a covering decision resume");
    assert_eq!(
        resumed.work, receipt.work,
        "a resume continues the same work identity"
    );

    let done = handle
        .wait(&resumed.work, Duration::from_secs(5))
        .await
        .expect("the resumed work is observable");
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    match &done.observation.finished {
        Some(FinishedKind::Completed { final_output }) => {
            assert_eq!(final_output.response.text.0, "done");
        }
        other => panic!("expected a completed resume, got {other:?}"),
    }

    // Continuation, not restart: the original paused round was resumed (one
    // more model call), not re-run from round 0.
    assert_eq!(
        gateway.recorded().len(),
        2,
        "a resume continues the paused turn: exactly one more model call"
    );

    // The turn committed once: a probe work's merged frame sees exactly one
    // committed history entry.
    assert_eq!(
        history_len_seen_by_probe(&handle, &gateway, "p1").await,
        1,
        "the resumed turn committed exactly once"
    );
}

#[tokio::test]
async fn repeated_resume_key_does_not_reexecute() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let session = pausing_session(
        "dedup",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    let original = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("the paused revision and a covering decision resume");

    // Same key and revision: resolves to the original receipt, never a second
    // execution.
    let retried = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("the same key and revision resolve to the original receipt");
    assert_eq!(retried, original);

    let done = handle
        .wait(&original.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    assert_eq!(
        gateway.recorded().len(),
        2,
        "the deduped resume must not add a model call"
    );
}

#[tokio::test]
async fn resume_with_a_new_key_after_completion_is_not_paused() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let session = pausing_session(
        "notpaused",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    let resumed = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("the paused revision and a covering decision resume");
    let done = handle
        .wait(&resumed.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);

    // The work is terminal and the session idle, so a fresh key cannot resume
    // it — the state implies NotPaused.
    let err = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r2".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect_err("a finished work is not resumable");
    match err {
        SessionError::NotPaused(work) => assert_eq!(work, receipt.work),
        SessionError::Busy { active } => assert_eq!(active, receipt.work),
        other => panic!("expected NotPaused (or Busy), got {other:?}"),
    }
    assert_eq!(
        gateway.recorded().len(),
        2,
        "the rejected resume must not run the model again"
    );
}

#[tokio::test]
async fn resume_foreign_ref_is_not_found() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let session = pausing_session(
        "foreign",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    // Same turn id, a different conversation: never routed to the real work.
    let foreign = WorkRef {
        conversation_id: ConversationId("other".into()),
        turn_id: receipt.work.turn_id.clone(),
    };
    let err = handle
        .resume(
            &foreign,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect_err("a foreign conversation ref is unknown");
    assert!(
        matches!(err, SessionError::NotFound(ref w) if *w == foreign),
        "expected NotFound for the foreign ref, got {err:?}"
    );

    // The genuine paused work is untouched.
    assert_eq!(
        handle.observe(&receipt.work).unwrap().state,
        WorkState::Paused
    );
    assert_eq!(gateway.recorded().len(), 1);
}

// ---- the deadline includes pause time --------------------------------------

#[tokio::test]
async fn work_deadline_includes_pause_time() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let (tool, executions) = CountingEcho::new();
    let config = SessionConfig {
        retained_work_capacity: 256,
        work_deadline: Some(Duration::from_millis(150)),
    };
    let session = pausing_session("deadline-pause", gateway.clone(), vec![tool], config);
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    // Sleep past the accept-time deadline while the work is paused.
    tokio::time::sleep(Duration::from_millis(250)).await;

    // The resume is accepted (validation is not a deadline check), but the
    // expired deadline dispatches nothing.
    let resumed = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("a stale deadline does not reject the resume request itself");
    let done = handle
        .wait(&resumed.work, Duration::from_secs(5))
        .await
        .expect("the resumed work is observable");
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    match done.observation.finished {
        Some(FinishedKind::Interrupted { cause, .. }) => assert_eq!(
            cause,
            TurnInterruption::TurnDeadlineExceeded,
            "the deadline measured from acceptance still bounds the resume"
        ),
        other => panic!("expected a deadline interruption, got {other:?}"),
    }

    assert_eq!(
        executions.load(Ordering::SeqCst),
        0,
        "the expired deadline dispatched no tool call"
    );
    assert_eq!(
        gateway.recorded().len(),
        1,
        "the expired deadline made no further model call"
    );
}

#[tokio::test]
async fn wait_does_not_extend_the_deadline() {
    let config = SessionConfig {
        retained_work_capacity: 256,
        work_deadline: Some(Duration::from_millis(50)),
    };
    let session = Session::new(
        ConversationState::new(ConversationId("deadline-wait".into())),
        Arc::new(runner_with(Arc::new(SlowGateway), vec![Arc::new(EchoTool)])),
        TurnRunOptions::default(),
        config,
    )
    .expect("an idle state is a valid session base");
    let handle = session.handle();

    let receipt = handle
        .submit(session_req("s1", "slow"))
        .expect("an idle session accepts the work");

    // A finite wait times out while the work is still inside its slow model
    // call — timing out grants no extra time.
    let timed = handle
        .wait(&receipt.work, Duration::from_millis(20))
        .await
        .expect("the accepted work is observable");
    assert_eq!(timed.end, WaitEnd::TimedOut);
    assert!(
        matches!(
            timed.observation.state,
            WorkState::Accepted | WorkState::Running
        ),
        "the timed-out wait sees a non-terminal work, got {:?}",
        timed.observation.state
    );

    // The accept-time deadline still wins: the work ends
    // Interrupted { TurnDeadlineExceeded }.
    let done = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the work is observable");
    assert_eq!(done.end, WaitEnd::ReachedState);
    assert_eq!(done.observation.state, WorkState::Finished);
    match done.observation.finished {
        Some(FinishedKind::Interrupted { cause, .. }) => assert_eq!(
            cause,
            TurnInterruption::TurnDeadlineExceeded,
            "the wait must not extend the accept-time deadline"
        ),
        other => panic!("expected a deadline interruption, got {other:?}"),
    }
}

// ---- the tool count is consumed, not re-derived ----------------------------
#[tokio::test]
async fn resume_keeps_the_consumed_tool_count() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
        Ok(endturn_output("probe")),
    ]);
    let (tool, executions) = CountingEcho::new();
    let session = pausing_session(
        "toolcount",
        gateway.clone(),
        vec![tool],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    let resumed = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("the paused revision and a covering decision resume");
    let done = handle
        .wait(&resumed.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    assert!(matches!(
        done.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));

    // Once across the pause and its resume: the count continues from the
    // continuation rather than being re-derived, and no restart re-runs the
    // paused round.
    assert_eq!(
        executions.load(Ordering::SeqCst),
        1,
        "the consumed tool call must not run again on resume"
    );
    assert_eq!(
        gateway.recorded().len(),
        2,
        "a resumed turn makes one more model call, never a restart"
    );
    assert_eq!(
        history_len_seen_by_probe(&handle, &gateway, "p1").await,
        1,
        "the resumed turn committed once, not twice"
    );
}

// ---- the dedup key covers the whole request, not just work + revision -----

#[tokio::test]
async fn resume_same_key_different_decision_conflicts() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let session = pausing_session(
        "conflict-decision",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    let original = handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("the first resume is accepted");

    // Same key, same work and revision, but an extra injection: a different
    // request, so the retry is a key-reuse bug rather than a lost receipt.
    let mut other = approve(vec![awaiting_echo()]);
    other.inject = vec![TextPayload::new("late addition")];
    let err = handle
        .resume(&receipt.work, paused.revision, "r1".into(), other)
        .expect_err("the same key with a different request is a conflict");
    assert!(
        matches!(err, SessionError::Conflict),
        "expected Conflict, got {err:?}"
    );

    // The conflict executed nothing: the original resume still owns the work
    // and completes it exactly once.
    let done = handle
        .wait(&original.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    assert!(matches!(
        done.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));
    assert_eq!(
        gateway.recorded().len(),
        2,
        "the conflicting retry must not add a model call"
    );
}

#[tokio::test]
async fn resume_same_key_for_another_work_conflicts_before_not_found() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let session = pausing_session(
        "conflict-work",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;
    handle
        .resume(
            &receipt.work,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("the first resume is accepted");

    // The key is taken even though the named work was never accepted: dedup
    // resolves before the ref lookup, so a reused key can never be mistaken
    // for a fresh request on another work.
    let unknown = WorkRef {
        conversation_id: ConversationId("conflict-work".into()),
        turn_id: TurnId::new("never-accepted"),
    };
    let err = handle
        .resume(
            &unknown,
            paused.revision,
            "r1".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect_err("a reused key is a conflict, not a new lookup");
    assert!(
        matches!(err, SessionError::Conflict),
        "expected Conflict, got {err:?}"
    );
}

#[tokio::test]
async fn submit_and_resume_keys_live_in_separate_namespaces() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
    ]);
    let session = pausing_session(
        "namespaces",
        gateway.clone(),
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "k").await;

    // The same key string under the resume operation is a fresh key, not a
    // conflict with the submit that used it.
    let resumed = handle
        .resume(
            &receipt.work,
            paused.revision,
            "k".into(),
            approve(vec![awaiting_echo()]),
        )
        .expect("per-operation namespaces keep the same key string distinct");
    let done = handle
        .wait(&resumed.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
}

// ---- a resume may inject new inputs before the next round -------------------

#[tokio::test]
async fn resume_with_inject_appends_before_the_next_round() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("approval?", "echo", serde_json::json!({}))),
        Ok(endturn_output("done")),
        Ok(endturn_output("probe")),
    ]);
    let (tool, executions) = CountingEcho::new();
    let session = pausing_session(
        "inject",
        gateway.clone(),
        vec![tool],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, paused) = submit_to_pause(&handle, "s1").await;

    let mut request = approve(vec![awaiting_echo()]);
    request.inject = vec![TextPayload::new("extra-note")];
    let resumed = handle
        .resume(&receipt.work, paused.revision, "r1".into(), request)
        .expect("a covering decision with an injection resumes");
    let done = handle
        .wait(&resumed.work, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(done.observation.state, WorkState::Finished);
    assert!(matches!(
        done.observation.finished,
        Some(FinishedKind::Completed { .. })
    ));

    // The injected input reached the next model round, and the approved tool
    // still ran exactly once.
    assert_eq!(gateway.recorded().len(), 2);
    let second = format!("{:?}", gateway.recorded()[1].frame);
    assert!(
        second.contains("extra-note"),
        "the injection must reach the next round's frame: {second}"
    );
    assert_eq!(
        executions.load(Ordering::SeqCst),
        1,
        "the approved call runs once across the injecting resume"
    );
    assert_eq!(
        history_len_seen_by_probe(&handle, &gateway, "p1").await,
        1,
        "the injecting resume committed exactly once"
    );
}
