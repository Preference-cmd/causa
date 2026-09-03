//! Recoverable-interruption tests (Slice 7): the pause gate
//! (`TurnInteraction::decide_batch` → `TurnResult::Paused`), the resume
//! paths (`TurnRunner::resume` / `resume_turn` with the withheld
//! `HookOutcome` or steering injection), and the round-boundary steering
//! pull (`TurnInteraction::pending_inputs`).

mod common;

use causa_kernel::{
    BatchDecision, ConversationError, ConversationId, ConversationState, RoundId, SealedResult,
    TextPayload, ToolCallPayload, ToolExecutionOutcome, ToolOutput, ToolResultPayload,
    ToolResultStatus, TurnContext, TurnId, TurnInteraction,
};
use causa_runtime::{
    HookOutcome, PausedReason, ResumeRequest, TurnOutcome, TurnResult, TurnRunOptions, TurnRunner,
    TurnTrace, resume_turn,
};
use common::{
    EchoTool, RecordingGateway, ctrl, endturn_output, options_with_limits, runner_with,
    tooluse_output,
};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The pause gate: every batch pauses (deadline-less).
struct PauseOnBatch;
#[async_trait::async_trait]
impl TurnInteraction for PauseOnBatch {
    async fn decide_batch(&self, _calls: &[ToolCallPayload]) -> BatchDecision {
        BatchDecision::Pause { deadline: None }
    }
}

/// Steering pull that yields one input on the first call, then nothing.
/// Counts the pulls so tests can assert the boundary behavior.
struct OneShotSteering {
    pulls: AtomicUsize,
}
#[async_trait::async_trait]
impl TurnInteraction for OneShotSteering {
    async fn pending_inputs(&self) -> Vec<TextPayload> {
        if self.pulls.fetch_add(1, Ordering::SeqCst) == 0 {
            vec![TextPayload::new("focus on the config file")]
        } else {
            vec![]
        }
    }
}

fn pause_options() -> TurnRunOptions {
    let mut options = options_with_limits(5, 10);
    options.interaction = Arc::new(PauseOnBatch);
    options
}

fn plain_options() -> TurnRunOptions {
    options_with_limits(5, 10)
}

/// Round 0 emits a tool call (the pause lands in its tool phase); round 1
/// completes after resume.
fn paused_runner() -> (TurnRunner, Arc<RecordingGateway>) {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("calling echo", "echo", json!({"a": 1}))),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gateway.clone(), vec![Arc::new(EchoTool)]);
    (runner, gateway)
}

fn seeded_state() -> ConversationState {
    let mut state = ConversationState::new(ConversationId("conv-7".into()));
    state.begin_turn(TurnId::new("t1")).unwrap();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    state
}

async fn run_to_pause(runner: &TurnRunner) -> TurnOutcome {
    let mut ctx = TurnContext::new(TurnId::new("t1"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();
    let out = runner.run(ctx, pause_options(), ctrl()).await;
    assert!(
        matches!(out.result, TurnResult::Paused { .. }),
        "expected Paused, got {:?}",
        out.result
    );
    out
}

/// Split a paused outcome into (context, reason, trace).
fn decompose_pause(out: TurnOutcome) -> (TurnContext, PausedReason, TurnTrace) {
    match out.result {
        TurnResult::Paused { reason, .. } => (out.context, reason, out.trace),
        other => panic!("expected Paused, got {other:?}"),
    }
}

fn expect_pending_calls(reason: &PausedReason) -> Vec<ToolCallPayload> {
    match reason {
        PausedReason::AwaitingApproval { pending_calls, .. } => pending_calls.clone(),
        other => panic!("expected AwaitingApproval, got {other:?}"),
    }
}

fn text_facts(context: &TurnContext) -> Vec<String> {
    context
        .blocks()
        .iter()
        .filter_map(|b| match &b.content {
            causa_kernel::BlockContent::Text(t) => Some(t.0.clone()),
            _ => None,
        })
        .collect()
}

fn tool_result_fact(context: &TurnContext) -> causa_kernel::ToolResultPayload {
    context
        .blocks()
        .iter()
        .find_map(|b| match &b.content {
            causa_kernel::BlockContent::ToolResult(r) => Some(r.clone()),
            _ => None,
        })
        .expect("tool result fact")
}

// ---- the pause gate -------------------------------------------------------------

#[tokio::test]
async fn decide_batch_pause_leaves_the_turn_open() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;

    match &out.result {
        TurnResult::Paused {
            snapshot,
            reason:
                PausedReason::AwaitingApproval {
                    pending_calls,
                    deadline,
                },
        } => {
            assert_eq!(pending_calls.len(), 1);
            assert_eq!(pending_calls[0].tool_name, "echo");
            assert_eq!(*deadline, None);
            assert!(!snapshot.sealed, "the paused snapshot records an open turn");
        }
        other => panic!("expected Paused, got {other:?}"),
    }
    // The context itself is NOT sealed — the turn is still alive.
    assert!(!out.context.is_sealed());
    // The model's tool-call block is a committed fact; no tool result yet.
    assert!(
        out.context
            .blocks()
            .iter()
            .any(|b| matches!(b.content, causa_kernel::BlockContent::ToolCall(_)))
    );
    assert!(
        !out.context
            .blocks()
            .iter()
            .any(|b| matches!(b.content, causa_kernel::BlockContent::ToolResult(_)))
    );
    assert_eq!(gateway.recorded().len(), 1, "no second round after a pause");
    // The paused round is traced; its tool batch is empty (never ran).
    assert_eq!(out.trace.rounds.len(), 1);
    assert!(out.trace.rounds[0].tool_batch.is_none());
}

#[tokio::test]
async fn conversation_pause_stamps_paused_and_rejects_commit() {
    let (runner, _gw) = paused_runner();
    let mut out = runner
        .run_in_conversation(seeded_state(), pause_options(), ctrl())
        .await
        .unwrap();

    assert!(matches!(out.result, TurnResult::Paused { .. }));
    assert!(!out.state.active_turn().unwrap().is_sealed());
    assert_eq!(out.state.sealed_result(), Some(SealedResult::Paused));
    // commit refuses a paused turn; abort clears the slot and the id is
    // reusable.
    assert!(matches!(
        out.state.commit(TurnId::new("t1")),
        Err(ConversationError::TurnPaused(_))
    ));
    let mut state = out.state;
    state.abort_turn(TurnId::new("t1")).unwrap();
    assert_eq!(state.snapshot_count(), 0);
    state.begin_turn(TurnId::new("t1")).unwrap();
}

#[tokio::test]
async fn fresh_run_rejects_a_paused_state() {
    let (runner, _gw) = paused_runner();
    let out = runner
        .run_in_conversation(seeded_state(), pause_options(), ctrl())
        .await
        .unwrap();
    // Fresh entries must not double-run a paused turn.
    assert!(matches!(
        runner
            .run_in_conversation(out.state, plain_options(), ctrl())
            .await,
        Err(ConversationError::TurnAlreadyActive)
    ));
}

// ---- resume: approval decisions reuse HookOutcome -------------------------------

#[tokio::test]
async fn resume_with_passthrough_executes_and_completes() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;
    let (context, reason, trace) = decompose_pause(out);
    let pending = expect_pending_calls(&reason);

    let resumed = runner
        .resume(
            context,
            ResumeRequest {
                pending: reason,
                trace,
                withheld: HookOutcome::passthrough(pending),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await;
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    assert!(resumed.context.is_sealed(), "terminal outcome seals");
    // The tool result is now a fact.
    assert_eq!(
        tool_result_fact(&resumed.context).status,
        ToolResultStatus::Succeeded
    );
    assert_eq!(gateway.recorded().len(), 2);
}

#[tokio::test]
async fn resume_with_reject_records_rejected_results() {
    let (runner, _gw) = paused_runner();
    let out = run_to_pause(&runner).await;
    let (context, reason, trace) = decompose_pause(out);
    let pending = expect_pending_calls(&reason);

    let rejected = ToolExecutionOutcome::new(ToolResultPayload {
        call_id: pending[0].call_id.clone(),
        status: ToolResultStatus::Rejected,
        output: ToolOutput::new(json!({"error": "denied by operator"})),
    });
    let resumed = runner
        .resume(
            context,
            ResumeRequest {
                pending: reason,
                trace,
                withheld: HookOutcome {
                    to_execute: vec![],
                    rejected: vec![rejected],
                },
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await;
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    let rejected_fact = tool_result_fact(&resumed.context);
    assert_eq!(rejected_fact.status, ToolResultStatus::Rejected);
    assert_eq!(rejected_fact.call_id, pending[0].call_id);
}

#[tokio::test]
async fn resume_with_rewrite_executes_the_edited_arguments() {
    let (runner, _gw) = paused_runner();
    let out = run_to_pause(&runner).await;
    let (context, reason, trace) = decompose_pause(out);
    let pending = expect_pending_calls(&reason);

    let mut rewritten = pending[0].clone();
    rewritten.arguments = json!({"a": 1, "approved": true});
    let resumed = runner
        .resume(
            context,
            ResumeRequest {
                pending: reason,
                trace,
                withheld: HookOutcome::passthrough(vec![rewritten]),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await;
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    // The executor ran the rewritten arguments (EchoTool echoes them back).
    assert_eq!(
        tool_result_fact(&resumed.context).output.content,
        json!({"echo": {"a": 1, "approved": true}})
    );
}

#[tokio::test]
async fn resume_appends_steering_injection_before_next_round() {
    let (runner, _gw) = paused_runner();
    let out = run_to_pause(&runner).await;
    let (context, reason, trace) = decompose_pause(out);
    let pending = expect_pending_calls(&reason);

    let resumed = runner
        .resume(
            context,
            ResumeRequest {
                pending: reason,
                trace,
                withheld: HookOutcome::passthrough(pending),
                inject: vec![TextPayload::new("also check the logs")],
            },
            plain_options(),
            ctrl(),
        )
        .await;
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    assert!(
        text_facts(&resumed.context)
            .iter()
            .any(|t| t == "also check the logs"),
        "steering input missing from facts"
    );
}

#[tokio::test]
async fn resumed_trace_appends_rounds_and_keeps_totals() {
    let (runner, _gw) = paused_runner();
    let out = run_to_pause(&runner).await;
    let paused_total = out.trace.tool_calls_total;
    let (context, reason, trace) = decompose_pause(out);
    let pending = expect_pending_calls(&reason);

    let resumed = runner
        .resume(
            context,
            ResumeRequest {
                pending: reason,
                trace,
                withheld: HookOutcome::passthrough(pending),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await;
    // Pause and resume are two phases of one turn: rounds continue, not reset.
    assert_eq!(resumed.trace.rounds.len(), 2);
    assert_eq!(resumed.trace.rounds[0].round_id, RoundId(0));
    assert_eq!(resumed.trace.rounds[1].round_id, RoundId(1));
    // The paused phase had counted the pending batch; the resume must not
    // re-count it, and the executed batch is now real.
    assert_eq!(resumed.trace.tool_calls_total, paused_total);
    assert_eq!(resumed.trace.tool_calls_total, 1);
    // The paused round's tool batch is filled in by the resume prologue.
    assert!(resumed.trace.rounds[0].tool_batch.is_some());
}

// ---- resume: steering reason ----------------------------------------------------

#[tokio::test]
async fn steering_resume_injects_and_continues() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;
    let paused_total = out.trace.tool_calls_total;
    // The host pauses via the approval gate but resumes with a steering
    // reason: no batch executes, the next round continues with the
    // injection.
    let resumed = runner
        .resume(
            out.context,
            ResumeRequest {
                pending: PausedReason::PausedForSteering {
                    queued_inputs: vec![],
                    pending_round_id: RoundId(1),
                },
                trace: out.trace,
                withheld: HookOutcome::passthrough(vec![]),
                inject: vec![TextPayload::new("drop the tool call, just answer")],
            },
            plain_options(),
            ctrl(),
        )
        .await;
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    assert_eq!(gateway.recorded().len(), 2);
    assert_eq!(resumed.trace.tool_calls_total, paused_total);
    // No tool result ever landed: the withheld batch was dropped.
    assert!(
        !resumed
            .context
            .blocks()
            .iter()
            .any(|b| matches!(b.content, causa_kernel::BlockContent::ToolResult(_)))
    );
}

// ---- resume validation ----------------------------------------------------------

#[tokio::test]
async fn conversation_resume_validates_the_paused_stamp() {
    let (runner, _gw) = paused_runner();
    let steering = || ResumeRequest {
        pending: PausedReason::PausedForSteering {
            queued_inputs: vec![],
            pending_round_id: RoundId(0),
        },
        trace: TurnTrace::new(),
        withheld: HookOutcome::passthrough(vec![]),
        inject: vec![],
    };
    // Not paused (no stamp at all) → NotPaused.
    assert!(matches!(
        resume_turn(&runner, seeded_state(), steering(), plain_options(), ctrl()).await,
        Err(ConversationError::NotPaused(_))
    ));
    // Sealed + completed → TurnAlreadySealed.
    let mut state = seeded_state();
    state
        .seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    assert!(matches!(
        resume_turn(&runner, state, steering(), plain_options(), ctrl()).await,
        Err(ConversationError::TurnAlreadySealed)
    ));
}

#[tokio::test]
async fn conversation_resume_path_completes_and_commits() {
    let (runner, _gw) = paused_runner();
    let out = runner
        .run_in_conversation(seeded_state(), pause_options(), ctrl())
        .await
        .unwrap();
    let reason = match &out.result {
        TurnResult::Paused { reason, .. } => reason.clone(),
        other => panic!("expected Paused, got {other:?}"),
    };
    let pending = expect_pending_calls(&reason);
    let resumed = resume_turn(
        &runner,
        out.state,
        ResumeRequest {
            pending: reason,
            trace: out.trace,
            withheld: HookOutcome::passthrough(pending),
            inject: vec![],
        },
        plain_options(),
        ctrl(),
    )
    .await
    .unwrap();
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    // The completed turn commits normally after the resume.
    let mut resumed = resumed;
    let snap = resumed.state.commit(TurnId::new("t1")).unwrap();
    assert_eq!(snap.turn_sequence.0, 0);
    assert_eq!(resumed.state.snapshot_count(), 1);
}

// ---- steering pull (Phase D) ----------------------------------------------------

#[tokio::test]
async fn pending_inputs_are_pulled_at_round_boundaries() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("calling echo", "echo", json!({"a": 1}))),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gateway, vec![Arc::new(EchoTool)]);
    let steering = Arc::new(OneShotSteering {
        pulls: AtomicUsize::new(0),
    });
    let mut options = plain_options();
    options.interaction = steering.clone();
    let mut ctx = TurnContext::new(TurnId::new("t1"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();

    let out: TurnOutcome = runner.run(ctx, options, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert!(
        steering.pulls.load(Ordering::SeqCst) >= 2,
        "pulled each round"
    );
    // The steering input is a fact the model saw from round 0 on.
    assert!(text_facts(&out.context).contains(&"focus on the config file".to_string()));
}
