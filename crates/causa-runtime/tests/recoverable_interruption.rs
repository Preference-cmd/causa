//! Recoverable-interruption tests (Slice 7, reworked by Slice 6.5): the
//! pause gate (`TurnInteraction::decide_batch` → `TurnResult::Paused`),
//! the resume entries (`TurnRunner::resume` / `resume_turn` consuming the
//! complete paused outcome + `ResumeRequest`), continuation validation,
//! prepared hook work, and the round-boundary steering pull.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use causa_kernel::{
    BlockContent, ContextVersion, ConversationId, InvocationId, RoundId, TextPayload, Tool,
    ToolCallContext, ToolCallDraft, ToolCallId, ToolCallPayload, ToolDefinition, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnContext, TurnId,
};
use causa_runtime::{
    BatchDecision, Continuation, ConversationError, ConversationOutcome, ConversationState,
    HookCtx, HookOutcome, ModelRoundTrace, PausePoint, PreparedApproval, ResumeRequest,
    SealedResult, ToolExecutor, ToolUseHook, TurnInteraction, TurnInterruption, TurnOutcome,
    TurnResult, TurnRunOptions, TurnRunner, TurnTrace, resume_turn,
};
use common::{
    EchoTool, RecordingGateway, ctrl, endturn_output, options_with_limits, runner_with,
    tooluse_calls_output, tooluse_output,
};
use serde_json::json;

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

/// A hook that refuses `danger` calls outright and rewrites every other
/// call's arguments. Counts its applications — the resume must never
/// re-run it.
struct RejectDangerRewriteRest {
    applications: AtomicUsize,
}
#[async_trait::async_trait]
impl ToolUseHook for RejectDangerRewriteRest {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        self.applications.fetch_add(1, Ordering::SeqCst);
        let mut to_execute = Vec::new();
        let mut rejected = Vec::new();
        for mut payload in calls {
            if payload.tool_name == "danger" {
                rejected.push(ToolResultPayload {
                    call_id: payload.call_id.clone(),
                    status: ToolResultStatus::Rejected,
                    output: ToolOutput::new(json!({"error": "hook denied danger"})),
                    media: Vec::new(),
                });
            } else {
                payload.arguments = json!({"hooked": true});
                to_execute.push(payload);
            }
        }
        HookOutcome {
            to_execute,
            rejected,
            unknown_decisions: Vec::new(),
        }
    }
}

/// A tool the hook always refuses — if it ever executes, the test fails
/// loudly through its result.
struct DangerTool;
#[async_trait::async_trait]
impl Tool for DangerTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "danger".into(),
            description: "must never run".into(),
            parameters: json!({"type": "object"}),
        }
    }
    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _c: &causa_kernel::CallControl,
    ) -> ToolResultPayload {
        ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({"boom": true})),
            media: Vec::new(),
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

/// Split a paused outcome into (context, continuation, trace).
fn decompose_pause(out: TurnOutcome) -> (TurnContext, Continuation, TurnTrace) {
    match out.result {
        TurnResult::Paused { continuation } => (out.context, continuation, out.trace),
        other => panic!("expected Paused, got {other:?}"),
    }
}

fn expect_awaiting(continuation: &Continuation) -> Vec<ToolCallPayload> {
    match &continuation.pause_point {
        PausePoint::AwaitingApproval { prepared, .. } => prepared.awaiting.clone(),
        other => panic!("expected AwaitingApproval, got {other:?}"),
    }
}

fn approve(awaiting: Vec<ToolCallPayload>) -> Option<HookOutcome> {
    Some(HookOutcome::passthrough(awaiting))
}

fn text_facts(context: &TurnContext) -> Vec<String> {
    context
        .blocks()
        .iter()
        .flat_map(|b| match &b.content {
            BlockContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    causa_kernel::ContentPart::Text(t) => Some(t.0.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn result_facts(context: &TurnContext) -> Vec<causa_kernel::ToolResultPayload> {
    context
        .blocks()
        .iter()
        .filter_map(|b| match &b.content {
            BlockContent::ToolResult(r) => Some(r.clone()),
            _ => None,
        })
        .collect()
}

/// A minimal observational trace entry for a fabricated checkpoint.
fn round_trace(round: u32, turn: &str) -> ModelRoundTrace {
    ModelRoundTrace {
        round_id: RoundId(round),
        invocation_id: InvocationId {
            turn_id: TurnId::new(turn),
            round_id: RoundId(round),
        },
        frame_version: ContextVersion(1),
        attempts: vec![],
        output_summary: None,
        applied_block_ids: vec![],
        tool_batch: None,
    }
}

// ---- the pause gate -------------------------------------------------------------

#[tokio::test]
async fn decide_batch_pause_leaves_the_turn_open() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;

    match &out.result {
        TurnResult::Paused { continuation } => {
            let PausePoint::AwaitingApproval { prepared, deadline } = &continuation.pause_point
            else {
                panic!(
                    "expected AwaitingApproval, got {:?}",
                    continuation.pause_point
                );
            };
            assert_eq!(prepared.awaiting.len(), 1);
            assert_eq!(prepared.awaiting[0].tool_name, "echo");
            assert!(
                prepared.rejected.is_empty(),
                "passthrough hook rejects nothing"
            );
            assert_eq!(*deadline, None);
            assert_eq!(continuation.round, 0);
            assert_eq!(continuation.accounted_tool_calls, 1);
        }
        other => panic!("expected Paused, got {other:?}"),
    }
    // The context itself is NOT sealed — the turn is still alive — and it
    // is the single fact source: the Paused variant carries no snapshot.
    assert!(!out.context.is_sealed());
    assert!(
        out.context
            .blocks()
            .iter()
            .any(|b| matches!(b.content, BlockContent::ToolCall(_)))
    );
    assert!(
        !out.context
            .blocks()
            .iter()
            .any(|b| matches!(b.content, BlockContent::ToolResult(_)))
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
    assert_eq!(state.history_len(), 0);
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

// ---- resume: the outcome is the checkpoint --------------------------------------

#[tokio::test]
async fn resume_with_passthrough_executes_and_completes() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;

    // An approval pause without a decision is an input error, not a
    // terminal outcome — the material comes back untouched.
    let rejection = runner
        .resume(
            out,
            ResumeRequest {
                decision: None,
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect_err("missing decision must be rejected");
    assert!(rejection.reason.contains("requires a decision"));
    let out = rejection.into_outcome();
    assert!(!out.context.is_sealed());
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };
    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("valid approval");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    assert!(resumed.context.is_sealed(), "terminal outcome seals");
    // The tool result is now a fact.
    assert_eq!(
        result_facts(&resumed.context)[0].status,
        ToolResultStatus::Succeeded
    );
    assert_eq!(gateway.recorded().len(), 2);
}

#[tokio::test]
async fn resume_with_reject_records_rejected_results() {
    let (runner, _gw) = paused_runner();
    let out = run_to_pause(&runner).await;
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };

    let rejected = ToolResultPayload {
        call_id: awaiting[0].call_id.clone(),
        status: ToolResultStatus::Rejected,
        output: ToolOutput::new(json!({"error": "denied by operator"})),
        media: Vec::new(),
    };
    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: Some(HookOutcome {
                    to_execute: vec![],
                    rejected: vec![rejected],
                    unknown_decisions: Vec::new(),
                }),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("valid rejection decision");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    let rejected_fact = &result_facts(&resumed.context)[0];
    assert_eq!(rejected_fact.status, ToolResultStatus::Rejected);
    assert_eq!(rejected_fact.call_id, awaiting[0].call_id);
}

#[tokio::test]
async fn resume_with_rewrite_executes_the_edited_arguments() {
    let (runner, _gw) = paused_runner();
    let out = run_to_pause(&runner).await;
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };

    let mut rewritten = awaiting[0].clone();
    rewritten.arguments = json!({"a": 1, "approved": true});
    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: Some(HookOutcome::passthrough(vec![rewritten])),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("valid rewrite decision");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    // The executor ran the rewritten arguments (EchoTool echoes them back).
    assert_eq!(
        result_facts(&resumed.context)[0].output.content,
        json!({"echo": {"a": 1, "approved": true}})
    );
}

#[tokio::test]
async fn resume_appends_steering_injection_before_next_round() {
    let (runner, _gw) = paused_runner();
    let out = run_to_pause(&runner).await;
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };

    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![TextPayload::new("also check the logs")],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("valid approval with inject");
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
    let accounted = match &out.result {
        TurnResult::Paused { continuation } => continuation.accounted_tool_calls,
        other => panic!("expected Paused, got {other:?}"),
    };
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };

    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("valid approval");
    // Pause and resume are two phases of one turn: rounds continue, not reset.
    assert_eq!(resumed.trace.rounds.len(), 2);
    assert_eq!(resumed.trace.rounds[0].round_id, RoundId(0));
    assert_eq!(resumed.trace.rounds[1].round_id, RoundId(1));
    // The paused phase had counted the pending batch; the resume must not
    // re-count it, and the executed batch is now real.
    assert_eq!(resumed.trace.tool_calls_total, accounted);
    assert_eq!(resumed.trace.tool_calls_total, 1);
    // The paused round's tool batch is filled in by the resume prologue.
    assert!(resumed.trace.rounds[0].tool_batch.is_some());
}

/// Acceptance: trimming or clearing the trace must not change where the
/// resume continues or what it may spend — rounds and counts live in the
/// continuation only.
#[tokio::test]
async fn trimmed_trace_does_not_change_resume_rounds_or_counts() {
    let (runner, gateway) = paused_runner();
    let mut out = run_to_pause(&runner).await;
    // The host trims the trace to nothing before resuming.
    out.trace = TurnTrace::new();
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };

    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("trimmed trace is still a valid checkpoint");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    // The batch ran from the continuation (not the trace): its result is a
    // fact and the quota count survived the trim.
    assert_eq!(result_facts(&resumed.context).len(), 1);
    assert_eq!(resumed.trace.tool_calls_total, 1);
    // Round numbering continues from the continuation, not the trace.
    assert_eq!(resumed.trace.rounds.len(), 1);
    assert_eq!(resumed.trace.rounds[0].round_id, RoundId(1));
    assert_eq!(gateway.recorded().len(), 2);
}

// ---- prepared hook work survives the pause --------------------------------------

/// Acceptance: the hook rejects `danger` and rewrites `echo`; the gate
/// pauses; the host approves. The saved rejections commit verbatim under
/// the original ids, only the saved-and-approved arguments execute, the
/// hook is never re-run, and every batch item lands exactly once.
#[tokio::test]
async fn pause_preserves_hook_rejections_and_rewrites() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "two calls",
            vec![
                common::draft("danger", json!({"path": "etc"})),
                common::draft("echo", json!({"a": 1})),
            ],
        )),
        Ok(endturn_output("done")),
    ]);
    let hook = Arc::new(RejectDangerRewriteRest {
        applications: AtomicUsize::new(0),
    });
    let runner = TurnRunner::with_hook(
        gateway.clone(),
        Arc::new(ToolExecutor::from_vec(vec![
            Arc::new(DangerTool),
            Arc::new(EchoTool),
        ])),
        hook.clone(),
    );
    let mut ctx = TurnContext::new(TurnId::new("t1"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();
    let out = runner.run(ctx, pause_options(), ctrl()).await;
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => match &continuation.pause_point {
            PausePoint::AwaitingApproval { prepared, .. } => {
                assert_eq!(
                    prepared.awaiting.len(),
                    1,
                    "danger was rejected by the hook"
                );
                assert_eq!(prepared.awaiting[0].tool_name, "echo");
                assert_eq!(prepared.awaiting[0].arguments, json!({"hooked": true}));
                assert_eq!(prepared.rejected.len(), 1, "the hook rejection is saved");
                // The saved rejection keeps the original call id — and it is
                // NOT one of the awaiting ids.
                assert_ne!(prepared.rejected[0].call_id, prepared.awaiting[0].call_id);
                prepared.awaiting.clone()
            }
            other => panic!("expected AwaitingApproval, got {other:?}"),
        },
        other => panic!("expected Paused, got {other:?}"),
    };

    let danger_id = match &out.result {
        TurnResult::Paused { continuation } => match &continuation.pause_point {
            PausePoint::AwaitingApproval { prepared, .. } => prepared.rejected[0].call_id.clone(),
            other => panic!("expected AwaitingApproval, got {other:?}"),
        },
        other => panic!("expected Paused, got {other:?}"),
    };
    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("approval of the prepared work");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    // The hook ran exactly once — at the original emission, never on resume.
    assert_eq!(hook.applications.load(Ordering::SeqCst), 1);
    // Every batch item committed exactly once, with the saved decisions.
    let results = result_facts(&resumed.context);
    assert_eq!(results.len(), 2);
    let danger = results.iter().find(|r| r.call_id == danger_id).unwrap();
    assert_eq!(danger.status, ToolResultStatus::Rejected);
    assert_eq!(
        danger.output.content,
        json!({"error": "hook denied danger"})
    );
    let echo = results.iter().find(|r| r.call_id != danger_id).unwrap();
    assert_eq!(echo.status, ToolResultStatus::Succeeded);
    assert_eq!(echo.output.content, json!({"echo": {"hooked": true}}));
    assert_eq!(gateway.recorded().len(), 2);
}

// ---- steering resumes (fabricated checkpoints) ----------------------------------

/// A steering continuation commits its queued inputs first, then the
/// request's inject, then pulls — identical texts are independent inputs
/// and nothing is deduped.
#[tokio::test]
async fn steering_resume_commits_queued_then_inject_in_order() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let runner = runner_with(gateway.clone(), vec![Arc::new(EchoTool)]);

    let mut ctx = TurnContext::new(TurnId::new("t1"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();
    let mut trace = TurnTrace::new();
    trace.rounds.push(round_trace(0, "t1"));
    let outcome = TurnOutcome {
        context: ctx,
        result: TurnResult::Paused {
            continuation: Continuation {
                pause_point: PausePoint::PausedForSteering,
                round: 1,
                accounted_tool_calls: 0,
                queued_inputs: vec![TextPayload::new("nudge")],
            },
        },
        trace,
    };

    let resumed = runner
        .resume(
            outcome,
            ResumeRequest {
                decision: None,
                inject: vec![TextPayload::new("nudge"), TextPayload::new("and the logs")],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("valid steering resume");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    assert_eq!(gateway.recorded().len(), 1);
    // Three steering blocks: queued "nudge", then two injected ones — the
    // same text is NOT deduped, and the order is queued → injected.
    let steering_texts: Vec<&str> = resumed
        .context
        .blocks()
        .iter()
        .filter(|b| b.meta.source.as_deref() == Some("user.steering"))
        .filter_map(|b| match &b.content {
            BlockContent::Parts(parts) => match &parts[0] {
                causa_kernel::ContentPart::Text(t) => Some(t.0.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(steering_texts, vec!["nudge", "nudge", "and the logs"]);
}

/// A decision on a steering pause — or a steering continuation over
/// unanswered calls — is rejected before anything executes.
#[tokio::test]
async fn steering_resume_rejects_decisions_and_unanswered_calls() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let runner = runner_with(gateway.clone(), vec![Arc::new(EchoTool)]);

    // A decision on a steering pause.
    let mut ctx = TurnContext::new(TurnId::new("t1"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();
    let outcome = TurnOutcome {
        context: ctx,
        result: TurnResult::Paused {
            continuation: Continuation {
                pause_point: PausePoint::PausedForSteering,
                round: 1,
                accounted_tool_calls: 0,
                queued_inputs: vec![],
            },
        },
        trace: TurnTrace::new(),
    };
    let err = runner
        .resume(
            outcome,
            ResumeRequest {
                decision: Some(HookOutcome::passthrough(vec![])),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect_err("decision on steering pause must be rejected");
    assert!(err.reason.contains("takes no decision"));

    // A steering continuation over unanswered calls would drop the batch.
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let runner = runner_with(gateway, vec![Arc::new(EchoTool)]);
    let mut ctx = TurnContext::new(TurnId::new("t2"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();
    ctx.append_model_output(
        InvocationId {
            turn_id: TurnId::new("t2"),
            round_id: RoundId(0),
        },
        &causa_kernel::ModelResponse {
            text: TextPayload::new(String::new()),
            tool_calls: vec![ToolCallDraft {
                tool_name: "echo".into(),
                arguments: json!({"a": 1}),
                provider_call_id: None,
            }],
        },
        causa_kernel::ModelStopReason::ToolUse,
    )
    .unwrap();
    let outcome = TurnOutcome {
        context: ctx,
        result: TurnResult::Paused {
            continuation: Continuation {
                pause_point: PausePoint::PausedForSteering,
                round: 1,
                accounted_tool_calls: 1,
                queued_inputs: vec![],
            },
        },
        trace: TurnTrace::new(),
    };
    let err = runner
        .resume(
            outcome,
            ResumeRequest {
                decision: None,
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect_err("steering over unanswered calls must be rejected");
    assert!(err.reason.contains("unanswered tool calls"));
}

// ---- conversation resume entry --------------------------------------------------

#[tokio::test]
async fn conversation_resume_rejects_non_paused_material_and_returns_it() {
    let (runner, _gw) = paused_runner();
    let request = || ResumeRequest {
        decision: None,
        inject: vec![],
    };
    // Not paused (no stamp at all): the outcome comes back intact.
    let fresh = ConversationOutcome {
        state: seeded_state(),
        result: TurnResult::Interrupted {
            cause: TurnInterruption::MaxModelRounds { limit: 1 },
        },
        trace: TurnTrace::new(),
    };
    let err = resume_turn(&runner, fresh, request(), plain_options(), ctrl())
        .await
        .expect_err("non-paused outcome must be rejected");
    assert!(err.reason.contains("paused outcome"));
    assert!(
        matches!(
            err.outcome.result,
            TurnResult::Interrupted {
                cause: TurnInterruption::MaxModelRounds { .. }
            }
        ),
        "the returned material is untouched"
    );

    // Open but unstamped state with a paused result: inconsistent material.
    let inconsistent = ConversationOutcome {
        state: seeded_state(),
        result: TurnResult::Paused {
            continuation: Continuation {
                pause_point: PausePoint::PausedForSteering,
                round: 1,
                accounted_tool_calls: 0,
                queued_inputs: vec![],
            },
        },
        trace: TurnTrace::new(),
    };
    let err = resume_turn(&runner, inconsistent, request(), plain_options(), ctrl())
        .await
        .expect_err("inconsistent stamp must be rejected");
    assert!(err.reason.contains("stamped Paused"));
    // The material is returned whole: the open active turn survives.
    assert!(!err.outcome.state.active_turn().unwrap().is_sealed());

    // A sealed active turn behind a paused result is rejected too.
    let mut state = seeded_state();
    state
        .seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    let sealed = ConversationOutcome {
        state,
        result: TurnResult::Paused {
            continuation: Continuation {
                pause_point: PausePoint::PausedForSteering,
                round: 1,
                accounted_tool_calls: 0,
                queued_inputs: vec![],
            },
        },
        trace: TurnTrace::new(),
    };
    let err = resume_turn(&runner, sealed, request(), plain_options(), ctrl())
        .await
        .expect_err("sealed active turn must be rejected");
    assert!(err.reason.contains("must be open"));
}

#[tokio::test]
async fn conversation_resume_path_completes_and_commits() {
    let (runner, _gw) = paused_runner();
    let out = runner
        .run_in_conversation(seeded_state(), pause_options(), ctrl())
        .await
        .unwrap();
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };
    let resumed = resume_turn(
        &runner,
        out,
        ResumeRequest {
            decision: approve(awaiting),
            inject: vec![],
        },
        plain_options(),
        ctrl(),
    )
    .await
    .expect("valid approval");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    // The completed turn commits normally after the resume.
    let mut resumed = resumed;
    let entry = resumed.state.commit(TurnId::new("t1")).unwrap();
    assert_eq!(entry.sequence.0, 0);
    assert_eq!(resumed.state.history_len(), 1);
}

// ---- resume validation: input errors never execute ------------------------------
//
// A continuation that does not match the facts, or a decision that does
// not cover the awaiting calls exactly, is rejected at the entry — no
// model call, no tool execution, no fact appended or sealed — and the
// paused material comes back for correction.

#[tokio::test]
async fn resume_with_an_incomplete_decision_is_rejected_before_execution() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;
    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                // Neither executes nor rejects: the batch would be dropped
                // on the floor, leaving the committed calls permanently
                // unanswered.
                decision: Some(HookOutcome {
                    to_execute: vec![],
                    rejected: vec![],
                    unknown_decisions: Vec::new(),
                }),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await;
    let err = resumed.expect_err("incomplete batch must be rejected");
    assert!(err.reason.contains("does not cover every awaiting call"));
    // No side effects: the tool never ran, no second model request fired,
    // the returned turn is still open with its unanswered call intact.
    assert_eq!(gateway.recorded().len(), 1);
    let outcome = err.into_outcome();
    assert!(!outcome.context.is_sealed());
    assert!(
        !outcome
            .context
            .blocks()
            .iter()
            .any(|b| matches!(b.content, BlockContent::ToolResult(_)))
    );

    // After correcting the request, the same material resumes cleanly.
    let awaiting = match &outcome.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };
    let (runner, _gateway) = paused_runner();
    let resumed = runner
        .resume(
            outcome,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("corrected request resumes the same material");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
}

#[tokio::test]
async fn resume_with_a_continuation_mismatching_the_facts_is_rejected() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;
    // The host fabricates a batch the turn never emitted: the checkpoint's
    // awaiting set cannot match the facts.
    let fabricated = Continuation {
        pause_point: PausePoint::AwaitingApproval {
            prepared: PreparedApproval {
                awaiting: vec![ToolCallPayload {
                    call_id: ToolCallId("forged".into()),
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                }],
                rejected: vec![],
                unknown_decisions: Vec::new(),
            },
            deadline: None,
        },
        round: 0,
        accounted_tool_calls: 1,
        queued_inputs: vec![],
    };
    let (context, _continuation, trace) = decompose_pause(out);
    let outcome = TurnOutcome {
        context,
        result: TurnResult::Paused {
            continuation: fabricated,
        },
        trace,
    };
    let decision = match &outcome.result {
        TurnResult::Paused { continuation } => match &continuation.pause_point {
            PausePoint::AwaitingApproval { prepared, .. } => {
                Some(HookOutcome::passthrough(prepared.awaiting.clone()))
            }
            other => panic!("expected AwaitingApproval, got {other:?}"),
        },
        other => panic!("expected Paused, got {other:?}"),
    };
    let err = runner
        .resume(
            outcome,
            ResumeRequest {
                decision,
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect_err("fabricated continuation must be rejected");
    assert!(
        err.reason
            .contains("does not match the turn's unanswered tool calls")
    );
    assert_eq!(gateway.recorded().len(), 1);
}

#[tokio::test]
async fn resume_with_a_duplicated_decision_is_rejected() {
    let (runner, _gateway) = paused_runner();
    let out = run_to_pause(&runner).await;
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };
    let first = awaiting[0].clone();
    // The same call covered twice — a duplicated decision would pair the
    // committed call block with two results.
    let err = runner
        .resume(
            out,
            ResumeRequest {
                decision: Some(HookOutcome {
                    to_execute: vec![first.clone(), first],
                    rejected: vec![],
                    unknown_decisions: Vec::new(),
                }),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect_err("duplicated coverage must be rejected");
    assert!(err.reason.contains("more than once"));
}

// ---- lower limits stop before external execution --------------------------------

#[tokio::test]
async fn lowered_tool_quota_stops_before_the_batch_executes() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };
    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![],
            },
            options_with_limits(5, 0),
            ctrl(),
        )
        .await;
    // The quota is spent: the turn stops without executing the batch —
    // a limit outcome, not an input rejection.
    let outcome = resumed.expect("material itself is valid");
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::MaxToolCalls { limit: 0 }
        }
    ));
    assert_eq!(gateway.recorded().len(), 1, "no second model call");
    assert!(
        !outcome
            .context
            .blocks()
            .iter()
            .any(|b| matches!(b.content, BlockContent::ToolResult(_))),
        "no tool executed"
    );
}

#[tokio::test]
async fn lowered_round_budget_stops_before_the_batch_executes() {
    let (runner, gateway) = paused_runner();
    let out = run_to_pause(&runner).await;
    let awaiting = match &out.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };
    let resumed = runner
        .resume(
            out,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![],
            },
            options_with_limits(0, 10),
            ctrl(),
        )
        .await;
    let outcome = resumed.expect("material itself is valid");
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::MaxModelRounds { limit: 0 }
        }
    ));
    assert_eq!(gateway.recorded().len(), 1);
    assert!(
        !outcome
            .context
            .blocks()
            .iter()
            .any(|b| matches!(b.content, BlockContent::ToolResult(_)))
    );
}

// ---- media facts survive a pause/resume ------------------------------------------

/// Acceptance: a paused outcome whose facts carry a media reference
/// round-trips through JSON (the host's checkpoint document) and resumes;
/// the reference is the only media content anywhere on the wire — the
/// continuation carries no bytes and no duplicate fact snapshot.
#[tokio::test]
async fn paused_outcome_with_media_round_trips_and_resumes() {
    let (runner, _gateway) = paused_runner();
    let mut ctx = TurnContext::new(TurnId::new("t1"));
    ctx.append_parts(
        vec![
            causa_kernel::ContentPart::Text(TextPayload::new("read the chart")),
            causa_kernel::ContentPart::Media(causa_kernel::MediaRef::new("image/png", "asset-1")),
        ],
        "user",
    )
    .unwrap();
    let paused = runner.run(ctx, pause_options(), ctrl()).await;
    assert!(matches!(paused.result, TurnResult::Paused { .. }));

    // The host saves the complete paused outcome as its checkpoint.
    let json = serde_json::to_string(&paused).expect("checkpoint serializes");
    assert!(json.contains(r#""reference":"asset-1""#), "{json}");
    // The Paused variant carries no snapshot: the fact state appears once.
    assert!(!json.contains(r#""snapshot""#), "{json}");

    let restored: TurnOutcome = serde_json::from_str(&json).expect("checkpoint reloads");
    let awaiting = match &restored.result {
        TurnResult::Paused { continuation } => expect_awaiting(continuation),
        other => panic!("expected Paused, got {other:?}"),
    };
    let resumed = runner
        .resume(
            restored,
            ResumeRequest {
                decision: approve(awaiting),
                inject: vec![],
            },
            plain_options(),
            ctrl(),
        )
        .await
        .expect("restored checkpoint resumes");
    assert!(matches!(resumed.result, TurnResult::Completed { .. }));
    // The media reference is still a fact after the resume; bytes never
    // entered the facts, the continuation, or the trace.
    let wire = serde_json::to_string(&resumed).unwrap();
    assert!(wire.contains(r#""reference":"asset-1""#));
}

// ---- steering pull ---------------------------------------------------------------

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
