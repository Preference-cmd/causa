//! Staged runtime tests — the reference driver, config axes, executor
//! dispatch, cancellation, and traces. These exercise `internal/` wiring
//! through the root facade and change with the perimeter, not the contract.

mod common;

use common::{
    DropAllCompaction, EchoTool, FailTool, RecordingGateway, UnknownStopTool, ctrl, ctx, draft,
    endturn_output, options_with_limits, runner_with, runner_with_dedup, tooluse_calls_output,
    tooluse_output,
};
use reimagine_context_kernel::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, AttemptNumber, BlockContent,
    CallControl, ContextBlock, ExecutionOptions, FramePolicy, ModelInvokeErrorKind, ModelOutput,
    ModelResponse, ModelStopReason, ModelUsage, ReasoningPayload, RetryPolicy, RunControl,
    StoreError, TextPayload, Tool, ToolCallContext, ToolDefinition, ToolExecutionOutcome,
    ToolOutput, ToolOutputLimits, ToolResultPayload, ToolResultStatus, Truncation,
    TurnInterruption, TurnResult, TurnRunOptions, UnknownOutcomePolicy, WindowBudget,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn final_assistant_completes_once() {
    let c = ctx("t1");
    let runner = runner_with(
        RecordingGateway::scripted(vec![Ok(endturn_output("final"))]),
        vec![],
    );
    let out = runner.run(c, options_with_limits(5, 10), ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 1);
    assert_eq!(out.context.blocks().len(), 1);
}

#[tokio::test]
async fn tool_calls_drive_next_frame_and_causality() {
    let c = ctx("t1");
    let runner = runner_with(
        RecordingGateway::scripted(vec![
            Ok(tooluse_output("call echo", "echo", json!({"a":1}))),
            Ok(endturn_output("done")),
        ]),
        vec![Arc::new(EchoTool)],
    );
    let out = runner.run(c, options_with_limits(5, 10), ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // blocks: assistant tool call + tool result + final assistant (with optional assistant text blocks)
    // Should have at least 3 blocks, causality: tool.result follows tool.call
    let blocks = out.context.blocks();
    let pos_call = blocks
        .iter()
        .position(|b| matches!(b.content, BlockContent::ToolCall(_)))
        .unwrap();
    let pos_result = blocks
        .iter()
        .position(|b| matches!(b.content, BlockContent::ToolResult(_)))
        .unwrap();
    assert!(pos_result > pos_call);
}

#[tokio::test]
async fn retry_same_frame_only_attempt_increments_and_no_block_on_failure() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Err(ModelInvokeErrorKind::Transient),
        Ok(endturn_output("ok")),
    ]);
    let runner = runner_with(gw.clone(), vec![]);
    let mut cfg = options_with_limits(5, 10);
    cfg.policy.retry = RetryPolicy {
        max_retries: 1,
        retry_timeouts: false,
    };
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // Recorded 2 attempts with same invocation and frame
    let rec = gw.recorded();
    assert_eq!(rec.len(), 2);
    assert_eq!(rec[0].invocation_id, rec[1].invocation_id);
    assert_eq!(rec[0].frame.frame_id, rec[1].frame.frame_id);
    assert_eq!(rec[0].attempt, AttemptNumber(1));
    assert_eq!(rec[1].attempt, AttemptNumber(2));
    // Only one assistant block (no failed attempt block)
    assert_eq!(out.context.blocks().len(), 1);
}

#[tokio::test]
async fn retry_exhaustion_interrupted_and_counts() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Err(ModelInvokeErrorKind::Transient),
        Err(ModelInvokeErrorKind::Transient),
    ]);
    let runner = runner_with(gw.clone(), vec![]);
    let mut cfg = options_with_limits(5, 10);
    cfg.policy.retry = RetryPolicy {
        max_retries: 1,
        retry_timeouts: false,
    };
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::RetryExhausted { .. }
        }
    ));
    let rec = gw.recorded();
    assert_eq!(rec.len(), 2);
}

#[tokio::test]
async fn tool_failure_is_observation_not_terminal_and_next_round() {
    let c = ctx("t1");
    let runner = runner_with(
        RecordingGateway::scripted(vec![
            Ok(tooluse_output("call fail", "fail", json!({}))),
            Ok(endturn_output("done after fail")),
        ]),
        vec![Arc::new(FailTool)],
    );
    let cfg = options_with_limits(5, 10);
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 2);
}

#[tokio::test]
async fn dedup_same_batch_rejected_and_parallel_single_failure_not_abort() {
    let c = ctx("t1");
    // Two identical echo calls in same batch => one rejected. This is a
    // test-only dedup hook — TurnRunner::new defaults to passthrough
    // (no opinion); the host composes filters via the framework layer.
    let runner = runner_with_dedup(
        RecordingGateway::scripted(vec![
            Ok(tooluse_calls_output(
                "dup",
                vec![draft("echo", json!({"x":1})), draft("echo", json!({"x":1}))],
            )),
            Ok(endturn_output("done")),
        ]),
        vec![Arc::new(EchoTool)],
    );
    let cfg = options_with_limits(5, 10);
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // Should have 2 tool calls blocks + 2 results (one rejected) + final assistant?
    // At least check trace shows rejected handling via tool result status
    let last_round = &out.trace.rounds[0];
    assert!(last_round.tool_batch.is_some());
    let batch = last_round.tool_batch.as_ref().unwrap();
    // completion_order should have 1 (only non-rejected executed)
    assert_eq!(batch.completion_order.len(), 1);
    assert_eq!(batch.calls.len(), 2);
}

#[tokio::test]
async fn unknown_outcome_stop_interrupts() {
    let c = ctx("t1");
    let runner = runner_with(
        RecordingGateway::scripted(vec![Ok(tooluse_output("call unk", "unk", json!({})))]),
        vec![Arc::new(UnknownStopTool)],
    );
    let cfg = options_with_limits(5, 10);
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::UnsafeUnknownOutcome { .. }
        }
    ));
}

#[tokio::test]
async fn same_input_same_fake_output_same_snapshot() {
    async fn run_once() -> Vec<ContextBlock> {
        let c = ctx("t1");
        let runner = runner_with(
            RecordingGateway::scripted(vec![
                Ok(tooluse_output("hi", "echo", json!({"k":1}))),
                Ok(endturn_output("bye")),
            ]),
            vec![Arc::new(EchoTool)],
        );
        let out = runner.run(c, options_with_limits(5, 10), ctrl()).await;
        out.context.snapshot_blocks()
    }
    let a = run_once().await;
    let b = run_once().await;
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap()
    );
}

#[tokio::test]
async fn token_limits_and_artifact_truncation() {
    struct BigTool;
    #[async_trait::async_trait]
    impl Tool for BigTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "big".into(),
                description: "big".into(),
                parameters: json!({"type":"object"}),
            }
        }
        fn output_limits(&self) -> Option<ToolOutputLimits> {
            Some(ToolOutputLimits { max_tokens: 10 })
        }
        async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
            let big = "a".repeat(1000);
            ToolExecutionOutcome::new(ToolResultPayload {
                call_id: ctx.call_id.clone(),
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!(big)),
            })
        }
    }
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_output("call big", "big", json!({}))),
        Ok(endturn_output("done")),
    ]);
    struct MemStore;
    #[async_trait::async_trait]
    impl ArtifactStore for MemStore {
        async fn persist(
            &self,
            data: &[u8],
            _hint: ArtifactHint,
        ) -> Result<ArtifactRef, StoreError> {
            Ok(ArtifactRef {
                id: blake3::hash(data).to_hex().to_string()[..8].into(),
                size_bytes: data.len(),
                kind: ArtifactKind::FullOutput,
                persisted: true,
            })
        }
        async fn read(
            &self,
            _id: &str,
            _range: Option<std::ops::Range<u64>>,
        ) -> Result<Vec<u8>, StoreError> {
            Ok(vec![])
        }
    }
    let runner = runner_with(gw, vec![Arc::new(BigTool)]);
    let mut cfg = options_with_limits(5, 10);
    cfg.execution.tool_output_limits = ToolOutputLimits { max_tokens: 10 };
    cfg.execution.artifact_store = Some(Arc::new(MemStore));
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // Find truncated output
    let result_block = out
        .context
        .blocks()
        .iter()
        .find(|b| {
            matches!(&b.content, BlockContent::ToolResult(r) if r.output.truncation == Truncation::Middle)
        })
        .expect("truncated");
    if let BlockContent::ToolResult(r) = &result_block.content {
        assert!(r.output.artifact.is_some());
    } else {
        panic!()
    }
}

#[tokio::test]
async fn parent_cancellation_interrupted() {
    let c = ctx("t1");
    let token = tokio_util::sync::CancellationToken::new();
    token.cancel();
    let gw = RecordingGateway::scripted(vec![Ok(endturn_output("should not"))]);
    let runner = runner_with(gw, vec![]);
    let out = runner
        .run(c, options_with_limits(5, 10), RunControl::new(token, None))
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ExplicitCancellation
        }
    ));
}

// ---------------------------------------------------------------------------
// Alignment coverage (2026-08-28 review): acceptance gaps + regressions

// ---------------------------------------------------------------------------

// [P2 acceptance #10] max_retries = 0 → initial attempt only

#[tokio::test]
async fn max_retries_zero_does_single_attempt() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![Err(ModelInvokeErrorKind::Transient)]);
    let runner = runner_with(gw.clone(), vec![]);
    let mut cfg = options_with_limits(5, 10);
    cfg.policy.retry = RetryPolicy {
        max_retries: 0,
        retry_timeouts: false,
    };
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::RetryExhausted {
                last_kind: ModelInvokeErrorKind::Transient,
                ..
            }
        }
    ));
    assert_eq!(gw.recorded().len(), 1);
}

// [P2 acceptance #13] turn deadline exceeded → TurnDeadlineExceeded

#[tokio::test]
async fn turn_deadline_yields_deadline_exceeded() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![]);
    let runner = runner_with(gw.clone(), vec![]);
    let cfg = TurnRunOptions::default();
    let ctrl = RunControl::new(
        tokio_util::sync::CancellationToken::new(),
        Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
    );
    let out = runner.run(c, cfg, ctrl).await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::TurnDeadlineExceeded
        }
    ));
    assert_eq!(gw.recorded().len(), 0);
}

// [P2 acceptance #13] non-retryable model failure → Interrupted with round trace

#[tokio::test]
async fn non_retryable_error_records_round_trace() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![Err(ModelInvokeErrorKind::Permanent)]);
    let runner = runner_with(gw, vec![]);
    let cfg = TurnRunOptions::default();
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::RetryExhausted {
                last_kind: ModelInvokeErrorKind::Permanent,
                ..
            }
        }
    ));
    assert_eq!(out.trace.rounds.len(), 1);
    assert_eq!(out.trace.rounds[0].attempts.len(), 1);
    assert!(!out.trace.rounds[0].attempts[0].is_retryable);
    assert!(out.trace.rounds[0].attempts[0].kind.is_some());
    assert_eq!(out.trace.tool_calls_total, 0);
    assert!(out.context.is_sealed());
}

// [P0 regression] cross-batch identical call must not collide (proposal §5.3 recovery path)

#[tokio::test]
async fn cross_batch_identical_call_does_not_collide() {
    let c = ctx("t1");
    let same_call = || tooluse_output("retry read", "echo", json!({"path": "A"}));
    let gw = RecordingGateway::scripted(vec![
        Ok(same_call()),
        Ok(same_call()),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gw, vec![Arc::new(EchoTool)]);
    let cfg = TurnRunOptions::default();
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(
        matches!(out.result, TurnResult::Completed { .. }),
        "expected completion, got {:?}",
        out.result
    );
    let call_ids: Vec<_> = out
        .context
        .blocks()
        .iter()
        .filter_map(|b| match &b.content {
            BlockContent::ToolCall(tc) => Some(tc.call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(call_ids.len(), 2);
    assert_ne!(call_ids[0], call_ids[1], "round-salted ids must differ");
}

// [P1 regression] MaxTokens / Refusal reach their dedicated interruption causes

#[tokio::test]
async fn max_tokens_and_refusal_yield_dedicated_causes_without_blocks() {
    for (stop, expect) in [
        (ModelStopReason::MaxTokens, "max_tokens"),
        (ModelStopReason::Refusal, "refusal"),
    ] {
        let c = ctx("t1");
        let gw = RecordingGateway::scripted(vec![Ok(ModelOutput {
            response: ModelResponse {
                text: TextPayload::new("cut"),
                tool_calls: vec![],
            },
            usage: None,
            stop_reason: stop,
            reasoning: None,
        })]);
        let runner = runner_with(gw, vec![]);
        let cfg = TurnRunOptions::default();
        let out = runner.run(c, cfg, ctrl()).await;
        match (&out.result, expect) {
            (TurnResult::Interrupted { cause }, "max_tokens") => {
                assert!(matches!(cause, TurnInterruption::ModelMaxTokens))
            }
            (TurnResult::Interrupted { cause }, "refusal") => {
                assert!(matches!(cause, TurnInterruption::ModelRefusal))
            }
            _ => panic!("unexpected outcome for {expect}"),
        }
        // MaxTokens/Refusal never persist blocks
        assert_eq!(out.context.blocks().len(), 0);
        assert!(out.context.is_sealed());
        assert_eq!(out.trace.rounds.len(), 1);
        assert!(out.trace.rounds[0].output_summary.is_some());
    }
}

// [P3 acceptance #14] UnknownOutcomePolicy::Continue → append result and continue

#[tokio::test]
async fn unknown_outcome_continue_continues_turn() {
    struct UnknownContinueTool;
    #[async_trait::async_trait]
    impl Tool for UnknownContinueTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "unkc".into(),
                description: "unkc".into(),
                parameters: json!({"type":"object"}),
            }
        }
        fn unknown_outcome_policy(&self) -> UnknownOutcomePolicy {
            UnknownOutcomePolicy::Continue
        }
        async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
            ToolExecutionOutcome::new(ToolResultPayload {
                call_id: ctx.call_id.clone(),
                status: ToolResultStatus::UnknownOutcome,
                output: ToolOutput::new(json!({"unk": true})),
            })
        }
    }
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_output("call unkc", "unkc", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gw, vec![Arc::new(UnknownContinueTool)]);
    let cfg = TurnRunOptions::default();
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert!(out.context.blocks().iter().any(
        |b| matches!(&b.content, BlockContent::ToolResult(r) if r.status == ToolResultStatus::UnknownOutcome)
    ));
}

// [P3 §6.1.1] hung tool: executor call-deadline backstop yields UnknownOutcome;
// Stop policy interrupts the turn.

#[tokio::test]
async fn hung_tool_stop_policy_interrupts_with_unknown_outcome() {
    struct HungTool;
    #[async_trait::async_trait]
    impl Tool for HungTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "hung".into(),
                description: "hung".into(),
                parameters: json!({"type":"object"}),
            }
        }
        async fn execute(&self, _ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
            std::future::pending::<()>().await;
            unreachable!()
        }
    }
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![Ok(tooluse_output("call hung", "hung", json!({})))]);
    let runner = runner_with(gw, vec![Arc::new(HungTool)]);
    let cfg = TurnRunOptions {
        execution: ExecutionOptions {
            call_timeout: Some(std::time::Duration::from_millis(50)),
            ..Default::default()
        },
        ..Default::default()
    };
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::UnsafeUnknownOutcome { .. }
        }
    ));
    let batch = out.trace.rounds[0].tool_batch.as_ref().unwrap();
    assert_eq!(batch.calls[0].status, ToolResultStatus::UnknownOutcome);
}

// [P3 §6.1.1] hung tool with Continue declaration → backstop UnknownOutcome is
// committed and the turn proceeds to the next round.

#[tokio::test]
async fn hung_tool_continue_policy_still_completes() {
    struct HungContinueTool;
    #[async_trait::async_trait]
    impl Tool for HungContinueTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "hungc".into(),
                description: "hungc".into(),
                parameters: json!({"type":"object"}),
            }
        }
        fn unknown_outcome_policy(&self) -> UnknownOutcomePolicy {
            UnknownOutcomePolicy::Continue
        }
        async fn execute(&self, _ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
            std::future::pending::<()>().await;
            unreachable!()
        }
    }
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_output("call hungc", "hungc", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gw, vec![Arc::new(HungContinueTool)]);
    let cfg = TurnRunOptions {
        execution: ExecutionOptions {
            call_timeout: Some(std::time::Duration::from_millis(50)),
            ..Default::default()
        },
        ..Default::default()
    };
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 2);
}

// [P3 acceptance #14] parallel batch: one failing call does not abort the others

#[tokio::test]
async fn parallel_batch_partial_failure_does_not_abort() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "two calls",
            vec![draft("echo", json!({"a": 1})), draft("fail", json!({}))],
        )),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gw, vec![Arc::new(EchoTool), Arc::new(FailTool)]);
    let cfg = TurnRunOptions::default();
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    let statuses: Vec<_> = out
        .context
        .blocks()
        .iter()
        .filter_map(|b| match &b.content {
            BlockContent::ToolResult(r) => Some(r.status.clone()),
            _ => None,
        })
        .collect();
    assert!(statuses.contains(&ToolResultStatus::Succeeded));
    assert!(statuses.contains(&ToolResultStatus::Failed));
}

// [§6.1.1] completion_order reflects real completion, not submission order

#[tokio::test]
async fn completion_order_reflects_real_completion() {
    struct SlowTool;
    #[async_trait::async_trait]
    impl Tool for SlowTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "slow".into(),
                description: "slow".into(),
                parameters: json!({"type":"object"}),
            }
        }
        async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            ToolExecutionOutcome::new(ToolResultPayload {
                call_id: ctx.call_id.clone(),
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("slow")),
            })
        }
    }
    let c = ctx("t1");
    // slow is dispatched first (position 0) but completes last
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "mixed",
            vec![draft("slow", json!({})), draft("echo", json!({}))],
        )),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gw, vec![Arc::new(SlowTool), Arc::new(EchoTool)]);
    let cfg = TurnRunOptions::default();
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    let batch = out.trace.rounds[0].tool_batch.as_ref().unwrap();
    assert_eq!(batch.calls.len(), 2);
    assert_eq!(batch.completion_order.len(), 2);
    // results are committed in original order (slow first)…
    assert_eq!(batch.calls[0].tool_name, "slow");
    assert_eq!(batch.calls[1].tool_name, "echo");
    // …but completion_order starts with the fast echo call
    let fast_id = &batch.calls[1].call_id;
    assert_eq!(&batch.completion_order[0], fast_id);
    assert!(batch.calls[1].duration_ms < batch.calls[0].duration_ms);
}

// [P1 Phase 1] append_model_output validation branches

#[tokio::test]
async fn max_tool_calls_interrupt_records_total() {
    let c = ctx("t1");
    let two_calls = tooluse_calls_output(
        "two",
        vec![
            draft("echo", json!({"a": 1})),
            draft("echo", json!({"a": 1})),
        ],
    );
    let gw = RecordingGateway::scripted(vec![Ok(two_calls)]);
    let runner = runner_with(gw, vec![Arc::new(EchoTool)]);
    let cfg = options_with_limits(5, 1);
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::MaxToolCalls { limit: 1 }
        }
    ));
    // raw count (incl. the duplicate) is recorded even on the interrupt path
    assert_eq!(out.trace.tool_calls_total, 2);
    assert_eq!(out.trace.rounds.len(), 1);
    assert!(out.trace.rounds[0].tool_batch.is_none());
}

// ---- Phase C: frame policy is driver-owned; evaluation stays canonical ----

#[tokio::test]
async fn frame_policy_from_options_shapes_projection_without_touching_facts() {
    let mut c = ctx("t1");
    c.append_input(TextPayload::new("hello"), "user").unwrap();
    // any non-empty content trips the placeholder trigger
    let frame_policy = FramePolicy {
        window_budget: WindowBudget {
            model_window_limit: 100,
            compaction_trigger: 1,
        },
        compaction: Some(Arc::new(DropAllCompaction)),
        token_counter: None,
    };
    let gw = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let runner = runner_with(gw.clone(), vec![]);
    let cfg = TurnRunOptions {
        frame: frame_policy,
        ..Default::default()
    };
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // the gateway saw the COMPACTED projection (empty blocks)
    let recorded = gw.recorded();
    assert_eq!(recorded.len(), 1);
    assert!(recorded[0].frame.model_context.blocks.is_empty());
    drop(recorded);
    // the fact state is untouched — compaction is frame-local, never writes
    // back; blocks are still [input text, response text]
    assert_eq!(out.context.blocks().len(), 2);
    assert!(matches!(
        out.context.blocks()[0].content,
        BlockContent::Text(_)
    ));
}

// ---- Phase D boundary: artifact failure and loss-free fidelity ----

struct FailingStore;
#[async_trait::async_trait]
impl ArtifactStore for FailingStore {
    async fn persist(&self, _data: &[u8], _hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        Err(StoreError::Persist("disk full".into()))
    }
    async fn read(
        &self,
        _id: &str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError> {
        Err(StoreError::Read("missing".into()))
    }
}

#[tokio::test]
async fn artifact_store_failure_still_truncates_without_artifact() {
    let c = ctx("t1");
    let big = "x".repeat(400);
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_output("big", "echo", json!({"pad": big}))),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gw, vec![Arc::new(EchoTool)]);
    let mut cfg = options_with_limits(5, 10);
    cfg.execution.tool_output_limits = ToolOutputLimits { max_tokens: 10 };
    cfg.execution.artifact_store = Some(Arc::new(FailingStore));
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    let result_block = out
        .context
        .blocks()
        .iter()
        .find(|b| {
            matches!(
                &b.content,
                BlockContent::ToolResult(r) if r.output.truncation == Truncation::Middle
            )
        })
        .expect("truncated");
    if let BlockContent::ToolResult(r) = &result_block.content {
        // persist failed -> observation degrades to head+tail, no artifact ref
        assert!(r.output.artifact.is_none());
        assert!(r.output.is_truncated());
    } else {
        panic!()
    }
}

#[tokio::test]
async fn completed_output_carries_reasoning_and_usage_unchanged() {
    // gate 12: reasoning signature + rich usage survive the staged driver losslessly
    let final_output = ModelOutput {
        response: ModelResponse {
            text: TextPayload::new("final"),
            tool_calls: vec![],
        },
        usage: Some(ModelUsage {
            input_tokens: 42,
            output_tokens: 7,
            cache_read_tokens: Some(11),
            cache_write_tokens: Some(3),
            reasoning_tokens: Some(5),
        }),
        stop_reason: ModelStopReason::EndTurn,
        reasoning: Some(ReasoningPayload {
            text: "thinking".into(),
            signature: Some("sig-xyz".into()),
        }),
    };
    let gw = RecordingGateway::scripted(vec![Ok(final_output.clone())]);
    let runner = runner_with(gw, vec![]);
    let cfg = TurnRunOptions::default();
    let out = runner.run(ctx("t1"), cfg, ctrl()).await;
    let completed = match out.result {
        TurnResult::Completed { final_output } => final_output,
        other => panic!("expected completion, got {other:?}"),
    };
    assert_eq!(
        serde_json::to_string(&completed).unwrap(),
        serde_json::to_string(&final_output).unwrap()
    );
}
