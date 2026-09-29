//! Driver-stack tests — the execution stack, config axes, executor
//! dispatch, cancellation, and traces. They exercise the runtime wiring,
//! not the kernel contract.

mod common;

use causa_kernel::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, AttemptNumber, BlockContent,
    CallControl, ContextBlock, DynamicToolSource, ModelInvokeErrorKind, ModelOutput, ModelResponse,
    ModelStopReason, ModelUsage, ProcessorContext, ProcessorError, ReasoningPayload, SourceError,
    StoreError, TextPayload, Tool, ToolBatch, ToolBatchProcessor, ToolCallContext, ToolDefinition,
    ToolExecutionError, ToolOutput, ToolResultPayload, ToolResultStatus, Truncation,
};
use causa_runtime::{
    DeduplicateProcessor, ExecutionOptions, FramePolicy, RetryPolicy, RunControl, ToolExecutor,
    ToolOutputBudgetProcessor, ToolProcessingChain, TurnInterruption, TurnPolicy, TurnResult,
    TurnRunOptions, TurnRunner, UnknownOutcomeConfig, UnknownOutcomePolicy, WindowBudget,
};
use common::{
    DropAllCompaction, EchoTool, FailTool, RecordingGateway, UnknownStopTool, ctrl, ctx, draft,
    endturn_output, options_with_limits, runner_with, tooluse_calls_output, tooluse_output,
};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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
        .position(|b| matches!(b.content(), BlockContent::ToolCall(_)))
        .unwrap();
    let pos_result = blocks
        .iter()
        .position(|b| matches!(b.content(), BlockContent::ToolResult(_)))
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
        backoff_base_ms: 0,
        backoff_max_ms: 0,
    };
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // Recorded 2 attempts with same invocation and frame
    let rec = gw.recorded();
    assert_eq!(rec.len(), 2);
    assert_eq!(rec[0].invocation_id, rec[1].invocation_id);
    assert_eq!(rec[0].frame.scope, rec[1].frame.scope);
    assert_eq!(rec[0].frame.round_id, rec[1].frame.round_id);
    assert_eq!(
        rec[0].frame.model_context.blocks.len(),
        rec[1].frame.model_context.blocks.len()
    );
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
        backoff_base_ms: 0,
        backoff_max_ms: 0,
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
    // Two identical echo calls in same batch => one rejected by an ordinary pre-processor.
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "dup",
            vec![draft("echo", json!({"x":1})), draft("echo", json!({"x":1}))],
        )),
        Ok(endturn_output("done")),
    ]);
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(EchoTool)])),
        ToolProcessingChain::builder()
            .before(Arc::new(DeduplicateProcessor))
            .build(),
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
async fn same_input_output_preserves_content_with_fresh_ids() {
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
        out.context.blocks().to_vec()
    }
    let a = run_once().await;
    let b = run_once().await;
    let normalize = |blocks: &[ContextBlock]| {
        blocks
            .iter()
            .map(|block| match block.content() {
                BlockContent::Parts(parts) => json!({"parts": parts}),
                BlockContent::ToolCall(call) => {
                    json!({"tool_name": call.tool_name, "arguments": call.arguments})
                }
                BlockContent::ToolResult(result) => json!({
                    "status": result.status,
                    "output": result.output.content,
                    "notes": result.notes,
                    "media": result.media,
                }),
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(normalize(&a), normalize(&b));
    assert_ne!(
        a[0].id(),
        b[0].id(),
        "runtime block IDs are fresh UUIDv7 values"
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
        async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
            let big = "a".repeat(1000);
            ToolResultPayload {
                call_block_id: ctx.call_block_id,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!(big)),
                media: Vec::new(),
                notes: Vec::new(),
            }
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
    let runner = TurnRunner::with_tool_processors(
        gw,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(BigTool)])),
        ToolProcessingChain::builder()
            .after(Arc::new(
                ToolOutputBudgetProcessor::new(10).with_artifact_store(Arc::new(MemStore)),
            ))
            .build(),
    );
    let out = runner.run(c, options_with_limits(5, 10), ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // Find truncated output
    let result_block = out
        .context
        .blocks()
        .iter()
        .find(|b| {
            matches!(b.content(), BlockContent::ToolResult(r) if r.output.truncation == Truncation::Middle)
        })
        .expect("truncated");
    if let BlockContent::ToolResult(r) = result_block.content() {
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

#[tokio::test]
async fn turn_deadline_notifies_inflight_tool_and_keeps_deadline_cause() {
    struct ParkUntilCancelled(Arc<tokio::sync::Notify>);
    #[async_trait::async_trait]
    impl Tool for ParkUntilCancelled {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "park".into(),
                description: "park until cancelled".into(),
                parameters: json!({"type":"object"}),
            }
        }

        async fn execute(
            &self,
            _ctx: &ToolCallContext,
            control: &CallControl,
        ) -> ToolResultPayload {
            let token = control.cancellation_token().clone();
            let notified = self.0.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                notified.notify_one();
            });
            std::future::pending().await
        }
    }

    let notified = Arc::new(tokio::sync::Notify::new());
    let gateway = RecordingGateway::scripted(vec![Ok(tooluse_output("", "park", json!({})))]);
    let runner = runner_with(
        gateway,
        vec![Arc::new(ParkUntilCancelled(notified.clone()))],
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(20);
    let out = runner
        .run(
            ctx("deadline-tool"),
            TurnRunOptions::default(),
            RunControl::new(tokio_util::sync::CancellationToken::new(), Some(deadline)),
        )
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::TurnDeadlineExceeded
        }
    ));
    tokio::time::timeout(std::time::Duration::from_secs(1), notified.notified())
        .await
        .expect("in-flight tool receives batch cancellation");
    let batch = out
        .uncommitted_tool_batch
        .expect("deadline keeps current U");
    assert_eq!(
        batch.results()[0].result().unwrap().1.status,
        ToolResultStatus::UnknownOutcome
    );
}

// ---------------------------------------------------------------------------
// Alignment coverage: acceptance gaps + regressions

// max_retries = 0 → initial attempt only

#[tokio::test]
async fn max_retries_zero_does_single_attempt() {
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![Err(ModelInvokeErrorKind::Transient)]);
    let runner = runner_with(gw.clone(), vec![]);
    let mut cfg = options_with_limits(5, 10);
    cfg.policy.retry = RetryPolicy {
        max_retries: 0,
        retry_timeouts: false,
        backoff_base_ms: 0,
        backoff_max_ms: 0,
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

// turn deadline exceeded → TurnDeadlineExceeded

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

// non-retryable model failure → Interrupted with round trace

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

// cross-batch identical call must not collide

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
        .filter_map(|b| match b.content() {
            BlockContent::ToolCall(_) => Some(b.id()),
            _ => None,
        })
        .collect();
    assert_eq!(call_ids.len(), 2);
    assert_ne!(call_ids[0], call_ids[1], "round-salted ids must differ");
}

// MaxTokens / Refusal reach their dedicated interruption causes

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

// UnknownOutcomePolicy::Continue → append result and continue

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
        async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
            ToolResultPayload {
                call_block_id: ctx.call_block_id,
                status: ToolResultStatus::UnknownOutcome,
                output: ToolOutput::new(json!({"unk": true})),
                media: Vec::new(),
                notes: Vec::new(),
            }
        }
    }
    let c = ctx("t1");
    let gw = RecordingGateway::scripted(vec![
        Ok(tooluse_output("call unkc", "unkc", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gw, vec![Arc::new(UnknownContinueTool)]);
    // The Continue action is host configuration, keyed by the executed
    // tool name — not a tool declaration.
    let mut unknown_outcome = UnknownOutcomeConfig::default();
    unknown_outcome
        .overrides
        .insert("unkc".into(), UnknownOutcomePolicy::Continue);
    let cfg = TurnRunOptions {
        policy: TurnPolicy {
            unknown_outcome,
            ..Default::default()
        },
        ..Default::default()
    };
    let out = runner.run(c, cfg, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert!(out.context.blocks().iter().any(
        |b| matches!(b.content(), BlockContent::ToolResult(r) if r.status == ToolResultStatus::UnknownOutcome)
    ));
}

// hung tool: executor call-deadline backstop yields UnknownOutcome;
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
        async fn execute(&self, _ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
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

// hung tool with Continue policy → backstop UnknownOutcome is
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
        async fn execute(&self, _ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
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
    let mut unknown_outcome = UnknownOutcomeConfig::default();
    unknown_outcome
        .overrides
        .insert("hungc".into(), UnknownOutcomePolicy::Continue);
    let cfg = TurnRunOptions {
        policy: TurnPolicy {
            unknown_outcome,
            ..Default::default()
        },
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

// parallel batch: one failing call does not abort the others

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
        .filter_map(|b| match b.content() {
            BlockContent::ToolResult(r) => Some(r.status.clone()),
            _ => None,
        })
        .collect();
    assert!(statuses.contains(&ToolResultStatus::Succeeded));
    assert!(statuses.contains(&ToolResultStatus::Failed));
}

// completion_order reflects real completion, not submission order

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
        async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            ToolResultPayload {
                call_block_id: ctx.call_block_id,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("slow")),
                media: Vec::new(),
                notes: Vec::new(),
            }
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
    // Trace calls are reported in declaration order…
    assert_eq!(batch.calls[0].tool_name, "slow");
    assert_eq!(batch.calls[1].tool_name, "echo");
    // …but completion_order starts with the fast echo call
    let fast_id = &batch.calls[1].call_block_id;
    assert_eq!(&batch.completion_order[0], fast_id);
    assert!(batch.calls[1].duration_ms < batch.calls[0].duration_ms);
}

// append_model_output validation branches

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

// ---- frame policy is driver-owned; evaluation stays canonical ----

#[tokio::test]
async fn frame_policy_from_options_shapes_projection_without_touching_facts() {
    let mut c = ctx("t1");
    c.append_input(
        causa_runtime::new_block_id(),
        TextPayload::new("hello"),
        "user",
    )
    .unwrap();
    // any non-empty content trips the trigger when the host
    // wires a real `TokenCounter` -- an absent counter estimates 0
    // and never trips the budget.
    struct CountPlusOne;
    impl causa_runtime::TokenCounter for CountPlusOne {
        fn estimate(&self, blocks: &[ContextBlock]) -> usize {
            blocks.len() + 100
        }
        fn estimate_value(&self, _value: &serde_json::Value) -> usize {
            1
        }
    }
    let frame_policy = FramePolicy {
        window_budget: WindowBudget {
            model_window_limit: 100,
            compaction_trigger: 1,
        },
        compaction: Some(Arc::new(DropAllCompaction)),
        token_counter: Some(Arc::new(CountPlusOne)),
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
        out.context.blocks()[0].content(),
        BlockContent::Parts(_)
    ));
}

#[tokio::test]
async fn completed_output_carries_reasoning_and_usage_unchanged() {
    // reasoning signature + rich usage survive the driver losslessly
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

#[tokio::test]
async fn processor_failure_returns_uncommitted_batch() {
    struct FailingProcessor;
    #[async_trait::async_trait]
    impl ToolBatchProcessor for FailingProcessor {
        async fn process(
            &self,
            batch: &mut ToolBatch,
            _ctx: &ProcessorContext<'_>,
        ) -> Result<(), ProcessorError> {
            batch.calls_mut()[0].push_note(TextPayload::new("kept on failure"));
            Err(ProcessorError::Failed("test failure".into()))
        }
    }

    let gateway = RecordingGateway::scripted(vec![Ok(tooluse_output(
        "call echo",
        "echo",
        json!({"a": 1}),
    ))]);
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(EchoTool)])),
        ToolProcessingChain::builder()
            .before(Arc::new(FailingProcessor))
            .build(),
    );
    let out = runner
        .run(ctx("t1"), options_with_limits(5, 10), ctrl())
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ProcessorFailed { .. }
        }
    ));
    let batch = out.uncommitted_tool_batch.expect("failure batch retained");
    assert_eq!(batch.completed_len(), 0);
    assert_eq!(batch.calls()[0].call().result_notes[0].0, "kept on failure");
    assert!(
        !out.context
            .blocks()
            .iter()
            .any(|block| matches!(block.content(), BlockContent::ToolResult(_)))
    );
}

#[tokio::test]
async fn preprocessor_error_preserves_mutations_for_explicit_rebuild() {
    struct MutateRejectThenFail;
    #[async_trait::async_trait]
    impl ToolBatchProcessor for MutateRejectThenFail {
        async fn process(
            &self,
            batch: &mut ToolBatch,
            _ctx: &ProcessorContext<'_>,
        ) -> Result<(), ProcessorError> {
            batch.calls_mut()[0]
                .input_mut()
                .expect("pending call input")
                .arguments["rewritten"] = json!(true);
            batch.calls_mut()[0].push_note(TextPayload::new("preflight note"));

            batch.calls_mut()[1].push_note(TextPayload::new("rejection note"));
            let rejected_call_id = batch.calls()[1].call().call_block_id;
            batch
                .resolve_at(
                    1,
                    causa_runtime::new_block_id(),
                    ToolResultPayload {
                        call_block_id: rejected_call_id,
                        status: ToolResultStatus::Rejected,
                        output: ToolOutput::new(json!("rejected during preflight")),
                        media: Vec::new(),
                        notes: Vec::new(),
                    },
                )
                .expect("the second declaration is pending");
            Err(ProcessorError::Failed(
                "preflight failed after edits".into(),
            ))
        }
    }

    struct CountTool(Arc<AtomicUsize>);
    #[async_trait::async_trait]
    impl Tool for CountTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "count".into(),
                description: "counts calls".into(),
                parameters: json!({"type": "object"}),
            }
        }

        async fn execute(
            &self,
            call: &ToolCallContext,
            _control: &CallControl,
        ) -> ToolResultPayload {
            self.0.fetch_add(1, Ordering::SeqCst);
            ToolResultPayload {
                call_block_id: call.call_block_id,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("executed")),
                media: Vec::new(),
                notes: Vec::new(),
            }
        }
    }

    struct CountProcessor(Arc<AtomicUsize>);
    #[async_trait::async_trait]
    impl ToolBatchProcessor for CountProcessor {
        async fn process(
            &self,
            _batch: &mut ToolBatch,
            _ctx: &ProcessorContext<'_>,
        ) -> Result<(), ProcessorError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let tool_calls = vec![
        draft("count", json!({"item": 1})),
        draft("count", json!({"item": 2})),
    ];
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output("", tool_calls.clone())),
        Ok(endturn_output("must not be requested")),
    ]);
    let tool_invocations = Arc::new(AtomicUsize::new(0));
    let later_processors = Arc::new(AtomicUsize::new(0));
    let failing = Arc::new(MutateRejectThenFail);
    let runner = TurnRunner::with_tool_processors(
        gateway.clone(),
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(CountTool(
            tool_invocations.clone(),
        ))])),
        ToolProcessingChain::builder()
            .before(failing.clone())
            .before(Arc::new(CountProcessor(later_processors.clone())))
            .after(Arc::new(CountProcessor(later_processors.clone())))
            .build(),
    );

    let out = runner
        .run(ctx("preflight-rebuild"), options_with_limits(5, 10), ctrl())
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ProcessorFailed { .. }
        }
    ));
    assert_eq!(gateway.recorded().len(), 1, "no next model round starts");
    assert_eq!(
        tool_invocations.load(Ordering::SeqCst),
        0,
        "executor is skipped"
    );
    assert_eq!(
        later_processors.load(Ordering::SeqCst),
        0,
        "later stages are skipped"
    );

    let uncommitted = out
        .uncommitted_tool_batch
        .expect("the edited batch is returned to the caller");
    assert_eq!(uncommitted.completed_len(), 1);
    let rejected = uncommitted.results()[0]
        .result()
        .expect("second declaration has its rejection result")
        .1;
    assert_eq!(rejected.status, ToolResultStatus::Rejected);
    assert_eq!(rejected.notes, [TextPayload::new("rejection note")]);
    let first = &uncommitted.calls()[0];
    assert_eq!(
        first.call().input.arguments,
        json!({"item": 1, "rewritten": true})
    );
    assert_eq!(
        first.call().result_notes,
        [TextPayload::new("preflight note")]
    );

    let declarations = out
        .context
        .blocks()
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolCall(payload) => Some((block.id(), payload.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(declarations.len(), 2);
    let returned_ids = uncommitted.declaration_ids();
    assert_eq!(returned_ids.len(), declarations.len());
    assert!(declarations.iter().all(|(id, _)| returned_ids.contains(id)));
    assert_eq!(
        declarations
            .iter()
            .map(|(_, call)| (&call.tool_name, &call.arguments))
            .collect::<Vec<_>>(),
        tool_calls
            .iter()
            .map(|call| (&call.tool_name, &call.arguments))
            .collect::<Vec<_>>(),
        "preflight edits do not rewrite committed model declarations"
    );
    assert!(
        out.context
            .blocks()
            .iter()
            .all(|block| !matches!(block.content(), BlockContent::ToolResult(_)))
    );

    // Recovery is explicit: rebuild from the original committed declarations,
    // then invoke the processor again. The original IDs and clean notes prove
    // no hidden continuation state or duplicate context registration exists.
    let rebuilt = ToolBatch::new(
        declarations
            .iter()
            .map(|(id, payload)| ToolCallContext::from_declaration(*id, payload))
            .collect(),
    )
    .unwrap();
    assert_eq!(
        rebuilt.declaration_ids(),
        declarations.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        "rebuilding from context restores declaration order"
    );
    assert!(
        rebuilt
            .calls()
            .iter()
            .all(|entry| entry.call().result_notes.is_empty())
    );

    let turn_id = ctx("preflight-rebuild").turn_id();
    let ids = rebuilt.declaration_ids();
    let recovery_control = CallControl::new(tokio_util::sync::CancellationToken::new(), None);
    let recovery_context = ProcessorContext {
        conversation_id: None,
        turn_id: &turn_id,
        round_id: causa_kernel::RoundId(0),
        declaration_order: &ids,
        control: &recovery_control,
    };
    let mut retry = rebuilt;
    assert!(
        failing
            .process(&mut retry, &recovery_context)
            .await
            .is_err()
    );
    assert_eq!(retry.completed_len(), 1);
    assert_eq!(
        retry.calls()[0].call().result_notes,
        [TextPayload::new("preflight note")]
    );
    assert_eq!(
        retry.results()[0].result().unwrap().1.notes,
        [TextPayload::new("rejection note")]
    );
    assert_eq!(
        out.context.blocks().len(),
        2,
        "explicit processor retry does not register duplicate context blocks"
    );
}

// ---- per-round tool-surface refresh ------------------------------------------------

#[tokio::test]
async fn dynamic_catalog_refreshes_between_model_rounds() {
    // A dynamic source whose listing changes when one of its tools runs
    // must reach the NEXT round's model request through the driver — the
    // run-start snapshot alone is not the catalog.
    struct Catalog {
        version: AtomicU64,
    }
    #[async_trait::async_trait]
    impl DynamicToolSource for Catalog {
        fn id(&self) -> &str {
            "catalog"
        }
        fn version(&self) -> u64 {
            self.version.load(Ordering::SeqCst)
        }
        async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
            Ok(vec![ToolDefinition {
                name: format!("tool_v{}", self.version()),
                description: String::new(),
                parameters: json!({"type": "object"}),
            }])
        }
        async fn invoke(
            &self,
            call: &ToolCallContext,
            _control: &CallControl,
        ) -> Result<ToolResultPayload, ToolExecutionError> {
            self.version.store(1, Ordering::SeqCst);
            Ok(ToolResultPayload {
                call_block_id: call.call_block_id,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("updated")),
                media: Vec::new(),
                notes: Vec::new(),
            })
        }
    }
    let source = Arc::new(Catalog {
        version: AtomicU64::new(0),
    });
    let executor = Arc::new(ToolExecutor::from_vec(vec![]));
    executor.register_dynamic(source).unwrap();
    let mut options = options_with_limits(5, 10);
    options.invocation.tool_surface = executor.tool_surface().await;
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("", "tool_v0", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let runner = TurnRunner::new(gateway.clone(), executor);
    let out = runner.run(ctx("catalog"), options, ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    let recorded = gateway.recorded();
    assert_eq!(recorded[0].tool_surface.definitions[0].name, "tool_v0");
    assert_eq!(
        recorded[1].tool_surface.definitions[0].name, "tool_v1",
        "second round ran on a stale catalog"
    );
}
