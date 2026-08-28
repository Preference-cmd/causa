use reimagine_context_kernel::*;
use serde_json::json;
use std::sync::{Arc, Mutex};

fn turn_id(s: &str) -> TurnId {
    TurnId::new(s)
}
fn ctx(s: &str) -> TurnContext {
    TurnContext::new(turn_id(s))
}

fn assistant_endturn(text: &str) -> ModelOutput {
    ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new(text),
            tool_calls: vec![],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    }
}
fn assistant_tooluse(text: &str, tool_name: &str, args: serde_json::Value) -> ModelOutput {
    ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new(text),
            tool_calls: vec![ToolCallDraft {
                tool_name: tool_name.into(),
                arguments: args,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    }
}

// ---- FakeGateway that records requests ----
struct RecordingGateway {
    outputs: Mutex<Vec<Result<ModelOutput, ModelInvokeErrorKind>>>,
    recorded: Mutex<Vec<ModelRequest>>,
}
impl RecordingGateway {
    fn new(outputs: Vec<Result<ModelOutput, ModelInvokeErrorKind>>) -> Self {
        Self {
            outputs: Mutex::new(outputs),
            recorded: Mutex::new(vec![]),
        }
    }
}
#[async_trait::async_trait]
impl ModelGateway for RecordingGateway {
    async fn invoke(
        &self,
        req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.recorded.lock().unwrap().push(req.clone());
        let mut g = self.outputs.lock().unwrap();
        if g.is_empty() {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                "no outputs",
            ));
        }
        match g.remove(0) {
            Ok(o) => Ok(o),
            Err(k) => Err(ModelInvokeError::new(k, "fake")),
        }
    }
}

struct EchoTool;
#[async_trait::async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echo".into(),
            parameters: json!({"type":"object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput {
                content: json!({"echo": ctx.arguments}),
                truncation: Truncation::None,
                meta: None,
                artifact: None,
            },
        })
    }
}

struct FailTool;
#[async_trait::async_trait]
impl Tool for FailTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fail".into(),
            description: "fail".into(),
            parameters: json!({"type":"object"}),
        }
    }
    async fn execute(&self, _ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: _ctx.call_id.clone(),
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(json!({"err": "fail"})),
        })
    }
}

struct UnknownStopTool;
#[async_trait::async_trait]
impl Tool for UnknownStopTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "unk".into(),
            description: "unk".into(),
            parameters: json!({"type":"object"}),
        }
    }
    fn unknown_outcome_policy(&self) -> UnknownOutcomePolicy {
        UnknownOutcomePolicy::Stop
    }
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::UnknownOutcome,
            output: ToolOutput::new(json!({"unk": true})),
        })
        .with_policy(UnknownOutcomePolicy::Stop)
    }
}

fn limits(r: u32, t: u32) -> TurnLimits {
    TurnLimits {
        max_model_rounds: r,
        max_tool_calls: t,
    }
}

#[tokio::test]
async fn empty_frame_deterministic() {
    let c = ctx("t1");
    let f0 = c.frame_sync(RoundId(0)).unwrap();
    let f1 = c.frame_sync(RoundId(0)).unwrap();
    assert_eq!(f0.frame_id, f1.frame_id);
    assert!(f0.model_context.blocks.is_empty());
}

#[tokio::test]
async fn append_input_and_frame_order() {
    let mut c = ctx("t1");
    c.apply_inputs(vec![
        InputPayload::RequestUser(TextPayload::new("hello")),
        InputPayload::InstructionSystem(TextPayload::new("sys")),
    ])
    .unwrap();
    let f = c.frame_sync(RoundId(0)).unwrap();
    assert_eq!(f.model_context.blocks.len(), 2);
    // Order preserved
    assert!(matches!(
        f.model_context.blocks[0].payload,
        BlockPayload::RequestUser(_)
    ));
    assert!(matches!(
        f.model_context.blocks[1].payload,
        BlockPayload::InstructionSystem(_)
    ));
    // After frame, seal should make further append fail
    let mut c2 =
        TurnContext::from_validated_blocks(turn_id("t1"), c.snapshot_blocks(), c.version())
            .unwrap();
    // not sealed, should allow
    c2.append_input(InputPayload::RequestUser(TextPayload::new("more")))
        .unwrap();
}

#[tokio::test]
async fn sealed_turn_append_closed() {
    let mut c = ctx("t1");
    c.append_input(InputPayload::RequestUser(TextPayload::new("hi")))
        .unwrap();
    let gateway = Arc::new(RecordingGateway::new(vec![Ok(assistant_endturn("done"))]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![]));
    let runner = TurnRunner::new(gateway, exec);
    let cfg = TurnRunConfig {
        limits: limits(5, 10),
        ..Default::default()
    };
    let outcome = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    assert!(outcome.context.is_sealed());
    let mut sealed = outcome.context;
    assert!(matches!(
        sealed.append_input(InputPayload::RequestUser(TextPayload::new("x"))),
        Err(ContextError::SealedTurn)
    ));
    assert!(matches!(
        sealed.apply_model_output(
            InvocationId {
                turn_id: turn_id("t1"),
                round_id: RoundId(1)
            },
            assistant_endturn("y")
        ),
        Err(ContextError::SealedTurn)
    ));
}

#[tokio::test]
async fn final_assistant_completes_once() {
    let c = ctx("t1");
    let gw = Arc::new(RecordingGateway::new(vec![Ok(assistant_endturn("final"))]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![]));
    let runner = TurnRunner::new(gw, exec);
    let cfg = TurnRunConfig {
        limits: limits(5, 10),
        ..Default::default()
    };
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 1);
    assert_eq!(out.context.blocks().len(), 1);
}

#[tokio::test]
async fn tool_calls_drive_next_frame_and_causality() {
    let c = ctx("t1");
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(assistant_tooluse("call echo", "echo", json!({"a":1}))),
        Ok(assistant_endturn("done")),
    ]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![Arc::new(EchoTool)]));
    let runner = TurnRunner::new(gw, exec);
    let cfg = TurnRunConfig {
        limits: limits(5, 10),
        ..Default::default()
    };
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // blocks: assistant tool call + tool result + final assistant (with optional assistant text blocks)
    // Should have at least 3 blocks, causality: tool.result follows tool.call
    let blocks = out.context.blocks();
    let pos_call = blocks
        .iter()
        .position(|b| matches!(&b.payload, BlockPayload::ToolCall(_)))
        .unwrap();
    let pos_result = blocks
        .iter()
        .position(|b| matches!(&b.payload, BlockPayload::ToolResult(_)))
        .unwrap();
    assert!(pos_result > pos_call);
}

#[tokio::test]
async fn retry_same_frame_only_attempt_increments_and_no_block_on_failure() {
    let c = ctx("t1");
    let gw = Arc::new(RecordingGateway::new(vec![
        Err(ModelInvokeErrorKind::Transient),
        Ok(assistant_endturn("ok")),
    ]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![]));
    let runner = TurnRunner::new(gw.clone(), exec);
    let cfg = TurnRunConfig {
        retry: RetryPolicy {
            max_retries: 1,
            retry_timeouts: false,
        },
        limits: limits(5, 10),
        ..Default::default()
    };
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // Recorded 2 attempts with same invocation and frame
    let rec = gw.recorded.lock().unwrap();
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
    let gw = Arc::new(RecordingGateway::new(vec![
        Err(ModelInvokeErrorKind::Transient),
        Err(ModelInvokeErrorKind::Transient),
    ]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![]));
    let runner = TurnRunner::new(gw.clone(), exec);
    let cfg = TurnRunConfig {
        retry: RetryPolicy {
            max_retries: 1,
            retry_timeouts: false,
        },
        limits: limits(5, 10),
        ..Default::default()
    };
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::RetryExhausted { .. }
        }
    ));
    let rec = gw.recorded.lock().unwrap();
    assert_eq!(rec.len(), 2);
}

#[tokio::test]
async fn tool_failure_is_observation_not_terminal_and_next_round() {
    let c = ctx("t1");
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(assistant_tooluse("call fail", "fail", json!({}))),
        Ok(assistant_endturn("done after fail")),
    ]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![Arc::new(FailTool)]));
    let runner = TurnRunner::new(gw, exec);
    let cfg = TurnRunConfig {
        limits: limits(5, 10),
        ..Default::default()
    };
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 2);
}

#[tokio::test]
async fn dedup_same_batch_rejected_and_parallel_single_failure_not_abort() {
    let c = ctx("t1");
    // Two identical echo calls in same batch => one rejected
    let out1 = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("dup"),
            tool_calls: vec![
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"x":1}),
                },
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"x":1}),
                },
            ],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(out1),
        Ok(assistant_endturn("done")),
    ]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![Arc::new(EchoTool)]));
    let runner = TurnRunner::new(gw, exec);
    let cfg = TurnRunConfig {
        limits: limits(5, 10),
        ..Default::default()
    };
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
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
    let gw = Arc::new(RecordingGateway::new(vec![Ok(assistant_tooluse(
        "call unk",
        "unk",
        json!({}),
    ))]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![Arc::new(UnknownStopTool)]));
    let runner = TurnRunner::new(gw, exec);
    let cfg = TurnRunConfig {
        limits: limits(5, 10),
        ..Default::default()
    };
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
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
        let gw = Arc::new(RecordingGateway::new(vec![
            Ok(assistant_tooluse("hi", "echo", json!({"k":1}))),
            Ok(assistant_endturn("bye")),
        ]));
        let exec = Arc::new(ToolExecutor::from_vec(vec![Arc::new(EchoTool)]));
        let runner = TurnRunner::new(gw, exec);
        let cfg = TurnRunConfig {
            limits: limits(5, 10),
            ..Default::default()
        };
        let out = runner
            .run(
                c,
                cfg,
                RunControl::new(tokio_util::sync::CancellationToken::new(), None),
            )
            .await;
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
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(assistant_tooluse("call big", "big", json!({}))),
        Ok(assistant_endturn("done")),
    ]));
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
    let exec = Arc::new(ToolExecutor::from_vec(vec![Arc::new(BigTool)]));
    let runner = TurnRunner::new(gw, exec);
    let cfg = TurnRunConfig {
        limits: limits(5, 10),
        tool_output_limits: ToolOutputLimits { max_tokens: 10 },
        artifact_store: Some(Arc::new(MemStore)),
        ..Default::default()
    };
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    // Find truncated output
    let result_block = out.context.blocks().iter().find(|b| matches!(&b.payload, BlockPayload::ToolResult(r) if r.output.truncation == Truncation::Middle)).expect("truncated");
    if let BlockPayload::ToolResult(r) = &result_block.payload {
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
    let gw = Arc::new(RecordingGateway::new(vec![Ok(assistant_endturn(
        "should not",
    ))]));
    let exec = Arc::new(ToolExecutor::from_vec(vec![]));
    let runner = TurnRunner::new(gw, exec);
    let cfg = TurnRunConfig {
        limits: limits(5, 10),
        ..Default::default()
    };
    let out = runner.run(c, cfg, RunControl::new(token, None)).await;
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

fn runner_with(
    gw: Arc<RecordingGateway>,
    tools: Vec<Arc<dyn Tool>>,
    cfg: TurnRunConfig,
) -> (TurnRunner, TurnRunConfig) {
    (
        TurnRunner::new(gw, Arc::new(ToolExecutor::from_vec(tools))),
        cfg,
    )
}

// [P2 acceptance #10] max_retries = 0 → initial attempt only
#[tokio::test]
async fn max_retries_zero_does_single_attempt() {
    let c = ctx("t1");
    let gw = Arc::new(RecordingGateway::new(vec![Err(
        ModelInvokeErrorKind::Transient,
    )]));
    let (runner, cfg) = runner_with(
        gw.clone(),
        vec![],
        TurnRunConfig {
            retry: RetryPolicy {
                max_retries: 0,
                retry_timeouts: false,
            },
            limits: limits(5, 10),
            ..Default::default()
        },
    );
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::RetryExhausted {
                last_kind: ModelInvokeErrorKind::Transient,
                ..
            }
        }
    ));
    assert_eq!(gw.recorded.lock().unwrap().len(), 1);
}

// [P2 acceptance #13] turn deadline exceeded → TurnDeadlineExceeded
#[tokio::test]
async fn turn_deadline_yields_deadline_exceeded() {
    let c = ctx("t1");
    let gw = Arc::new(RecordingGateway::new(vec![]));
    let (runner, cfg) = runner_with(gw.clone(), vec![], TurnRunConfig::default());
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
    assert_eq!(gw.recorded.lock().unwrap().len(), 0);
}

// [P2 acceptance #13] non-retryable model failure → Interrupted with round trace
#[tokio::test]
async fn non_retryable_error_records_round_trace() {
    let c = ctx("t1");
    let gw = Arc::new(RecordingGateway::new(vec![Err(
        ModelInvokeErrorKind::Permanent,
    )]));
    let (runner, cfg) = runner_with(gw, vec![], TurnRunConfig::default());
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
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
    let same_call = || assistant_tooluse("retry read", "echo", json!({"path": "A"}));
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(same_call()),
        Ok(same_call()),
        Ok(assistant_endturn("done")),
    ]));
    let (runner, cfg) = runner_with(gw, vec![Arc::new(EchoTool)], TurnRunConfig::default());
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(
        matches!(out.result, TurnResult::Completed { .. }),
        "expected completion, got {:?}",
        out.result
    );
    let call_ids: Vec<_> = out
        .context
        .blocks()
        .iter()
        .filter_map(|b| match &b.payload {
            BlockPayload::ToolCall(tc) => Some(tc.call_id.clone()),
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
        let gw = Arc::new(RecordingGateway::new(vec![Ok(ModelOutput {
            assistant: AssistantPayload {
                text: TextPayload::new("cut"),
                tool_calls: vec![],
            },
            usage: None,
            stop_reason: stop,
            reasoning: None,
        })]));
        let (runner, cfg) = runner_with(gw, vec![], TurnRunConfig::default());
        let out = runner
            .run(
                c,
                cfg,
                RunControl::new(tokio_util::sync::CancellationToken::new(), None),
            )
            .await;
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
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(assistant_tooluse("call unkc", "unkc", json!({}))),
        Ok(assistant_endturn("done")),
    ]));
    let (runner, cfg) = runner_with(
        gw,
        vec![Arc::new(UnknownContinueTool)],
        TurnRunConfig::default(),
    );
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert!(out.context.blocks().iter().any(
        |b| matches!(&b.payload, BlockPayload::ToolResult(r) if r.status == ToolResultStatus::UnknownOutcome)
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
    let gw = Arc::new(RecordingGateway::new(vec![Ok(assistant_tooluse(
        "call hung",
        "hung",
        json!({}),
    ))]));
    let (runner, cfg) = runner_with(
        gw,
        vec![Arc::new(HungTool)],
        TurnRunConfig {
            call_timeout: Some(std::time::Duration::from_millis(50)),
            ..Default::default()
        },
    );
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
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
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(assistant_tooluse("call hungc", "hungc", json!({}))),
        Ok(assistant_endturn("done")),
    ]));
    let (runner, cfg) = runner_with(
        gw,
        vec![Arc::new(HungContinueTool)],
        TurnRunConfig {
            call_timeout: Some(std::time::Duration::from_millis(50)),
            ..Default::default()
        },
    );
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 2);
}

// [P3 acceptance #14] parallel batch: one failing call does not abort the others
#[tokio::test]
async fn parallel_batch_partial_failure_does_not_abort() {
    let c = ctx("t1");
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(ModelOutput {
            assistant: AssistantPayload {
                text: TextPayload::new("two calls"),
                tool_calls: vec![
                    ToolCallDraft {
                        tool_name: "echo".into(),
                        arguments: json!({"a": 1}),
                    },
                    ToolCallDraft {
                        tool_name: "fail".into(),
                        arguments: json!({}),
                    },
                ],
            },
            usage: None,
            stop_reason: ModelStopReason::ToolUse,
            reasoning: None,
        }),
        Ok(assistant_endturn("done")),
    ]));
    let (runner, cfg) = runner_with(
        gw,
        vec![Arc::new(EchoTool), Arc::new(FailTool)],
        TurnRunConfig::default(),
    );
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    let statuses: Vec<_> = out
        .context
        .blocks()
        .iter()
        .filter_map(|b| match &b.payload {
            BlockPayload::ToolResult(r) => Some(r.status.clone()),
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
    let gw = Arc::new(RecordingGateway::new(vec![
        Ok(ModelOutput {
            assistant: AssistantPayload {
                text: TextPayload::new("mixed"),
                tool_calls: vec![
                    ToolCallDraft {
                        tool_name: "slow".into(),
                        arguments: json!({}),
                    },
                    ToolCallDraft {
                        tool_name: "echo".into(),
                        arguments: json!({}),
                    },
                ],
            },
            usage: None,
            stop_reason: ModelStopReason::ToolUse,
            reasoning: None,
        }),
        Ok(assistant_endturn("done")),
    ]));
    let (runner, cfg) = runner_with(
        gw,
        vec![Arc::new(SlowTool), Arc::new(EchoTool)],
        TurnRunConfig::default(),
    );
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
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

// [P1 Phase 1] apply_model_output validation branches
#[tokio::test]
async fn apply_model_output_rejects_invalid_outputs() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    // EndTurn must not carry tool_calls
    let bad_endturn = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![ToolCallDraft {
                tool_name: "echo".into(),
                arguments: json!({}),
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    };
    assert!(matches!(
        c.apply_model_output(inv.clone(), bad_endturn),
        Err(ContextError::InvalidSequence(_))
    ));
    // ToolUse requires non-empty tool name
    let empty_name = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![ToolCallDraft {
                tool_name: "  ".into(),
                arguments: json!({}),
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    assert!(matches!(
        c.apply_model_output(inv.clone(), empty_name),
        Err(ContextError::InvalidSequence(_))
    ));
    // ToolUse requires object arguments
    let non_object = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![ToolCallDraft {
                tool_name: "echo".into(),
                arguments: json!(42),
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    assert!(matches!(
        c.apply_model_output(inv.clone(), non_object),
        Err(ContextError::InvalidSequence(_))
    ));
    // identical (tool, args) twice in one batch is fine — positions differ
    let dup_batch = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                },
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                },
            ],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    assert!(c.apply_model_output(inv.clone(), dup_batch).is_ok());
    // sealed turn rejects everything — obtain a sealed context via a completed run
    let gw = Arc::new(RecordingGateway::new(vec![Ok(assistant_endturn("final"))]));
    let (runner, cfg) = runner_with(gw, vec![], TurnRunConfig::default());
    let out = runner
        .run(
            ctx("t2"),
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
    let mut sealed = out.context;
    assert!(sealed.is_sealed());
    let sealed_inv = InvocationId {
        turn_id: turn_id("t2"),
        round_id: RoundId(1),
    };
    assert!(matches!(
        sealed.apply_model_output(sealed_inv, assistant_endturn("y")),
        Err(ContextError::SealedTurn)
    ));
}

// [P1 Phase 1] from_validated_blocks rejects corrupt state
#[tokio::test]
async fn from_validated_blocks_rejects_corrupt_state() {
    fn block(seq: u64, payload: BlockPayload) -> ContextBlock {
        let tid = turn_id("t1");
        ContextBlock {
            id: BlockId {
                turn_id: tid,
                sequence: BlockSequence(seq),
            },
            sequence: BlockSequence(seq),
            meta: BlockMeta::default(),
            payload,
        }
    }
    // wrong turn_id
    let mut b = block(0, BlockPayload::RequestUser(TextPayload::new("hi")));
    b.id.turn_id = turn_id("other");
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), vec![b], ContextVersion(1)),
        Err(ContextError::InvalidSequence(_))
    ));
    // non-contiguous sequence
    let blocks = vec![
        block(0, BlockPayload::RequestUser(TextPayload::new("a"))),
        block(2, BlockPayload::RequestUser(TextPayload::new("b"))),
    ];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(2)),
        Err(ContextError::InvalidSequence(_))
    ));
    // duplicate tool.call ids
    let blocks = vec![
        block(
            0,
            BlockPayload::ToolCall(ToolCallPayload {
                call_id: ToolCallId::new("dup"),
                tool_name: "echo".into(),
                arguments: json!({}),
            }),
        ),
        block(
            1,
            BlockPayload::ToolCall(ToolCallPayload {
                call_id: ToolCallId::new("dup"),
                tool_name: "echo".into(),
                arguments: json!({}),
            }),
        ),
    ];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(2)),
        Err(ContextError::DuplicateToolCallId(_))
    ));
    // unpaired tool.result
    let blocks = vec![block(
        0,
        BlockPayload::ToolResult(ToolResultPayload {
            call_id: ToolCallId::new("ghost"),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({})),
        }),
    )];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(1)),
        Err(ContextError::UnpairedToolResult(_))
    ));
}

// [P3] MaxToolCalls counts raw calls (incl. rejected duplicates) and records trace
#[tokio::test]
async fn max_tool_calls_interrupt_records_total() {
    let c = ctx("t1");
    let two_calls = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("two"),
            tool_calls: vec![
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                },
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                },
            ],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    let gw = Arc::new(RecordingGateway::new(vec![Ok(two_calls)]));
    let (runner, cfg) = runner_with(
        gw,
        vec![Arc::new(EchoTool)],
        TurnRunConfig {
            limits: limits(5, 1), // 2 raw calls > limit 1
            ..Default::default()
        },
    );
    let out = runner
        .run(
            c,
            cfg,
            RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        )
        .await;
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
