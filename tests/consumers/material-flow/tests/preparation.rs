//! C3/E3 input ownership and C6/E4 a concrete request-budget consumer.
#![cfg(feature = "runtime")]

use async_trait::async_trait;
use causa::kernel::{
    BlockContent, BlockId, BlockMeta, CallControl, CancellationToken, ContentPart, Context,
    ContextBlock, ContextFrame, ContextPreparer, GenerationOptions, MediaRef, ModelGateway,
    ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRef, ModelRequest, ModelResponse,
    ModelStopReason, PrepareError, RoundInfo, TextPayload, Tool, ToolCallContext, ToolCallDraft,
    ToolDefinition, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use causa::runtime::{
    RunControl, ToolExecutor, ToolExecutorOptions, TurnInterruption, TurnResult, TurnRunOptions,
    TurnRunner,
};
use serde_json::json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn id(value: u128) -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(value))
}
fn text(value: u128, content: &str) -> ContextBlock {
    ContextBlock::new(
        id(value),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(content))]),
        BlockMeta::default(),
    )
}
fn output(tool_use: bool) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(if tool_use { "checking" } else { "done" }),
            tool_calls: if tool_use {
                vec![ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({}),
                    provider_call_id: Some("provider-echo".into()),
                }]
            } else {
                vec![]
            },
        },
        usage: None,
        stop_reason: if tool_use {
            ModelStopReason::ToolUse
        } else {
            ModelStopReason::EndTurn
        },
        reasoning: None,
    }
}
struct CapturingGateway {
    requests: Mutex<Vec<ModelRequest>>,
    responses: Mutex<VecDeque<Result<ModelOutput, ModelInvokeError>>>,
}
#[async_trait]
impl ModelGateway for CapturingGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.requests.lock().unwrap().push(request.clone());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected model request")
    }
}
fn gateway(responses: Vec<Result<ModelOutput, ModelInvokeError>>) -> Arc<CapturingGateway> {
    Arc::new(CapturingGateway {
        requests: Mutex::new(vec![]),
        responses: Mutex::new(responses.into()),
    })
}
struct Retry(Arc<CapturingGateway>);
#[async_trait]
impl ModelGateway for Retry {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        for ordinal in 0..3 {
            control.check().unwrap();
            match self.0.invoke(request, control).await {
                Err(error) if error.kind == ModelInvokeErrorKind::Transient && ordinal < 2 => {
                    continue;
                }
                result => return result,
            }
        }
        unreachable!()
    }
}
struct Echo {
    definition: ToolDefinition,
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for Echo {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }
    async fn execute(&self, call: &ToolCallContext, _: &CallControl) -> ToolResultPayload {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ToolResultPayload {
            call_block_id: call.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("echoed")),
            media: vec![],
            notes: vec![],
        }
    }
}
fn echo(definition: ToolDefinition, calls: Arc<AtomicUsize>) -> Arc<ToolExecutor> {
    Arc::new(
        ToolExecutor::new(
            vec![Arc::new(Echo { definition, calls })],
            ToolExecutorOptions::default(),
        )
        .unwrap(),
    )
}
fn definition() -> ToolDefinition {
    ToolDefinition {
        name: "echo".into(),
        description: "echo tool".into(),
        parameters: json!({"type":"object"}),
    }
}

#[derive(Default)]
struct InputState {
    pending: VecDeque<ContextBlock>,
    stages: Vec<&'static str>,
    rounds: Vec<u32>,
}
struct InputPreparer {
    source: Mutex<InputState>,
    fail_after_commit: bool,
}
#[async_trait]
impl ContextPreparer for InputPreparer {
    async fn prepare(
        &self,
        context: &mut Context,
        round: &RoundInfo<'_>,
        control: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        control.check().unwrap();
        let mut source = self.source.lock().unwrap();
        source.rounds.push(round.invocation_id.round_id.0);
        if let Some(block) = source.pending.front().cloned() {
            // This source owns the pending block until Context has accepted it.
            source.stages.push("read");
            context
                .apply(Vec::new(), vec![block])
                .map_err(|error| PrepareError {
                    message: error.to_string(),
                })?;
            source.stages.push("committed");
            source.pending.pop_front();
            source.stages.push("confirmed");
            if self.fail_after_commit {
                return Err(PrepareError {
                    message: "failure after committed input".into(),
                });
            }
        }
        let mut frame = context.frame();
        frame.blocks.push(text(
            1000 + u128::from(round.invocation_id.round_id.0),
            &format!("observation round {}", round.invocation_id.round_id.0),
        ));
        Ok(frame)
    }
}
fn input_preparer(blocks: Vec<ContextBlock>, fail_after_commit: bool) -> Arc<InputPreparer> {
    Arc::new(InputPreparer {
        source: Mutex::new(InputState {
            pending: blocks.into(),
            ..Default::default()
        }),
        fail_after_commit,
    })
}
fn includes_text(request: &ModelRequest, wanted: &str) -> usize {
    request
        .frame
        .blocks
        .iter()
        .filter(|block| match block.content() {
            BlockContent::Parts(parts) => parts
                .iter()
                .any(|part| matches!(part, ContentPart::Text(value) if value.0 == wanted)),
            _ => false,
        })
        .count()
}
fn control() -> RunControl {
    RunControl::new(CancellationToken::new(), None)
}

#[tokio::test]
async fn source_commits_and_confirms_each_input_once_and_observation_refreshes_per_invocation() {
    let source = input_preparer(
        vec![text(100, "first input"), text(101, "second input")],
        false,
    );
    let inner = gateway(vec![
        Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Transient,
            "first failure",
        )),
        Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Transient,
            "second failure",
        )),
        Ok(output(true)),
        Ok(output(false)),
    ]);
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = TurnRunner::new(
        Arc::new(Retry(inner.clone())),
        echo(definition(), calls.clone()),
    );
    let mut options = TurnRunOptions::new(ModelRef::new("input-model"));
    options.preparer = Some(source.clone());
    let outcome = runner
        .run(
            causa::kernel::TurnId::new("input"),
            Context::new(),
            options,
            control(),
        )
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let source_state = source.source.lock().unwrap();
    assert!(source_state.pending.is_empty());
    assert_eq!(source_state.rounds, vec![0, 1]);
    assert_eq!(
        source_state.stages,
        vec![
            "read",
            "committed",
            "confirmed",
            "read",
            "committed",
            "confirmed"
        ]
    );
    let requests = inner.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    for request in &requests[..3] {
        assert_eq!(includes_text(request, "first input"), 1);
        assert_eq!(includes_text(request, "second input"), 0);
        assert_eq!(includes_text(request, "observation round 0"), 1);
        assert_eq!(request.invocation_id, requests[0].invocation_id);
        assert_eq!(request.frame.blocks, requests[0].frame.blocks);
    }
    assert_eq!(includes_text(&requests[3], "first input"), 1);
    assert_eq!(includes_text(&requests[3], "second input"), 1);
    assert_eq!(includes_text(&requests[3], "observation round 0"), 0);
    assert_eq!(includes_text(&requests[3], "observation round 1"), 1);
    assert!(
        outcome
            .context
            .blocks()
            .iter()
            .all(|block| block.id() != id(1000) && block.id() != id(1001))
    );
    assert_eq!(
        outcome
            .context
            .blocks()
            .iter()
            .filter(|block| block.id() == id(100))
            .count(),
        1
    );
    assert_eq!(
        outcome
            .context
            .blocks()
            .iter()
            .filter(|block| block.id() == id(101))
            .count(),
        1
    );
}

#[tokio::test]
async fn failed_material_commit_keeps_source_pending_and_does_not_confirm_or_invoke() {
    let initial = text(100, "existing");
    let source = input_preparer(vec![text(100, "conflicting input")], false);
    let gateway = gateway(vec![]);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap()),
    );
    let mut options = TurnRunOptions::new(ModelRef::new("input-model"));
    options.preparer = Some(source.clone());
    let outcome = runner
        .run(
            causa::kernel::TurnId::new("failure"),
            Context::from_blocks(vec![initial.clone()]).unwrap(),
            options,
            control(),
        )
        .await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::PrepareFailed { .. }
        }
    ));
    assert_eq!(outcome.context.blocks(), &[initial]);
    let state = source.source.lock().unwrap();
    assert_eq!(state.pending.len(), 1);
    assert_eq!(state.pending[0], text(100, "conflicting input"));
    assert_eq!(state.stages, vec!["read"]);
    assert!(gateway.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn preparation_failure_after_input_commit_returns_that_material_and_source_confirmation() {
    let source = input_preparer(vec![text(100, "accepted input")], true);
    let gateway = gateway(vec![]);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap()),
    );
    let mut options = TurnRunOptions::new(ModelRef::new("input-model"));
    options.preparer = Some(source.clone());
    let outcome = runner
        .run(
            causa::kernel::TurnId::new("post-commit-failure"),
            Context::new(),
            options,
            control(),
        )
        .await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::PrepareFailed { .. }
        }
    ));
    assert_eq!(outcome.context.blocks(), &[text(100, "accepted input")]);
    let state = source.source.lock().unwrap();
    assert!(state.pending.is_empty());
    assert_eq!(state.stages, vec!["read", "committed", "confirmed"]);
    assert!(gateway.requests.lock().unwrap().is_empty());
}

/// Budget units are serialized bytes plus explicit media/output surcharges.
/// They estimate request cost; they are not a tokenizer or a token guarantee.
#[derive(Debug, Clone)]
struct Estimate {
    blocks: usize,
    tools: usize,
    media: usize,
    output: usize,
    model: usize,
}
impl Estimate {
    fn total(&self) -> usize {
        self.blocks + self.tools + self.media + self.output + self.model
    }
}
fn estimate(context: &Context, round: &RoundInfo<'_>) -> Estimate {
    let media_count = context
        .blocks()
        .iter()
        .map(|block| match block.content() {
            BlockContent::Parts(parts) => parts
                .iter()
                .filter(|part| matches!(part, ContentPart::Media(_)))
                .count(),
            BlockContent::ToolResult(result) => result.media.len(),
            _ => 0,
        })
        .sum::<usize>();
    Estimate {
        blocks: serde_json::to_vec(context.blocks()).unwrap().len(),
        tools: serde_json::to_vec(&round.tool_surface.definitions)
            .unwrap()
            .len(),
        media: media_count * 256,
        output: usize::try_from(round.generation.max_tokens.unwrap_or(64)).unwrap() * 4,
        model: round.model.0.len(),
    }
}
struct BudgetPreparer {
    maximum: usize,
    estimates: Mutex<Vec<Estimate>>,
}
#[async_trait]
impl ContextPreparer for BudgetPreparer {
    async fn prepare(
        &self,
        context: &mut Context,
        round: &RoundInfo<'_>,
        _: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        let estimate = estimate(context, round);
        let total = estimate.total();
        self.estimates.lock().unwrap().push(estimate);
        if total > self.maximum {
            return Err(PrepareError {
                message: format!(
                    "estimated request cost {total} exceeds {} units",
                    self.maximum
                ),
            });
        }
        Ok(context.frame())
    }
}
#[tokio::test]
async fn schema_only_threshold_crossing_changes_request_admission_with_identical_blocks() {
    let context = Context::from_blocks(vec![ContextBlock::new(
        id(100),
        BlockContent::Parts(vec![
            ContentPart::Text(TextPayload::new("question")),
            ContentPart::Media(MediaRef::new("image/png", "asset-reference")),
        ]),
        BlockMeta::default(),
    )])
    .unwrap();
    let baseline_blocks = context.blocks().to_vec();
    let model = ModelRef::new("budget-model");
    let generation = GenerationOptions {
        max_tokens: Some(32),
        ..Default::default()
    };
    let small = definition();
    let mut large = small.clone();
    large.parameters = json!({"type":"object","properties":{"detail":{"type":"string","description":"schema explanation ".repeat(200)}}});
    let small_tools = causa::kernel::ToolSurface::from_definitions(vec![small.clone()]);
    let large_tools = causa::kernel::ToolSurface::from_definitions(vec![large.clone()]);
    let invocation = causa::kernel::InvocationId {
        turn_id: causa::kernel::TurnId::new("budget"),
        round_id: causa::kernel::RoundId(0),
    };
    let small_cost = estimate(
        &context,
        &RoundInfo {
            invocation_id: &invocation,
            model: &model,
            tool_surface: &small_tools,
            generation: &generation,
        },
    );
    let large_cost = estimate(
        &context,
        &RoundInfo {
            invocation_id: &invocation,
            model: &model,
            tool_surface: &large_tools,
            generation: &generation,
        },
    );
    assert_eq!(small_cost.blocks, large_cost.blocks);
    assert_eq!(small_cost.media, 256);
    assert_eq!(small_cost.output, 128);
    assert!(large_cost.tools > small_cost.tools);
    let maximum = (small_cost.total() + large_cost.total()) / 2;
    let budget = Arc::new(BudgetPreparer {
        maximum,
        estimates: Mutex::new(vec![]),
    });
    let admitted_gateway = gateway(vec![Ok(output(false))]);
    let rejected_gateway = gateway(vec![]);
    for (definition, gateway, expected) in [
        (small, admitted_gateway.clone(), true),
        (large, rejected_gateway.clone(), false),
    ] {
        let runner = TurnRunner::new(gateway, echo(definition, Arc::new(AtomicUsize::new(0))));
        let mut options = TurnRunOptions::new(model.clone());
        options.generation = generation.clone();
        options.preparer = Some(budget.clone());
        let outcome = runner
            .run(
                causa::kernel::TurnId::new("budget"),
                context.clone(),
                options,
                control(),
            )
            .await;
        assert_eq!(
            matches!(outcome.result, TurnResult::Completed { .. }),
            expected
        );
        if !expected {
            assert!(matches!(
                outcome.result,
                TurnResult::Interrupted {
                    cause: TurnInterruption::PrepareFailed { .. }
                }
            ));
            assert_eq!(outcome.context.blocks(), baseline_blocks);
        }
    }
    let requests = admitted_gateway.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].frame.blocks, baseline_blocks);
    assert_eq!(requests[0].tool_surface, small_tools);
    assert_eq!(requests[0].generation, generation);
    assert!(rejected_gateway.requests.lock().unwrap().is_empty());
    let estimates = budget.estimates.lock().unwrap();
    assert_eq!(estimates.len(), 2);
    assert!(estimates[0].total() < maximum && estimates[1].total() > maximum);
    let mut description_changed = small_tools;
    description_changed.definitions[0].description = "long description ".repeat(200);
    let description_cost = estimate(
        &context,
        &RoundInfo {
            invocation_id: &invocation,
            model: &model,
            tool_surface: &description_changed,
            generation: &generation,
        },
    );
    assert!(description_cost.tools > estimates[0].tools);
}
