//! External material flow assembled solely from public kernel contracts.

mod common;

use causa_kernel::{
    BlockContent, BlockMeta, CacheDirective, CallControl, CancellationToken, Context, ContextFrame,
    ContextPreparer, GenerationOptions, InvocationId, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelOutput, ModelRef, ModelRequest, ModelResponse, ModelStopReason,
    ModelUsage, PrepareError, ReasoningPayload, RoundId, RoundInfo, TextPayload, ToolOutput,
    ToolResultPayload, ToolResultStatus, ToolSurface, TurnId, validate_tool_result_append,
};
use serde_json::json;
use std::sync::Mutex;

struct OneShotGateway {
    output: ModelOutput,
}

#[async_trait::async_trait]
impl ModelGateway for OneShotGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        assert_eq!(request.invocation_id.round_id, RoundId(0));
        assert_eq!(request.frame.blocks.len(), 1);
        control.check().unwrap();
        Ok(self.output.clone())
    }
}

#[tokio::test]
async fn external_single_shot_driver_uses_independent_execution_identity_and_direct_serde() {
    let mut context = Context::from_blocks(vec![common::text_block(
        common::block_id(),
        TextPayload::new("hi"),
        "user",
    )])
    .unwrap();
    let gateway = OneShotGateway {
        output: ModelOutput {
            response: ModelResponse {
                text: TextPayload::new("hello"),
                tool_calls: vec![],
            },
            usage: Some(ModelUsage {
                input_tokens: 10,
                output_tokens: 2,
                cache_read_tokens: Some(4),
                cache_write_tokens: None,
                reasoning_tokens: Some(1),
            }),
            stop_reason: ModelStopReason::EndTurn,
            reasoning: Some(ReasoningPayload {
                text: "thinking".into(),
                signature: Some("sig".into()),
            }),
        },
    };
    let request = ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("ext-1"),
            round_id: RoundId(0),
        },
        frame: context.frame(),
        model: ModelRef::new("external-model"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions::default(),
        cache: CacheDirective::None,
    };
    let control = CallControl::new(CancellationToken::new(), None);
    let output = gateway.invoke(&request, &control).await.unwrap();
    let ids = common::block_ids_for(&output.response);
    context
        .apply(
            Vec::new(),
            output.response.to_blocks(output.stop_reason, &ids).unwrap(),
        )
        .unwrap();
    assert_eq!(output.usage.unwrap().cache_read_tokens, Some(4));
    assert_eq!(output.reasoning.unwrap().signature.as_deref(), Some("sig"));
    let restored: Context =
        serde_json::from_str(&serde_json::to_string(&context).unwrap()).unwrap();
    assert_eq!(restored.blocks(), context.blocks());
    context
        .apply(
            Vec::new(),
            vec![common::text_block(
                common::block_id(),
                TextPayload::new("more"),
                "user",
            )],
        )
        .unwrap();
    assert_eq!(restored.blocks().len(), 2);
    assert_eq!(context.blocks().len(), 3);
    assert_eq!(request.frame.blocks.len(), 1);
}

#[test]
fn external_tool_driver_converts_only_new_material_and_validates_results_before_commit() {
    let old = ModelResponse {
        text: TextPayload::new(""),
        tool_calls: vec![common::draft("old", json!({}))],
    };
    let mut context = Context::from_blocks(
        old.to_blocks(ModelStopReason::ToolUse, &common::block_ids_for(&old))
            .unwrap(),
    )
    .unwrap();
    let response = ModelResponse {
        text: TextPayload::new("new"),
        tool_calls: vec![common::draft("echo", json!({"message":"hi"}))],
    };
    let ids = common::block_ids_for(&response);
    let new_blocks = response.to_blocks(ModelStopReason::ToolUse, &ids).unwrap();
    let new_calls = new_blocks
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolCall(call) => Some((block.id(), call.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(new_calls.len(), 1);
    assert_eq!(new_calls[0].1.tool_name, "echo");
    context.apply(Vec::new(), new_blocks).unwrap();
    let result = ToolResultPayload {
        call_block_id: new_calls[0].0,
        status: ToolResultStatus::Succeeded,
        output: ToolOutput::new(json!("hi")),
        media: vec![],
        notes: vec![],
    };
    let result_id = common::block_id();
    validate_tool_result_append(context.blocks(), &[(result_id, result.clone())]).unwrap();
    context
        .apply(
            Vec::new(),
            vec![causa_kernel::ContextBlock::new(
                result_id,
                BlockContent::ToolResult(result),
                BlockMeta::default(),
            )],
        )
        .unwrap();
    assert_eq!(context.blocks().len(), 4);
}

struct Preparing;

#[async_trait::async_trait]
impl ContextPreparer for Preparing {
    async fn prepare(
        &self,
        context: &mut Context,
        round: &RoundInfo<'_>,
        control: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        control.check().unwrap();
        assert_eq!(round.invocation_id.turn_id, TurnId::new("prepare"));
        assert_eq!(round.model.0, "actual-model");
        assert_eq!(round.generation.max_tokens, Some(32));
        assert!(round.tool_surface.definitions.is_empty());
        context
            .edit()
            .append(vec![common::text_block(
                common::block_id(),
                TextPayload::new("durable input"),
                "user",
            )])
            .commit()
            .unwrap();
        let mut frame = context.frame();
        frame.blocks.push(common::text_block(
            common::block_id(),
            TextPayload::new("temporary observation"),
            "host",
        ));
        Ok(frame)
    }
}

#[tokio::test]
async fn external_preparer_sees_actual_round_and_selects_temporary_material() {
    let invocation_id = InvocationId {
        turn_id: TurnId::new("prepare"),
        round_id: RoundId(0),
    };
    let model = ModelRef::new("actual-model");
    let tool_surface = ToolSurface::empty();
    let generation = GenerationOptions {
        max_tokens: Some(32),
        ..Default::default()
    };
    let round = RoundInfo {
        invocation_id: &invocation_id,
        model: &model,
        tool_surface: &tool_surface,
        generation: &generation,
    };
    let mut context = Context::new();
    let preparer: &dyn ContextPreparer = &Preparing;
    let frame = preparer
        .prepare(
            &mut context,
            &round,
            &CallControl::new(CancellationToken::new(), None),
        )
        .await
        .unwrap();
    assert_eq!(frame.blocks.len(), 2);
    assert_eq!(context.blocks().len(), 1);
}

/// A scripted underlying gateway really fails twice before returning output.
struct FlakyGateway {
    requests: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl ModelGateway for FlakyGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut requests = self.requests.lock().unwrap();
        requests.push(format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{}",
            request.invocation_id,
            request.model,
            request.tool_surface,
            request.generation,
            request.cache,
            serde_json::to_string(&request.frame.blocks).unwrap()
        ));
        if requests.len() < 3 {
            Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Transient,
                "scripted failure",
            ))
        } else {
            Ok(common::endturn_output("after retries"))
        }
    }
}

/// A caller's concrete wrapper demonstrates retry without a public attempt field.
struct RetryGateway {
    inner: FlakyGateway,
}

#[async_trait::async_trait]
impl ModelGateway for RetryGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        for ordinal in 0..3 {
            control.check().map_err(|error| {
                ModelInvokeError::new(ModelInvokeErrorKind::Cancelled, error.to_string())
            })?;
            match self.inner.invoke(request, control).await {
                Err(error) if error.kind == ModelInvokeErrorKind::Transient && ordinal < 2 => {
                    continue;
                }
                result => return result,
            }
        }
        unreachable!()
    }
}

#[tokio::test]
async fn logical_gateway_wrapper_keeps_invocation_request_values_stable() {
    let gateway = RetryGateway {
        inner: FlakyGateway {
            requests: Mutex::new(Vec::new()),
        },
    };
    let request = ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("wrapped"),
            round_id: RoundId(0),
        },
        frame: Context::new().frame(),
        model: ModelRef::new("model"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions::default(),
        cache: CacheDirective::StablePrefix,
    };
    let result = gateway
        .invoke(&request, &CallControl::new(CancellationToken::new(), None))
        .await
        .unwrap();
    assert_eq!(result.response.text.0, "after retries");
    let requests = gateway.inner.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(requests.windows(2).all(|pair| pair[0] == pair[1]));
}
