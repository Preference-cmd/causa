//! Malformed selected request material is rejected before any transport request.
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, CallControl, CancellationToken, Context,
    ContextBlock, GenerationOptions, InvocationId, ModelGateway, ModelInvokeErrorKind, ModelRef,
    ModelRequest, RoundId, ToolCallPayload, ToolOutput, ToolResultPayload, ToolResultStatus,
    ToolSurface, TurnId,
};
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use serde_json::json;
use wiremock::MockServer;

fn id(value: u128) -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(value))
}
fn call(value: u128, wire: &str) -> ContextBlock {
    ContextBlock::new(
        id(value),
        BlockContent::ToolCall(ToolCallPayload {
            tool_name: "read".into(),
            arguments: json!({}),
        }),
        BlockMeta {
            provider_call_id: Some(wire.into()),
            ..Default::default()
        },
    )
}
fn result(value: u128, target: u128) -> ContextBlock {
    ContextBlock::new(
        id(value),
        BlockContent::ToolResult(ToolResultPayload {
            call_block_id: id(target),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("ok")),
            media: vec![],
            notes: vec![],
        }),
        BlockMeta::default(),
    )
}

#[tokio::test]
async fn invalid_tool_material_never_reaches_http_for_any_gateway() {
    let server = MockServer::start().await;
    let gateways: Vec<Box<dyn ModelGateway>> = vec![
        Box::new(AnthropicMessagesGateway::new("test").with_base_url(server.uri())),
        Box::new(OpenAiChatCompletionsGateway::new("test").with_base_url(server.uri())),
        Box::new(OpenAiResponsesGateway::new("test").with_base_url(server.uri())),
    ];
    let cases = [
        vec![result(2, 1)],
        vec![call(1, "a")],
        vec![call(1, "a"), result(2, 1), result(3, 1)],
        vec![call(1, "same"), call(2, "same"), result(3, 1), result(4, 2)],
    ];
    let control = CallControl::new(CancellationToken::new(), None);
    for blocks in cases {
        let context = Context::from_blocks(blocks).expect("legal partial material");
        let request = ModelRequest {
            invocation_id: InvocationId {
                turn_id: TurnId::new("validation"),
                round_id: RoundId(0),
            },
            frame: context.frame(),
            model: ModelRef::new("test"),
            tool_surface: ToolSurface::empty(),
            generation: GenerationOptions::default(),
            cache: CacheDirective::None,
        };
        for gateway in &gateways {
            let error = gateway.invoke(&request, &control).await.unwrap_err();
            assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
            let error = match gateway.stream(&request, &control).await {
                Ok(_) => panic!("invalid request unexpectedly produced a stream"),
                Err(error) => error,
            };
            assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
        }
    }
    assert_eq!(server.received_requests().await.unwrap().len(), 0);
}
