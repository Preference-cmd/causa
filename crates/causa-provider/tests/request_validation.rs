//! Malformed selected request material is rejected before any transport request.
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CallControl, CancellationToken, Context, ContextBlock,
    GenerationOptions, InvocationId, ModelGateway, ModelInvokeErrorKind, ModelRef, ModelRequest,
    RoundId, ToolCallPayload, ToolOutput, ToolResultPayload, ToolResultStatus, TurnId,
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
        BlockContent::ToolCall(ToolCallPayload::new("read", json!({}))),
        {
            let mut meta = BlockMeta::default();
            meta.provider_call_id = Some(wire.into());
            meta
        },
    )
}
fn result(value: u128, target: u128) -> ContextBlock {
    ContextBlock::new(
        id(value),
        BlockContent::ToolResult(ToolResultPayload::new(
            id(target),
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!("ok")),
        )),
        BlockMeta::default(),
    )
}

#[tokio::test]
async fn invalid_tool_material_never_reaches_http_for_any_gateway() {
    let server = MockServer::start().await;
    let gateways: Vec<Box<dyn ModelGateway>> = vec![
        Box::new(
            AnthropicMessagesGateway::new("test").with_base_url(format!("{}/v1", server.uri())),
        ),
        Box::new(
            OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", server.uri())),
        ),
        Box::new(OpenAiResponsesGateway::new("test").with_base_url(format!("{}/v1", server.uri()))),
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
        let request = {
            let mut request = ModelRequest::new(
                InvocationId {
                    turn_id: TurnId::new("validation"),
                    round_id: RoundId(0),
                },
                context.frame(),
                ModelRef::new("test"),
            );
            // A limit, so every rejection below is the frame's, not the
            // Anthropic wire's required `max_tokens`.
            request.generation = {
                let mut generation = GenerationOptions::default();
                generation.max_tokens = Some(64);
                generation
            };
            request
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
