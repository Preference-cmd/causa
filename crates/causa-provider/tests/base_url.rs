//! `with_base_url` appends the protocol path and nothing else: the base URL
//! carries its own version segment.
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, CallControl, CancellationToken, ContentPart,
    Context, ContextBlock, GenerationOptions, InvocationId, ModelGateway, ModelRef, ModelRequest,
    RoundId, TextPayload, ToolSurface, TurnId,
};
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use wiremock::MockServer;

fn request() -> ModelRequest {
    let block = ContextBlock::new(
        BlockId::new(uuid::Uuid::from_u128(1)),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("hi"))]),
        BlockMeta {
            source: Some("user".into()),
            ..Default::default()
        },
    );
    ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("base-url"),
            round_id: RoundId(0),
        },
        frame: Context::from_blocks(vec![block]).unwrap().frame(),
        model: ModelRef::new("test"),
        tool_surface: ToolSurface::empty(),
        // The Anthropic leg requires `max_tokens`; the OpenAI legs ignore it.
        generation: GenerationOptions {
            max_tokens: Some(1024),
            ..GenerationOptions::default()
        },
        cache: CacheDirective::None,
    }
}

async fn requested_path(build: impl Fn(String) -> Box<dyn ModelGateway>, suffix: &str) -> String {
    let server = MockServer::start().await;
    let gateway = build(format!("{}{suffix}", server.uri()));
    let control = CallControl::new(CancellationToken::new(), None);
    // No mock is mounted: the call fails, and the path it used is recorded.
    let _ = gateway.invoke(&request(), &control).await;
    let received = server.received_requests().await.expect("recording enabled");
    assert_eq!(received.len(), 1, "one request for base suffix {suffix:?}");
    received[0].url.path().to_string()
}

#[tokio::test]
async fn base_url_supplies_the_version_segment() {
    type Build = fn(String) -> Box<dyn ModelGateway>;
    let gateways: [(Build, &str); 3] = [
        (
            |base| Box::new(AnthropicMessagesGateway::new("k").with_base_url(base)),
            "/messages",
        ),
        (
            |base| Box::new(OpenAiChatCompletionsGateway::new("k").with_base_url(base)),
            "/chat/completions",
        ),
        (
            |base| Box::new(OpenAiResponsesGateway::new("k").with_base_url(base)),
            "/responses",
        ),
    ];
    for (build, path) in gateways {
        // A base ending in the version segment is not doubled.
        assert_eq!(requested_path(build, "/v1").await, format!("/v1{path}"));
        // A provider prefix before the version segment is kept.
        assert_eq!(
            requested_path(build, "/provider/v1").await,
            format!("/provider/v1{path}")
        );
        // A trailing slash on the base is not duplicated.
        assert_eq!(requested_path(build, "/v1/").await, format!("/v1{path}"));
        // A base without a version segment is used as given.
        assert_eq!(requested_path(build, "").await, path);
    }
}
