//! Verification of error body behavior across invoke and stream entry points.
//!
//! Rule:
//! - With documented error envelope (`{"error": {"message": "..."}}`), `invoke`
//!   and `stream` produce the same message: `"{provider} error {status}: {error.message}"`.
//! - Without that envelope (e.g. HTML or unrecognized JSON), both produce
//!   `"{provider} error {status}"` with no body text.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CallControl, CancellationToken, ContextBlock, ContextFrame,
    GenerationOptions, InvocationId, ModelGateway, ModelRef, ModelRequest, RoundId, TextPayload,
    ToolSurface, TurnId,
};
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request() -> ModelRequest {
    ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("error-bodies"),
            round_id: RoundId(0),
        },
        frame: ContextFrame {
            blocks: vec![ContextBlock::new(
                BlockId::new(uuid::Uuid::from_u128(1)),
                BlockContent::Parts(vec![causa_kernel::ContentPart::Text(TextPayload::new(
                    "hello",
                ))]),
                BlockMeta {
                    provider_call_id: None,
                    source: Some("user".into()),
                    wire: None,
                },
            )],
        },
        model: ModelRef::new("test-model"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions {
            max_tokens: Some(1024),
            ..GenerationOptions::default()
        },
        cache: causa_kernel::CacheDirective::None,
    }
}

fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

fn unwrap_stream_err(
    res: Result<causa_kernel::ModelStream, causa_kernel::ModelInvokeError>,
) -> causa_kernel::ModelInvokeError {
    match res {
        Err(e) => e,
        Ok(_) => panic!("expected stream error"),
    }
}

#[tokio::test]
async fn documented_envelope_produces_identical_messages_on_invoke_and_stream() {
    let server = MockServer::start().await;
    let envelope_body = serde_json::json!({
        "error": {
            "message": "rate limit exceeded: retry later"
        }
    })
    .to_string();

    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("content-type", "application/json")
                .set_body_string(envelope_body),
        )
        .mount(&server)
        .await;

    // 1. Anthropic
    let anthropic =
        AnthropicMessagesGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let invoke_err = anthropic
        .invoke(&request(), &control())
        .await
        .expect_err("expected error");
    let stream_err = unwrap_stream_err(anthropic.stream(&request(), &control()).await);
    assert_eq!(
        invoke_err.message,
        "anthropic error 429: rate limit exceeded: retry later"
    );
    assert_eq!(stream_err.message, invoke_err.message);

    // 2. Chat
    let chat = OpenAiChatCompletionsGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let invoke_err = chat
        .invoke(&request(), &control())
        .await
        .expect_err("expected error");
    let stream_err = unwrap_stream_err(chat.stream(&request(), &control()).await);
    assert_eq!(
        invoke_err.message,
        "openai error 429: rate limit exceeded: retry later"
    );
    assert_eq!(stream_err.message, invoke_err.message);

    // 3. Responses
    let responses = OpenAiResponsesGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let invoke_err = responses
        .invoke(&request(), &control())
        .await
        .expect_err("expected error");
    let stream_err = unwrap_stream_err(responses.stream(&request(), &control()).await);
    assert_eq!(
        invoke_err.message,
        "openai error 429: rate limit exceeded: retry later"
    );
    assert_eq!(stream_err.message, invoke_err.message);
}

#[tokio::test]
async fn non_envelope_body_omits_body_text_on_invoke_and_stream() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(502)
                .insert_header("content-type", "text/html")
                .set_body_string("<html><head><title>502 Bad Gateway</title></head><body>Bad Gateway</body></html>"),
        )
        .mount(&server)
        .await;

    // 1. Anthropic
    let anthropic =
        AnthropicMessagesGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let invoke_err = anthropic
        .invoke(&request(), &control())
        .await
        .expect_err("expected error");
    let stream_err = unwrap_stream_err(anthropic.stream(&request(), &control()).await);
    assert_eq!(invoke_err.message, "anthropic error 502");
    assert_eq!(stream_err.message, invoke_err.message);

    // 2. Chat
    let chat = OpenAiChatCompletionsGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let invoke_err = chat
        .invoke(&request(), &control())
        .await
        .expect_err("expected error");
    let stream_err = unwrap_stream_err(chat.stream(&request(), &control()).await);
    assert_eq!(invoke_err.message, "openai error 502");
    assert_eq!(stream_err.message, invoke_err.message);

    // 3. Responses
    let responses = OpenAiResponsesGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let invoke_err = responses
        .invoke(&request(), &control())
        .await
        .expect_err("expected error");
    let stream_err = unwrap_stream_err(responses.stream(&request(), &control()).await);
    assert_eq!(invoke_err.message, "openai error 502");
    assert_eq!(stream_err.message, invoke_err.message);
}
