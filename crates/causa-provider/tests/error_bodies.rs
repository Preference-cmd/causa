//! Verification of error body behavior across invoke and stream entry points.
//!
//! Rule:
//! - With documented error envelope (`{"error": {"message": "..."}}`), `invoke`
//!   and `stream` produce the same message: `"{provider} error {status}: {error.message}"`.
//! - Without that envelope (e.g. HTML or unrecognized JSON), both produce
//!   `"{provider} error {status}"` with no body text.
//! - The status and the provider's error code are also carried as data, so a
//!   caller can tell the causes apart without reading the message.
//! - An error event inside a successful stream is read by the same rule as an
//!   HTTP error body, and no provider text keeps a control character.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CallControl, CancellationToken, ContextBlock, ContextFrame,
    GenerationOptions, InvocationId, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelRef, ModelRequest, RoundId, StreamDelta, TextPayload, TurnId,
};
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use futures_util::StreamExt;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request() -> ModelRequest {
    {
        let mut request = ModelRequest::new(
            InvocationId {
                turn_id: TurnId::new("error-bodies"),
                round_id: RoundId(0),
            },
            ContextFrame {
                blocks: vec![ContextBlock::new(
                    BlockId::new(uuid::Uuid::from_u128(1)),
                    BlockContent::Parts(vec![causa_kernel::ContentPart::Text(TextPayload::new(
                        "hello",
                    ))]),
                    BlockMeta::from_source("user"),
                )],
            },
            ModelRef::new("test-model"),
        );
        request.generation = {
            let mut generation = GenerationOptions::default();
            generation.max_tokens = Some(1024);
            generation
        };
        request
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

async fn respond(status: u16, content_type: &str, body: String) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status).set_body_raw(body, content_type))
        .mount(&server)
        .await;
    server
}

async fn invoke_error(status: u16, body: serde_json::Value) -> ModelInvokeError {
    let server = respond(status, "application/json", body.to_string()).await;
    OpenAiChatCompletionsGateway::new("k")
        .with_base_url(format!("{}/v1", server.uri()))
        .invoke(&request(), &control())
        .await
        .expect_err("expected error")
}

/// The first error a successful stream with this body yields.
async fn in_stream_error(gateway: &dyn ModelGateway, uri: &str) -> ModelInvokeError {
    let mut stream = match gateway.stream(&request(), &control()).await {
        Ok(stream) => stream,
        Err(error) => panic!("stream to {uri} was not established: {error}"),
    };
    loop {
        match stream.next().await {
            Some(StreamDelta::Error { error }) => return error,
            Some(_) => {}
            None => panic!("stream ended without an error"),
        }
    }
}

#[tokio::test]
async fn causes_are_told_apart_by_status_code_and_kind_without_the_message() {
    let credential = invoke_error(
        401,
        serde_json::json!({"error":{"message":"bad key","type":"authentication_error"}}),
    )
    .await;
    let missing = invoke_error(
        404,
        serde_json::json!({"error":{"message":"no such model","code":"model_not_found"}}),
    )
    .await;
    let unreadable = invoke_error(200, serde_json::json!({"choices": "not an array"})).await;

    assert_eq!(credential.kind, ModelInvokeErrorKind::Permanent);
    assert_eq!(credential.status, Some(401));
    assert_eq!(credential.code.as_deref(), Some("authentication_error"));

    assert_eq!(missing.kind, ModelInvokeErrorKind::Permanent);
    assert_eq!(missing.status, Some(404));
    assert_eq!(missing.code.as_deref(), Some("model_not_found"));

    assert_eq!(unreadable.kind, ModelInvokeErrorKind::Undecodable);
    assert_eq!(unreadable.status, None);
    assert_eq!(unreadable.code, None);

    // A body that is not the envelope still carries its status.
    let server = respond(403, "text/html", "<html>Forbidden</html>".into()).await;
    let plain = AnthropicMessagesGateway::new("k")
        .with_base_url(format!("{}/v1", server.uri()))
        .invoke(&request(), &control())
        .await
        .expect_err("expected error");
    assert_eq!(plain.status, Some(403));
    assert_eq!(plain.code, None);
}

#[tokio::test]
async fn a_stream_error_event_reads_like_the_same_envelope_over_http() {
    let envelope = serde_json::json!({
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "slow down"}
    });
    let sse = |data: &serde_json::Value| format!("data: {data}\n\n");

    let server = respond(429, "application/json", envelope.to_string()).await;
    let http = OpenAiChatCompletionsGateway::new("k")
        .with_base_url(format!("{}/v1", server.uri()))
        .invoke(&request(), &control())
        .await
        .expect_err("expected error");
    assert_eq!(http.message, "openai error 429: slow down");
    let text = http.message.strip_prefix("openai error 429: ").unwrap();

    let server = respond(200, "text/event-stream", sse(&envelope)).await;
    let chat = OpenAiChatCompletionsGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let anthropic =
        AnthropicMessagesGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let flat = serde_json::json!({
        "type": "error", "code": "rate_limit_error", "message": "slow down"
    });
    let flat_server = respond(200, "text/event-stream", sse(&flat)).await;
    let responses =
        OpenAiResponsesGateway::new("k").with_base_url(format!("{}/v1", flat_server.uri()));

    for gateway in [
        &chat as &dyn ModelGateway,
        &anthropic as &dyn ModelGateway,
        &responses as &dyn ModelGateway,
    ] {
        let event = in_stream_error(gateway, &server.uri()).await;
        assert_eq!(event.kind, http.kind);
        assert_eq!(event.code, http.code);
        assert_eq!(event.status, None);
        assert_eq!(
            event.message,
            format!("provider stream error (rate_limit_error): {text}")
        );
    }
}

#[tokio::test]
async fn provider_text_never_keeps_a_control_character() {
    let envelope = serde_json::json!({
        "error": {
            "type": "invalid_request_error",
            "message": "first line\r\nsecond\u{1b}[2J line\ttab"
        }
    });
    let http = invoke_error(400, envelope.clone()).await;
    assert_eq!(http.kind, ModelInvokeErrorKind::InvalidRequest);
    assert_eq!(
        http.message,
        "openai error 400: first line second [2J line tab"
    );

    let server = respond(200, "text/event-stream", format!("data: {envelope}\n\n")).await;
    let chat = OpenAiChatCompletionsGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let event = in_stream_error(&chat, &server.uri()).await;
    assert_eq!(
        event.message,
        "provider stream error (invalid_request_error): first line second [2J line tab"
    );
    for message in [&http.message, &event.message] {
        assert!(!message.chars().any(char::is_control), "{message:?}");
    }
}
