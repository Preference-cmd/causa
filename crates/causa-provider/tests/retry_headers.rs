//! End-to-end retry-header handling for the three built-in HTTP gateways.
//!
//! Acceptance: the status table classifies 408 / 409 / 429 / 5xx as
//! `Transient` and 400 / 422 as `InvalidRequest`; `x-should-retry`
//! overrides that rule; every gateway fills [`ModelInvokeError::retry_delay`]
//! from `retry-after-ms` then `retry-after`; and the stream path reads the
//! same headers as the invoke path.

use std::time::Duration;

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CallControl, CancellationToken, ContextBlock, ContextFrame,
    GenerationOptions, InvocationId, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelRef, ModelRequest, RoundId, TextPayload, TurnId,
};
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request() -> ModelRequest {
    {
        let mut built = ModelRequest::new(
            InvocationId {
                turn_id: TurnId::new("retry-headers"),
                round_id: RoundId(0),
            },
            ContextFrame {
                blocks: vec![ContextBlock::new(
                    BlockId::new(uuid::Uuid::from_u128(1)),
                    BlockContent::Parts(vec![causa_kernel::ContentPart::Text(TextPayload::new(
                        "hi",
                    ))]),
                    BlockMeta::from_source("user"),
                )],
            },
            ModelRef::new("retry-test-model"),
        );
        // The Anthropic leg requires `max_tokens`; the transport behavior
        // under test is reached only when the request renders.
        built.generation = GenerationOptions::default().with_max_tokens(1024);
        built
    }
}

fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

async fn mount(server: &MockServer, status: u16, headers: &[(&str, &str)]) {
    let mut template = ResponseTemplate::new(status);
    for (name, value) in headers {
        template = template.insert_header(*name, *value);
    }
    Mock::given(method("POST"))
        .respond_with(template)
        .mount(server)
        .await;
}

fn check(
    error: &ModelInvokeError,
    kind: &ModelInvokeErrorKind,
    delay: Option<Duration>,
    label: &str,
) {
    assert_eq!(&error.kind, kind, "{label}");
    assert_eq!(error.retry_delay, delay, "{label}");
}

/// One invoke against a fresh server, per concrete gateway type.
macro_rules! invoke_case {
    ($gateway:expr, $status:expr, $headers:expr) => {{
        let server = MockServer::start().await;
        mount(&server, $status, $headers).await;
        let gateway = $gateway.with_base_url(format!("{}/v1", server.uri()));
        gateway.invoke(&request(), &control()).await.unwrap_err()
    }};
}

#[tokio::test]
async fn all_three_gateways_fill_the_server_retry_delay() {
    for (name, status, headers, delay) in [
        (
            "retry-after-ms",
            429u16,
            vec![("retry-after-ms", "1500")],
            Duration::from_millis(1500),
        ),
        (
            "retry-after seconds",
            503u16,
            vec![("retry-after", "3")],
            Duration::from_secs(3),
        ),
        (
            "retry-after-ms wins",
            429u16,
            vec![("retry-after-ms", "250"), ("retry-after", "9")],
            Duration::from_millis(250),
        ),
    ] {
        let anthropic = invoke_case!(AnthropicMessagesGateway::new("k"), status, &headers);
        check(
            &anthropic,
            &ModelInvokeErrorKind::Transient,
            Some(delay),
            &format!("anthropic {name}"),
        );
        let chat = invoke_case!(OpenAiChatCompletionsGateway::new("k"), status, &headers);
        check(
            &chat,
            &ModelInvokeErrorKind::Transient,
            Some(delay),
            &format!("chat {name}"),
        );
        let responses = invoke_case!(OpenAiResponsesGateway::new("k"), status, &headers);
        check(
            &responses,
            &ModelInvokeErrorKind::Transient,
            Some(delay),
            &format!("responses {name}"),
        );
    }
}

#[tokio::test]
async fn statuses_classify_per_row_end_to_end() {
    for (status, kind) in [
        (408u16, ModelInvokeErrorKind::Transient),
        (409, ModelInvokeErrorKind::Transient),
        (429, ModelInvokeErrorKind::Transient),
        (500, ModelInvokeErrorKind::Transient),
        (503, ModelInvokeErrorKind::Transient),
        (400, ModelInvokeErrorKind::InvalidRequest),
        (422, ModelInvokeErrorKind::InvalidRequest),
        (401, ModelInvokeErrorKind::Permanent),
        (404, ModelInvokeErrorKind::Permanent),
    ] {
        let error = invoke_case!(OpenAiChatCompletionsGateway::new("k"), status, &[]);
        check(&error, &kind, None, &format!("chat {status}"));
        let error = invoke_case!(AnthropicMessagesGateway::new("k"), status, &[]);
        check(&error, &kind, None, &format!("anthropic {status}"));
    }
}

#[tokio::test]
async fn should_retry_header_overrides_statuses_end_to_end() {
    let error = invoke_case!(
        OpenAiChatCompletionsGateway::new("k"),
        429,
        &[("x-should-retry", "false")]
    );
    check(
        &error,
        &ModelInvokeErrorKind::Permanent,
        None,
        "chat 429 refused",
    );
    let error = invoke_case!(
        AnthropicMessagesGateway::new("k"),
        503,
        &[("x-should-retry", "false")]
    );
    check(
        &error,
        &ModelInvokeErrorKind::Permanent,
        None,
        "anthropic 503 refused",
    );
    let error = invoke_case!(
        OpenAiResponsesGateway::new("k"),
        400,
        &[("x-should-retry", "true")]
    );
    check(
        &error,
        &ModelInvokeErrorKind::Transient,
        None,
        "responses 400 forced",
    );
    let error = invoke_case!(
        OpenAiResponsesGateway::new("k"),
        400,
        &[("x-should-retry", "maybe")]
    );
    check(
        &error,
        &ModelInvokeErrorKind::InvalidRequest,
        None,
        "responses 400 unrecognized",
    );
}

#[tokio::test]
async fn stream_establishment_reads_the_same_headers() {
    let server = MockServer::start().await;
    mount(&server, 409, &[("retry-after-ms", "750")]).await;
    let gateway = AnthropicMessagesGateway::new("k").with_base_url(format!("{}/v1", server.uri()));
    let error = gateway
        .stream(&request(), &control())
        .await
        .err()
        .expect("non-2xx establishment fails");
    check(
        &error,
        &ModelInvokeErrorKind::Transient,
        Some(Duration::from_millis(750)),
        "stream 409",
    );
}
