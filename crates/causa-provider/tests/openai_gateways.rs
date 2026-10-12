//! Wiremock-driven integration tests for the OpenAI kernel gateways —
//! structurally the same coverage as `anthropic_gateway.rs`, exercised
//! through both the Chat Completions and Responses call paths.

use std::time::{Duration, Instant};

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CallControl, CancellationToken, ContextBlock, ContextFrame,
    InvocationId, ModelGateway, ModelInvokeErrorKind, ModelRef, ModelRequest, ModelStopReason,
    RoundId, TextPayload, TurnId,
};
use causa_provider::{OpenAiChatCompletionsGateway, OpenAiResponsesGateway};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY: &str = "sk-test-openai";

fn ctrl(deadline: Option<Instant>) -> CallControl {
    CallControl::new(CancellationToken::new(), deadline)
}

fn user_frame() -> ContextFrame {
    ContextFrame {
        blocks: vec![ContextBlock::new(
            BlockId::new(uuid::Uuid::from_u128(1)),
            BlockContent::Parts(vec![causa_kernel::ContentPart::Text(TextPayload::new(
                "hi",
            ))]),
            BlockMeta::from_source("user"),
        )],
    }
}

fn request(frame: ContextFrame) -> ModelRequest {
    ModelRequest::new(
        InvocationId {
            turn_id: TurnId::new("t1"),
            round_id: RoundId(0),
        },
        frame,
        ModelRef::new("gpt-test"),
    )
}

fn post_mock(status: u16, path_suffix: &str, body: Value) -> Mock {
    Mock::given(method("POST"))
        .and(path(path_suffix))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
}

fn delayed_post_mock(status: u16, path_suffix: &str, body: Value, delay: Duration) -> Mock {
    Mock::given(method("POST"))
        .and(path(path_suffix))
        .respond_with(
            ResponseTemplate::new(status)
                .set_body_json(body)
                .set_delay(delay),
        )
}

#[tokio::test]
async fn unset_max_tokens_still_sends_on_the_openai_wires() {
    // Only Anthropic requires the output limit; on the OpenAI wires an unset
    // `max_tokens` stays absent and the request is sent.
    let server = MockServer::start().await;
    let cases = [
        (
            "/v1/chat/completions",
            json!({"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}),
        ),
        (
            "/v1/responses",
            json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}),
        ),
    ];
    for (endpoint, response) in cases {
        server.reset().await;
        Mock::given(method("POST"))
            .and(path(endpoint))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response))
            .expect(1)
            .mount(&server)
            .await;
        let gateway: Box<dyn ModelGateway> = if endpoint.ends_with("chat/completions") {
            Box::new(
                OpenAiChatCompletionsGateway::new(KEY)
                    .with_base_url(format!("{}/v1", server.uri())),
            )
        } else {
            Box::new(OpenAiResponsesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri())))
        };
        let request = request(user_frame());
        assert!(request.generation.max_tokens.is_none());
        gateway.invoke(&request, &ctrl(None)).await.unwrap();
        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1, "{endpoint}");
        let body: Value = serde_json::from_slice(&received[0].body).unwrap();
        assert!(body.get("max_completion_tokens").is_none(), "{endpoint}");
        assert!(body.get("max_output_tokens").is_none(), "{endpoint}");
    }
}
// === Chat Completions ========================================================

#[tokio::test]
async fn chat_invoke_round_trips_complete_model_output() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .and(body_partial_json(json!({
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "hi"}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "reading",
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "read", "arguments": "{\"path\": \"a\"}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })))
        .mount(&server)
        .await;

    let gw = OpenAiChatCompletionsGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
    let out = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap();
    assert_eq!(out.response.text(), "reading");
    assert_eq!(out.response.tool_calls().count(), 1);
    assert_eq!(
        out.response
            .tool_calls()
            .next()
            .unwrap()
            .provider_call_id
            .as_deref(),
        Some("call_1")
    );
    assert_eq!(
        out.response.tool_calls().next().unwrap().arguments,
        json!({"path": "a"})
    );
    assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
    let usage = out.usage.unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (3, 2));
}

#[tokio::test]
async fn chat_error_rows_map_to_their_kinds() {
    let body = json!({"error": {"message": "boom", "type": "server_error"}});
    for (status, expected) in [
        (408, ModelInvokeErrorKind::Transient),
        (409, ModelInvokeErrorKind::Transient),
        (429, ModelInvokeErrorKind::Transient),
        (500, ModelInvokeErrorKind::Transient),
        (400, ModelInvokeErrorKind::InvalidRequest),
        (422, ModelInvokeErrorKind::InvalidRequest),
        (401, ModelInvokeErrorKind::Permanent),
        (403, ModelInvokeErrorKind::Permanent),
        (404, ModelInvokeErrorKind::Permanent),
    ] {
        let server = MockServer::start().await;
        post_mock(status, "/v1/chat/completions", body.clone())
            .mount(&server)
            .await;
        let gw =
            OpenAiChatCompletionsGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
        let e = gw
            .invoke(&request(user_frame()), &ctrl(None))
            .await
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(&e.kind),
            std::mem::discriminant(&expected),
            "status {status}: got {e}"
        );
        assert!(e.message.contains("boom"), "message: {}", e.message);
    }
}

#[tokio::test]
async fn chat_unparseable_2xx_body_is_undecodable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;
    let gw = OpenAiChatCompletionsGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
    let e = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind, ModelInvokeErrorKind::Undecodable),
        "got {e}"
    );
}

// === Responses ==============================================================

#[tokio::test]
async fn responses_invoke_round_trips_complete_model_output() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .and(body_partial_json(json!({
            "model": "gpt-test",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "reading"}]},
                {"type": "function_call", "call_id": "call_2", "name": "read",
                 "arguments": "{\"path\": \"a\"}"},
            ],
            "status": "completed",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        })))
        .mount(&server)
        .await;

    let gw = OpenAiResponsesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
    let out = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap();
    assert_eq!(out.response.text(), "reading");
    assert_eq!(out.response.tool_calls().count(), 1);
    assert_eq!(
        out.response
            .tool_calls()
            .next()
            .unwrap()
            .provider_call_id
            .as_deref(),
        Some("call_2")
    );
    assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
    let usage = out.usage.unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (3, 2));
}

#[tokio::test]
async fn responses_error_rows_map_to_their_kinds() {
    let body = json!({"error": {"message": "boom", "type": "server_error"}});
    for (status, expected) in [
        (408, ModelInvokeErrorKind::Transient),
        (409, ModelInvokeErrorKind::Transient),
        (429, ModelInvokeErrorKind::Transient),
        (500, ModelInvokeErrorKind::Transient),
        (400, ModelInvokeErrorKind::InvalidRequest),
        (422, ModelInvokeErrorKind::InvalidRequest),
        (401, ModelInvokeErrorKind::Permanent),
        (403, ModelInvokeErrorKind::Permanent),
        (404, ModelInvokeErrorKind::Permanent),
    ] {
        let server = MockServer::start().await;
        post_mock(status, "/v1/responses", body.clone())
            .mount(&server)
            .await;
        let gw = OpenAiResponsesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
        let e = gw
            .invoke(&request(user_frame()), &ctrl(None))
            .await
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(&e.kind),
            std::mem::discriminant(&expected),
            "status {status}: got {e}"
        );
    }
}

#[tokio::test]
async fn responses_unparseable_2xx_body_is_undecodable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;
    let gw = OpenAiResponsesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
    let e = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind, ModelInvokeErrorKind::Undecodable),
        "got {e}"
    );
}

// === Shared transport behavior (per protocol path) ===========================

#[tokio::test]
async fn connect_error_is_transient_on_both_paths() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let chat = OpenAiChatCompletionsGateway::new(KEY)
        .with_endpoint(format!("http://{addr}/v1/chat/completions"));
    let e = chat
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(matches!(e.kind, ModelInvokeErrorKind::Transient), "got {e}");

    let responses =
        OpenAiResponsesGateway::new(KEY).with_endpoint(format!("http://{addr}/v1/responses"));
    let e = responses
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(matches!(e.kind, ModelInvokeErrorKind::Transient), "got {e}");
}

#[tokio::test]
async fn deadline_yields_timed_out_promptly_on_both_paths() {
    let body = json!({"content": [], "stop_reason": "end_turn"});
    for path_suffix in ["/v1/chat/completions", "/v1/responses"] {
        let server = MockServer::start().await;
        delayed_post_mock(200, path_suffix, body.clone(), Duration::from_secs(5))
            .mount(&server)
            .await;
        let gw: Box<dyn ModelGateway> = match path_suffix {
            "/v1/chat/completions" => Box::new(
                OpenAiChatCompletionsGateway::new(KEY)
                    .with_base_url(format!("{}/v1", server.uri())),
            ),
            _ => Box::new(
                OpenAiResponsesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri())),
            ),
        };
        let deadline = Instant::now() + Duration::from_millis(100);
        let started = Instant::now();
        let e = gw
            .invoke(&request(user_frame()), &ctrl(Some(deadline)))
            .await
            .unwrap_err();
        assert!(
            matches!(e.kind, ModelInvokeErrorKind::TimedOut),
            "{path_suffix}: got {e}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{path_suffix}: took {:?}",
            started.elapsed()
        );
    }
}

#[tokio::test]
async fn cancelled_token_yields_cancelled_promptly_on_both_paths() {
    let body = json!({"content": [], "stop_reason": "end_turn"});
    for path_suffix in ["/v1/chat/completions", "/v1/responses"] {
        let server = MockServer::start().await;
        delayed_post_mock(200, path_suffix, body.clone(), Duration::from_secs(5))
            .mount(&server)
            .await;
        let token = CancellationToken::new();
        let canceller = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            canceller.cancel();
        });
        let attempt = CallControl::new(token.clone(), None);
        let gw: Box<dyn ModelGateway> = match path_suffix {
            "/v1/chat/completions" => Box::new(
                OpenAiChatCompletionsGateway::new(KEY)
                    .with_base_url(format!("{}/v1", server.uri())),
            ),
            _ => Box::new(
                OpenAiResponsesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri())),
            ),
        };
        let started = Instant::now();
        let e = gw
            .invoke(&request(user_frame()), &attempt)
            .await
            .unwrap_err();
        assert!(
            matches!(e.kind, ModelInvokeErrorKind::Cancelled),
            "{path_suffix}: got {e}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{path_suffix}: took {:?}",
            started.elapsed()
        );
    }
}

#[tokio::test]
async fn pre_cancelled_control_never_reaches_the_wire_on_both_paths() {
    // No mock mounted: any request would 404 (→ Permanent). The
    // short-circuit must return Cancelled instead.
    let server = MockServer::start().await;
    let token = CancellationToken::new();
    token.cancel();
    let attempt = CallControl::new(token.clone(), None);

    let chat = OpenAiChatCompletionsGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
    let e = chat
        .invoke(&request(user_frame()), &attempt)
        .await
        .unwrap_err();
    assert!(matches!(e.kind, ModelInvokeErrorKind::Cancelled), "got {e}");

    let responses = OpenAiResponsesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
    let e = responses
        .invoke(&request(user_frame()), &attempt)
        .await
        .unwrap_err();
    assert!(matches!(e.kind, ModelInvokeErrorKind::Cancelled), "got {e}");
}
