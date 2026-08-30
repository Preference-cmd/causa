//! Wiremock-driven integration tests for the OpenAI kernel gateways
//! (Slice 3 Phase C) — structurally the same acceptance coverage as
//! `anthropic_gateway.rs` (Phase B), exercised through both the Chat
//! Completions and Responses call paths.

use std::time::{Duration, Instant};

use reimagine_agent_provider::{OpenAiChatCompletionsGateway, OpenAiResponsesGateway};
use reimagine_context_kernel::{
    AttemptControl, AttemptNumber, BlockContent, BlockId, BlockMeta, BlockSequence,
    CancellationToken, ContextBlock, ContextFrame, ContextVersion, FrameId, FrameScope,
    GenerationOptions, InvocationId, ModelContext, ModelGateway, ModelInvokeErrorKind, ModelRef,
    ModelRequest, ModelStopReason, RoundId, RunControl, TextPayload, ToolSurface, TurnId,
};
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY: &str = "sk-test-openai";

fn ctrl(deadline: Option<Instant>) -> AttemptControl {
    RunControl::new(CancellationToken::new(), deadline).for_attempt(None)
}

fn user_frame() -> ContextFrame {
    let scope = FrameScope::Turn {
        turn_id: TurnId::new("t1"),
        source_version: ContextVersion(1),
    };
    ContextFrame {
        frame_id: FrameId::from_scope(&scope, RoundId(0)),
        scope,
        round_id: RoundId(0),
        model_context: ModelContext {
            blocks: vec![ContextBlock {
                id: BlockId {
                    turn_id: TurnId::new("t1"),
                    sequence: BlockSequence(0),
                },
                sequence: BlockSequence(0),
                content: BlockContent::Text(TextPayload::new("hi")),
                meta: BlockMeta {
                    provider_call_id: None,
                    source: Some("user".into()),
                },
            }],
        },
    }
}

fn request(frame: ContextFrame) -> ModelRequest {
    ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("t1"),
            round_id: RoundId(0),
        },
        attempt: AttemptNumber(1),
        frame,
        model: ModelRef::new("gpt-test"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions::default(),
    }
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

    let gw = OpenAiChatCompletionsGateway::new(KEY).with_base_url(server.uri());
    let out = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap();
    assert_eq!(out.response.text.0, "reading");
    assert_eq!(out.response.tool_calls.len(), 1);
    assert_eq!(
        out.response.tool_calls[0].provider_call_id.as_deref(),
        Some("call_1")
    );
    assert_eq!(out.response.tool_calls[0].arguments, json!({"path": "a"}));
    assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
    let usage = out.usage.unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (3, 2));
}

#[tokio::test]
async fn chat_error_rows_map_to_their_kinds() {
    let body = json!({"error": {"message": "boom", "type": "server_error"}});
    for (status, expected) in [
        (408, ModelInvokeErrorKind::Transient),
        (429, ModelInvokeErrorKind::Transient),
        (500, ModelInvokeErrorKind::Transient),
        (400, ModelInvokeErrorKind::InvalidRequest),
        (422, ModelInvokeErrorKind::InvalidRequest),
        (401, ModelInvokeErrorKind::Permanent),
        (403, ModelInvokeErrorKind::Permanent),
        (404, ModelInvokeErrorKind::Permanent),
        (409, ModelInvokeErrorKind::Permanent),
    ] {
        let server = MockServer::start().await;
        post_mock(status, "/v1/chat/completions", body.clone())
            .mount(&server)
            .await;
        let gw = OpenAiChatCompletionsGateway::new(KEY).with_base_url(server.uri());
        let e = gw
            .invoke(&request(user_frame()), &ctrl(None))
            .await
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(e.kind()),
            std::mem::discriminant(&expected),
            "status {status}: got {e}"
        );
        assert!(e.message.contains("boom"), "message: {}", e.message);
    }
}

#[tokio::test]
async fn chat_unparseable_2xx_body_is_permanent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;
    let gw = OpenAiChatCompletionsGateway::new(KEY).with_base_url(server.uri());
    let e = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Permanent),
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

    let gw = OpenAiResponsesGateway::new(KEY).with_base_url(server.uri());
    let out = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap();
    assert_eq!(out.response.text.0, "reading");
    assert_eq!(out.response.tool_calls.len(), 1);
    assert_eq!(
        out.response.tool_calls[0].provider_call_id.as_deref(),
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
        (429, ModelInvokeErrorKind::Transient),
        (500, ModelInvokeErrorKind::Transient),
        (400, ModelInvokeErrorKind::InvalidRequest),
        (422, ModelInvokeErrorKind::InvalidRequest),
        (401, ModelInvokeErrorKind::Permanent),
        (403, ModelInvokeErrorKind::Permanent),
        (404, ModelInvokeErrorKind::Permanent),
        (409, ModelInvokeErrorKind::Permanent),
    ] {
        let server = MockServer::start().await;
        post_mock(status, "/v1/responses", body.clone())
            .mount(&server)
            .await;
        let gw = OpenAiResponsesGateway::new(KEY).with_base_url(server.uri());
        let e = gw
            .invoke(&request(user_frame()), &ctrl(None))
            .await
            .unwrap_err();
        assert_eq!(
            std::mem::discriminant(e.kind()),
            std::mem::discriminant(&expected),
            "status {status}: got {e}"
        );
    }
}

#[tokio::test]
async fn responses_unparseable_2xx_body_is_permanent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;
    let gw = OpenAiResponsesGateway::new(KEY).with_base_url(server.uri());
    let e = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Permanent),
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
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Transient),
        "got {e}"
    );

    let responses =
        OpenAiResponsesGateway::new(KEY).with_endpoint(format!("http://{addr}/v1/responses"));
    let e = responses
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Transient),
        "got {e}"
    );
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
            "/v1/chat/completions" => {
                Box::new(OpenAiChatCompletionsGateway::new(KEY).with_base_url(server.uri()))
            }
            _ => Box::new(OpenAiResponsesGateway::new(KEY).with_base_url(server.uri())),
        };
        let deadline = Instant::now() + Duration::from_millis(100);
        let started = Instant::now();
        let e = gw
            .invoke(&request(user_frame()), &ctrl(Some(deadline)))
            .await
            .unwrap_err();
        assert!(
            matches!(e.kind(), ModelInvokeErrorKind::TimedOut),
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
        let run = RunControl::new(token, None);
        let gw: Box<dyn ModelGateway> = match path_suffix {
            "/v1/chat/completions" => {
                Box::new(OpenAiChatCompletionsGateway::new(KEY).with_base_url(server.uri()))
            }
            _ => Box::new(OpenAiResponsesGateway::new(KEY).with_base_url(server.uri())),
        };
        let started = Instant::now();
        let e = gw
            .invoke(&request(user_frame()), &run.for_attempt(None))
            .await
            .unwrap_err();
        assert!(
            matches!(e.kind(), ModelInvokeErrorKind::Cancelled),
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
    let run = RunControl::new(token, None);

    let chat = OpenAiChatCompletionsGateway::new(KEY).with_base_url(server.uri());
    let e = chat
        .invoke(&request(user_frame()), &run.for_attempt(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Cancelled),
        "got {e}"
    );

    let responses = OpenAiResponsesGateway::new(KEY).with_base_url(server.uri());
    let e = responses
        .invoke(&request(user_frame()), &run.for_attempt(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Cancelled),
        "got {e}"
    );
}
