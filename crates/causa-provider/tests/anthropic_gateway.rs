//! Wiremock-driven integration tests for `AnthropicMessagesGateway`
//! (Slice 3 Phase B).
//!
//! Coverage targets the Slice 3 acceptance rows: the §4 error mapping
//! table (every row), cancellation and deadline behavior, and the
//! end-to-end `invoke` returning a complete kernel `ModelOutput`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use causa_kernel::{
    AttemptControl, AttemptNumber, BlockContent, BlockId, BlockMeta, BlockSequence,
    CancellationToken, ContextBlock, ContextFrame, ContextVersion, FrameId, FrameScope,
    GenerationOptions, InvocationId, ModelContext, ModelGateway, ModelInvokeErrorKind, ModelRef,
    ModelRequest, ModelStopReason, RoundId, TextPayload, ToolSurface, TurnId,
};
use causa_provider::AnthropicMessagesGateway;
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY: &str = "sk-test-anthropic";

fn ctrl(deadline: Option<Instant>) -> AttemptControl {
    AttemptControl::new(CancellationToken::new(), deadline)
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
                content: BlockContent::Parts(vec![causa_kernel::ContentPart::Text(
                    TextPayload::new("hi"),
                )]),
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
        model: ModelRef::new("claude-test"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions::default(),
        cache: causa_kernel::CacheDirective::None,
    }
}

fn gateway(server: &MockServer) -> AnthropicMessagesGateway {
    AnthropicMessagesGateway::new(KEY).with_base_url(server.uri())
}

fn messages_mock(status: u16, body: Value) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
}

fn delayed_mock(status: u16, body: Value, delay: Duration) -> Mock {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(status)
                .set_body_json(body)
                .set_delay(delay),
        )
}

// --- acceptance: end-to-end invoke returns a complete ModelOutput -----------

#[tokio::test]
async fn invoke_round_trips_complete_model_output() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", KEY))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_partial_json(json!({
            "model": "claude-test",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [
                {"type": "text", "text": "reading"},
                {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        })))
        .mount(&server)
        .await;

    let out = gateway(&server)
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap();
    assert_eq!(out.response.text.0, "reading");
    assert_eq!(out.response.tool_calls.len(), 1);
    assert_eq!(
        out.response.tool_calls[0].provider_call_id.as_deref(),
        Some("toolu_1")
    );
    assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
    let usage = out.usage.unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (3, 2));
}

// --- acceptance: §4 error mapping table, every row --------------------------

async fn status_case(status: u16, expected: ModelInvokeErrorKind, error_body: Value) {
    let server = MockServer::start().await;
    messages_mock(status, error_body).mount(&server).await;
    let e = gateway(&server)
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert_eq!(
        std::mem::discriminant(e.kind()),
        std::mem::discriminant(&expected),
        "status {status}: got {e}"
    );
}

#[tokio::test]
async fn http_error_rows_map_to_their_kinds() {
    let transient_body =
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}});
    let schema_body = json!({"type": "error", "error": {"type": "invalid_request_error", "message": "bad tools"}});
    let auth_body =
        json!({"type": "error", "error": {"type": "authentication_error", "message": "bad key"}});
    status_case(408, ModelInvokeErrorKind::Transient, transient_body.clone()).await;
    status_case(429, ModelInvokeErrorKind::Transient, transient_body.clone()).await;
    status_case(500, ModelInvokeErrorKind::Transient, transient_body.clone()).await;
    status_case(503, ModelInvokeErrorKind::Transient, transient_body.clone()).await;
    status_case(
        400,
        ModelInvokeErrorKind::InvalidRequest,
        schema_body.clone(),
    )
    .await;
    status_case(422, ModelInvokeErrorKind::InvalidRequest, schema_body).await;
    status_case(401, ModelInvokeErrorKind::Permanent, auth_body.clone()).await;
    status_case(403, ModelInvokeErrorKind::Permanent, auth_body.clone()).await;
    status_case(404, ModelInvokeErrorKind::Permanent, auth_body).await;
    // §4 "其余" row: any undocumented status is Permanent
    status_case(
        409,
        ModelInvokeErrorKind::Permanent,
        json!({"type": "error", "error": {"type": "conflict", "message": "conflict"}}),
    )
    .await;
    // error messages surface the provider's own text
    let server = MockServer::start().await;
    messages_mock(429, transient_body).mount(&server).await;
    let e = gateway(&server)
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(e.message.contains("Overloaded"), "message: {}", e.message);
}

#[tokio::test]
async fn oversized_response_body_is_permanent_not_oom() {
    // A response over the transport cap must abort before buffering the
    // whole payload (P1-2). Content-Length is present here, so the header
    // path rejects without reading.
    let server = MockServer::start().await;
    let body = "x".repeat(33 * 1024 * 1024);
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;
    let started = std::time::Instant::now();
    let e = gateway(&server)
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Permanent),
        "got {e}"
    );
    assert!(e.message.contains("too large"), "message: {}", e.message);
    // far below the old unbounded-read cost; the reject happens early
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn unparseable_2xx_body_is_permanent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json at all"))
        .mount(&server)
        .await;
    let e = gateway(&server)
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Permanent),
        "got {e}"
    );

    // valid JSON but body-schema violation (missing stop_reason) → Permanent
    let server = MockServer::start().await;
    messages_mock(200, json!({"content": []}))
        .mount(&server)
        .await;
    let e = gateway(&server)
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Permanent),
        "got {e}"
    );
}

#[tokio::test]
async fn connect_error_is_transient() {
    // bind then drop: an address that refuses connections
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let gw = AnthropicMessagesGateway::new(KEY).with_endpoint(format!("http://{addr}/v1/messages"));
    let e = gw
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Transient),
        "got {e}"
    );
}

// --- acceptance: cancellation and deadline return immediately ---------------

#[tokio::test]
async fn attempt_deadline_yields_timed_out_promptly() {
    let server = MockServer::start().await;
    delayed_mock(
        200,
        json!({"content": [], "stop_reason": "end_turn"}),
        Duration::from_secs(5),
    )
    .mount(&server)
    .await;
    let deadline = Instant::now() + Duration::from_millis(100);
    let started = Instant::now();
    let e = gateway(&server)
        .invoke(&request(user_frame()), &ctrl(Some(deadline)))
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::TimedOut),
        "got {e}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "timeout took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn cancelled_token_yields_cancelled_promptly() {
    let server = MockServer::start().await;
    delayed_mock(
        200,
        json!({"content": [], "stop_reason": "end_turn"}),
        Duration::from_secs(5),
    )
    .mount(&server)
    .await;
    let token = CancellationToken::new();
    let canceller = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        canceller.cancel();
    });
    let attempt = AttemptControl::new(token.clone(), None);
    let started = Instant::now();
    let e = gateway(&server)
        .invoke(&request(user_frame()), &attempt)
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Cancelled),
        "got {e}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "cancellation took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn pre_cancelled_control_never_reaches_the_wire() {
    // No mock is mounted: any request would 404 (→ Permanent). The
    // short-circuit must return Cancelled instead.
    let server = MockServer::start().await;
    let token = CancellationToken::new();
    token.cancel();
    let attempt = AttemptControl::new(token.clone(), None);
    let e = gateway(&server)
        .invoke(&request(user_frame()), &attempt)
        .await
        .unwrap_err();
    assert!(
        matches!(e.kind(), ModelInvokeErrorKind::Cancelled),
        "got {e}"
    );
}

/// Appends every write into the shared capture buffer.
struct SpanSink(Arc<Mutex<String>>);
impl std::io::Write for SpanSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap()
            .push_str(&String::from_utf8_lossy(buf));
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn invoke_emits_the_agent_http_span() {
    let server = MockServer::start().await;
    server
        .register(messages_mock(
            200,
            json!({
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            }),
        ))
        .await;

    let captured: Arc<Mutex<String>> = Arc::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::NEW)
        .with_writer({
            let sink = captured.clone();
            move || SpanSink(sink.clone())
        })
        .finish();
    // Global default: hyper's connection events fire off the test thread,
    // so a thread-local default misses the span lifecycle lines.
    tracing::subscriber::set_global_default(subscriber)
        .expect("span test owns the global subscriber");

    gateway(&server)
        .invoke(&request(user_frame()), &ctrl(None))
        .await
        .unwrap();

    let text = captured.lock().unwrap().clone();
    assert!(text.contains("agent.http"), "missing agent.http: {text}");
    assert!(
        text.contains("provider=\"anthropic\""),
        "missing provider field: {text}"
    );
    assert!(
        text.contains("path=\"/v1/messages\""),
        "missing path field: {text}"
    );
}

// ---- media resolution (Slice 6.5) -----------------------------------------------

use causa_kernel::{ContentPart, MediaRef};
use causa_protocol::translation::media::MediaPayload;
use causa_provider::MediaResolver;
use std::collections::HashMap;

fn media_frame() -> ContextFrame {
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
                content: BlockContent::Parts(vec![
                    ContentPart::Text(TextPayload::new("look")),
                    ContentPart::Media(MediaRef::new("image/png", "asset-1")),
                ]),
                meta: BlockMeta {
                    provider_call_id: None,
                    source: Some("user".into()),
                },
            }],
        },
    }
}

fn ok_response() -> Value {
    json!({
        "content": [{"type": "text", "text": "seen"}],
        "stop_reason": "end_turn",
    })
}

#[tokio::test]
async fn resolved_media_rides_the_wire_as_image_blocks() {
    let server = MockServer::start().await;
    let asset_table: HashMap<String, MediaPayload> = [(
        "asset-1".to_string(),
        MediaPayload::new("image/png", "AAAA"),
    )]
    .into();
    let mock = Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "messages": [{"content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            ]}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()));
    server.register(mock).await;

    let gw = gateway(&server).with_media_resolver(Arc::new(asset_table));
    gw.invoke(&request(media_frame()), &ctrl(None))
        .await
        .unwrap();
}

#[tokio::test]
async fn unresolved_media_degrades_to_placeholder_on_the_wire() {
    let server = MockServer::start().await;
    // no resolver wired: the reference degrades to the text placeholder
    let mock = Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_string_contains("[media: image/png asset-1]"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()));
    server.register(mock).await;

    let gw = gateway(&server);
    gw.invoke(&request(media_frame()), &ctrl(None))
        .await
        .unwrap();
}

#[tokio::test]
async fn oversized_media_degrades_like_a_miss() {
    let server = MockServer::start().await;
    // payload of 4 MiB decoded bytes exceeds nothing at the default 5 MiB
    // ceiling — so shrink the ceiling instead and watch the big payload go
    let huge = "A".repeat(1024);
    let asset_table: HashMap<String, MediaPayload> =
        [("asset-1".to_string(), MediaPayload::new("image/png", huge))].into();
    let mock = Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_string_contains("[media: image/png asset-1]"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response()));
    server.register(mock).await;

    let gw = gateway(&server)
        .with_media_resolver(Arc::new(asset_table))
        .with_max_inline_media_bytes(16);
    gw.invoke(&request(media_frame()), &ctrl(None))
        .await
        .unwrap();
}

#[tokio::test]
async fn hash_map_asset_table_resolves_by_reference() {
    // the blanket impl: a plain table IS a MediaResolver
    let table: HashMap<String, MediaPayload> = [(
        "asset-1".to_string(),
        MediaPayload::new("image/png", "AAAA"),
    )]
    .into();
    let payload = MediaResolver::resolve(&table, &MediaRef::new("image/png", "asset-1"));
    assert_eq!(payload.unwrap().data_base64, "AAAA");
    assert!(MediaResolver::resolve(&table, &MediaRef::new("image/png", "nope")).is_none());
}
