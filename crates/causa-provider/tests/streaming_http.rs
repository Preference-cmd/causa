//! Real chunked HTTP tests: the server holds termination until the consumer acknowledges a delta.
use causa_kernel::{
    CacheDirective, CallControl, CancellationToken, ContextFrame, GenerationOptions, InvocationId,
    ModelGateway, ModelInvokeErrorKind, ModelRef, ModelRequest, RoundId, StreamDelta, ToolSurface,
    TurnId,
};
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway, StreamLimits,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
};

const WAIT: Duration = Duration::from_secs(2);
fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}
fn request() -> ModelRequest {
    {
        let mut request = ModelRequest::new(
            InvocationId {
                turn_id: TurnId::new("stream-http"),
                round_id: RoundId(0),
            },
            ContextFrame {
                blocks: vec![causa_kernel::ContextBlock::new(
                    causa_kernel::BlockId::new(uuid::Uuid::from_u128(100)),
                    causa_kernel::BlockContent::Parts(vec![causa_kernel::ContentPart::Text(
                        causa_kernel::TextPayload::new("hi"),
                    )]),
                    causa_kernel::BlockMeta::from_source("user"),
                )],
            },
            ModelRef::new("test-model"),
        );
        // Anthropic requires `max_tokens`; the OpenAI wires render it as their
        // own limit key, which no case here asserts on.
        request.generation = GenerationOptions::default().with_max_tokens(1024);
        request
    }
}
fn chat_delta() -> String {
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"你🙂\"},\"finish_reason\":null}]}\n\n".into()
}
fn chat_terminal() -> String {
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into()
}
fn event(name: &str, body: Value) -> String {
    format!("event: {name}\ndata: {body}\n\n")
}

async fn read_request(socket: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut byte = [0];
        assert_eq!(socket.read(&mut byte).await.unwrap(), 1);
        bytes.push(byte[0]);
        if bytes.ends_with(b"\r\n\r\n") {
            break bytes.len();
        }
    };
    let headers = String::from_utf8(bytes.clone()).unwrap();
    let length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    bytes.resize(header_end + length, 0);
    socket.read_exact(&mut bytes[header_end..]).await.unwrap();
    String::from_utf8(bytes).unwrap()
}
async fn chunk(socket: &mut TcpStream, text: &str) {
    if !text.is_empty() {
        socket
            .write_all(format!("{:x}\r\n{text}\r\n", text.len()).as_bytes())
            .await
            .unwrap();
        socket.flush().await.unwrap();
    }
}

struct Fixture {
    url: String,
    request: oneshot::Receiver<String>,
    advance: Option<oneshot::Sender<()>>,
    closed: oneshot::Receiver<()>,
    count: Arc<AtomicUsize>,
    task: JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Fixture {
    async fn new(
        status: u16,
        content_type: &str,
        first: String,
        tail: String,
        wait_headers: bool,
        end_body: bool,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (request_tx, request) = oneshot::channel();
        let (advance_tx, advance_rx) = oneshot::channel();
        let (closed_tx, closed) = oneshot::channel();
        let count = Arc::new(AtomicUsize::new(0));
        let requests = count.clone();
        let content_type = content_type.to_owned();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            requests.fetch_add(1, Ordering::SeqCst);
            request_tx.send(read_request(&mut socket).await).unwrap();
            let mut advance_rx = Some(advance_rx);
            if wait_headers {
                // No headers will be sent: control must interrupt send itself.
                let mut byte = [0];
                let _ = socket.read(&mut byte).await;
                let _ = closed_tx.send(());
                return;
            }
            socket.write_all(format!("HTTP/1.1 {status} fixture\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\n\r\n").as_bytes()).await.unwrap();
            chunk(&mut socket, &first).await;
            if advance_rx.take().unwrap().await.is_ok() {
                chunk(&mut socket, &tail).await;
                if end_body {
                    socket.write_all(b"0\r\n\r\n").await.unwrap();
                }
            }
            let mut byte = [0];
            let _ = socket.read(&mut byte).await;
            let _ = closed_tx.send(());
            // A second logical request would connect here; record it if any.
            if tokio::time::timeout(Duration::from_millis(50), listener.accept())
                .await
                .is_ok()
            {
                requests.fetch_add(1, Ordering::SeqCst);
            }
        });
        Self {
            url,
            request,
            advance: Some(advance_tx),
            closed,
            count,
            task,
        }
    }
    fn advance(&mut self) {
        self.advance.take().unwrap().send(()).unwrap();
    }
    async fn assert_closed(&mut self) {
        tokio::time::timeout(WAIT, &mut self.closed)
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn all_protocols_yield_before_server_releases_terminal_and_close_without_eof() {
    let sequences = [
        (chat_delta(), chat_terminal()),
        (
            event(
                "response.output_item.added",
                json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m1","role":"assistant","content":[]}}),
            ) + &event(
                "response.content_part.added",
                json!({"type":"response.content_part.added","output_index":0,"item_id":"m1","content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}),
            ) + &event(
                "response.output_text.delta",
                json!({"type":"response.output_text.delta","output_index":0,"item_id":"m1","content_index":0,"delta":"你🙂"}),
            ),
            event(
                "response.completed",
                json!({"type":"response.completed","response":{"id":"r1","status":"completed","output":[{"type":"message","id":"m1","role":"assistant","content":[{"type":"output_text","text":"你🙂","annotations":[]}]}]}}),
            ),
        ),
        (
            event(
                "message_start",
                json!({"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"test","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}),
            ) + &event(
                "content_block_start",
                json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            ) + &event(
                "content_block_delta",
                json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你🙂"}}),
            ),
            event(
                "content_block_stop",
                json!({"type":"content_block_stop","index":0}),
            ) + &event(
                "message_delta",
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}),
            ) + &event("message_stop", json!({"type":"message_stop"})),
        ),
    ];
    for (index, (first, tail)) in sequences.into_iter().enumerate() {
        let mut fixture = Fixture::new(
            200,
            "text/event-stream; charset=utf-8",
            first,
            tail + "data: malformed unread tail\n\n",
            false,
            false,
        )
        .await;
        let gateway: Box<dyn ModelGateway> = match index {
            0 => Box::new(
                OpenAiChatCompletionsGateway::new("test-key")
                    .with_base_url(format!("{}/v1", fixture.url)),
            ),
            1 => Box::new(
                OpenAiResponsesGateway::new("test-key")
                    .with_base_url(format!("{}/v1", fixture.url)),
            ),
            _ => Box::new(
                AnthropicMessagesGateway::new("test-key")
                    .with_base_url(format!("{}/v1", fixture.url)),
            ),
        };
        let mut stream = gateway.stream(&request(), &control()).await.unwrap();
        let captured = (&mut fixture.request).await.unwrap();
        let (headers, body) = captured.split_once("\r\n\r\n").unwrap();
        let body: Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(body["model"], "test-model");
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("accept: text/event-stream")
        );
        if index == 0 {
            assert_eq!(body["stream_options"]["include_usage"], true);
        }
        if index == 2 {
            assert!(headers.contains("x-api-key: test-key"));
            assert!(headers.contains("anthropic-version: 2023-06-01"));
        } else {
            assert!(headers.contains("authorization: Bearer test-key"));
        }
        loop {
            match tokio::time::timeout(WAIT, stream.next())
                .await
                .unwrap()
                .unwrap()
            {
                StreamDelta::TextDelta { delta } => {
                    assert_eq!(delta, "你🙂");
                    break;
                }
                StreamDelta::Error { error } => panic!("{error:?}"),
                StreamDelta::Done { .. } => panic!("terminal reached before release"),
                _ => {}
            }
        }
        // Consumption can pause without a background reader completing the call.
        assert!(
            tokio::time::timeout(Duration::from_millis(20), stream.next())
                .await
                .is_err()
        );
        fixture.advance();
        loop {
            match tokio::time::timeout(WAIT, stream.next())
                .await
                .unwrap()
                .unwrap()
            {
                StreamDelta::Done { final_output, .. } => {
                    assert!(!final_output.response.text().is_empty());
                    break;
                }
                StreamDelta::Error { error } => panic!("{error:?}"),
                _ => {}
            }
        }
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
        fixture.assert_closed().await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(fixture.count.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn standalone_control_covers_waiting_headers_and_silent_body_with_timeout_free_client() {
    for wait_headers in [true, false] {
        for cancellation in [true, false] {
            let mut fixture = Fixture::new(
                200,
                "text/event-stream",
                String::new(),
                String::new(),
                wait_headers,
                false,
            )
            .await;
            let token = CancellationToken::new();
            let call = CallControl::new(
                token.clone(),
                (!cancellation).then(|| Instant::now() + Duration::from_millis(100)),
            );
            let gateway = OpenAiChatCompletionsGateway::new("test")
                .with_base_url(format!("{}/v1", fixture.url))
                .with_http_client(reqwest::Client::builder().build().unwrap());
            let req = request();
            let cancel = async {
                if cancellation {
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    token.cancel();
                }
            };
            let consume = async {
                match gateway.stream(&req, &call).await {
                    Err(error) => {
                        assert!(wait_headers);
                        error.kind
                    }
                    Ok(mut stream) => {
                        assert!(!wait_headers);
                        let kind = match stream.next().await.unwrap() {
                            StreamDelta::Error { error, .. } => error.kind,
                            other => panic!("unexpected delta: {other:?}"),
                        };
                        assert!(stream.next().await.is_none());
                        assert!(stream.next().await.is_none());
                        kind
                    }
                }
            };
            let (kind, ()) = tokio::time::timeout(WAIT, async { tokio::join!(consume, cancel) })
                .await
                .unwrap();
            assert_eq!(
                kind,
                if cancellation {
                    ModelInvokeErrorKind::Cancelled
                } else {
                    ModelInvokeErrorKind::TimedOut
                }
            );
            // A body fixture waits for acknowledgment, release it without data
            // so it can observe the already closed client socket.
            if !wait_headers {
                fixture.advance();
            }
            fixture.assert_closed().await;
        }
    }
}

#[tokio::test]
async fn dropping_stream_closes_unfinished_response_without_polling_background_reader() {
    let mut fixture = Fixture::new(
        200,
        "text/event-stream",
        chat_delta(),
        String::new(),
        false,
        false,
    )
    .await;
    let gateway =
        OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", fixture.url));
    let stream = gateway.stream(&request(), &control()).await.unwrap();
    drop(stream);
    fixture.advance();
    fixture.assert_closed().await;
    assert_eq!(fixture.count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn http_and_content_type_errors_are_pre_stream_failures_and_read_errors_end_once() {
    for (status, content_type, body, kind) in [
        (
            200,
            "application/json",
            "{}",
            ModelInvokeErrorKind::Undecodable,
        ),
        (
            401,
            "application/json",
            "{\"error\":{\"message\":\"bad key\"}}",
            ModelInvokeErrorKind::Permanent,
        ),
        (
            422,
            "application/json",
            "bad request",
            ModelInvokeErrorKind::InvalidRequest,
        ),
        (
            429,
            "application/json",
            "rate limit",
            ModelInvokeErrorKind::Transient,
        ),
        (
            503,
            "text/html",
            "unavailable",
            ModelInvokeErrorKind::Transient,
        ),
    ] {
        let mut fixture = Fixture::new(
            status,
            content_type,
            body.into(),
            String::new(),
            false,
            true,
        )
        .await;
        fixture.advance();
        let gateway =
            OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", fixture.url));
        let error = match gateway.stream(&request(), &control()).await {
            Err(error) => error,
            Ok(_) => panic!("invalid response accepted"),
        };
        assert_eq!(error.kind, kind);
    }
    for (body, kind) in [
        (
            "data: nonsense\n\n".to_owned(),
            ModelInvokeErrorKind::Undecodable,
        ),
        (chat_delta(), ModelInvokeErrorKind::UnknownOutcome),
        (
            "data: {\"choices\":[]}\n".into(),
            ModelInvokeErrorKind::UnknownOutcome,
        ),
    ] {
        let mut fixture =
            Fixture::new(200, "text/event-stream", body, String::new(), false, true).await;
        fixture.advance();
        let gateway =
            OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", fixture.url));
        let mut stream = gateway.stream(&request(), &control()).await.unwrap();
        let actual = loop {
            match stream.next().await.unwrap() {
                StreamDelta::Error { error, .. } => break error.kind,
                StreamDelta::Done { .. } => panic!("invalid stream completed"),
                _ => {}
            }
        };
        assert_eq!(actual, kind);
        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
    }
}

#[tokio::test]
async fn decoder_limits_and_ready_ignored_events_preserve_control_fairness() {
    let mut fixture = Fixture::new(
        200,
        "text/event-stream",
        ": heartbeat\n\n".repeat(100_000),
        String::new(),
        false,
        false,
    )
    .await;
    let gateway =
        OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", fixture.url));
    let token = CancellationToken::new();
    let call = CallControl::new(token.clone(), None);
    let mut stream = gateway.stream(&request(), &call).await.unwrap();
    tokio::spawn(async move {
        tokio::task::yield_now().await;
        token.cancel();
    });
    assert!(matches!(
        tokio::time::timeout(WAIT, stream.next()).await.unwrap(),
        Some(StreamDelta::Error { error }) if error.kind == ModelInvokeErrorKind::Cancelled
    ));
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none());
    // Avoid writing after cancellation: this fixture has no tail.
    fixture.advance();

    let mut fixture = Fixture::new(
        200,
        "text/event-stream",
        chat_delta(),
        String::new(),
        false,
        false,
    )
    .await;
    let limits = {
        let mut limits = StreamLimits::default();
        limits.max_total_bytes = 10;
        limits
    };
    let gateway = OpenAiChatCompletionsGateway::new("test")
        .with_base_url(format!("{}/v1", fixture.url))
        .with_stream_limits(limits);
    let mut stream = gateway.stream(&request(), &control()).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(StreamDelta::Error { error }) if error.kind == ModelInvokeErrorKind::Undecodable
    ));
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none());
    fixture.advance();
    fixture.assert_closed().await;
}

#[tokio::test]
async fn stream_and_invoke_share_media_tools_cache_generation_headers_and_endpoint() {
    use causa_kernel::{
        BlockContent, BlockId, BlockMeta, ContentPart, ContextBlock, MediaRef, TextPayload,
        ToolDefinition,
    };
    use causa_protocol::translation::media::MediaPayload;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_partial_json, method},
    };
    let mut req = request();
    req.frame.blocks.push(ContextBlock::new(
        BlockId::new(uuid::Uuid::from_u128(1)),
        BlockContent::Parts(vec![
            ContentPart::Text(TextPayload::new("look")),
            ContentPart::Media(MediaRef::new("image/png", "asset")),
        ]),
        BlockMeta::from_source("user"),
    ));
    req.tool_surface = ToolSurface::from_definitions(vec![ToolDefinition::new(
        "read",
        "read something",
        json!({"type":"object","properties":{"path":{"type":"string"}}}),
    )]);
    req.cache = CacheDirective::StablePrefix;
    req.generation.temperature = Some(0.25);
    req.generation.max_tokens = Some(123);
    let media: Arc<std::collections::HashMap<String, MediaPayload>> =
        Arc::new([("asset".into(), MediaPayload::new("image/png", "AAAA"))].into());
    for index in 0..3 {
        let server = MockServer::start().await;
        let complete = match index {
            0 => {
                json!({"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})
            }
            1 => {
                json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]})
            }
            _ => json!({"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}),
        };
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(complete))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"stream":true})))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(": heartbeat\n\n", "text/event-stream"),
            )
            .with_priority(1)
            .mount(&server)
            .await;
        let gateway: Box<dyn ModelGateway> = match index {
            0 => Box::new(
                OpenAiChatCompletionsGateway::new("parity")
                    .with_base_url(format!("{}/v1", server.uri()))
                    .with_media_resolver(media.clone()),
            ),
            1 => Box::new(
                OpenAiResponsesGateway::new("parity")
                    .with_base_url(format!("{}/v1", server.uri()))
                    .with_media_resolver(media.clone()),
            ),
            _ => Box::new(
                AnthropicMessagesGateway::new("parity")
                    .with_base_url(format!("{}/v1", server.uri()))
                    .with_media_resolver(media.clone())
                    .with_anthropic_version("2023-06-01"),
            ),
        };
        gateway.invoke(&req, &control()).await.unwrap();
        drop(gateway.stream(&req, &control()).await.unwrap());
        let captured = server.received_requests().await.unwrap();
        assert_eq!(captured.len(), 2);
        let invoke_body: Value = serde_json::from_slice(&captured[0].body).unwrap();
        let mut stream_body: Value = serde_json::from_slice(&captured[1].body).unwrap();
        assert_eq!(
            stream_body.as_object_mut().unwrap().remove("stream"),
            Some(json!(true))
        );
        if index == 0 {
            assert_eq!(
                stream_body
                    .as_object_mut()
                    .unwrap()
                    .remove("stream_options"),
                Some(json!({"include_usage":true}))
            );
        }
        assert_eq!(invoke_body, stream_body);
        assert!(invoke_body.to_string().contains("AAAA"));
        assert!(invoke_body.to_string().contains("read"));
        assert_eq!(captured[0].url, captured[1].url);
        for name in [
            "authorization",
            "x-api-key",
            "anthropic-version",
            "content-type",
        ] {
            assert_eq!(captured[0].headers.get(name), captured[1].headers.get(name));
        }
    }
}

#[tokio::test]
async fn ready_deltas_yield_to_cancellation_and_slow_consumption_defers_parsing() {
    // Establish and consume once so subsequent calls can stay ready from the
    // same network allocation. Cancellation requires a scheduler opportunity.
    let mut fixture = Fixture::new(
        200,
        "text/event-stream",
        chat_delta().repeat(5_000),
        String::new(),
        false,
        false,
    )
    .await;
    let gateway =
        OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", fixture.url));
    let token = CancellationToken::new();
    let call = CallControl::new(token.clone(), None);
    let mut stream = gateway.stream(&request(), &call).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(StreamDelta::TextDelta { .. })
    ));
    tokio::spawn(async move {
        token.cancel();
    });
    let mut observed = 1;
    loop {
        match tokio::time::timeout(WAIT, stream.next())
            .await
            .unwrap()
            .unwrap()
        {
            StreamDelta::TextDelta { .. } => observed += 1,
            StreamDelta::Error { error, .. } => {
                assert_eq!(error.kind, ModelInvokeErrorKind::Cancelled);
                break;
            }
            delta => panic!("unexpected delta: {delta:?}"),
        }
    }
    assert!(observed < 5_000, "ready decoding starved cancellation");
    assert!(stream.next().await.is_none());
    fixture.advance();

    // A malformed second event is already delivered by the server, but the
    // gateway must not parse it while the consumer is paused after one delta.
    let mut fixture = Fixture::new(
        200,
        "text/event-stream",
        chat_delta() + "data: malformed\n\n",
        String::new(),
        false,
        false,
    )
    .await;
    let gateway =
        OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", fixture.url));
    let call = control();
    let mut stream = gateway.stream(&request(), &call).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(StreamDelta::TextDelta { .. })
    ));
    tokio::time::sleep(Duration::from_millis(10)).await;
    call.cancellation_token().cancel();
    assert!(matches!(
        stream.next().await,
        Some(StreamDelta::Error { error }) if error.kind == ModelInvokeErrorKind::Cancelled
    ));
    assert!(stream.next().await.is_none());
    fixture.advance();
    fixture.assert_closed().await;
}

#[tokio::test]
async fn bounded_error_bodies_honor_control_and_never_echo_request_material() {
    let secret = "DO_NOT_ECHO_PRIVATE_PROMPT_OR_TOOL_ARGUMENTS";
    // Raw body (not the documented envelope) must never be echoed in error.message
    let mut fixture = Fixture::new(
        400,
        "text/plain",
        format!("raw server error echoing: {secret}"),
        String::new(),
        false,
        true,
    )
    .await;
    fixture.advance();
    let gateway =
        OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", fixture.url));
    let error = match gateway.stream(&request(), &control()).await {
        Err(error) => error,
        Ok(_) => panic!("HTTP error accepted"),
    };
    assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
    assert_eq!(error.message, "openai error 400");
    assert!(!error.message.contains(secret));

    let mut fixture = Fixture::new(
        429,
        "application/json",
        "{\"error\":".into(),
        String::new(),
        false,
        false,
    )
    .await;
    let gateway = OpenAiChatCompletionsGateway::new("test")
        .with_base_url(format!("{}/v1", fixture.url))
        .with_http_client(reqwest::Client::new());
    let call = CallControl::new(
        CancellationToken::new(),
        Some(Instant::now() + Duration::from_millis(50)),
    );
    let error = match tokio::time::timeout(WAIT, gateway.stream(&request(), &call))
        .await
        .unwrap()
    {
        Err(error) => error,
        Ok(_) => panic!("unfinished error response accepted"),
    };
    assert_eq!(error.kind, ModelInvokeErrorKind::TimedOut);
    fixture.advance();
    fixture.assert_closed().await;

    // Reject a declared over-cap error body without waiting for its bytes.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/test", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_request(&mut socket).await;
        socket
            .write_all(b"HTTP/1.1 500 Error\r\nContent-Length: 33554433\r\n\r\n")
            .await
            .unwrap();
        let mut byte = [0];
        socket.read(&mut byte).await.unwrap()
    });
    let gateway = OpenAiChatCompletionsGateway::new("test").with_endpoint(endpoint);
    let error = match tokio::time::timeout(WAIT, gateway.stream(&request(), &control()))
        .await
        .unwrap()
    {
        Err(error) => error,
        Ok(_) => panic!("oversized error accepted"),
    };
    assert_eq!(error.kind, ModelInvokeErrorKind::Undecodable);
    assert!(error.message.contains("too large"));
    assert_eq!(
        tokio::time::timeout(WAIT, server).await.unwrap().unwrap(),
        0
    );
}

#[tokio::test]
async fn invoke_error_envelope_diagnostics_are_bounded() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(500)
                .set_body_json(json!({"error":{"message":"x".repeat(10_000)}})),
        )
        .mount(&server)
        .await;
    let gateway =
        OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", server.uri()));
    let error = gateway.invoke(&request(), &control()).await.unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::Transient);
    assert!(error.message.len() < 300);
    assert!(error.message.ends_with('…'));
}

#[tokio::test]
async fn truncated_chunked_http_body_is_transient_and_emits_one_error() {
    let fixture = Fixture::new(
        200,
        "text/event-stream",
        chat_delta(),
        String::new(),
        false,
        false,
    )
    .await;
    let gateway =
        OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", fixture.url));
    let mut stream = gateway.stream(&request(), &control()).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(StreamDelta::TextDelta { .. })
    ));
    fixture.task.abort();
    assert!(matches!(
        tokio::time::timeout(WAIT, stream.next()).await.unwrap(),
        Some(StreamDelta::Error { error }) if error.kind == ModelInvokeErrorKind::Transient
    ));
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn default_client_does_not_follow_redirects_or_resubmit_http_errors() {
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};
    for status in [302, 429, 503] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(status)
                    .insert_header("location", format!("{}/redirected", server.uri()))
                    .set_body_string("rejected"),
            )
            .mount(&server)
            .await;
        let gateway =
            OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", server.uri()));
        assert!(gateway.stream(&request(), &control()).await.is_err());
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }
}

#[tokio::test]
async fn invalid_stream_limits_fail_before_http() {
    let server = wiremock::MockServer::start().await;
    let limits = {
        let mut limits = StreamLimits::default();
        limits.max_pending_deltas = 0;
        limits
    };
    let gateway = OpenAiChatCompletionsGateway::new("test")
        .with_base_url(format!("{}/v1", server.uri()))
        .with_stream_limits(limits);
    let error = match gateway.stream(&request(), &control()).await {
        Err(error) => error,
        Ok(_) => panic!("zero limit accepted"),
    };
    assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
    assert!(server.received_requests().await.unwrap().is_empty());
}
