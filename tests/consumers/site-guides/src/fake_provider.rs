//! A local HTTP server that answers like the three provider APIs. The site
//! never shows this module: it lets a guide's program send real requests
//! through the gateways without a key or a network. Streaming requests are
//! answered for Chat Completions only.

use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// One request the server received.
#[derive(Clone, Debug)]
pub struct Received {
    /// The request path, such as `/v1/messages`.
    pub path: String,
    /// The header that carried the key: `x-api-key` or `authorization`.
    pub auth_header: &'static str,
    /// The JSON body.
    pub body: Value,
}

/// The running server.
pub struct FakeProvider {
    /// Base URL with a version segment, such as `http://127.0.0.1:4000/v1`.
    pub base_url: String,
    received: Arc<Mutex<Vec<Received>>>,
}

impl FakeProvider {
    /// Starts the server on a free local port.
    pub async fn start() -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        let received = Arc::new(Mutex::new(Vec::new()));
        let log = received.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let log = log.clone();
                tokio::spawn(async move {
                    let _ = answer(stream, log).await;
                });
            }
        });
        Ok(Self { base_url, received })
    }

    /// The last request received.
    pub fn last(&self) -> Option<Received> {
        self.received.lock().ok()?.last().cloned()
    }
}

async fn answer(mut stream: TcpStream, log: Arc<Mutex<Vec<Received>>>) -> std::io::Result<()> {
    let mut data = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Ok(());
        }
        data.extend_from_slice(&chunk[..n]);
        if let Some(at) = data.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
    };
    let head = String::from_utf8_lossy(&data[..header_end]).to_ascii_lowercase();
    let length = head
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while data.len() < header_end + length {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&chunk[..n]);
    }
    let path = head
        .split_whitespace()
        .nth(1)
        .unwrap_or_default()
        .to_owned();
    let auth_header = if head.contains("\r\nx-api-key:") {
        "x-api-key"
    } else {
        "authorization"
    };
    let body: Value = serde_json::from_slice(&data[header_end..]).unwrap_or(Value::Null);

    let reply = if path.ends_with("/messages") {
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{ "type": "text", "text": "Hello from Anthropic Messages." }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 5, "output_tokens": 5 }
        })
    } else if path.ends_with("/chat/completions") {
        json!({
            "id": "chat_1", "object": "chat.completion", "model": body["model"],
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": { "role": "assistant", "content": "Hello from Chat Completions." }
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10 }
        })
    } else {
        json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": body["model"],
            "output": [{
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{ "type": "output_text", "text": "Hello from Responses.", "annotations": [] }]
            }],
            "usage": { "input_tokens": 5, "output_tokens": 5, "total_tokens": 10 }
        })
    };
    let streamed = body["stream"] == json!(true);
    if let Ok(mut log) = log.lock() {
        log.push(Received {
            path,
            auth_header,
            body,
        });
    }
    let (content_type, reply) = if streamed {
        ("text/event-stream", chat_events())
    } else {
        ("application/json", reply.to_string())
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
        reply.len()
    );
    stream.write_all(response.as_bytes()).await?;
    stream.shutdown().await
}

/// A streamed Chat Completions answer: three text fragments, the finish
/// reason, the usage, and the end marker.
fn chat_events() -> String {
    let chunk = |delta: Value, finish: Value| {
        json!({
            "id": "chat_1", "object": "chat.completion.chunk", "model": "model",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
        })
    };
    let mut events = vec![chunk(
        json!({ "role": "assistant", "content": "" }),
        Value::Null,
    )];
    for text in ["Hel", "lo ", "there."] {
        events.push(chunk(json!({ "content": text }), Value::Null));
    }
    events.push(chunk(json!({}), json!("stop")));
    events.push(json!({
        "id": "chat_1", "object": "chat.completion.chunk", "model": "model", "choices": [],
        "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 }
    }));
    let mut body: String = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect();
    body.push_str("data: [DONE]\n\n");
    body
}
