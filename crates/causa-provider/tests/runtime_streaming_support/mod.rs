//! A real chunked HTTP server whose terminal bytes wait for a consumer signal.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{Value, json};

pub struct Round {
    pub prefix: String,
    pub suffix: String,
    pub gate: Option<mpsc::Receiver<()>>,
}

impl Round {
    pub fn immediate(body: String) -> Self {
        Self {
            prefix: body,
            suffix: String::new(),
            gate: None,
        }
    }

    pub fn gated(prefix: String, suffix: String) -> (Self, mpsc::Sender<()>) {
        let (send, receive) = mpsc::channel();
        (
            Self {
                prefix,
                suffix,
                gate: Some(receive),
            },
            send,
        )
    }
}

pub struct Server {
    pub url: String,
    pub requests: Arc<Mutex<Vec<Value>>>,
    worker: Option<JoinHandle<Result<(), String>>>,
}

impl Server {
    pub fn start(rounds: Vec<Round>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let received = requests.clone();
        let worker = std::thread::spawn(move || {
            for round in rounds {
                let until = std::time::Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            if std::time::Instant::now() >= until {
                                return Err("HTTP request never arrived".into());
                            }
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        Err(error) => return Err(error.to_string()),
                    }
                };
                // Accepted sockets inherit nonblocking mode on some platforms.
                // The dedicated fixture thread performs bounded blocking reads.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                received.lock().unwrap().push(read_request(&mut stream)?);
                stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").map_err(|e| e.to_string())?;
                chunk(&mut stream, round.prefix.as_bytes())?;
                if let Some(gate) = round.gate {
                    gate.recv_timeout(Duration::from_secs(4)).map_err(|_| {
                        "consumer did not observe the prefix before terminal release".to_owned()
                    })?;
                }
                chunk(&mut stream, round.suffix.as_bytes())?;
                // Cancellation is allowed to close the local stream first.
                let _ = stream.write_all(b"0\r\n\r\n");
            }
            Ok(())
        });
        Self {
            url,
            requests,
            worker: Some(worker),
        }
    }

    pub fn finish(&mut self) {
        self.worker.take().unwrap().join().unwrap().unwrap();
    }
}

fn read_request(stream: &mut TcpStream) -> Result<Value, String> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 4096];
    let (start, length) = loop {
        let count = stream.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            return Err("request ended before headers".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(index) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
            let header = std::str::from_utf8(&bytes[..index]).map_err(|e| e.to_string())?;
            let length = header
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .ok_or("missing content length")?;
            break (index + 4, length);
        }
    };
    while bytes.len() < start + length {
        let count = stream.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            return Err("request ended before body".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    serde_json::from_slice(&bytes[start..start + length]).map_err(|e| e.to_string())
}

fn chunk(stream: &mut TcpStream, bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    write!(stream, "{:x}\r\n", bytes.len()).map_err(|e| e.to_string())?;
    stream.write_all(bytes).map_err(|e| e.to_string())?;
    stream.write_all(b"\r\n").map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

pub fn event(value: Value) -> String {
    format!("data: {value}\n\n")
}

pub fn named_event(kind: &str, value: Value) -> String {
    format!("event: {kind}\ndata: {value}\n\n")
}

pub fn chat_text(text: &str) -> String {
    event(
        json!({"id":"chat1","choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]}),
    )
}

pub fn chat_terminal(reason: &str) -> String {
    event(json!({"id":"chat1","choices":[{"index":0,"delta":{},"finish_reason":reason}]}))
        + "data: [DONE]\n\n"
}

pub fn chat_tool(arguments: &str) -> String {
    event(
        json!({"id":"chat1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_wire","type":"function","function":{"name":"echo","arguments":arguments}}]},"finish_reason":null}]}),
    )
}

pub fn responses_prefix(text: &str) -> String {
    named_event(
        "response.created",
        json!({"type":"response.created","response":{"id":"resp1","status":"in_progress","output":[]}}),
    ) + &named_event(
        "response.output_item.added",
        json!({"type":"response.output_item.added","output_index":0,"item":{"id":"msg1","type":"message","role":"assistant","content":[]}}),
    ) + &named_event(
        "response.content_part.added",
        json!({"type":"response.content_part.added","output_index":0,"item_id":"msg1","content_index":0,"part":{"type":"output_text","text":""}}),
    ) + &named_event(
        "response.output_text.delta",
        json!({"type":"response.output_text.delta","output_index":0,"item_id":"msg1","content_index":0,"delta":text}),
    )
}

pub fn responses_terminal(text: &str) -> String {
    named_event(
        "response.completed",
        json!({"type":"response.completed","response":{"id":"resp1","status":"completed","output":[{"id":"msg1","type":"message","role":"assistant","content":[{"type":"output_text","text":text}]}],"usage":{"input_tokens":7,"output_tokens":3}}}),
    )
}

pub fn anthropic_prefix(text: &str) -> String {
    named_event(
        "message_start",
        json!({"type":"message_start","message":{"id":"msg1","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":7,"output_tokens":0}}}),
    ) + &named_event(
        "content_block_start",
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
    ) + &named_event(
        "content_block_delta",
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}}),
    )
}

pub fn anthropic_terminal() -> String {
    named_event(
        "content_block_stop",
        json!({"type":"content_block_stop","index":0}),
    ) + &named_event(
        "message_delta",
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}),
    ) + &named_event("message_stop", json!({"type":"message_stop"}))
}
