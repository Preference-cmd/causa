//! Reasoning material through the real gateways: sealed provider items from a
//! first round are sent back unchanged on the next one, and the render-option
//! setters change exactly the wire fields they name.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CallControl, CancellationToken, ContentPart, ContextBlock,
    ContextFrame, GenerationOptions, InvocationId, ModelGateway, ModelInvokeErrorKind, ModelOutput,
    ModelRef, ModelRequest, ReasoningEffort, RoundId, SealedPart, StreamDelta, TextPayload,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnId,
};
use causa_protocol::translation::render_options::PrefixMismatchBehavior;
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY: &str = "sk-test";

static NEXT_ID: AtomicU64 = AtomicU64::new(1_000);

/// Fresh block identities for the second round of a loop.
fn next_id() -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(u128::from(
        NEXT_ID.fetch_add(1, Ordering::Relaxed),
    )))
}

fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

fn user_block() -> ContextBlock {
    ContextBlock::new(
        BlockId::new(uuid::Uuid::from_u128(1)),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("hi"))]),
        BlockMeta::from_source("user"),
    )
}

fn user_frame() -> ContextFrame {
    ContextFrame {
        blocks: vec![user_block()],
    }
}

fn request(frame: ContextFrame, model: &str) -> ModelRequest {
    {
        let mut request = ModelRequest::new(
            InvocationId {
                turn_id: TurnId::new("replay"),
                round_id: RoundId(0),
            },
            frame,
            ModelRef::new(model),
        );
        // The Anthropic wire requires a limit; the other two ignore it.
        request.generation = {
            let mut generation = GenerationOptions::default();
            generation.max_tokens = Some(1024);
            generation
        };
        request
    }
}

/// The assistant blocks of the first round plus the answer to its tool call:
/// the frame of the second round of a thinking-enabled tool loop.
fn answered_frame(frame: &ContextFrame, output: &ModelOutput) -> ContextFrame {
    let ids: Vec<BlockId> = (0..output.response.block_count())
        .map(|_| next_id())
        .collect();
    let blocks = output
        .response
        .to_blocks(output.stop_reason, &ids)
        .expect("conversion");
    let call = blocks
        .iter()
        .find(|b| matches!(b.content(), BlockContent::ToolCall(_)))
        .map(|b| b.id())
        .expect("the first round ends in a tool call");
    let mut all = frame.blocks.clone();
    all.extend(blocks);
    all.push(ContextBlock::new(
        next_id(),
        BlockContent::ToolResult(ToolResultPayload::new(
            call,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!("ok")),
        )),
        BlockMeta::default(),
    ));
    ContextFrame { blocks: all }
}

fn anthropic_sse(turn: &Value) -> String {
    let mut sse = String::from(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
    );
    for (i, block) in turn["content"].as_array().unwrap().iter().enumerate() {
        let kind = block["type"].as_str().unwrap();
        let start = match kind {
            "thinking" => json!({"type": "thinking", "thinking": ""}),
            "text" => json!({"type": "text", "text": ""}),
            _ => block.clone(),
        };
        sse.push_str(&format!(
            "event: content_block_start\ndata: {}\n\n",
            json!({"type": "content_block_start", "index": i, "content_block": start})
        ));
        match kind {
            "thinking" => {
                sse.push_str(&format!(
                    "event: content_block_delta\ndata: {}\n\n",
                    json!({"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": block["thinking"]}})
                ));
                sse.push_str(&format!(
                    "event: content_block_delta\ndata: {}\n\n",
                    json!({"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": block["signature"]}})
                ));
            }
            "text" => {
                sse.push_str(&format!(
                    "event: content_block_delta\ndata: {}\n\n",
                    json!({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}})
                ));
            }
            _ => {}
        }
        sse.push_str(&format!(
            "event: content_block_stop\ndata: {}\n\n",
            json!({"type": "content_block_stop", "index": i})
        ));
    }
    sse.push_str(&format!(
        "event: message_delta\ndata: {}\n\n",
        json!({"type": "message_delta", "delta": {"stop_reason": turn["stop_reason"]}, "usage": {"output_tokens": 1}})
    ));
    sse.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    sse
}

/// A Responses SSE turn: declare every output item, then send the authoritative
/// terminal snapshot the decoder validates against the declarations.
fn responses_sse(output: &Value) -> String {
    let mut sse = format!(
        "event: response.created\ndata: {}\n\n",
        json!({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}})
    );
    for (i, item) in output.as_array().unwrap().iter().enumerate() {
        for kind in ["response.output_item.added", "response.output_item.done"] {
            sse.push_str(&format!(
                "event: {kind}\ndata: {}\n\n",
                json!({"type": kind, "output_index": i, "item": item})
            ));
        }
    }
    sse.push_str(&format!(
        "event: response.completed\ndata: {}\n\n",
        json!({"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": output}})
    ));
    sse
}

fn text_response(content: &str) -> Value {
    json!({"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"})
}

async fn only_request_body(server: &MockServer) -> Value {
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1, "exactly one request");
    serde_json::from_slice(&received[0].body).unwrap()
}

fn anthropic_sealed(item: Value) -> ContextBlock {
    ContextBlock::new(
        next_id(),
        BlockContent::Parts(vec![ContentPart::Sealed(SealedPart::new(
            causa_protocol::Protocol::AnthropicMessages.as_str(),
            item,
        ))]),
        BlockMeta::default(),
    )
}

#[tokio::test]
async fn anthropic_thinking_tool_loop_replays_sealed_blocks_verbatim() {
    let turn = json!({
        "content": [
            {"type": "thinking", "thinking": "Compare.", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "enc-1"},
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a"}},
        ],
        "stop_reason": "tool_use",
    });
    for streaming in [false, true] {
        let server = MockServer::start().await;
        let response = if streaming {
            ResponseTemplate::new(200).set_body_raw(anthropic_sse(&turn), "text/event-stream")
        } else {
            ResponseTemplate::new(200).set_body_json(&turn)
        };
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(response)
            .mount(&server)
            .await;
        let gateway =
            AnthropicMessagesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
        let first = if streaming {
            let mut stream = gateway
                .stream(&request(user_frame(), "claude-test"), &control())
                .await
                .unwrap();
            let mut terminal = None;
            while let Some(delta) = stream.next().await {
                if let StreamDelta::Done { final_output, .. } = delta {
                    terminal = Some(final_output);
                }
            }
            terminal.expect("the stream completed")
        } else {
            gateway
                .invoke(&request(user_frame(), "claude-test"), &control())
                .await
                .unwrap()
        };
        assert_eq!(first.response.items.len(), 4, "streaming {streaming}");
        server.reset().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(text_response("done")))
            .mount(&server)
            .await;
        let second = request(answered_frame(&user_frame(), &first), "claude-test");
        gateway.invoke(&second, &control()).await.unwrap();
        let body = only_request_body(&server).await;
        let messages = body["messages"].as_array().unwrap();
        let assistant = messages
            .iter()
            .find(|m| m["role"] == "assistant")
            .expect("assistant message");
        let content = assistant["content"].as_array().unwrap();
        assert_eq!(
            content[0],
            json!({"type": "thinking", "thinking": "Compare.", "signature": "sig-1"}),
            "streaming {streaming}"
        );
        assert_eq!(
            content[1],
            json!({"type": "redacted_thinking", "data": "enc-1"}),
            "streaming {streaming}"
        );
        assert_eq!(content[2], json!({"type": "text", "text": "looking"}));
        assert_eq!(content[3]["type"], "tool_use");
    }
}

#[tokio::test]
async fn a_sealed_block_the_endpoint_rejects_is_a_classified_error_not_a_silent_drop() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "unknown block type"},
        })))
        .mount(&server)
        .await;
    let gateway = AnthropicMessagesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
    let frame = ContextFrame {
        blocks: vec![
            user_block(),
            anthropic_sealed(json!({"type": "server_tool_use", "id": "x", "name": "web_search"})),
        ],
    };
    let error = gateway
        .invoke(&request(frame, "claude-test"), &control())
        .await
        .unwrap_err();
    // The frame was sent (not dropped) and the provider's rejection is
    // classified from the status, never swallowed.
    assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn anthropic_beta_values_are_one_optional_comma_separated_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("done")))
        .mount(&server)
        .await;
    let plain = AnthropicMessagesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
    plain
        .invoke(&request(user_frame(), "claude-test"), &control())
        .await
        .unwrap();
    assert!(
        server.received_requests().await.unwrap()[0]
            .headers
            .get("anthropic-beta")
            .is_none(),
        "no beta header by default"
    );
    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("done")))
        .expect(1)
        .mount(&server)
        .await;
    let beta = AnthropicMessagesGateway::new(KEY)
        .with_base_url(format!("{}/v1", server.uri()))
        .with_beta("prompt-caching-2024-07-31")
        .with_beta("context-1m-2025-08-07");
    beta.invoke(&request(user_frame(), "claude-test"), &control())
        .await
        .unwrap();
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    // Both values ride one header, in the order the host listed them.
    assert_eq!(
        received[0]
            .headers
            .get("anthropic-beta")
            .and_then(|value| value.to_str().ok()),
        Some("prompt-caching-2024-07-31,context-1m-2025-08-07")
    );
}

#[tokio::test]
async fn anthropic_prefix_mismatch_behavior_renders_only_with_a_thinking_object() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("done")))
        .mount(&server)
        .await;
    let gateway = AnthropicMessagesGateway::new(KEY)
        .with_base_url(format!("{}/v1", server.uri()))
        .with_prefix_mismatch_behavior(Some(PrefixMismatchBehavior::DropBlock));
    // No effort: no thinking object, so the binding cannot render.
    gateway
        .invoke(&request(user_frame(), "claude-test"), &control())
        .await
        .unwrap();
    let body = only_request_body(&server).await;
    assert!(body.get("thinking").is_none());
    assert!(
        server.received_requests().await.unwrap()[0]
            .headers
            .get("anthropic-beta")
            .is_none(),
        "the option does not add the beta header"
    );
    server.reset().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(text_response("done")))
        .mount(&server)
        .await;
    let mut with_effort = request(user_frame(), "claude-test");
    with_effort.generation.reasoning_effort = Some(ReasoningEffort::High);
    gateway.invoke(&with_effort, &control()).await.unwrap();
    let body = only_request_body(&server).await;
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"], json!({"effort": "high"}));
    assert_eq!(
        body["thinking"]["block_binding"],
        json!({"prefix_mismatch_behavior": "drop_block"})
    );
}
/// A completed first round (no tool call): the frame of the next request.
fn assistant_frame(frame: &ContextFrame, output: &ModelOutput) -> ContextFrame {
    let ids: Vec<BlockId> = (0..output.response.block_count())
        .map(|_| next_id())
        .collect();
    let mut all = frame.blocks.clone();
    all.extend(
        output
            .response
            .to_blocks(output.stop_reason, &ids)
            .expect("conversion"),
    );
    ContextFrame { blocks: all }
}

#[tokio::test]
async fn responses_reasoning_tool_loop_replays_the_item_whole_with_defaults() {
    let output = json!([
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc-1",
            "summary": [{"type": "summary_text", "text": "hmm"}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "read",
            "arguments": "{\"path\":\"a\"}",
        },
    ]);
    let turn = json!({"status": "completed", "output": output});
    for streaming in [false, true] {
        let server = MockServer::start().await;
        let response = if streaming {
            ResponseTemplate::new(200).set_body_raw(responses_sse(&output), "text/event-stream")
        } else {
            ResponseTemplate::new(200).set_body_json(&turn)
        };
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(response)
            .mount(&server)
            .await;
        let gateway =
            OpenAiResponsesGateway::new(KEY).with_base_url(format!("{}/v1", server.uri()));
        let first = if streaming {
            let mut stream = gateway
                .stream(&request(user_frame(), "gpt-test"), &control())
                .await
                .unwrap();
            let mut terminal = None;
            while let Some(delta) = stream.next().await {
                if let StreamDelta::Done { final_output, .. } = delta {
                    terminal = Some(final_output);
                }
            }
            terminal.expect("the stream completed")
        } else {
            gateway
                .invoke(&request(user_frame(), "gpt-test"), &control())
                .await
                .unwrap()
        };
        assert_eq!(first.response.items.len(), 2, "streaming {streaming}");
        server.reset().await;
        let done = json!({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]});
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&done))
            .mount(&server)
            .await;
        let second = request(answered_frame(&user_frame(), &first), "gpt-test");
        gateway.invoke(&second, &control()).await.unwrap();
        let body = only_request_body(&server).await;
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["store"], json!(false));
        let reasoning = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("the reasoning item is replayed");
        assert_eq!(reasoning, &output[0], "streaming {streaming}");
    }
}

#[tokio::test]
async fn responses_options_change_exactly_their_own_field() {
    let output = json!([
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc-1",
            "summary": [{"type": "summary_text", "text": "hmm"}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "read",
            "arguments": "{\"path\":\"a\"}",
        },
    ]);
    let server = MockServer::start().await;
    let base = format!("{}/v1", server.uri());
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"status": "completed", "output": output})),
        )
        .mount(&server)
        .await;
    let gateway = OpenAiResponsesGateway::new(KEY).with_base_url(base.clone());
    let first = gateway
        .invoke(&request(user_frame(), "gpt-test"), &control())
        .await
        .unwrap();
    server.reset().await;
    let done = json!({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]});
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&done))
        .mount(&server)
        .await;
    let second = request(answered_frame(&user_frame(), &first), "gpt-test");
    let variants: Vec<Box<dyn ModelGateway>> = vec![
        Box::new(OpenAiResponsesGateway::new(KEY).with_base_url(base.clone())),
        Box::new(
            OpenAiResponsesGateway::new(KEY)
                .with_base_url(base.clone())
                .with_encrypted_reasoning(false),
        ),
        Box::new(
            OpenAiResponsesGateway::new(KEY)
                .with_base_url(base.clone())
                .with_store(None),
        ),
        Box::new(
            OpenAiResponsesGateway::new(KEY)
                .with_base_url(base.clone())
                .with_store(Some(true)),
        ),
    ];
    for variant in &variants {
        variant.invoke(&second, &control()).await.unwrap();
    }
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), variants.len());
    let bodies: Vec<Value> = received
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect();
    // Default: encrypted reasoning included, store off.
    assert_eq!(bodies[0]["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(bodies[0]["store"], json!(false));
    // Opt out of `include` only.
    assert!(bodies[1].get("include").is_none());
    assert_eq!(bodies[1]["store"], json!(false));
    // Omit `store` only.
    assert_eq!(bodies[2]["include"], json!(["reasoning.encrypted_content"]));
    assert!(bodies[2].get("store").is_none());
    // Set `store` only.
    assert_eq!(bodies[3]["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(bodies[3]["store"], json!(true));
    // No variant disturbs the replayed item.
    for body in &bodies {
        let reasoning = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "reasoning")
            .expect("the reasoning item is replayed");
        assert_eq!(reasoning, &output[0]);
    }
}

#[tokio::test]
async fn chat_replay_opt_out_removes_only_reasoning_content() {
    let server = MockServer::start().await;
    let base = format!("{}/v1", server.uri());
    let message = json!({
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "think",
        "x_extra": {"a": 1},
    });
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": message, "finish_reason": "stop"}],
        })))
        .mount(&server)
        .await;
    let first = OpenAiChatCompletionsGateway::new(KEY)
        .with_base_url(base.clone())
        .invoke(&request(user_frame(), "gpt-test"), &control())
        .await
        .unwrap();
    server.reset().await;
    let done = json!({"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]});
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&done))
        .mount(&server)
        .await;
    let second = request(assistant_frame(&user_frame(), &first), "gpt-test");
    OpenAiChatCompletionsGateway::new(KEY)
        .with_base_url(base.clone())
        .invoke(&second, &control())
        .await
        .unwrap();
    OpenAiChatCompletionsGateway::new(KEY)
        .with_base_url(base.clone())
        .with_reasoning_content_replay(false)
        .invoke(&second, &control())
        .await
        .unwrap();
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 2);
    let bodies: Vec<Value> = received
        .iter()
        .map(|request| serde_json::from_slice(&request.body).unwrap())
        .collect();
    let assistant = |body: &Value| {
        body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "assistant")
            .cloned()
            .expect("assistant message")
    };
    let replay = assistant(&bodies[0]);
    assert_eq!(replay["content"], "answer");
    assert_eq!(replay["reasoning_content"], "think");
    assert_eq!(replay["x_extra"], json!({"a": 1}));
    let opt_out = assistant(&bodies[1]);
    assert_eq!(opt_out["content"], "answer");
    assert!(opt_out.get("reasoning_content").is_none());
    assert_eq!(opt_out["x_extra"], json!({"a": 1}));
    // Exactly one key differs: every other sealed message field still replays.
    let mut expected = replay.clone();
    expected
        .as_object_mut()
        .unwrap()
        .remove("reasoning_content");
    assert_eq!(opt_out, expected);
}
