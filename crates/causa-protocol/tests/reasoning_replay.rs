//! Reasoning material, translation level: parsers keep provider items whole as
//! sealed items in place and each renderer writes its own format back verbatim
//! (with the Chat replay opt-out removing only `reasoning_content`).

use causa_kernel::{
    BlockId, CacheDirective, ContentPart, ContextBlock, ContextFrame, GenerationOptions,
    ModelOutput, ModelRef, Role, StreamDelta,
};
use causa_protocol::Protocol;
use causa_protocol::translation::anthropic::{parse_anthropic_response, render_anthropic_messages};
use causa_protocol::translation::media::MediaSet;
use causa_protocol::translation::openai_chat::{
    parse_openai_chat_response, render_openai_chat_messages,
};
use causa_protocol::translation::openai_responses::{
    parse_openai_responses_output, render_openai_responses_input,
};
use causa_protocol::translation::render_options::{
    AnthropicRenderOptions, ChatRenderOptions, ResponsesRenderOptions,
};
use causa_protocol::translation::streaming::{StreamDecoder, StreamLimits};
use serde_json::{Value, json};

fn user_block() -> ContextBlock {
    ContextBlock::text(BlockId::new(uuid::Uuid::from_u128(1)), Role::User, "hi")
}

/// The frame of the next turn for a parsed output: the user block, the
/// assistant blocks in item order, and no tool result (the fixtures end in
/// text).
fn frame_of(output: &ModelOutput) -> ContextFrame {
    let ids: Vec<BlockId> = (0..output.response.block_count())
        .map(|n| BlockId::new(uuid::Uuid::from_u128(100 + n as u128)))
        .collect();
    let mut blocks = vec![user_block()];
    blocks.extend(
        output
            .response
            .to_blocks(output.stop_reason, &ids)
            .expect("conversion"),
    );
    ContextFrame { blocks }
}

fn generation() -> GenerationOptions {
    GenerationOptions::default().with_max_tokens(1024)
}

fn anthropic(frame: &ContextFrame) -> Value {
    render_anthropic_messages(
        frame,
        &MediaSet::new(),
        &generation(),
        &ModelRef::new("m"),
        CacheDirective::None,
        &AnthropicRenderOptions::default(),
    )
    .unwrap()
}

fn responses(frame: &ContextFrame) -> Value {
    render_openai_responses_input(
        frame,
        &MediaSet::new(),
        &generation(),
        &ModelRef::new("m"),
        CacheDirective::None,
        &ResponsesRenderOptions::default(),
    )
    .unwrap()
}

fn chat(frame: &ContextFrame, options: &ChatRenderOptions) -> Value {
    render_openai_chat_messages(
        frame,
        &MediaSet::new(),
        &generation(),
        &ModelRef::new("m"),
        CacheDirective::None,
        options,
    )
    .unwrap()
}

fn sealed_items(output: &ModelOutput) -> Vec<Value> {
    output
        .response
        .items
        .iter()
        .flat_map(|item| match &item.content {
            causa_kernel::ModelItemContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Sealed(sealed) => Some(sealed.item.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn assistant_message(body: &Value, protocol: Protocol) -> &Value {
    match protocol {
        Protocol::AnthropicMessages => body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "assistant")
            .expect("assistant message"),
        Protocol::OpenAiChatCompletions => body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "assistant")
            .expect("assistant message"),
        Protocol::OpenAiResponses => body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["role"] == "assistant")
            .expect("assistant message"),
    }
}

#[test]
fn anthropic_unknown_and_redacted_blocks_round_trip_verbatim() {
    let content = json!([
        {"type": "redacted_thinking", "data": "enc"},
        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"q": "x"}},
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "answer"},
    ]);
    let body = json!({"content": content, "stop_reason": "end_turn"});
    let out = parse_anthropic_response(&body).unwrap();
    // One item per content block, in order: nothing is skipped and nothing is
    // joined into the reasoning observation.
    assert_eq!(out.response.items.len(), 4);
    assert_eq!(
        sealed_items(&out),
        vec![content[0].clone(), content[1].clone(), content[2].clone()]
    );
    assert_eq!(out.reasoning.as_ref().unwrap().text, "hmm");
    let rendered = anthropic(&frame_of(&out));
    assert_eq!(
        assistant_message(&rendered, Protocol::AnthropicMessages)["content"],
        content
    );
}

#[test]
fn responses_unknown_items_round_trip_verbatim() {
    let output = json!([
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc",
            "summary": [{"type": "summary_text", "text": "hmm"}],
        },
        {"type": "web_search_call", "id": "ws_1", "status": "completed"},
        {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
    ]);
    let body = json!({"status": "completed", "output": output});
    let out = parse_openai_responses_output(&body).unwrap();
    assert_eq!(out.response.items.len(), 3);
    assert_eq!(
        sealed_items(&out),
        vec![output[0].clone(), output[1].clone()]
    );
    assert_eq!(out.reasoning.as_ref().unwrap().text, "hmm");
    let rendered = responses(&frame_of(&out));
    // `input[0]` is the user turn; the assistant items follow it in order.
    let input = &rendered["input"].as_array().unwrap()[1..];
    // The reasoning item and the unknown item replay whole and in place; the
    // message renders from its facts (equal to the recorded item here).
    assert_eq!(input[0], output[0]);
    assert_eq!(input[1], output[1]);
    assert_eq!(input[2]["content"], output[2]["content"]);
}

#[test]
fn chat_reasoning_content_replays_by_default_and_can_be_opted_out() {
    let message = json!({
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "think",
    });
    let body = json!({"choices": [{"message": message, "finish_reason": "stop"}]});
    let out = parse_openai_chat_response(&body).unwrap();
    assert_eq!(
        sealed_items(&out),
        vec![json!({"reasoning_content": "think"})]
    );
    let frame = frame_of(&out);
    let rendered = chat(&frame, &ChatRenderOptions::default());
    let assistant = assistant_message(&rendered, Protocol::OpenAiChatCompletions);
    assert_eq!(assistant["content"], "answer");
    assert_eq!(assistant["reasoning_content"], "think");

    let opted_out = chat(&frame, &{
        let mut options = ChatRenderOptions::default();
        options.replay_reasoning_content = false;
        options
    });
    let assistant = assistant_message(&opted_out, Protocol::OpenAiChatCompletions);
    assert_eq!(assistant["content"], "answer");
    assert!(assistant.get("reasoning_content").is_none());
    // The streamed decoder finalizes through the batch parser, so the same
    // frame comes out of an SSE turn.
    let bytes = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".to_vec();
    let mut decoder =
        StreamDecoder::new(Protocol::OpenAiChatCompletions, StreamLimits::default()).unwrap();
    let mut terminal = None;
    let mut input: &[u8] = &bytes;
    while let Some(delta) = decoder.decode(&mut input).unwrap() {
        if let StreamDelta::Done { final_output, .. } = delta {
            terminal = Some(final_output);
        }
    }
    decoder.finish().unwrap();
    let streamed = terminal.expect("terminal");
    assert_eq!(
        sealed_items(&streamed),
        vec![json!({"reasoning_content": "think"})]
    );
}
