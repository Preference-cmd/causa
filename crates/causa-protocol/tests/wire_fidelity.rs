//! Wire fidelity: model output the host has not edited is rendered back
//! value-equal to what the provider returned, item for item and in order, on
//! the format it came from.
//!
//! Each fixture is a recorded-shape provider response. The round trip is
//! parse (batch, and streamed through `StreamDecoder`) -> `to_blocks` ->
//! append to a context after a user block -> render -> compare the assistant
//! items of the rendered body with the recorded ones. Items carry fields the
//! translators have never heard of (`x_unknown`), which must survive.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, ContentPart, ContextBlock, ContextFrame,
    GenerationOptions, ModelInvokeErrorKind, ModelOutput, ModelRef, SealedPart, StreamDelta,
    TextPayload, ToolOutput, ToolResultPayload, ToolResultStatus, ToolSurface, WireMeta,
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
use std::sync::atomic::{AtomicU64, Ordering};

const ALL: [Protocol; 3] = [
    Protocol::AnthropicMessages,
    Protocol::OpenAiChatCompletions,
    Protocol::OpenAiResponses,
];

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(
        u128::from(NEXT_ID.fetch_add(1, Ordering::Relaxed)) + 1_000_000,
    ))
}

fn user_block(text: &str) -> ContextBlock {
    ContextBlock::new(
        next_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
        BlockMeta::from_source("user"),
    )
}

fn parse(protocol: Protocol, body: &Value) -> ModelOutput {
    match protocol {
        Protocol::AnthropicMessages => parse_anthropic_response(body),
        Protocol::OpenAiChatCompletions => parse_openai_chat_response(body),
        Protocol::OpenAiResponses => parse_openai_responses_output(body),
    }
    .unwrap()
}

fn render(
    protocol: Protocol,
    frame: &ContextFrame,
) -> Result<Value, causa_kernel::ModelInvokeError> {
    let media = MediaSet::new();
    let surface = ToolSurface::empty();
    let generation = GenerationOptions::default().with_max_tokens(1024);
    let model = ModelRef::new("m");
    match protocol {
        Protocol::AnthropicMessages => render_anthropic_messages(
            frame,
            &media,
            &surface,
            &generation,
            &model,
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        ),
        Protocol::OpenAiChatCompletions => render_openai_chat_messages(
            frame,
            &media,
            &surface,
            &generation,
            &model,
            CacheDirective::None,
            &ChatRenderOptions::default(),
        ),
        Protocol::OpenAiResponses => render_openai_responses_input(
            frame,
            &media,
            &surface,
            &generation,
            &model,
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        ),
    }
}

/// Blocks for a model output, in item order.
fn blocks_of(output: &ModelOutput) -> Vec<ContextBlock> {
    let ids: Vec<BlockId> = (0..output.response.block_count())
        .map(|_| next_id())
        .collect();
    output
        .response
        .to_blocks(output.stop_reason, &ids)
        .expect("conversion")
}

/// A frame of one user block, the assistant blocks, and a result for every
/// tool call (the renderers reject unanswered calls).
fn frame_of(assistant: Vec<ContextBlock>) -> ContextFrame {
    let results: Vec<ContextBlock> = assistant
        .iter()
        .filter(|b| matches!(b.content(), BlockContent::ToolCall(_)))
        .map(|call| {
            ContextBlock::new(
                next_id(),
                BlockContent::ToolResult(ToolResultPayload::new(
                    call.id(),
                    ToolResultStatus::Succeeded,
                    ToolOutput::new(json!("ok")),
                )),
                BlockMeta::default(),
            )
        })
        .collect();
    let mut blocks = vec![user_block("question")];
    blocks.extend(assistant);
    blocks.extend(results);
    ContextFrame { blocks }
}

/// The assistant items of a rendered body, in the wire format's own terms.
fn assistant_items(protocol: Protocol, body: &Value) -> Vec<Value> {
    match protocol {
        Protocol::OpenAiResponses => body["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["role"] == "assistant" || item["type"] == "function_call")
            .cloned()
            .collect(),
        Protocol::OpenAiChatCompletions => body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["role"] == "assistant")
            .cloned()
            .collect(),
        Protocol::AnthropicMessages => body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["role"] == "assistant")
            .flat_map(|message| message["content"].as_array().unwrap().clone())
            .collect(),
    }
}

fn rendered_assistant(protocol: Protocol, output: &ModelOutput) -> Vec<Value> {
    let frame = frame_of(blocks_of(output));
    assistant_items(protocol, &render(protocol, &frame).unwrap())
}

fn decode(protocol: Protocol, events: &[Value], chunk: usize) -> ModelOutput {
    let bytes: Vec<u8> = {
        let mut bytes: Vec<u8> = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>()
            .into_bytes();
        if protocol == Protocol::OpenAiChatCompletions {
            bytes.extend_from_slice(b"data: [DONE]\n\n");
        }
        bytes
    };
    let mut decoder = StreamDecoder::new(protocol, StreamLimits::default()).unwrap();
    let mut done = None;
    for piece in bytes.chunks(chunk) {
        let mut input = piece;
        while let Some(delta) = decoder.decode(&mut input).unwrap() {
            if let StreamDelta::Done { final_output, .. } = delta {
                done = Some(final_output);
            }
        }
    }
    decoder.finish().unwrap();
    done.expect("stream terminal")
}

/// Split a string in two at a character boundary.
fn halves(text: &str) -> (String, String) {
    let at = text
        .char_indices()
        .nth(text.chars().count() / 2)
        .map_or(text.len(), |(i, _)| i);
    (text[..at].to_owned(), text[at..].to_owned())
}

// --- fixtures -------------------------------------------------------------

/// A Responses turn: reasoning, a commentary message (two text parts, an
/// annotation, an unknown field), a function call with whitespace and
/// non-alphabetical key order in `arguments`, and a final-answer message.
fn responses_turn() -> Value {
    json!({
        "id": "resp_1",
        "status": "completed",
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Oslo is further north."}],
                "status": "completed",
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "commentary",
                "x_unknown": {"a": 1},
                "content": [
                    {
                        "type": "output_text",
                        "text": "Oslo is further north. ",
                        "annotations": [{"type": "url_citation", "url": "https://example.test"}],
                    },
                    {"type": "output_text", "text": "Checking the weather.\n", "annotations": []},
                ],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{\"unit\": \"celsius\",\n  \"location\": \"Oslo\"}",
                "status": "completed",
                "x_unknown": {"a": 1},
            },
            {
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "It is 3 degrees.", "annotations": []}],
            },
        ],
    })
}

fn responses_assistant(body: &Value) -> Vec<Value> {
    body["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] != "reasoning")
        .cloned()
        .collect()
}

fn responses_events(response: &Value) -> Vec<Value> {
    let mut events = vec![json!({
        "type": "response.created",
        "response": {"id": "resp_1", "status": "in_progress"},
    })];
    for (i, item) in response["output"].as_array().unwrap().iter().enumerate() {
        let id = item["id"].clone();
        let mut skeleton = item.clone();
        match item["type"].as_str().unwrap() {
            "message" => {
                skeleton["content"] = json!([]);
                skeleton["status"] = json!("in_progress");
                events.push(json!({"type": "response.output_item.added", "output_index": i, "item": skeleton}));
                for (j, part) in item["content"].as_array().unwrap().iter().enumerate() {
                    let text = part["text"].as_str().unwrap();
                    let (a, b) = halves(text);
                    let at = |kind: &str| json!({"type": kind, "output_index": i, "item_id": id, "content_index": j});
                    let mut added = at("response.content_part.added");
                    added["part"] = json!({"type": "output_text", "text": ""});
                    events.push(added);
                    for piece in [a, b] {
                        let mut delta = at("response.output_text.delta");
                        delta["delta"] = json!(piece);
                        events.push(delta);
                    }
                    let mut done = at("response.output_text.done");
                    done["text"] = json!(text);
                    events.push(done);
                    let mut part_done = at("response.content_part.done");
                    part_done["part"] = part.clone();
                    events.push(part_done);
                }
            }
            "function_call" => {
                let raw = item["arguments"].as_str().unwrap();
                skeleton["arguments"] = json!("");
                skeleton["status"] = json!("in_progress");
                events.push(json!({"type": "response.output_item.added", "output_index": i, "item": skeleton}));
                let (a, b) = halves(raw);
                for piece in [a, b] {
                    events.push(json!({
                        "type": "response.function_call_arguments.delta",
                        "output_index": i, "item_id": id, "delta": piece,
                    }));
                }
                events.push(json!({
                    "type": "response.function_call_arguments.done",
                    "output_index": i, "item_id": id, "name": item["name"], "arguments": raw,
                }));
            }
            _ => {
                skeleton["summary"] = json!([]);
                events.push(json!({"type": "response.output_item.added", "output_index": i, "item": skeleton}));
            }
        }
        events.push(json!({"type": "response.output_item.done", "output_index": i, "item": item}));
    }
    events.push(json!({"type": "response.completed", "response": response}));
    events
}

/// A Chat message the stream can deliver in full: reasoning, content, and a
/// tool call whose arguments carry whitespace and non-alphabetical keys.
fn chat_message() -> Value {
    json!({
        "role": "assistant",
        "content": "Oslo is further north. Let me check.",
        "reasoning_content": "We need to compare latitudes.",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"unit\": \"celsius\",\n  \"location\": \"Oslo\"}",
            },
        }],
    })
}

fn chat_body(message: &Value) -> Value {
    json!({
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
    })
}

fn chat_events(message: &Value) -> Vec<Value> {
    let chunk = |delta: Value| json!({"id": "chat_1", "choices": [{"index": 0, "delta": delta}]});
    let mut events = vec![chunk(json!({"role": "assistant"}))];
    if let Some(reasoning) = message["reasoning_content"].as_str() {
        let (a, b) = halves(reasoning);
        events.push(chunk(json!({"reasoning_content": a})));
        events.push(chunk(json!({"reasoning_content": b})));
    }
    if let Some(content) = message["content"].as_str() {
        let (a, b) = halves(content);
        events.push(chunk(json!({"content": a})));
        events.push(chunk(json!({"content": b})));
    }
    for (i, call) in message["tool_calls"].as_array().unwrap().iter().enumerate() {
        let (a, b) = halves(call["function"]["arguments"].as_str().unwrap());
        events.push(chunk(json!({"tool_calls": [{
            "index": i, "id": call["id"], "type": "function",
            "function": {"name": call["function"]["name"], "arguments": a},
        }]})));
        events.push(chunk(
            json!({"tool_calls": [{"index": i, "function": {"arguments": b}}]}),
        ));
    }
    events.push(json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}));
    events
}

/// An Anthropic turn: thinking (sealed material, replayed verbatim), two text
/// blocks (one with a citation and an unknown field) and a `tool_use` with
/// extra fields.
fn anthropic_turn() -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "stop_reason": "tool_use",
        "content": [
            {"type": "thinking", "thinking": "Compare latitudes.", "signature": "sig"},
            {"type": "text", "text": "Oslo is further north.", "x_unknown": {"a": 1}},
            {
                "type": "text",
                "text": "Let me look up the weather.\n",
                "citations": [{"type": "char_location", "cited_text": "Oslo"}],
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"unit": "celsius", "location": "Oslo"},
                "caller": {"type": "direct"},
                "x_unknown": {"a": 1},
            },
        ],
    })
}

/// Every assistant content block round-trips now, `thinking` included: it is
/// sealed material written back verbatim, not an observation to drop.
fn anthropic_assistant(body: &Value) -> Vec<Value> {
    body["content"].as_array().unwrap().clone()
}

fn anthropic_events(response: &Value) -> Vec<Value> {
    let mut events = vec![json!({
        "type": "message_start",
        "message": {"content": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
    })];
    for (i, block) in response["content"].as_array().unwrap().iter().enumerate() {
        let delta =
            |delta: Value| json!({"type": "content_block_delta", "index": i, "delta": delta});
        match block["type"].as_str().unwrap() {
            "thinking" => {
                events.push(json!({"type": "content_block_start", "index": i,
                    "content_block": {"type": "thinking", "thinking": ""}}));
                events.push(delta(
                    json!({"type": "thinking_delta", "thinking": block["thinking"]}),
                ));
                events.push(delta(
                    json!({"type": "signature_delta", "signature": block["signature"]}),
                ));
            }
            "text" => {
                let mut start = block.clone();
                start["text"] = json!("");
                start.as_object_mut().unwrap().remove("citations");
                events.push(
                    json!({"type": "content_block_start", "index": i, "content_block": start}),
                );
                let (a, b) = halves(block["text"].as_str().unwrap());
                events.push(delta(json!({"type": "text_delta", "text": a})));
                for citation in block["citations"].as_array().into_iter().flatten() {
                    events.push(delta(
                        json!({"type": "citations_delta", "citation": citation}),
                    ));
                }
                events.push(delta(json!({"type": "text_delta", "text": b})));
            }
            _ => {
                let mut start = block.clone();
                start["input"] = json!({});
                events.push(
                    json!({"type": "content_block_start", "index": i, "content_block": start}),
                );
                let (a, b) = halves(&block["input"].to_string());
                events.push(delta(
                    json!({"type": "input_json_delta", "partial_json": a}),
                ));
                events.push(delta(
                    json!({"type": "input_json_delta", "partial_json": b}),
                ));
            }
        }
        events.push(json!({"type": "content_block_stop", "index": i}));
    }
    events.push(json!({
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"output_tokens": 5},
    }));
    events.push(json!({"type": "message_stop"}));
    events
}

// --- the round-trip property ------------------------------------------------

#[test]
fn responses_round_trip_batch_and_streaming() {
    let body = responses_turn();
    let expected = responses_assistant(&body);
    let batch = parse(Protocol::OpenAiResponses, &body);
    assert_eq!(
        rendered_assistant(Protocol::OpenAiResponses, &batch),
        expected
    );
    for chunk in [1, 5, 4096] {
        let streamed = decode(Protocol::OpenAiResponses, &responses_events(&body), chunk);
        assert_eq!(
            rendered_assistant(Protocol::OpenAiResponses, &streamed),
            expected,
            "chunk {chunk}"
        );
    }
    // item boundaries and order: message, call, message; `phase` and `id`
    // survive on both messages
    assert_eq!(expected[0]["phase"], "commentary");
    assert_eq!(expected[2]["phase"], "final_answer");
}

#[test]
fn responses_function_call_keeps_the_original_arguments_string() {
    let body = responses_turn();
    let rendered = rendered_assistant(
        Protocol::OpenAiResponses,
        &parse(Protocol::OpenAiResponses, &body),
    );
    let call = &rendered[1];
    assert_eq!(
        call["arguments"],
        "{\"unit\": \"celsius\",\n  \"location\": \"Oslo\"}"
    );
    assert_eq!(call["id"], "fc_1");
}

#[test]
fn chat_round_trip_batch_and_streaming() {
    let message = chat_message();
    let expected = vec![message.clone()];
    let batch = parse(Protocol::OpenAiChatCompletions, &chat_body(&message));
    assert_eq!(
        rendered_assistant(Protocol::OpenAiChatCompletions, &batch),
        expected
    );
    for chunk in [1, 7, 4096] {
        let streamed = decode(
            Protocol::OpenAiChatCompletions,
            &chat_events(&message),
            chunk,
        );
        assert_eq!(
            rendered_assistant(Protocol::OpenAiChatCompletions, &streamed),
            expected,
            "chunk {chunk}"
        );
    }
}

#[test]
fn chat_batch_keeps_unknown_message_and_call_fields() {
    let message = json!({
        "role": "assistant",
        "content": "checking",
        "reasoning_content": "think",
        "refusal": null,
        "x_unknown": {"a": 1},
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "index": 0,
            "x_unknown": {"a": 1},
            "function": {"name": "get_weather", "arguments": "{ \"b\": 1, \"a\": 2 }"},
        }],
    });
    let batch = parse(Protocol::OpenAiChatCompletions, &chat_body(&message));
    assert_eq!(
        rendered_assistant(Protocol::OpenAiChatCompletions, &batch),
        vec![message]
    );
}

#[test]
fn chat_null_content_round_trips_in_batch() {
    let message = json!({
        "role": "assistant",
        "content": null,
        "reasoning_content": "think",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"b\": 1,\n \"a\": 2}"},
        }],
    });
    let batch = parse(Protocol::OpenAiChatCompletions, &chat_body(&message));
    assert_eq!(
        rendered_assistant(Protocol::OpenAiChatCompletions, &batch),
        vec![message.clone()]
    );
    // A stream never delivers `content: null`; it is not invented, so the
    // streamed turn is the same message without the key.
    let mut streamed_message = message;
    streamed_message.as_object_mut().unwrap().remove("content");
    let streamed = decode(
        Protocol::OpenAiChatCompletions,
        &chat_events(&streamed_message),
        64,
    );
    assert_eq!(
        rendered_assistant(Protocol::OpenAiChatCompletions, &streamed),
        vec![streamed_message]
    );
}

#[test]
fn anthropic_round_trip_batch_and_streaming() {
    let body = anthropic_turn();
    let expected = anthropic_assistant(&body);
    let batch = parse(Protocol::AnthropicMessages, &body);
    assert_eq!(
        rendered_assistant(Protocol::AnthropicMessages, &batch),
        expected
    );
    for chunk in [1, 11, 4096] {
        let streamed = decode(Protocol::AnthropicMessages, &anthropic_events(&body), chunk);
        assert_eq!(
            rendered_assistant(Protocol::AnthropicMessages, &streamed),
            expected,
            "chunk {chunk}"
        );
    }
    // two text blocks stay two content blocks, no joined text
    assert_eq!(expected.iter().filter(|b| b["type"] == "text").count(), 2);
}

#[test]
fn batch_and_streaming_parse_to_the_same_items() {
    for (protocol, body, events) in [
        (
            Protocol::OpenAiResponses,
            responses_turn(),
            responses_events(&responses_turn()),
        ),
        (
            Protocol::OpenAiChatCompletions,
            chat_body(&chat_message()),
            chat_events(&chat_message()),
        ),
        (
            Protocol::AnthropicMessages,
            anthropic_turn(),
            anthropic_events(&anthropic_turn()),
        ),
    ] {
        let batch = serde_json::to_value(&parse(protocol, &body).response).unwrap();
        let streamed = serde_json::to_value(&decode(protocol, &events, 17).response).unwrap();
        assert_eq!(batch, streamed, "{protocol}");
    }
}

// --- other formats, edits and malformed frames --------------------------------

fn parsed_blocks() -> Vec<(Protocol, Vec<ContextBlock>)> {
    vec![
        (
            Protocol::OpenAiResponses,
            blocks_of(&parse(Protocol::OpenAiResponses, &responses_turn())),
        ),
        (
            Protocol::OpenAiChatCompletions,
            blocks_of(&parse(
                Protocol::OpenAiChatCompletions,
                &chat_body(&chat_message()),
            )),
        ),
        (
            Protocol::AnthropicMessages,
            blocks_of(&parse(Protocol::AnthropicMessages, &anthropic_turn())),
        ),
    ]
}

fn without_wire(block: &ContextBlock) -> ContextBlock {
    let mut meta = block.meta().clone();
    meta.wire = None;
    ContextBlock::new(block.id(), block.content().clone(), meta)
}

fn without_sealed(block: &ContextBlock) -> Option<ContextBlock> {
    match block.content() {
        BlockContent::Parts(parts) => {
            let kept: Vec<ContentPart> = parts
                .iter()
                .filter(|p| !matches!(p, ContentPart::Sealed(_)))
                .cloned()
                .collect();
            (!kept.is_empty()).then(|| {
                ContextBlock::new(block.id(), BlockContent::Parts(kept), block.meta().clone())
            })
        }
        _ => Some(block.clone()),
    }
}

#[test]
fn blocks_of_another_format_render_from_facts_without_error() {
    for (source, blocks) in parsed_blocks() {
        assert!(
            blocks.iter().any(|b| b.meta().wire.is_some()
                || matches!(b.content(), BlockContent::Parts(p) if p.iter().any(|p| matches!(p, ContentPart::Sealed(_))))),
            "{source} fixture carries wire material"
        );
        for target in ALL.into_iter().filter(|target| *target != source) {
            let with_wire = render(target, &frame_of(blocks.clone())).unwrap();
            let facts_only: Vec<ContextBlock> = blocks
                .iter()
                .filter_map(without_sealed)
                .map(|b| without_wire(&b))
                .collect();
            // the same frame shape with every wire field removed; block ids
            // differ only in the appended results, which render identically
            let expected = render(target, &frame_of(facts_only)).unwrap();
            assert_eq!(
                assistant_items(target, &with_wire),
                assistant_items(target, &expected),
                "{source} blocks rendered as {target}"
            );
        }
    }
}

fn reencode_call(block: &ContextBlock, arguments: Value) -> ContextBlock {
    let BlockContent::ToolCall(call) = block.content() else {
        panic!("tool call block expected");
    };
    let mut call = call.clone();
    call.arguments = arguments;
    // a changed block takes a new identity
    ContextBlock::new(
        next_id(),
        BlockContent::ToolCall(call),
        block.meta().clone(),
    )
}

#[test]
fn edited_arguments_render_the_reencoded_facts() {
    for protocol in [Protocol::OpenAiResponses, Protocol::OpenAiChatCompletions] {
        let (_, blocks) = parsed_blocks()
            .into_iter()
            .find(|(source, _)| *source == protocol)
            .unwrap();
        let edited_args = json!({"unit": "kelvin", "location": "Oslo"});
        let blocks: Vec<ContextBlock> = blocks
            .iter()
            .map(|b| {
                if matches!(b.content(), BlockContent::ToolCall(_)) {
                    reencode_call(b, edited_args.clone())
                } else {
                    b.clone()
                }
            })
            .collect();
        let rendered = assistant_items(protocol, &render(protocol, &frame_of(blocks)).unwrap());
        let raw = match protocol {
            Protocol::OpenAiResponses => rendered
                .iter()
                .find(|i| i["type"] == "function_call")
                .unwrap()["arguments"]
                .clone(),
            _ => rendered[0]["tool_calls"][0]["function"]["arguments"].clone(),
        };
        assert_eq!(raw, json!(edited_args.to_string()), "{protocol}");
    }
}

#[test]
fn reformatted_but_equal_arguments_keep_the_raw_string_and_fields() {
    // facts equal to the raw string's value: the raw string is sent
    let (_, blocks) = parsed_blocks()
        .into_iter()
        .find(|(source, _)| *source == Protocol::OpenAiResponses)
        .unwrap();
    let rendered = assistant_items(
        Protocol::OpenAiResponses,
        &render(Protocol::OpenAiResponses, &frame_of(blocks)).unwrap(),
    );
    assert_eq!(
        rendered[1]["arguments"],
        "{\"unit\": \"celsius\",\n  \"location\": \"Oslo\"}"
    );
}

#[test]
fn a_part_count_that_no_longer_matches_the_recorded_slots_renders_from_facts() {
    // Responses: the recorded message has two output_text slots; a block
    // holding one text part cannot fill them.
    let (_, blocks) = parsed_blocks()
        .into_iter()
        .find(|(source, _)| *source == Protocol::OpenAiResponses)
        .unwrap();
    // The reasoning item is now its own sealed block and comes first; find the
    // message block by shape, not by position.
    let message = blocks
        .iter()
        .find(|b| {
            matches!(
                b.content(),
                BlockContent::Parts(parts)
                    if parts.len() == 2
                        && parts.iter().all(|p| matches!(p, ContentPart::Text(_)))
            )
        })
        .expect("the fixture holds one two-part message block");
    let BlockContent::Parts(parts) = message.content() else {
        panic!("message block");
    };
    assert_eq!(parts.len(), 2);
    let edited = ContextBlock::new(
        next_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("edited"))]),
        message.meta().clone(),
    );
    let rendered = assistant_items(
        Protocol::OpenAiResponses,
        &render(Protocol::OpenAiResponses, &frame_of(vec![edited])).unwrap(),
    );
    assert_eq!(
        rendered,
        vec![json!({"role": "assistant", "content": [{"type": "output_text", "text": "edited"}]})]
    );

    // Anthropic: fields apply only to a block of exactly one text part.
    let two_parts = ContextBlock::new(
        next_id(),
        BlockContent::Parts(vec![
            ContentPart::Text(TextPayload::new("a")),
            ContentPart::Text(TextPayload::new("b")),
        ]),
        BlockMeta::default().with_wire(WireMeta::new(
            "anthropic_messages",
            json!({"type": "text", "x_unknown": 1}),
        )),
    );
    let rendered = assistant_items(
        Protocol::AnthropicMessages,
        &render(Protocol::AnthropicMessages, &frame_of(vec![two_parts])).unwrap(),
    );
    assert_eq!(
        rendered,
        vec![
            json!({"type": "text", "text": "a"}),
            json!({"type": "text", "text": "b"})
        ]
    );
}

fn sealed_block(source: Option<&str>, format: &str, item: Value) -> ContextBlock {
    ContextBlock::new(
        next_id(),
        BlockContent::Parts(vec![ContentPart::Sealed(SealedPart::new(format, item))]),
        match source {
            Some(s) => BlockMeta::default().with_source(s),
            None => BlockMeta::default(),
        },
    )
}

#[test]
fn sealed_part_outside_assistant_position_is_an_invalid_request() {
    for protocol in ALL {
        for source in ["user", "system", "inject:note"] {
            for format in [protocol.as_str(), "some_other_format"] {
                let frame = ContextFrame {
                    blocks: vec![
                        user_block("hi"),
                        sealed_block(Some(source), format, json!({"reasoning_content": "x"})),
                    ],
                };
                let error = render(protocol, &frame).unwrap_err();
                assert_eq!(
                    error.kind,
                    ModelInvokeErrorKind::InvalidRequest,
                    "{protocol} {source} {format}"
                );
            }
        }
    }
}

#[test]
fn sealed_part_of_another_format_in_assistant_position_is_skipped() {
    for protocol in ALL {
        let other = ALL.into_iter().find(|p| *p != protocol).unwrap();
        for source in [None, Some("assistant")] {
            let with_sealed = ContextFrame {
                blocks: vec![
                    user_block("hi"),
                    sealed_block(source, other.as_str(), json!({"reasoning_content": "x"})),
                ],
            };
            let without = ContextFrame {
                blocks: vec![user_block("hi")],
            };
            assert_eq!(
                render(protocol, &with_sealed).unwrap(),
                render(protocol, &without).unwrap(),
                "{protocol} skips {other} sealed items"
            );
        }
    }
}

fn text_block(source: Option<&str>, text: &str) -> ContextBlock {
    ContextBlock::new(
        next_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
        match source {
            Some(s) => BlockMeta::default().with_source(s),
            None => BlockMeta::default(),
        },
    )
}

#[test]
fn chat_sealed_object_seeds_the_message_and_the_first_one_wins() {
    let chat = Protocol::OpenAiChatCompletions;
    let frame = ContextFrame {
        blocks: vec![
            user_block("hi"),
            sealed_block(
                None,
                chat.as_str(),
                json!({"content": null, "reasoning_content": "first"}),
            ),
            text_block(None, "a"),
            sealed_block(
                None,
                chat.as_str(),
                json!({"reasoning_content": "second", "other": 1}),
            ),
            text_block(None, "b"),
        ],
    };
    let rendered = assistant_items(chat, &render(chat, &frame).unwrap());
    assert_eq!(
        rendered,
        vec![json!({"role": "assistant", "content": "ab", "reasoning_content": "first"})]
    );
}

#[test]
fn chat_adjacent_assistant_text_blocks_concatenate_without_a_separator() {
    let chat = Protocol::OpenAiChatCompletions;
    let frame = ContextFrame {
        blocks: vec![
            user_block("hi"),
            text_block(None, "one "),
            text_block(Some("assistant"), "two"),
        ],
    };
    let rendered = assistant_items(chat, &render(chat, &frame).unwrap());
    assert_eq!(
        rendered,
        vec![json!({"role": "assistant", "content": "one two"})]
    );
}

#[test]
fn no_renderer_joins_text_across_block_seams() {
    for protocol in ALL {
        let frame = ContextFrame {
            blocks: vec![text_block(Some("user"), "x"), text_block(Some("user"), "y")],
        };
        let rendered = serde_json::to_string(&render(protocol, &frame).unwrap()).unwrap();
        assert!(!rendered.contains("x\\ny"), "{protocol}");
        assert!(
            rendered.contains("\"x\"") && rendered.contains("\"y\""),
            "{protocol}"
        );
    }
}
