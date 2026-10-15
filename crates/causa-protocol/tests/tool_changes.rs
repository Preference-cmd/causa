//! Tool changes recorded in the frame render on all three protocols: by
//! default `tools` carries the current tool state; in place, `tools` stays at
//! the declaration across rounds while the change renders at its position.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, ContextBlock, ContextFrame,
    GenerationOptions, ModelInvokeErrorKind, ModelRef, Role, ToolCallPayload, ToolChange,
    ToolDefinition, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use causa_protocol::translation::anthropic::render_anthropic_messages;
use causa_protocol::translation::media::MediaSet;
use causa_protocol::translation::openai_chat::render_openai_chat_messages;
use causa_protocol::translation::openai_responses::render_openai_responses_input;
use causa_protocol::translation::render_options::{
    AnthropicRenderOptions, ChatRenderOptions, ResponsesRenderOptions,
};
use serde_json::{Value, json};

fn id(label: &str) -> BlockId {
    let mut bytes = [0u8; 16];
    for (index, byte) in label.bytes().enumerate().take(16) {
        bytes[index] = byte;
    }
    BlockId::new(uuid::Uuid::from_bytes(bytes))
}

fn tool(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition::new(name, description, json!({"type": "object"}))
}

fn change(label: &str, added: Vec<ToolDefinition>, removed: &[&str]) -> ContextBlock {
    ContextBlock::new(
        id(label),
        BlockContent::ToolChange(ToolChange::new(
            added,
            removed.iter().map(|name| (*name).to_owned()).collect(),
        )),
        BlockMeta::default(),
    )
}

fn text(label: &str, text: &str, source: Option<&str>) -> ContextBlock {
    let role = match source {
        Some("system") => Role::System,
        Some("assistant") | None => Role::Assistant,
        _ => Role::User,
    };
    let meta = source.map_or_else(BlockMeta::default, BlockMeta::from_source);
    ContextBlock::new(id(label), BlockContent::text(role, text), meta)
}

fn call(label: &str, name: &str) -> ContextBlock {
    ContextBlock::new(
        id(label),
        BlockContent::ToolCall(ToolCallPayload::new(name, json!({}))),
        BlockMeta::default().with_provider_call_id(label),
    )
}

fn result(label: &str, call_label: &str) -> ContextBlock {
    ContextBlock::new(
        id(label),
        BlockContent::ToolResult(ToolResultPayload::new(
            id(call_label),
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!("done")),
        )),
        BlockMeta::default(),
    )
}

fn frame(blocks: &[ContextBlock]) -> ContextFrame {
    ContextFrame {
        blocks: blocks.to_vec(),
    }
}

fn anthropic_options(native: bool) -> AnthropicRenderOptions {
    let mut options = AnthropicRenderOptions::default();
    options.native_tool_changes = native;
    options
}

fn responses_options(additional_tools: bool) -> ResponsesRenderOptions {
    let mut options = ResponsesRenderOptions::default();
    options.additional_tools = additional_tools;
    options
}

fn anthropic(blocks: &[ContextBlock], native: bool, cache: CacheDirective) -> Value {
    render_anthropic_messages(
        &frame(blocks),
        &MediaSet::new(),
        &GenerationOptions::default().with_max_tokens(256),
        &ModelRef::new("claude-test"),
        cache,
        &anthropic_options(native),
    )
    .unwrap()
}

fn anthropic_error(blocks: &[ContextBlock]) -> ModelInvokeErrorKind {
    render_anthropic_messages(
        &frame(blocks),
        &MediaSet::new(),
        &GenerationOptions::default().with_max_tokens(256),
        &ModelRef::new("claude-test"),
        CacheDirective::None,
        &anthropic_options(true),
    )
    .unwrap_err()
    .kind
}

fn responses(blocks: &[ContextBlock], additional_tools: bool) -> Value {
    render_openai_responses_input(
        &frame(blocks),
        &MediaSet::new(),
        &GenerationOptions::default(),
        &ModelRef::new("responses-test"),
        CacheDirective::None,
        &responses_options(additional_tools),
    )
    .unwrap()
}

fn chat(blocks: &[ContextBlock]) -> Value {
    render_openai_chat_messages(
        &frame(blocks),
        &MediaSet::new(),
        &GenerationOptions::default(),
        &ModelRef::new("chat-test"),
        CacheDirective::None,
        &ChatRenderOptions::default(),
    )
    .unwrap()
}

fn tool_names(tools: &Value) -> Vec<&str> {
    tools
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| {
            tool["name"]
                .as_str()
                .or_else(|| tool["function"]["name"].as_str())
                .unwrap()
        })
        .collect()
}

/// Three rounds of one conversation: tool `b` is added before round two and
/// `a` withdrawn before round three, each change appended after the user turn
/// as the runner appends it.
fn rounds() -> [Vec<ContextBlock>; 3] {
    let one = vec![
        change("decl", vec![tool("a", "A")], &[]),
        text("u1", "first", Some("user")),
    ];
    let mut two = one.clone();
    two.extend([
        text("a1", "answer one", None),
        text("u2", "second", Some("user")),
        change("add-b", vec![tool("b", "B")], &[]),
    ]);
    let mut three = two.clone();
    three.extend([
        call("c1", "b"),
        result("r1", "c1"),
        change("drop-a", vec![], &["a"]),
    ]);
    [one, two, three]
}

#[test]
fn by_default_every_protocol_sends_the_current_tool_state() {
    let [one, two, three] = rounds();
    for (blocks, expected) in [
        (&one, vec!["a"]),
        (&two, vec!["a", "b"]),
        (&three, vec!["b"]),
    ] {
        assert_eq!(
            tool_names(&anthropic(blocks, false, CacheDirective::None)["tools"]),
            expected
        );
        assert_eq!(tool_names(&responses(blocks, false)["tools"]), expected);
        assert_eq!(tool_names(&chat(blocks)["tools"]), expected);
    }
    // Nothing renders at a change's position.
    let body = anthropic(&three, false, CacheDirective::None);
    assert!(!body.to_string().contains("tool_addition"));
    let body = responses(&three, false);
    assert!(!body.to_string().contains("additional_tools"));
}

#[test]
fn anthropic_in_place_keeps_tools_equal_across_rounds() {
    let [one, two, three] = rounds();
    let bodies = [&one, &two, &three].map(|b| anthropic(b, true, CacheDirective::None));
    assert_eq!(
        tool_names(&bodies[0]["tools"]),
        ["a", "__causa_deferred_placeholder__"]
    );
    assert_eq!(bodies[0]["tools"][1]["defer_loading"], json!(true));
    assert_eq!(bodies[0]["tools"], bodies[1]["tools"]);
    assert_eq!(bodies[1]["tools"], bodies[2]["tools"]);

    let messages = bodies[2]["messages"].as_array().unwrap();
    // user, assistant, user + system(addition), assistant call,
    // user result + system(removal)
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(
        roles,
        [
            "user",
            "assistant",
            "user",
            "system",
            "assistant",
            "user",
            "system"
        ]
    );
    assert_eq!(
        messages[3]["content"],
        json!([{
            "type": "tool_addition",
            "tool": {"type": "tool_definition", "definition": {
                "name": "b", "description": "B", "input_schema": {"type": "object"},
            }},
        }])
    );
    assert_eq!(
        messages[6]["content"],
        json!([{"type": "tool_removal", "tool": {"type": "tool_reference", "name": "a"}}])
    );
    // Round three repeats round two's messages unchanged.
    assert_eq!(bodies[1]["messages"].as_array().unwrap()[..], messages[..4]);
}

#[test]
fn anthropic_redefinition_adds_without_removing() {
    let blocks = vec![
        change("decl", vec![tool("a", "A")], &[]),
        text("u1", "first", Some("user")),
        change("redefine", vec![tool("a", "A2")], &["a", "missing"]),
    ];
    let body = anthropic(&blocks, true, CacheDirective::None);
    let content = &body["messages"][1]["content"];
    assert_eq!(content.as_array().unwrap().len(), 1);
    assert_eq!(content[0]["type"], "tool_addition");
    assert_eq!(content[0]["tool"]["definition"]["description"], "A2");
}

#[test]
fn anthropic_in_place_falls_back_on_an_empty_declaration() {
    let blocks = vec![
        change("decl", vec![], &[]),
        text("u1", "first", Some("user")),
        change("add-b", vec![tool("b", "B")], &[]),
    ];
    let body = anthropic(&blocks, true, CacheDirective::None);
    assert_eq!(tool_names(&body["tools"]), ["b"]);
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
}

#[test]
fn anthropic_in_place_rejects_a_change_it_cannot_place() {
    // Before any conversation content: it would be the first message.
    let first = vec![
        change("decl", vec![tool("a", "A")], &[]),
        change("early", vec![tool("b", "B")], &[]),
        text("u1", "first", Some("user")),
    ];
    assert_eq!(
        anthropic_error(&first),
        ModelInvokeErrorKind::InvalidRequest
    );
    // Right after assistant content.
    let after_assistant = vec![
        change("decl", vec![tool("a", "A")], &[]),
        text("u1", "first", Some("user")),
        text("a1", "answer", None),
        change("late", vec![tool("b", "B")], &[]),
        text("u2", "second", Some("user")),
    ];
    assert_eq!(
        anthropic_error(&after_assistant),
        ModelInvokeErrorKind::InvalidRequest
    );
    // The same frames render by default.
    anthropic(&first, false, CacheDirective::None);
    anthropic(&after_assistant, false, CacheDirective::None);
}

#[test]
fn anthropic_cache_anchor_skips_a_trailing_change() {
    let [_, two, _] = rounds();
    let body = anthropic(&two, true, CacheDirective::StablePrefix);
    let messages = body["messages"].as_array().unwrap();
    assert!(messages[3]["content"][0].get("cache_control").is_none());
    assert_eq!(
        messages[2]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    // The anchor on `tools` stays on the last declared tool, before the
    // placeholder.
    assert!(body["tools"][0].get("cache_control").is_some());
    assert!(body["tools"][1].get("cache_control").is_none());
}

#[test]
fn responses_additional_tools_keep_tools_equal_across_rounds() {
    let [one, two, _] = rounds();
    let mut more = two.clone();
    more.extend([
        text("a2", "answer two", None),
        text("u3", "third", Some("user")),
        change("add-c", vec![tool("c", "C")], &[]),
    ]);
    let bodies = [&one, &two, &more].map(|b| responses(b, true));
    assert_eq!(tool_names(&bodies[0]["tools"]), ["a"]);
    assert_eq!(bodies[0]["tools"], bodies[1]["tools"]);
    assert_eq!(bodies[1]["tools"], bodies[2]["tools"]);

    let input = bodies[2]["input"].as_array().unwrap();
    assert_eq!(
        input[3],
        json!({
            "type": "additional_tools",
            "role": "developer",
            "tools": [{
                "type": "function",
                "name": "b",
                "description": "B",
                "parameters": {"type": "object"},
            }],
        })
    );
    assert_eq!(input.last().unwrap()["type"], "additional_tools");
    assert_eq!(input.last().unwrap()["tools"][0]["name"], "c");
}

#[test]
fn responses_additional_tools_fall_back_on_removal_or_redefinition() {
    let [_, _, three] = rounds();
    let body = responses(&three, true);
    assert_eq!(tool_names(&body["tools"]), ["b"]);
    assert!(!body.to_string().contains("additional_tools"));

    let redefined = vec![
        change("decl", vec![tool("a", "A")], &[]),
        text("u1", "first", Some("user")),
        change("redefine", vec![tool("a", "A2")], &[]),
    ];
    let body = responses(&redefined, true);
    assert_eq!(body["tools"][0]["description"], "A2");
    assert!(!body.to_string().contains("additional_tools"));
}

#[test]
fn a_change_inside_an_open_exchange_is_rejected_everywhere() {
    let blocks = vec![
        change("decl", vec![tool("a", "A")], &[]),
        text("u1", "first", Some("user")),
        call("c1", "a"),
        change("inside", vec![tool("b", "B")], &[]),
        result("r1", "c1"),
    ];
    let frame = frame(&blocks);
    let media = MediaSet::new();
    let kinds = [
        render_anthropic_messages(
            &frame,
            &media,
            &GenerationOptions::default().with_max_tokens(256),
            &ModelRef::new("m"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap_err()
        .kind,
        render_openai_chat_messages(
            &frame,
            &media,
            &GenerationOptions::default(),
            &ModelRef::new("m"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap_err()
        .kind,
        render_openai_responses_input(
            &frame,
            &media,
            &GenerationOptions::default(),
            &ModelRef::new("m"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        )
        .unwrap_err()
        .kind,
    ];
    for kind in kinds {
        assert_eq!(kind, ModelInvokeErrorKind::InvalidRequest);
    }
}

#[test]
fn a_change_adding_one_name_twice_is_rejected() {
    let blocks = vec![
        change("decl", vec![tool("a", "A"), tool("a", "A2")], &[]),
        text("u1", "first", Some("user")),
    ];
    let error = render_openai_chat_messages(
        &frame(&blocks),
        &MediaSet::new(),
        &GenerationOptions::default(),
        &ModelRef::new("m"),
        CacheDirective::None,
        &ChatRenderOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
}
