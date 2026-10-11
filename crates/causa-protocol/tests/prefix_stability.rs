//! Prefix stability: the framework must not change an Anthropic request
//! prefix between two rounds of one run. Covers the two framework-made
//! changes this proposal fixes (system hoisting, whitespace-only assistant
//! text) and the cache anchor rules that go with them.
//!
//! The media half of the same property is asserted here too: every request
//! re-resolves every reference, so a context that is not edited must render
//! the same value for an earlier message on every round.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, ContentPart, ContextBlock, ContextFrame,
    GenerationOptions, MediaRef, ModelRef, SealedPart, TextPayload, ToolCallPayload,
    ToolDefinition, ToolSurface,
};
use causa_protocol::translation::anthropic::render_anthropic_messages;
use causa_protocol::translation::media::{MediaPayload, MediaSet};
use causa_protocol::translation::openai_chat::render_openai_chat_messages;
use causa_protocol::translation::openai_responses::render_openai_responses_input;
use causa_protocol::translation::render_options::{
    AnthropicRenderOptions, ChatRenderOptions, ResponsesRenderOptions,
};
use serde_json::{Value, json};

fn id(label: &str) -> BlockId {
    let mut value = 0_u128;
    for byte in label.bytes() {
        value = value.wrapping_mul(257).wrapping_add(u128::from(byte));
    }
    BlockId::new(uuid::Uuid::from_u128(value))
}

fn block(label: &str, content: BlockContent, source: Option<&str>) -> ContextBlock {
    ContextBlock::new(
        id(label),
        content,
        BlockMeta {
            source: source.map(String::from),
            ..BlockMeta::default()
        },
    )
}

fn text(label: &str, text: &str, source: Option<&str>) -> ContextBlock {
    block(
        label,
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
        source,
    )
}

fn call(label: &str, call_id: &str, name: &str) -> ContextBlock {
    ContextBlock::new(
        id(label),
        BlockContent::ToolCall(ToolCallPayload {
            tool_name: name.into(),
            arguments: json!({"path": "a"}),
        }),
        BlockMeta {
            provider_call_id: Some(call_id.into()),
            ..BlockMeta::default()
        },
    )
}

fn frame(blocks: Vec<ContextBlock>) -> ContextFrame {
    ContextFrame { blocks }
}

/// Anthropic requires `max_tokens`; the renderer tests always supply one.
fn anthropic_generation() -> GenerationOptions {
    GenerationOptions {
        max_tokens: Some(4096),
        ..GenerationOptions::default()
    }
}

fn anthropic(frame: &ContextFrame, cache: CacheDirective) -> Value {
    render_anthropic_messages(
        frame,
        &MediaSet::new(),
        &ToolSurface::empty(),
        &anthropic_generation(),
        &ModelRef::new("claude-test"),
        cache,
        &AnthropicRenderOptions::default(),
    )
    .unwrap()
}

/// The same render with a tool listing, so the \`tools\` clause of the
/// stability assertion is not comparing two nulls.
fn anthropic_with_tools(
    frame: &ContextFrame,
    cache: CacheDirective,
    surface: &ToolSurface,
) -> Value {
    render_anthropic_messages(
        frame,
        &MediaSet::new(),
        surface,
        &anthropic_generation(),
        &ModelRef::new("claude-test"),
        cache,
        &AnthropicRenderOptions::default(),
    )
    .unwrap()
}

fn chat(frame: &ContextFrame) -> Value {
    render_openai_chat_messages(
        frame,
        &MediaSet::new(),
        &ToolSurface::empty(),
        &GenerationOptions::default(),
        &ModelRef::new("chat-test"),
        CacheDirective::None,
        &ChatRenderOptions::default(),
    )
    .unwrap()
}

fn responses(frame: &ContextFrame) -> Value {
    render_openai_responses_input(
        frame,
        &MediaSet::new(),
        &ToolSurface::empty(),
        &GenerationOptions::default(),
        &ModelRef::new("responses-test"),
        CacheDirective::None,
        &ResponsesRenderOptions::default(),
    )
    .unwrap()
}

/// A system block before all conversation content, a user text, an assistant
/// whitespace-only text, and a normal assistant text.
fn prefix_frame() -> ContextFrame {
    frame(vec![
        text("s0", "be terse", Some("system")),
        text("u0", "hello", Some("user")),
        text("a-blank", "   \n  ", None),
        text("a0", "answer", None),
    ])
}

#[test]
fn whitespace_only_assistant_text_is_omitted_by_anthropic_and_chat_only() {
    let f = prefix_frame();

    let a = anthropic(&f, CacheDirective::None);
    let assistant_blocks: Vec<&Value> = a["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "assistant")
        .flat_map(|m| m["content"].as_array().unwrap().iter())
        .collect();
    assert!(
        assistant_blocks
            .iter()
            .all(|b| b["text"] != json!("   \n  ")),
        "Anthropic omits the blank part: {assistant_blocks:?}"
    );
    assert!(
        assistant_blocks
            .iter()
            .any(|b| b["text"] == json!("answer")),
        "every other part of the turn stays"
    );

    let c = chat(&f);
    let messages = c["messages"].as_array().unwrap();
    assert!(
        messages.iter().all(|m| m["content"] != json!("   \n  ")),
        "Chat omits the blank part"
    );
    assert!(messages.iter().any(|m| m["content"] == json!("answer")));

    // Responses keeps it, as in `pi-ai`.
    let r = responses(&f);
    let items = r["input"].as_array().unwrap();
    assert!(
        items
            .iter()
            .any(|i| i["content"][0]["text"] == json!("   \n  ")),
        "Responses keeps the blank part: {items:?}"
    );
}

#[test]
fn a_chat_turn_left_with_only_sealed_residue_renders_no_assistant_message() {
    // The message-level keys an endpoint returns beside whitespace-only
    // content are kept as a sealed item. With the text omitted, that residue
    // alone would render an assistant message with neither `content` nor
    // `tool_calls`, so the message is omitted as well.
    let f = frame(vec![
        text("u0", "hello", Some("user")),
        block(
            "a-residue",
            BlockContent::Parts(vec![ContentPart::Sealed(SealedPart {
                format: "openai_chat_completions".into(),
                item: json!({"refusal": null, "annotations": []}),
            })]),
            None,
        ),
        text("a-blank", "  \n", None),
        text("u1", "again", Some("user")),
    ]);
    let c = chat(&f);
    assert_eq!(
        c["messages"],
        json!([
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "again"},
        ])
    );

    // A sealed `content` the endpoint sent itself still renders its message.
    let kept = frame(vec![
        text("u0", "hello", Some("user")),
        block(
            "a-residue",
            BlockContent::Parts(vec![ContentPart::Sealed(SealedPart {
                format: "openai_chat_completions".into(),
                item: json!({"content": null, "reasoning_content": "thought"}),
            })]),
            None,
        ),
        text("u1", "again", Some("user")),
    ]);
    assert_eq!(
        chat(&kept)["messages"][1],
        json!({"role": "assistant", "content": null, "reasoning_content": "thought"})
    );
}

#[test]
fn top_level_system_is_the_array_form_with_and_without_a_cache_directive() {
    let f = prefix_frame();
    let plain = anthropic(&f, CacheDirective::None);
    assert_eq!(
        plain["system"],
        json!([{"type": "text", "text": "be terse"}]),
        "one block per leading system part, no join"
    );
    let cached = anthropic(&f, CacheDirective::StablePrefix);
    assert_eq!(
        cached["system"],
        json!([{
            "type": "text",
            "text": "be terse",
            "cache_control": {"type": "ephemeral"},
        }]),
        "the same shape, with the marker on the last block"
    );
    assert_eq!(plain["system"][0]["text"], cached["system"][0]["text"]);
}

#[test]
fn several_leading_system_parts_render_as_separate_blocks_without_a_join() {
    let f = frame(vec![
        text("s0", "be terse", Some("system")),
        text("s1", "cite sources", Some("system")),
        text("u0", "hello", Some("user")),
    ]);
    let v = anthropic(&f, CacheDirective::None);
    assert_eq!(
        v["system"],
        json!([
            {"type": "text", "text": "be terse"},
            {"type": "text", "text": "cite sources"},
        ]),
        "one block per leading system part; a join would collapse them into one"
    );
    // Responses keeps the listed exception: one \`instructions\` string.
    let r = responses(&f);
    assert_eq!(
        r["instructions"],
        json!(
            "be terse
cite sources"
        )
    );
}

#[test]
fn an_appended_system_block_leaves_the_top_level_system_unchanged() {
    let before = frame(vec![
        text("s0", "be terse", Some("system")),
        text("u0", "hello", Some("user")),
    ]);
    let after = frame(vec![
        text("s0", "be terse", Some("system")),
        text("u0", "hello", Some("user")),
        text("s1", "second instruction", Some("system")),
    ]);

    let before_a = anthropic(&before, CacheDirective::None);
    let after_a = anthropic(&after, CacheDirective::None);
    assert_eq!(
        before_a["system"], after_a["system"],
        "an append never rewrites the top-level system parameter"
    );
    let messages = after_a["messages"].as_array().unwrap();
    assert_eq!(
        messages.last().unwrap()["role"],
        json!("system"),
        "the appended block renders at its position"
    );
    assert_eq!(
        messages.last().unwrap()["content"],
        json!([{"type": "text", "text": "second instruction"}])
    );

    // Responses: leading system text stays in instructions; a later block
    // becomes an input item.
    let after_r = responses(&after);
    assert_eq!(after_r["instructions"], json!("be terse"));
    let items = after_r["input"].as_array().unwrap();
    assert_eq!(
        items.last().unwrap(),
        &json!({
            "role": "system",
            "content": [{"type": "input_text", "text": "second instruction"}],
        })
    );
}

#[test]
fn consecutive_renders_of_one_tool_loop_share_their_prefix() {
    // Round one: the system prefix, the user request, and the completed read
    // exchange. Round two appends the assistant answer and the next question.
    let round_one = frame(vec![
        text("s0", "be terse", Some("system")),
        text("u0", "read the file", Some("user")),
        call("c0", "toolu_1", "read"),
        block(
            "r0",
            BlockContent::ToolResult(causa_kernel::ToolResultPayload {
                call_block_id: id("c0"),
                status: causa_kernel::ToolResultStatus::Succeeded,
                output: causa_kernel::ToolOutput::new(json!("file-a")),
                media: Vec::new(),
                notes: Vec::new(),
            }),
            None,
        ),
        text("a0", "the file says a", None),
        text("u1", "and now?", Some("user")),
    ]);
    // Round two closes the exchange: the second read is declared and answered,
    // and the messages of round one are still at the front of the request.
    let mut blocks = round_one.blocks.clone();
    blocks.push(call("c1", "toolu_2", "read"));
    blocks.push(block(
        "r1",
        BlockContent::ToolResult(causa_kernel::ToolResultPayload {
            call_block_id: id("c1"),
            status: causa_kernel::ToolResultStatus::Succeeded,
            output: causa_kernel::ToolOutput::new(json!("file-b")),
            media: Vec::new(),
            notes: Vec::new(),
        }),
        None,
    ));
    blocks.push(text("a1", "the second file says b", None));
    let round_two = frame(blocks);

    let surface = ToolSurface::from_definitions(vec![ToolDefinition {
        name: "read".into(),
        description: "read a file".into(),
        parameters: json!({"type": "object"}),
    }]);
    let first = anthropic_with_tools(&round_one, CacheDirective::StablePrefix, &surface);
    let second = anthropic_with_tools(&round_two, CacheDirective::StablePrefix, &surface);

    assert_eq!(
        first["system"], second["system"],
        "the system value is stable across rounds"
    );
    assert!(
        first["tools"]
            .as_array()
            .is_some_and(|tools| !tools.is_empty()),
        "the listing must be non-empty or this assertion proves nothing"
    );
    assert_eq!(
        first["tools"], second["tools"],
        "an unchanged tool listing renders the same value"
    );
    // Every message the first round sent is still there, unchanged except that
    // the cache marker may move to a later message.
    let first_messages = first["messages"].as_array().unwrap();
    let second_messages = second["messages"].as_array().unwrap();
    assert!(second_messages.len() >= first_messages.len());
    for (index, message) in first_messages.iter().enumerate() {
        let mut a = message.clone();
        let mut b = second_messages[index].clone();
        for value in [&mut a, &mut b] {
            strip_cache_control(value);
        }
        assert_eq!(a, b, "message {index} changed between rounds");
    }
    // Rendering is deterministic: same context, same bytes.
    assert_eq!(
        anthropic_with_tools(&round_two, CacheDirective::StablePrefix, &surface),
        second
    );
    // The marker lands on a message block, never on a sealed item.
    let anchored = second_messages
        .iter()
        .filter_map(|message| message["content"].as_array())
        .flatten()
        .filter(|block| block.get("cache_control").is_some())
        .collect::<Vec<_>>();
    assert_eq!(anchored.len(), 1, "exactly one message marker");
}

#[test]
fn the_stable_prefix_anchor_skips_a_sealed_item_and_moves_backward() {
    // The assistant turn is one sealed thinking item: written verbatim, so the
    // marker may not be written into it, and the anchor has to move to an
    // earlier message.
    let f = frame(vec![
        text("u0", "look", Some("user")),
        block(
            "a0",
            BlockContent::Parts(vec![ContentPart::Sealed(SealedPart {
                format: "anthropic_messages".into(),
                item: json!({"type": "thinking", "thinking": "hmm", "signature": "sig"}),
            })]),
            None,
        ),
    ]);
    let v = anthropic(&f, CacheDirective::StablePrefix);
    let messages = v["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    // Verbatim: the sealed item is exactly what the parser stored.
    assert_eq!(
        messages[1]["content"][0],
        json!({"type": "thinking", "thinking": "hmm", "signature": "sig"}),
        "the sealed item is written verbatim"
    );
    assert!(
        messages[1]["content"][0].get("cache_control").is_none(),
        "no cache marker may be written into a sealed item"
    );
    assert!(
        messages[0]["content"][0].get("cache_control").is_some(),
        "the anchor moves backward to the last non-sealed block"
    );
    // Without the directive nothing is marked and the item is still verbatim.
    let plain = anthropic(&f, CacheDirective::None);
    assert_eq!(
        plain["messages"][1]["content"][0],
        messages[1]["content"][0]
    );
}

fn strip_cache_control(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("cache_control");
            for child in map.values_mut() {
                strip_cache_control(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_cache_control(item);
            }
        }
        _ => {}
    }
}

#[test]
fn an_unedited_media_reference_renders_the_same_value_on_every_round() {
    let mut media = MediaSet::new();
    media.insert("asset-1", MediaPayload::new("image/png", "AAAA"));
    let f = frame(vec![
        block(
            "u0",
            BlockContent::Parts(vec![
                ContentPart::Text(TextPayload::new("look")),
                ContentPart::Media(MediaRef::new("image/png", "asset-1")),
            ]),
            Some("user"),
        ),
        text("u1", "and now?", Some("user")),
    ]);
    let render = |f: &ContextFrame| {
        render_anthropic_messages(
            f,
            &media,
            &ToolSurface::empty(),
            &anthropic_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::StablePrefix,
            &AnthropicRenderOptions::default(),
        )
        .unwrap()
    };
    assert_eq!(render(&f), render(&f));
    assert_eq!(
        render(&f)["messages"][0]["content"][1]["type"],
        json!("image"),
        "the served reference renders as an image"
    );

    // A degraded reference is a value too, and it is stable for the same
    // reason: every request re-resolves from the same table. Two ways to
    // degrade: an unserved reference, and a payload whose type is not an image
    // (the walk inlines every `image/*`; the host pre-filters the rest).
    for set in [MediaSet::new(), {
        let mut unsupported = MediaSet::new();
        unsupported.insert("asset-1", MediaPayload::new("audio/mpeg", "AAAA"));
        unsupported
    }] {
        let degraded = |f: &ContextFrame| {
            render_anthropic_messages(
                f,
                &set,
                &ToolSurface::empty(),
                &anthropic_generation(),
                &ModelRef::new("claude-test"),
                CacheDirective::StablePrefix,
                &AnthropicRenderOptions::default(),
            )
            .unwrap()
        };
        let rendered = degraded(&f);
        assert_eq!(
            rendered,
            degraded(&f),
            "a degraded value is stable across renders"
        );
        assert_eq!(
            rendered["messages"][0]["content"][1],
            json!({"type": "text", "text": "[media: image/png asset-1]"}),
            "the reference degrades to the deterministic placeholder"
        );
    }
}
