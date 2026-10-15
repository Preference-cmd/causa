//! Anthropic Messages translation for the context kernel.
//!
//! Pure functions from [`causa_kernel::ContextFrame`] to an Anthropic
//! Messages request body, and from an Anthropic Messages response body
//! back to [`causa_kernel::ModelOutput`]. Transport-free — the reqwest
//! adapter in `causa-provider` owns HTTP.
//!
//! The renderer-independent policy (role mapping, empty-text skip,
//! no cross-block text join, tool id pairing, observation stringification) lives in
//! `super::context_frame`; this module only shapes the Anthropic wire.
//!
//! # Anthropic-specific structural rules
//!
//! - The leading run of `system` text segments moves to the top-level
//!   `system` parameter, always as a block array with no join; a `system`
//!   block after conversation content renders at its position as a
//!   `system` message, an append like any other. Media in a leading
//!   system-role block degrades to its placeholder text in the parameter —
//!   the parameter carries no image blocks.
//! - Consecutive segments with the same wire role merge into one
//!   message; Anthropic requires strictly alternating `user` /
//!   `assistant` roles.
//! - Resolved `image/*` media parts render as `image` content blocks
//!   (`base64` source) inside their role's message; unresolved or
//!   non-image media degrades to the shared placeholder text.
//! - Tool calls render as assistant `tool_use` content blocks with
//!   `input` as a JSON object; tool results render as `tool_result`
//!   content blocks in the following user message. Result media embeds
//!   natively: with attachments the `tool_result` `content` becomes a
//!   block array (text, then the images); without, it stays a string.
//!   Any status other than `Succeeded` sets `is_error: true` (the flag
//!   is Anthropic-only; OpenAI-family wires carry error information in
//!   the content).
//! - `GenerationOptions::max_tokens` is required by Anthropic; a `None`
//!   fails with `InvalidRequest` before HTTP. The adapter sends no default
//!   ceiling. [`AnthropicRenderOptions`] holds the choices only this
//!   protocol has.
//! - Tools: by default `tools` carries the frame's current tool state, the
//!   fold of its `ToolChange` blocks, so a change rewrites it. With
//!   [`AnthropicRenderOptions::native_tool_changes`], `tools` stays at the
//!   frame's first change (the declaration), followed by a deferred
//!   placeholder tool, and each later change renders at its position as a
//!   `system` message of `tool_removal` and `tool_addition` blocks (inline
//!   definitions), as `pi-ai` sends them. The request then needs
//!   [`INLINE_TOOLS_BETA`]. A change must follow a user turn; one that would
//!   render first or after assistant content is an `InvalidRequest`. An
//!   empty declaration renders as by default, because the API rejects a
//!   tool list in which every tool is deferred.
//! - [`CacheDirective::StablePrefix`] marks three `cache_control`
//!   breakpoints: the last tool definition, the `system` parameter
//!   (rendered as a block array so the anchor can attach — Anthropic
//!   cannot hang the marker off a string parameter), and the latest
//!   stable conversation message (the previous message when the final
//!   one carries this round's dispatched `tool_result` blocks, the
//!   final message otherwise). [`CacheDirective::None`] renders
//!   byte-identically to a body without any cache key.
//! - Wire fidelity. Each `text` and `tool_use` block parses to one item
//!   whose `wire.fields` is the block minus the keys its content holds
//!   (`text`; `id`, `name`, `input`); the renderer starts from those fields
//!   and writes the fact-derived keys over them. Text fields apply only to a
//!   block of exactly one text part. An own-format sealed part is emitted
//!   verbatim as an assistant content block.
//! - Reasoning is material. Every `thinking` block, `redacted_thinking` block
//!   and content block of a type this adapter has no vocabulary for is kept
//!   as one sealed item in place, with its stored value written back verbatim
//!   on the next turn; the readable `thinking` text additionally fills the
//!   reasoning observation, which is display-only.
//! - `reasoning_effort` is written as `output_config.effort` exactly as
//!   given. No `thinking` object is derived from it and no value is refused:
//!   a host that wants `thinking`, with or without `block_binding`, sends the
//!   object in `GenerationOptions::extra_body`, which is merged into the
//!   body beside what this renderer wrote. `output_schema` has no mapping in
//!   this adapter and is rejected before HTTP.

use serde_json::{Value, json};

use causa_kernel::{
    CacheDirective, ContextFrame, GenerationOptions, ModelInvokeError, ModelInvokeErrorKind,
    ModelItem, ModelOutput, ModelRef, ModelResponse, ReasoningPayload, ToolCallDraft,
    ToolDefinition, ToolResultStatus,
};

use super::context_frame::{self, ResolvedMedia, Role, Segment};
use super::media::MediaSet;
use super::render_options::AnthropicRenderOptions;
use crate::Protocol;

/// The `anthropic-beta` value that in-place tool changes need: it covers
/// `tool_addition` and `tool_removal` blocks, by reference and with inline
/// definitions.
pub const INLINE_TOOLS_BETA: &str = "inline-tools-2026-09-15";

/// Name of the deferred placeholder tool sent in in-place mode.
const DEFERRED_PLACEHOLDER: &str = "__causa_deferred_placeholder__";

/// A deferred tool that is never added, sent last in `tools` in in-place
/// mode. The API adds hidden prompt scaffolding for tool changes; declaring a
/// deferred tool from the first request keeps that scaffolding in the cached
/// prefix, so the first change does not miss the cache (as `pi-ai` reports
/// and does). The model cannot see or call it.
fn deferred_placeholder() -> Value {
    json!({
        "name": DEFERRED_PLACEHOLDER,
        "description": "Reserved placeholder. Never available. Never call this.",
        "input_schema": {"type": "object", "properties": {}},
        "defer_loading": true,
    })
}

/// Render a [`ContextFrame`] into an Anthropic Messages request body.
///
/// The body is complete: `model`, `max_tokens`, `messages`, plus `system`,
/// `temperature`, `tools` and `output_config` when the inputs call for them,
/// then the host's extra request fields. Media references resolve through `media`;
/// unresolvable or non-image references degrade to the shared text
/// placeholder. `options` holds the choices only this protocol has; see
/// [`AnthropicRenderOptions`]. Rendering is deterministic — the same frame,
/// generation, model, media table, cache directive and options always
/// produce byte-identical JSON.
pub fn render_anthropic_messages(
    frame: &ContextFrame,
    media: &MediaSet,
    generation: &GenerationOptions,
    model: &ModelRef,
    cache: CacheDirective,
    options: &AnthropicRenderOptions,
) -> Result<Value, ModelInvokeError> {
    super::generation::validate_anthropic(generation)?;
    super::tool_history::validate_frame(frame)?;
    let tool_plan = context_frame::plan_tools(
        frame,
        if options.native_tool_changes {
            context_frame::ToolPlacement::AnthropicInPlace
        } else {
            context_frame::ToolPlacement::Fold
        },
    );
    let normalized = context_frame::normalize(
        frame,
        media,
        Protocol::AnthropicMessages,
        tool_plan.in_place,
    );

    // Only the leading system run belongs in the top-level `system`
    // parameter; a system block appended after conversation content renders at
    // its position as a `system` message, exactly where the fact sits. Nothing
    // is hoisted.
    let segments = &normalized.segments;
    let leading = context_frame::leading_system_len(segments);
    let (leading_segments, conversation) = segments.split_at(leading);
    let mut system_blocks: Vec<Value> = leading_segments
        .iter()
        .filter_map(|segment| match segment {
            Segment::Text {
                role: Role::System,
                text,
                ..
            } => Some(json!({"type": "text", "text": text})),
            Segment::Media {
                role: Role::System,
                media: ResolvedMedia::Placeholder(text),
            } => Some(json!({"type": "text", "text": text})),
            _ => None,
        })
        .collect();

    // Group consecutive same-wire-role segments into one message
    // (Anthropic requires strictly alternating roles).
    let mut messages: Vec<(&'static str, Vec<Value>)> = Vec::new();
    // Positions of verbatim sealed blocks in `messages`; the cache anchor
    // never lands on one, because a sealed block is never modified. The
    // stored item can be any content-block type, so the position is tracked
    // rather than guessed from the block's `type`.
    let mut sealed_blocks: Vec<(usize, usize)> = Vec::new();
    // Positions of tool change blocks: not the place for the cache anchor
    // either, and a message holding only them does not end the stable
    // prefix.
    let mut change_blocks: Vec<(usize, usize)> = Vec::new();
    // The tool state the in-place changes apply to, from the declaration on.
    let mut tools_offered: Vec<ToolDefinition> = tool_plan.tools.clone();
    for segment in conversation {
        match segment {
            Segment::Text {
                role: Role::System,
                text,
                ..
            } => append_message(
                &mut messages,
                "system",
                json!({"type": "text", "text": text}),
            ),
            Segment::Text {
                role: Role::User,
                text,
                ..
            } => {
                append_message(&mut messages, "user", json!({"type": "text", "text": text}));
            }
            Segment::Text {
                role: Role::Assistant,
                text,
                origin,
            } => {
                // A whitespace-only assistant text part is omitted from the
                // request (as `pi-ai` does); the fact stays in the Context.
                // This is a listed exception to the round-trip property.
                if context_frame::is_blank_text(text) {
                    continue;
                }
                // The provider's fields ride along only for a block that is
                // exactly one text part: they describe that one wire block.
                let mut block = match origin.wire.as_ref().filter(|_| origin.parts == 1) {
                    Some(fields) => context_frame::residue_object(Some(fields)),
                    None => serde_json::Map::new(),
                };
                block.entry("type").or_insert_with(|| json!("text"));
                block.insert("text".into(), json!(text));
                append_message(&mut messages, "assistant", Value::Object(block));
            }
            Segment::Sealed(item) => {
                sealed_blocks.push(append_sealed(&mut messages, item.clone()));
            }
            Segment::Media { role, media } => match media {
                ResolvedMedia::Image {
                    media_type,
                    data_base64,
                } => append_message(
                    &mut messages,
                    role_name(*role),
                    image_block(media_type, data_base64),
                ),
                ResolvedMedia::Placeholder(text) => {
                    // A leading system placeholder went into the system
                    // parameter; one after conversation content stays at its
                    // position like any other system content.
                    append_message(
                        &mut messages,
                        role_name(*role),
                        json!({"type": "text", "text": text}),
                    );
                }
            },
            Segment::ToolCall(call) => {
                let mut block = context_frame::residue_object(call.wire.as_ref());
                block.entry("type").or_insert_with(|| json!("tool_use"));
                block.insert("id".into(), json!(call.wire_id));
                block.insert("name".into(), json!(call.name));
                block.insert("input".into(), call.arguments.clone());
                append_message(&mut messages, "assistant", Value::Object(block));
            }
            Segment::ToolResult {
                wire_id,
                status,
                content,
                notes,
                media,
            } => {
                let mut blocks = vec![json!({"type": "text", "text": content})];
                if !notes.is_empty() {
                    blocks.push(json!({"type": "text", "text": "Notes:"}));
                    blocks.extend(notes.iter().map(|note| {
                        json!({
                            "type": "text",
                            "text": format!("- {note}"),
                        })
                    }));
                }
                blocks.extend(media.iter().map(media_block_json));
                let content_json = if notes.is_empty() && media.is_empty() {
                    json!(content)
                } else {
                    json!(blocks)
                };
                let mut block_json = json!({
                    "type": "tool_result",
                    "tool_use_id": wire_id,
                    "content": content_json,
                });
                if *status != ToolResultStatus::Succeeded {
                    block_json["is_error"] = json!(true);
                }
                append_message(&mut messages, "user", block_json);
            }
            Segment::ToolChange(change) => {
                // The API takes a change only after a user turn (tool results
                // count) or within a system message; the renderer never moves
                // one there.
                if !matches!(messages.last(), Some(("user" | "system", _))) {
                    return Err(ModelInvokeError::new(
                        ModelInvokeErrorKind::InvalidRequest,
                        "an Anthropic tool change must follow a user turn",
                    ));
                }
                for block in tool_change_blocks(change, &tools_offered) {
                    append_message(&mut messages, "system", block);
                    let message = messages.len() - 1;
                    change_blocks.push((message, messages[message].1.len() - 1));
                }
                change.apply_to(&mut tools_offered);
            }
        }
    }

    super::tool_history::validate_anthropic(&messages)?;

    if messages.is_empty() {
        return Err(ModelInvokeError::new(
            ModelInvokeErrorKind::InvalidRequest,
            "frame rendered to zero messages; nothing to send",
        ));
    }

    // Stable-prefix anchors: within the system parameter the marker goes on
    // the last block; within the conversation on the last content block that is
    // not a sealed item, searching backward from the anchor message because a
    // sealed item is written verbatim and never modified.
    match cache {
        CacheDirective::None => {}
        CacheDirective::StablePrefix => {
            if let Some(last) = system_blocks.last_mut() {
                last["cache_control"] = json!({"type": "ephemeral"});
            }
            if let Some(last_block) = anchor_block(&mut messages, &sealed_blocks, &change_blocks) {
                last_block["cache_control"] = json!({"type": "ephemeral"});
            }
        }
    }

    // `validate_anthropic` rejected a missing limit, so this is always `Some`;
    // the error is repeated here rather than panicking if that invariant ever
    // changes.
    let max_tokens = generation.max_tokens.ok_or_else(|| {
        ModelInvokeError::new(
            ModelInvokeErrorKind::InvalidRequest,
            "max_tokens is required by the Anthropic Messages API; the adapter sends no default",
        )
    })?;
    let mut body = json!({
        "model": model.0,
        "max_tokens": max_tokens,
        "messages": messages
            .into_iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect::<Vec<_>>(),
    });
    if let Some(temperature) = generation.temperature {
        body["temperature"] = json!(temperature);
    }
    if !system_blocks.is_empty() {
        // Always the block-array form: one block per leading system part, no
        // join, so the value does not change shape with the cache directive.
        body["system"] = json!(system_blocks);
    }
    if !tool_plan.tools.is_empty() {
        let mut tools =
            context_frame::tool_definitions(&tool_plan.tools, context_frame::ToolShape::Anthropic);
        match cache {
            CacheDirective::None => {}
            CacheDirective::StablePrefix => {
                if let Some(last_tool) = tools.last_mut() {
                    last_tool["cache_control"] = json!({"type": "ephemeral"});
                }
            }
        }
        if tool_plan.in_place {
            tools.push(deferred_placeholder());
        }
        body["tools"] = json!(tools);
    }
    // The effort is written as given. No `thinking` object is derived from
    // it: a host that wants one sends it among the extra request fields.
    if let Some(effort) = &generation.reasoning_effort {
        body["output_config"] = json!({"effort": effort});
    }
    super::generation::merge_extra_body(&mut body, generation);
    Ok(body)
}

/// The content block the stable-prefix cache anchor lands on: the last block
/// of the anchor message that did not come from a sealed item, searching
/// backward through earlier messages when the anchor holds only sealed items.
/// A sealed item is written verbatim and must never be modified, so it cannot
/// carry the marker; `sealed` holds the `(message, block)` positions of those
/// blocks. The anchor message is the previous one when the final message
/// carries this round's dispatched `tool_result` blocks, the final message
/// otherwise: anchoring before the fresh dispatch keeps the cached prefix
/// clear of content the current round just appended.
fn anchor_block<'a>(
    messages: &'a mut [(&'static str, Vec<Value>)],
    sealed: &[(usize, usize)],
    changes: &[(usize, usize)],
) -> Option<&'a mut Value> {
    // Trailing messages that hold only tool changes are this round's
    // additions; the stable prefix ends before them.
    let end = messages
        .iter()
        .enumerate()
        .rposition(|(message_index, message)| {
            (0..message.1.len()).any(|block| !changes.contains(&(message_index, block)))
        })?;
    let last = &messages[end];
    let dispatching = last
        .1
        .iter()
        .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"));
    let start = if dispatching && end > 0 { end - 1 } else { end };
    for (message_index, message) in messages[..=start].iter_mut().enumerate().rev() {
        let found = message
            .1
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(block_index, _)| {
                !sealed.contains(&(message_index, *block_index))
                    && !changes.contains(&(message_index, *block_index))
            })
            .map(|(_, block)| block);
        if found.is_some() {
            return found;
        }
    }
    None
}

/// The content blocks of one in-place tool change, applied to `offered`:
/// a `tool_removal` per withdrawn tool that is offered and not defined again
/// in the same change (a new definition replaces the old one), then a
/// `tool_addition` with the inline definition per added tool, as `pi-ai`
/// renders them.
fn tool_change_blocks(change: &causa_kernel::ToolChange, offered: &[ToolDefinition]) -> Vec<Value> {
    let mut blocks: Vec<Value> = change
        .removed
        .iter()
        .filter(|name| offered.iter().any(|definition| &definition.name == *name))
        .filter(|name| {
            !change
                .added
                .iter()
                .any(|definition| &definition.name == *name)
        })
        .map(|name| {
            json!({
                "type": "tool_removal",
                "tool": {"type": "tool_reference", "name": name},
            })
        })
        .collect();
    blocks.extend(
        context_frame::tool_definitions(&change.added, context_frame::ToolShape::Anthropic)
            .into_iter()
            .map(|definition| {
                json!({
                    "type": "tool_addition",
                    "tool": {"type": "tool_definition", "definition": definition},
                })
            }),
    );
    blocks
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// The `image` content block for a resolved inline payload.
fn image_block(media_type: &str, data_base64: &str) -> Value {
    json!({
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data_base64},
    })
}

/// A resolved-or-placeholder attachment as a content block: images embed
/// inline, everything else degrades to its placeholder text.
fn media_block_json(media: &ResolvedMedia) -> Value {
    match media {
        ResolvedMedia::Image {
            media_type,
            data_base64,
        } => image_block(media_type, data_base64),
        ResolvedMedia::Placeholder(text) => json!({"type": "text", "text": text}),
    }
}

/// Parse an Anthropic Messages response body into a kernel
/// [`ModelOutput`].
///
/// Each `text` block becomes one text item and each `tool_use` block one
/// tool-call item (`provider_call_id` carries the provider id), in the
/// order returned. An item's `wire` holds the block minus the fields its
/// content holds (`text`; `id`, `name` and `input`), so the renderer can
/// send the block back as returned. Every `thinking` block,
/// `redacted_thinking` block and unknown block type becomes one sealed item
/// **in place**, holding the block exactly as returned, so the turn can be
/// echoed back; the readable `thinking` text additionally fills the reasoning
/// observation, which is display-only. `stop_reason` is required; a value the kernel has no meaning for maps to
/// [`causa_kernel::ModelStopReason::Unrecognized`] rather than being guessed, and the raw
/// wire value is preserved on [`ModelOutput::raw_stop_reason`]. One mapping
/// table is shared with the streaming decoder.
pub fn parse_anthropic_response(value: &Value) -> Result<ModelOutput, ModelInvokeError> {
    fn undecodable(message: impl Into<String>) -> ModelInvokeError {
        ModelInvokeError::new(ModelInvokeErrorKind::Undecodable, message)
    }

    let empty = Vec::new();
    let content = match value.get("content") {
        None | Some(Value::Null) => &empty,
        Some(Value::Array(items)) => items,
        Some(_) => return Err(undecodable("content: expected an array")),
    };

    let mut items: Vec<ModelItem> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| undecodable("text block: missing `text` string"))?;
                items.push({
                    let mut item = ModelItem::text(text);
                    item.wire = Some(context_frame::wire_meta(
                        Protocol::AnthropicMessages,
                        context_frame::without_keys(block, &["text"]),
                    ));
                    item
                });
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| undecodable("tool_use block: missing `id` string"))?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| undecodable("tool_use block: missing `name` string"))?;
                items.push({
                    let mut item = ModelItem::tool_call(
                        ToolCallDraft::new(
                            name.to_string(),
                            block.get("input").cloned().unwrap_or_else(|| json!({})),
                        )
                        .with_provider_call_id(id.to_string()),
                    );
                    item.wire = Some(context_frame::wire_meta(
                        Protocol::AnthropicMessages,
                        context_frame::without_keys(block, &["id", "name", "input"]),
                    ));
                    item
                });
            }
            Some("thinking") => {
                let text = block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .ok_or_else(|| undecodable("thinking block: missing `thinking` string"))?;
                thinking_parts.push(text.to_string());
                // The block, signature included, is sealed material; the text
                // above is the display-only observation.
                items.push(context_frame::sealed_item(
                    Protocol::AnthropicMessages,
                    block.clone(),
                ));
            }
            // `redacted_thinking` and every block type this adapter has no
            // vocabulary for are kept as sealed items in place, so the turn
            // can be echoed back as returned.
            _ => items.push(context_frame::sealed_item(
                Protocol::AnthropicMessages,
                block.clone(),
            )),
        }
    }

    let raw_stop_reason = match crate::translation::stop_reason::anthropic_raw(value) {
        Some(raw) => raw.to_owned(),
        None if value.get("stop_reason").is_some_and(|v| !v.is_null()) => {
            return Err(undecodable("stop_reason: expected a string"));
        }
        None => return Err(undecodable("missing stop_reason")),
    };
    let stop_reason = crate::translation::stop_reason::anthropic(&raw_stop_reason);

    let usage = value
        .get("usage")
        .filter(|v| v.is_object())
        .map(crate::translation::usage::model_usage_from_anthropic);

    Ok({
        let mut output = ModelOutput::new(ModelResponse { items }, stop_reason);
        output.usage = usage;
        output.raw_stop_reason = Some(raw_stop_reason);
        output.reasoning = if thinking_parts.is_empty() {
            None
        } else {
            Some(ReasoningPayload::new(thinking_parts.join("\n")))
        };
        output
    })
}

fn append_message(
    messages: &mut Vec<(&'static str, Vec<Value>)>,
    role: &'static str,
    block: Value,
) {
    match messages.last_mut() {
        Some((last_role, content)) if *last_role == role => content.push(block),
        _ => messages.push((role, vec![block])),
    }
}

/// Append one verbatim sealed block to the open assistant message, or open a
/// new assistant message for it. Returns its `(message, block)` position so
/// the cache anchor can skip it.
fn append_sealed(messages: &mut Vec<(&'static str, Vec<Value>)>, block: Value) -> (usize, usize) {
    if matches!(messages.last(), Some(("assistant", _))) {
        let content = &mut messages
            .last_mut()
            .expect("assistant message checked above")
            .1;
        let index = content.len();
        content.push(block);
        (messages.len() - 1, index)
    } else {
        messages.push(("assistant", vec![block]));
        (messages.len() - 1, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causa_kernel::{MediaRef, ModelStopReason, ModelUsage, ToolDefinition};

    use crate::translation::media::{MediaPayload, MediaSet};
    use crate::translation::test_support::{
        call, frame, media_part, parts, result, result_with_media, text, text_part,
        text_with_source, with_tools,
    };

    /// Anthropic requires `max_tokens`; the renderer tests always supply one.
    fn test_generation() -> GenerationOptions {
        GenerationOptions::default().with_max_tokens(4096)
    }

    fn render(frame: &ContextFrame) -> Value {
        render_anthropic_messages(
            frame,
            &MediaSet::new(),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap()
    }

    // --- rendering ---------------------------------------------------------

    #[test]
    fn role_mapping_and_source_independence() {
        let f = frame(vec![
            text(0, "be terse", Role::System),
            text(1, "model said", Role::Assistant),
            text(2, "user said", Role::User),
            text(3, "replayed", Role::Assistant),
            // source set to "system" but role is User: role wins, source ignored
            text_with_source(4, "user override", Role::User, Some("system")),
            // source absent and role is User: renders as user
            text(5, "user no source", Role::User),
        ]);
        let v = render(&f);
        // The top-level system parameter is always the block-array form.
        assert_eq!(v["system"], json!([{"type": "text", "text": "be terse"}]));
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(
            msgs[0]["content"],
            json!([{"type": "text", "text": "model said"}])
        );
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[3]["role"], "user");
        // adjacent same-role texts stay separate content blocks: no text is
        // joined across block seams
        assert_eq!(
            msgs[3]["content"],
            json!([
                {"type": "text", "text": "user override"},
                {"type": "text", "text": "user no source"},
            ])
        );
    }

    #[test]
    fn empty_text_blocks_are_skipped() {
        // The host door can commit empty texts; the renderer mirrors the
        // model door and skips them.
        let f = frame(vec![
            text(0, "", Role::User),
            text(1, "real", Role::User),
            text(2, "", Role::System),
            text(3, "still here", Role::Assistant),
        ]);
        let v = render(&f);
        // the skipped leading empty text must not leave an empty message
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(
            msgs[0]["content"],
            json!([{"type": "text", "text": "real"}])
        );
        assert!(v.get("system").is_none());
    }

    #[test]
    fn tool_round_trip_pairing_and_consecutive_merge() {
        let fallback_id = call(2, "kc2", None, "list", json!({})).id().0.to_string();
        let f = frame(vec![
            text(0, "reading now", Role::Assistant),
            call(1, "kc1", Some("toolu_a"), "read", json!({"path": "a"})),
            call(2, "kc2", None, "list", json!({})),
            result(3, "kc1", ToolResultStatus::Succeeded, json!("file-a")),
            result(4, "kc2", ToolResultStatus::Failed, json!({"error": "boom"})),
        ]);
        let v = render(&f);
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        // text + both calls merge into one assistant message; the wire id
        // is provider_call_id when present, declaration BlockId UUID otherwise
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(
            msgs[0]["content"][0],
            json!({"type": "text", "text": "reading now"})
        );
        assert_eq!(
            msgs[0]["content"][1],
            json!({"type": "tool_use", "id": "toolu_a", "name": "read", "input": {"path": "a"}})
        );
        assert_eq!(
            msgs[0]["content"][2],
            json!({"type": "tool_use", "id": fallback_id, "name": "list", "input": {}})
        );
        // both results merge into one user message, ids via the pairing map
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(
            msgs[1]["content"][0],
            json!({"type": "tool_result", "tool_use_id": "toolu_a", "content": "file-a"})
        );
        // non-string observations serialize to a string; Failed sets is_error
        assert_eq!(
            msgs[1]["content"][1],
            json!({
                "type": "tool_result",
                "tool_use_id": fallback_id,
                "content": "{\"error\":\"boom\"}",
                "is_error": true,
            })
        );
    }

    #[test]
    fn unpaired_tool_result_is_invalid_before_http() {
        let f = frame(vec![result(
            0,
            "orphan",
            ToolResultStatus::Succeeded,
            json!("x"),
        )]);
        let error = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &test_generation(),
            &ModelRef::new("test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
    }

    #[test]
    fn tools_generation_and_byte_determinism() {
        let surface = vec![ToolDefinition::new(
            "read",
            "read a file",
            json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )];
        let f = frame(vec![text(0, "hi", Role::User)]);

        let generation = GenerationOptions::default()
            .with_temperature(0.5)
            .with_max_tokens(4096);
        let v = render_anthropic_messages(
            &with_tools(&f, &surface),
            &MediaSet::new(),
            &generation,
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(v["model"], json!("claude-test"));
        assert_eq!(v["temperature"], json!(0.5));
        assert_eq!(v["max_tokens"], json!(4096));
        assert_eq!(
            v["tools"],
            json!([{
                "name": "read",
                "description": "read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }])
        );

        let generation = GenerationOptions::default()
            .with_temperature(0.5)
            .with_max_tokens(100);
        let v2 = render_anthropic_messages(
            &with_tools(&f, &surface),
            &MediaSet::new(),
            &generation,
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(v2["max_tokens"], json!(100));
        // byte determinism over the full body
        let again = render_anthropic_messages(
            &with_tools(&f, &surface),
            &MediaSet::new(),
            &generation,
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&v2).unwrap(),
            serde_json::to_string(&again).unwrap()
        );
    }

    // --- effort and extra request fields ----------------------------------

    fn render_with(
        generation: &GenerationOptions,
        options: &AnthropicRenderOptions,
    ) -> Result<Value, ModelInvokeError> {
        let f = frame(vec![text(0, "hi", Role::User)]);
        render_anthropic_messages(
            &f,
            &MediaSet::new(),
            generation,
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            options,
        )
    }

    #[test]
    fn effort_is_written_as_given_and_derives_no_thinking_object() {
        // unset: neither field
        let body = render_with(&test_generation(), &AnthropicRenderOptions::default()).unwrap();
        assert!(body.get("thinking").is_none());
        assert!(body.get("output_config").is_none());

        // Any string, including ones this adapter once refused or mapped.
        for effort in ["none", "minimal", "high", "a-level-of-tomorrow", ""] {
            let mut generation = test_generation();
            generation.reasoning_effort = Some(effort.into());
            let body = render_with(&generation, &AnthropicRenderOptions::default()).unwrap();
            assert_eq!(body["output_config"], json!({"effort": effort}), "{effort}");
            assert!(body.get("thinking").is_none(), "{effort}");
        }
    }

    #[test]
    fn extra_request_fields_are_copied_and_never_replace_rendered_keys() {
        let thinking = json!({
            "type": "adaptive",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            "unknown_to_this_adapter": [1, {"a": null}],
        });
        let mut generation = test_generation();
        generation.reasoning_effort = Some("high".into());
        generation.temperature = Some(0.5);
        generation
            .extra_body
            .insert("thinking".into(), thinking.clone());
        generation.extra_body.insert("model".into(), json!("other"));
        generation.extra_body.insert("messages".into(), json!([]));
        generation
            .extra_body
            .insert("output_config".into(), json!({"effort": "low"}));
        let body = render_with(&generation, &AnthropicRenderOptions::default()).unwrap();
        assert_eq!(body["thinking"], thinking);
        // An effort beside a temperature is the provider's to judge.
        assert_eq!(body["temperature"], json!(0.5));
        assert_eq!(body["model"], "claude-test");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["output_config"], json!({"effort": "high"}));
    }

    #[test]
    fn missing_max_tokens_still_fails_before_rendering() {
        let error = render_with(
            &GenerationOptions::default(),
            &AnthropicRenderOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
        assert!(error.message.contains("max_tokens"), "{}", error.message);
    }

    #[test]
    fn empty_frame_is_invalid_request() {
        let f = frame(vec![]);
        let e = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &test_generation(),
            &ModelRef::new("m"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(e.kind, ModelInvokeErrorKind::InvalidRequest));
    }

    #[test]
    fn cache_none_omits_every_cache_control_key() {
        let f = frame(vec![
            text(0, "be terse", Role::System),
            text(1, "hi", Role::User),
        ]);
        let surface = vec![ToolDefinition::new(
            "read",
            "read a file",
            json!({"type": "object"}),
        )];
        let v = render_anthropic_messages(
            &with_tools(&f, &surface),
            &MediaSet::new(),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert!(!serde_json::to_string(&v).unwrap().contains("cache_control"));
    }

    #[test]
    fn stable_prefix_marks_tools_system_and_previous_message() {
        let f = frame(vec![
            text(0, "be terse", Role::System),
            text(1, "hi", Role::User),
            call(2, "kc1", Some("toolu_a"), "read", json!({"path": "a"})),
            result(3, "kc1", ToolResultStatus::Succeeded, json!("file-a")),
        ]);
        let surface = vec![ToolDefinition::new(
            "read",
            "read a file",
            json!({"type": "object"}),
        )];
        let v = render_anthropic_messages(
            &with_tools(&f, &surface),
            &MediaSet::new(),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::StablePrefix,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        // system renders as a block array carrying the anchor
        assert_eq!(
            v["system"],
            json!([{
                "type": "text",
                "text": "be terse",
                "cache_control": {"type": "ephemeral"},
            }])
        );
        // the final message carries this round's dispatched tool results,
        // so the anchor lands on the last block of the assistant message
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert!(msgs[1]["content"][0].get("cache_control").is_some());
        // the last tool definition carries the anchor
        assert_eq!(v["tools"][0]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn stable_prefix_without_dispatch_anchors_the_final_message() {
        let f = frame(vec![text(0, "hi", Role::User)]);
        let v = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::StablePrefix,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert!(msgs[0]["content"][0].get("cache_control").is_some());
    }

    // --- media -------------------------------------------------------------

    fn resolved(reference: &str) -> MediaSet {
        let mut m = MediaSet::new();
        m.insert(reference, MediaPayload::new("image/png", "AAAA"));
        m
    }

    #[test]
    fn image_parts_render_as_image_blocks_in_user_position() {
        let f = frame(vec![parts(
            0,
            vec![
                text_part("caption"),
                media_part("image/png", "asset-1"),
                text_part("after"),
            ],
            Role::User,
        )]);
        let v = render_anthropic_messages(
            &f,
            &resolved("asset-1"),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        // part boundaries are preserved: text, image, text as separate blocks
        assert_eq!(
            v["messages"][0]["content"],
            json!([
                {"type": "text", "text": "caption"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                {"type": "text", "text": "after"},
            ])
        );
    }

    #[test]
    fn unresolved_media_degrades_to_the_deterministic_placeholder() {
        let f = frame(vec![parts(
            0,
            vec![media_part("image/png", "missing-asset")],
            Role::User,
        )]);
        let v = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            v["messages"][0]["content"],
            json!([{"type": "text", "text": "[media: image/png missing-asset]"}])
        );
    }

    #[test]
    fn non_image_media_degrades_even_when_resolved() {
        let f = frame(vec![parts(
            0,
            vec![media_part("application/pdf", "doc-1")],
            Role::User,
        )]);
        let mut media = MediaSet::new();
        media.insert("doc-1", MediaPayload::new("application/pdf", "AAAA"));
        let v = render_anthropic_messages(
            &f,
            &media,
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            v["messages"][0]["content"],
            json!([{"type": "text", "text": "[media: application/pdf doc-1]"}])
        );
    }

    #[test]
    fn media_in_system_position_degrades_to_placeholder_text() {
        let f = frame(vec![
            parts(0, vec![media_part("image/png", "asset-1")], Role::System),
            text(1, "hi", Role::User),
        ]);
        let v = render_anthropic_messages(
            &f,
            &resolved("asset-1"),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        // the system parameter is text-only; the walk degraded it, and the
        // block-array form carries the placeholder
        assert_eq!(
            v["system"],
            json!([{"type": "text", "text": "[media: image/png asset-1]"}])
        );
    }

    #[test]
    fn tool_result_media_embeds_in_the_tool_result_content() {
        let f = frame(vec![
            call(0, "kc1", Some("toolu_a"), "render", json!({})),
            result_with_media(
                1,
                "kc1",
                ToolResultStatus::Succeeded,
                json!("chart ready"),
                vec![
                    MediaRef::new("image/png", "asset-1"),
                    MediaRef::new("image/png", "asset-2"),
                ],
            ),
        ]);
        let mut media = MediaSet::new();
        media.insert("asset-1", MediaPayload::new("image/png", "AAAA"));
        // asset-2 deliberately unregistered -> placeholder inside the array
        let v = render_anthropic_messages(
            &f,
            &media,
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            v["messages"][1]["content"][0]["content"],
            json!([
                {"type": "text", "text": "chart ready"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                {"type": "text", "text": "[media: image/png asset-2]"},
            ])
        );
    }

    #[test]
    fn result_without_media_keeps_the_string_content_shape() {
        let f = frame(vec![
            call(0, "kc1", None, "read", json!({})),
            result(1, "kc1", ToolResultStatus::Succeeded, json!("ok")),
        ]);
        let v = render_anthropic_messages(
            &f,
            &resolved("unused"),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(v["messages"][1]["content"][0]["content"], json!("ok"));
    }

    #[test]
    fn stable_prefix_anchor_lands_on_parts_messages() {
        let f = frame(vec![
            text(0, "hi", Role::User),
            parts(
                1,
                vec![text_part("see"), media_part("image/png", "asset-1")],
                Role::User,
            ),
        ]);
        let v = render_anthropic_messages(
            &f,
            &resolved("asset-1"),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::StablePrefix,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        // same-role blocks share one message but their texts stay separate
        // content blocks: [text "hi", text "see", image]
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"][0], json!({"type": "text", "text": "hi"}));
        assert_eq!(
            msgs[0]["content"][1],
            json!({"type": "text", "text": "see"})
        );
        // the final message carries no tool_result, so it anchors itself
        assert!(
            msgs[0]["content"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .get("cache_control")
                .is_some()
        );
    }

    // --- parsing -----------------------------------------------------------

    #[test]
    fn parse_three_content_shapes_with_provider_id_passthrough() {
        let v = json!({
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig1"},
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_9", "name": "read", "input": {"path": "a"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        });
        let out = parse_anthropic_response(&v).unwrap();
        assert_eq!(out.response.text(), "Let me check.");
        assert_eq!(out.response.tool_calls().count(), 1);
        let tc = &out.response.tool_calls().next().unwrap();
        assert_eq!(tc.tool_name, "read");
        assert_eq!(tc.arguments, json!({"path": "a"}));
        assert_eq!(tc.provider_call_id.as_deref(), Some("toolu_9"));
        assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
        // The whole block, signature included, is sealed material in place:
        // item order is the wire order, and the text/tool items follow it.
        assert_eq!(out.response.items.len(), 3);
        assert_eq!(
            sealed_items(&out),
            vec![json!({
                "type": "thinking", "thinking": "hmm", "signature": "sig1"
            })]
        );
        let reasoning = out.reasoning.unwrap();
        assert_eq!(reasoning.text, "hmm");
    }

    /// The stored value of every sealed part of a response, in item order.
    fn sealed_items(out: &ModelOutput) -> Vec<Value> {
        out.response
            .items
            .iter()
            .flat_map(|item| match &item.content {
                causa_kernel::ModelItemContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|part| match part {
                        causa_kernel::ContentPart::Sealed(sealed) => Some(sealed.item.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect()
    }

    #[test]
    fn usage_mapping_includes_cache_fields() {
        let v = json!({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 11,
                "cache_creation_input_tokens": 3,
            },
        });
        let out = parse_anthropic_response(&v).unwrap();
        let u: ModelUsage = out.usage.unwrap();
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.output_tokens, 20);
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.cache_write_tokens, Some(3));
        // Anthropic does not expose reasoning tokens
        assert_eq!(u.reasoning_tokens, None);
    }

    #[test]
    fn absent_usage_is_none() {
        let v = json!({"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"});
        let out = parse_anthropic_response(&v).unwrap();
        assert!(out.usage.is_none());
    }

    #[test]
    fn stop_reason_mapping_full_and_unknown_is_unrecognized() {
        for (wire, expected) in [
            ("end_turn", ModelStopReason::EndTurn),
            ("stop_sequence", ModelStopReason::EndTurn),
            ("tool_use", ModelStopReason::ToolUse),
            ("max_tokens", ModelStopReason::MaxTokens),
            ("refusal", ModelStopReason::Refusal),
            (
                "model_context_window_exceeded",
                ModelStopReason::ContextWindowExceeded,
            ),
            ("pause_turn", ModelStopReason::Unrecognized),
            // An unknown value is reported as what it is, never guessed.
            ("model_context_overflow", ModelStopReason::Unrecognized),
        ] {
            let v = json!({"content": [], "stop_reason": wire});
            let out = parse_anthropic_response(&v).unwrap();
            assert_eq!(
                std::mem::discriminant(&out.stop_reason),
                std::mem::discriminant(&expected),
                "wire stop_reason = {wire}"
            );
            // The wire value reaches the host unchanged.
            assert_eq!(out.raw_stop_reason.as_deref(), Some(wire));
        }
    }

    #[test]
    fn missing_stop_reason_is_undecodable_error() {
        let v = json!({"content": []});
        let e = parse_anthropic_response(&v).unwrap_err();
        assert!(matches!(e.kind, ModelInvokeErrorKind::Undecodable));
        assert!(e.message.contains("stop_reason"));
        // Present but not a string is reported as such, not as missing.
        let e = parse_anthropic_response(&json!({"content": [], "stop_reason": 7})).unwrap_err();
        assert!(matches!(e.kind, ModelInvokeErrorKind::Undecodable));
        assert!(e.message.contains("expected a string"));
    }

    #[test]
    fn exotic_blocks_are_sealed_in_place_not_skipped() {
        let v = json!({
            "content": [
                {"type": "redacted_thinking", "data": "encrypted"},
                {"type": "text", "text": "a"},
                {"type": "server_tool_use", "id": "x", "name": "web_search"},
                {"type": "text", "text": "b"},
            ],
            "stop_reason": "end_turn",
        });
        let out = parse_anthropic_response(&v).unwrap();
        // one item per content block, in wire order, nothing skipped: the two
        // text blocks stay separate and the two unknown blocks are sealed
        assert_eq!(out.response.items.len(), 4);
        assert_eq!(out.response.text(), "ab");
        assert!(out.response.tool_calls().next().is_none());
        assert!(out.reasoning.is_none());
        assert_eq!(
            sealed_items(&out),
            vec![
                json!({"type": "redacted_thinking", "data": "encrypted"}),
                json!({"type": "server_tool_use", "id": "x", "name": "web_search"}),
            ]
        );
    }

    #[test]
    fn malformed_tool_use_is_undecodable_error() {
        let v = json!({"content": [{"type": "tool_use", "id": "x"}], "stop_reason": "tool_use"});
        let e = parse_anthropic_response(&v).unwrap_err();
        assert!(matches!(e.kind, ModelInvokeErrorKind::Undecodable));
        assert!(e.message.contains("name"));
    }
}
