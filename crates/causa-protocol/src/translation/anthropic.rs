//! Anthropic Messages translation for the context kernel.
//!
//! Pure functions from [`causa_kernel::ContextFrame`] to an Anthropic
//! Messages request body, and from an Anthropic Messages response body
//! back to [`causa_kernel::ModelOutput`]. Transport-free — the reqwest
//! adapter in `causa-provider` owns HTTP.
//!
//! The renderer-independent policy (source vocabulary, empty-text skip,
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
//!   ceiling. [`AnthropicRenderOptions`] carries the Anthropic-only choices.
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
//!   reasoning observation, which is display-only. `reasoning_effort` renders
//!   the effort table: unset leaves the `thinking` field out, `None` renders
//!   `thinking: {"type": "disabled"}`, and `Low`..`Max` render
//!   `thinking: {"type": "adaptive"}` plus `output_config.effort`.
//!   `Minimal` has no wire level and is rejected before HTTP, as is
//!   `output_schema` (no mapping in this adapter).
//! - `AnthropicRenderOptions::prefix_mismatch_behavior` renders as
//!   `thinking.block_binding.prefix_mismatch_behavior` whenever a `thinking`
//!   object is rendered; with `reasoning_effort` unset it renders nothing.

use serde_json::{Value, json};

use causa_kernel::{
    CacheDirective, ContextFrame, GenerationOptions, ModelInvokeError, ModelInvokeErrorKind,
    ModelItem, ModelOutput, ModelRef, ModelResponse, ReasoningEffort, ReasoningPayload,
    ToolCallDraft, ToolResultStatus, ToolSurface,
};

use super::context_frame::{self, ResolvedMedia, Role, Segment};
use super::media::MediaSet;
use super::render_options::AnthropicRenderOptions;
use crate::Protocol;

/// Render a [`ContextFrame`] into an Anthropic Messages request body.
///
/// The body is complete: `model`, `max_tokens`, `messages`, plus `system`,
/// `temperature`, `tools`, the effort `thinking` shape and `output_config`
/// when the inputs call for them. Media references resolve through `media`;
/// unresolvable or non-image references degrade to the shared text
/// placeholder. `options` carries the Anthropic-only choices; see
/// [`AnthropicRenderOptions`]. Rendering is deterministic — the same frame,
/// surface, generation, model, media table, cache directive and options
/// always produce byte-identical JSON.
pub fn render_anthropic_messages(
    frame: &ContextFrame,
    media: &MediaSet,
    tool_surface: &ToolSurface,
    generation: &GenerationOptions,
    model: &ModelRef,
    cache: CacheDirective,
    options: &AnthropicRenderOptions,
) -> Result<Value, ModelInvokeError> {
    super::generation::validate_anthropic(generation)?;
    super::tool_history::validate_frame(frame)?;
    let normalized = context_frame::normalize(frame, media, Protocol::AnthropicMessages);

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
    if matches!(cache, CacheDirective::StablePrefix) {
        if let Some(last) = system_blocks.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
        if let Some(last_block) = anchor_block(&mut messages, &sealed_blocks) {
            last_block["cache_control"] = json!({"type": "ephemeral"});
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
    if !tool_surface.definitions.is_empty() {
        let mut tools =
            context_frame::tool_definitions(tool_surface, context_frame::ToolShape::Anthropic);
        if matches!(cache, CacheDirective::StablePrefix)
            && let Some(last_tool) = tools.last_mut()
        {
            last_tool["cache_control"] = json!({"type": "ephemeral"});
        }
        body["tools"] = json!(tools);
    }
    // The effort table. `Minimal` was rejected by validation: the wire has no
    // such level. `None` disables thinking; every other level renders adaptive
    // thinking plus its `output_config.effort`.
    if let Some(effort) = generation.reasoning_effort {
        let mut thinking = if effort == ReasoningEffort::None {
            json!({"type": "disabled"})
        } else {
            body["output_config"] = json!({"effort": effort});
            json!({"type": "adaptive"})
        };
        // Rendered only alongside a `thinking` object; setting the option does
        // not add the beta header the field needs (the host's header list).
        if let Some(behavior) = options.prefix_mismatch_behavior {
            thinking["block_binding"] = json!({"prefix_mismatch_behavior": behavior.as_str()});
        }
        body["thinking"] = thinking;
    }
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
) -> Option<&'a mut Value> {
    let last = messages.last()?;
    let dispatching = last
        .1
        .iter()
        .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"));
    let start = if dispatching && messages.len() > 1 {
        messages.len() - 2
    } else {
        messages.len() - 1
    };
    for (message_index, message) in messages[..=start].iter_mut().enumerate().rev() {
        let found = message
            .1
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(block_index, _)| !sealed.contains(&(message_index, *block_index)))
            .map(|(_, block)| block);
        if found.is_some() {
            return found;
        }
    }
    None
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
                items.push(ModelItem {
                    wire: Some(context_frame::wire_meta(
                        Protocol::AnthropicMessages,
                        context_frame::without_keys(block, &["text"]),
                    )),
                    ..ModelItem::text(text)
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
                items.push(ModelItem {
                    wire: Some(context_frame::wire_meta(
                        Protocol::AnthropicMessages,
                        context_frame::without_keys(block, &["id", "name", "input"]),
                    )),
                    ..ModelItem::tool_call(ToolCallDraft {
                        tool_name: name.to_string(),
                        arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
                        provider_call_id: Some(id.to_string()),
                    })
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

    Ok(ModelOutput {
        response: ModelResponse { items },
        usage,
        stop_reason,
        raw_stop_reason: Some(raw_stop_reason),
        reasoning: if thinking_parts.is_empty() {
            None
        } else {
            Some(ReasoningPayload {
                text: thinking_parts.join("\n"),
            })
        },
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
    use crate::translation::render_options::PrefixMismatchBehavior;
    use crate::translation::test_support::{
        call, frame, media_part, parts, result, result_with_media, text, text_part,
    };

    /// Anthropic requires `max_tokens`; the renderer tests always supply one.
    fn test_generation() -> GenerationOptions {
        GenerationOptions {
            max_tokens: Some(4096),
            ..GenerationOptions::default()
        }
    }

    fn render(frame: &ContextFrame) -> Value {
        render_anthropic_messages(
            frame,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &test_generation(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap()
    }

    // --- rendering ---------------------------------------------------------

    #[test]
    fn source_vocabulary_full_branch_coverage() {
        let f = frame(vec![
            text(0, "be terse", Some("system")),
            text(1, "model said", None),
            text(2, "user said", Some("user")),
            text(3, "replayed", Some("assistant")),
            text(4, "injected", Some("inject:note")),
            text(5, "bare inject", Some("inject")),
            text(6, "unknown tag", Some("provider:gpt-x")),
        ]);
        let v = render(&f);
        // The top-level system parameter is always the block-array form.
        assert_eq!(v["system"], json!([{"type": "text", "text": "be terse"}]));
        let msgs = v["messages"].as_array().unwrap();
        // assistant(None) / user("user") / assistant("assistant") /
        // user("inject:note" + "inject" + unknown merged)
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
                {"type": "text", "text": "injected"},
                {"type": "text", "text": "bare inject"},
                {"type": "text", "text": "unknown tag"},
            ])
        );
    }

    #[test]
    fn empty_text_blocks_are_skipped() {
        // The host door can commit empty texts; the renderer mirrors the
        // model door and skips them.
        let f = frame(vec![
            text(0, "", Some("user")),
            text(1, "real", Some("user")),
            text(2, "", Some("system")),
            text(3, "still here", None),
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
            text(0, "reading now", None),
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
            &ToolSurface::empty(),
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
        let surface = ToolSurface::from_definitions(vec![ToolDefinition {
            name: "read".into(),
            description: "read a file".into(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        }]);
        let f = frame(vec![text(0, "hi", Some("user"))]);

        let generation = GenerationOptions {
            temperature: Some(0.5),
            max_tokens: Some(4096),
            ..GenerationOptions::default()
        };
        let v = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &surface,
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

        let generation = GenerationOptions {
            temperature: Some(0.5),
            max_tokens: Some(100),
            ..GenerationOptions::default()
        };
        let v2 = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &surface,
            &generation,
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(v2["max_tokens"], json!(100));
        // byte determinism over the full body
        let again = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &surface,
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

    // --- effort table and render options -----------------------------------

    fn render_with(
        generation: &GenerationOptions,
        options: &AnthropicRenderOptions,
    ) -> Result<Value, ModelInvokeError> {
        let f = frame(vec![text(0, "hi", Some("user"))]);
        render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
            generation,
            &ModelRef::new("claude-test"),
            CacheDirective::None,
            options,
        )
    }

    #[test]
    fn effort_table_renders_every_level() {
        // unset: no thinking field at all
        let body = render_with(&test_generation(), &AnthropicRenderOptions::default()).unwrap();
        assert!(body.get("thinking").is_none());
        assert!(body.get("output_config").is_none());

        // None: thinking explicitly disabled, no effort level
        let generation = GenerationOptions {
            max_tokens: Some(64),
            reasoning_effort: Some(ReasoningEffort::None),
            ..GenerationOptions::default()
        };
        let body = render_with(&generation, &AnthropicRenderOptions::default()).unwrap();
        assert_eq!(body["thinking"], json!({"type": "disabled"}));
        assert!(body.get("output_config").is_none());

        // Low..Max: adaptive thinking plus output_config.effort
        for (effort, wire) in [
            (ReasoningEffort::Low, "low"),
            (ReasoningEffort::Medium, "medium"),
            (ReasoningEffort::High, "high"),
            (ReasoningEffort::XHigh, "xhigh"),
            (ReasoningEffort::Max, "max"),
        ] {
            let generation = GenerationOptions {
                max_tokens: Some(64),
                reasoning_effort: Some(effort),
                ..GenerationOptions::default()
            };
            let body = render_with(&generation, &AnthropicRenderOptions::default()).unwrap();
            assert_eq!(body["thinking"], json!({"type": "adaptive"}), "{wire}");
            assert_eq!(body["output_config"], json!({"effort": wire}), "{wire}");
        }
    }

    #[test]
    fn minimal_effort_and_missing_max_tokens_fail_before_rendering() {
        let generation = GenerationOptions {
            max_tokens: Some(64),
            reasoning_effort: Some(ReasoningEffort::Minimal),
            ..GenerationOptions::default()
        };
        let error = render_with(&generation, &AnthropicRenderOptions::default()).unwrap_err();
        assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
        assert!(error.message.contains("minimal"), "{error}");

        // No default ceiling: a missing max_tokens is rejected, never filled.
        let error = render_with(
            &GenerationOptions::default(),
            &AnthropicRenderOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
        assert!(error.message.contains("max_tokens"), "{error}");

        // Active effort and temperature stay mutually exclusive; an explicit
        // `None` effort is not active.
        let generation = GenerationOptions {
            max_tokens: Some(64),
            temperature: Some(1.0),
            reasoning_effort: Some(ReasoningEffort::High),
            ..GenerationOptions::default()
        };
        assert_eq!(
            render_with(&generation, &AnthropicRenderOptions::default())
                .unwrap_err()
                .kind,
            ModelInvokeErrorKind::InvalidRequest
        );
        let generation = GenerationOptions {
            max_tokens: Some(64),
            temperature: Some(1.0),
            reasoning_effort: Some(ReasoningEffort::None),
            ..GenerationOptions::default()
        };
        assert!(render_with(&generation, &AnthropicRenderOptions::default()).is_ok());
    }

    #[test]
    fn prefix_mismatch_behavior_renders_only_with_a_thinking_object() {
        let options = AnthropicRenderOptions {
            prefix_mismatch_behavior: Some(PrefixMismatchBehavior::DropBlock),
        };
        // Effort unset: there is no thinking object to bind, so nothing renders.
        let body = render_with(&test_generation(), &options).unwrap();
        assert!(body.get("thinking").is_none());
        assert!(body.get("output_config").is_none());

        let generation = GenerationOptions {
            max_tokens: Some(64),
            reasoning_effort: Some(ReasoningEffort::High),
            ..GenerationOptions::default()
        };
        let body = render_with(&generation, &options).unwrap();
        assert_eq!(
            body["thinking"],
            json!({
                "type": "adaptive",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            })
        );
        // The default renders no block_binding at all.
        let body = render_with(&generation, &AnthropicRenderOptions::default()).unwrap();
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    }
    #[test]
    fn empty_frame_is_invalid_request() {
        let f = frame(vec![]);
        let e = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &test_generation(),
            &ModelRef::new("m"),
            CacheDirective::None,
            &AnthropicRenderOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::InvalidRequest));
    }

    #[test]
    fn cache_none_omits_every_cache_control_key() {
        let f = frame(vec![
            text(0, "be terse", Some("system")),
            text(1, "hi", Some("user")),
        ]);
        let surface = ToolSurface::from_definitions(vec![ToolDefinition {
            name: "read".into(),
            description: "read a file".into(),
            parameters: json!({"type": "object"}),
        }]);
        let v = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &surface,
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
            text(0, "be terse", Some("system")),
            text(1, "hi", Some("user")),
            call(2, "kc1", Some("toolu_a"), "read", json!({"path": "a"})),
            result(3, "kc1", ToolResultStatus::Succeeded, json!("file-a")),
        ]);
        let surface = ToolSurface::from_definitions(vec![ToolDefinition {
            name: "read".into(),
            description: "read a file".into(),
            parameters: json!({"type": "object"}),
        }]);
        let v = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &surface,
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
        let f = frame(vec![text(0, "hi", Some("user"))]);
        let v = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
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
            Some("user"),
        )]);
        let v = render_anthropic_messages(
            &f,
            &resolved("asset-1"),
            &ToolSurface::empty(),
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
            Some("user"),
        )]);
        let v = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
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
            Some("user"),
        )]);
        let mut media = MediaSet::new();
        media.insert("doc-1", MediaPayload::new("application/pdf", "AAAA"));
        let v = render_anthropic_messages(
            &f,
            &media,
            &ToolSurface::empty(),
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
            parts(0, vec![media_part("image/png", "asset-1")], Some("system")),
            text(1, "hi", Some("user")),
        ]);
        let v = render_anthropic_messages(
            &f,
            &resolved("asset-1"),
            &ToolSurface::empty(),
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
            &ToolSurface::empty(),
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
            &ToolSurface::empty(),
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
            text(0, "hi", Some("user")),
            parts(
                1,
                vec![text_part("see"), media_part("image/png", "asset-1")],
                Some("user"),
            ),
        ]);
        let v = render_anthropic_messages(
            &f,
            &resolved("asset-1"),
            &ToolSurface::empty(),
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
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Undecodable));
        assert!(e.message.contains("stop_reason"));
        // Present but not a string is reported as such, not as missing.
        let e = parse_anthropic_response(&json!({"content": [], "stop_reason": 7})).unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Undecodable));
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
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Undecodable));
        assert!(e.message.contains("name"));
    }
}
