//! Anthropic Messages translation for the context kernel.
//!
//! This is the kernel-native translation face (Slice 3): pure functions
//! from [`causa_kernel::ContextFrame`] to an Anthropic
//! Messages request body, and from an Anthropic Messages response body
//! back to [`causa_kernel::ModelOutput`]. Transport-free —
//! the reqwest adapter in `causa-provider` owns HTTP.
//!
//! The renderer-independent policy (source vocabulary, empty-text skip,
//! text joining, tool id pairing, observation stringification) lives in
//! [`super::context_frame`]; this module only shapes the Anthropic wire.
//!
//! # Anthropic-specific structural rules
//!
//! - `system` text segments move to the top-level `system` parameter
//!   (joined with `\n`); Anthropic has no system message role. Media in
//!   a system-role block degrades to its placeholder text there — the
//!   system parameter carries no image blocks.
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
//!   renders as [`DEFAULT_MAX_TOKENS`].
//! - [`CacheDirective::StablePrefix`] marks three `cache_control`
//!   breakpoints: the last tool definition, the `system` parameter
//!   (rendered as a block array so the anchor can attach — Anthropic
//!   cannot hang the marker off a string parameter), and the latest
//!   stable conversation message (the previous message when the final
//!   one carries this round's dispatched `tool_result` blocks, the
//!   final message otherwise). [`CacheDirective::None`] renders
//!   byte-identically to a body without any cache key.
//! - `reasoning` is parsed as a wire envelope only. The kernel does not
//!   persist reasoning as facts, so cross-turn thinking replay is out of
//!   scope here (the adapter sees fact-layer blocks each round).
//! - `redacted_thinking` and unknown content block types are skipped for
//!   forward compatibility with provider extensions.

use serde_json::{Value, json};

use causa_kernel::{
    CacheDirective, ContextFrame, GenerationOptions, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRef, ModelResponse, ModelStopReason, ReasoningPayload, TextPayload,
    ToolCallDraft, ToolResultStatus, ToolSurface,
};

use super::context_frame::{self, ResolvedMedia, Role, Segment};
use super::media::MediaSet;

/// Anthropic requires `max_tokens`; this is the documented default when
/// `GenerationOptions::max_tokens` is `None`.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Render a [`ContextFrame`] into an Anthropic Messages request body.
///
/// The body is complete: `model`, `max_tokens`, `messages`, plus `system`,
/// `temperature`, and `tools` when the inputs call for them. Media
/// references resolve through `media`; unresolvable or non-image
/// references degrade to the shared text placeholder. Rendering is
/// deterministic — the same frame, surface, generation, model, media
/// table, and cache directive always produce byte-identical JSON.
pub fn render_anthropic_messages(
    frame: &ContextFrame,
    media: &MediaSet,
    tool_surface: &ToolSurface,
    generation: &GenerationOptions,
    model: &ModelRef,
    cache: CacheDirective,
) -> Result<Value, ModelInvokeError> {
    let normalized = context_frame::normalize(frame, media);

    // Group consecutive same-wire-role segments into one message
    // (Anthropic requires strictly alternating roles).
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<(&'static str, Vec<Value>)> = Vec::new();
    for segment in &normalized.segments {
        match segment {
            Segment::Text {
                role: Role::System,
                text,
            } => system_parts.push(text.clone()),
            Segment::Text {
                role: Role::User,
                text,
            } => {
                append_message(&mut messages, "user", json!({"type": "text", "text": text}));
            }
            Segment::Text {
                role: Role::Assistant,
                text,
            } => {
                append_message(
                    &mut messages,
                    "assistant",
                    json!({"type": "text", "text": text}),
                );
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
                    // System media degraded in the walk; its placeholder
                    // joins the system parameter like any system text.
                    if *role == Role::System {
                        system_parts.push(text.clone());
                    } else {
                        append_message(
                            &mut messages,
                            role_name(*role),
                            json!({"type": "text", "text": text}),
                        );
                    }
                }
            },
            Segment::ToolCall(call) => append_message(
                &mut messages,
                "assistant",
                json!({
                    "type": "tool_use",
                    "id": call.wire_id,
                    "name": call.name,
                    "input": call.arguments,
                }),
            ),
            Segment::ToolResult {
                wire_id,
                status,
                content,
                media,
            } => {
                let content_json = if media.is_empty() {
                    json!(content)
                } else {
                    let mut blocks = vec![json!({"type": "text", "text": content})];
                    blocks.extend(media.iter().map(media_block_json));
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

    if messages.is_empty() {
        return Err(ModelInvokeError::new(
            ModelInvokeErrorKind::InvalidRequest,
            "frame rendered to zero messages; nothing to send",
        ));
    }

    // Stable-prefix anchor: the frame is append-only, so every rendered
    // message is byte-stable across rounds and retries; the anchor skips
    // the final message only when it carries this round's dispatched
    // tool results.
    if matches!(cache, CacheDirective::StablePrefix)
        && let Some(anchor_blocks) = stable_prefix_message(&mut messages)
        && let Some(last_block) = anchor_blocks.last_mut()
    {
        last_block["cache_control"] = json!({"type": "ephemeral"});
    }

    let mut body = json!({
        "model": model.0,
        "max_tokens": generation.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": messages
            .into_iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect::<Vec<_>>(),
    });
    if let Some(temperature) = generation.temperature {
        body["temperature"] = json!(temperature);
    }
    if !system_parts.is_empty() {
        if matches!(cache, CacheDirective::StablePrefix) {
            // A string parameter cannot carry the anchor; the block-array
            // form is the documented equivalent.
            body["system"] = json!([{
                "type": "text",
                "text": system_parts.join("\n"),
                "cache_control": {"type": "ephemeral"},
            }]);
        } else {
            body["system"] = json!(system_parts.join("\n"));
        }
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
    Ok(body)
}

/// The message the stable-prefix cache anchor lands on: the previous
/// message when the final one carries this round's dispatched
/// `tool_result` blocks, the final message otherwise. Anchoring the
/// message before the fresh dispatch keeps the cached prefix clear of
/// content the current round just appended.
fn stable_prefix_message<'a>(
    messages: &'a mut [(&'static str, Vec<Value>)],
) -> Option<&'a mut Vec<Value>> {
    let last = messages.last()?;
    let dispatching = last
        .1
        .iter()
        .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"));
    let index = if dispatching && messages.len() > 1 {
        messages.len() - 2
    } else {
        messages.len() - 1
    };
    Some(&mut messages[index].1)
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
/// `text` blocks join into the response text; `tool_use` blocks become
/// tool call drafts with `provider_call_id` carrying the provider id;
/// `thinking` blocks join into the reasoning envelope (last signature
/// wins). `stop_reason` is required; unknown values degrade to
/// [`ModelStopReason::EndTurn`] rather than inventing an interruption
/// the provider did not report.
pub fn parse_anthropic_response(value: &Value) -> Result<ModelOutput, ModelInvokeError> {
    fn permanent(message: impl Into<String>) -> ModelInvokeError {
        ModelInvokeError::new(ModelInvokeErrorKind::Permanent, message)
    }

    let empty = Vec::new();
    let content = match value.get("content") {
        None | Some(Value::Null) => &empty,
        Some(Value::Array(items)) => items,
        Some(_) => return Err(permanent("content: expected an array")),
    };

    let mut texts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCallDraft> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut signature: Option<String> = None;
    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| permanent("text block: missing `text` string"))?;
                texts.push(text.to_string());
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| permanent("tool_use block: missing `id` string"))?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| permanent("tool_use block: missing `name` string"))?;
                tool_calls.push(ToolCallDraft {
                    tool_name: name.to_string(),
                    arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
                    provider_call_id: Some(id.to_string()),
                });
            }
            Some("thinking") => {
                let text = block
                    .get("thinking")
                    .and_then(Value::as_str)
                    .ok_or_else(|| permanent("thinking block: missing `thinking` string"))?;
                thinking_parts.push(text.to_string());
                if let Some(sig) = block.get("signature").and_then(Value::as_str) {
                    signature = Some(sig.to_string());
                }
            }
            // redacted_thinking has no kernel representation; unknown
            // block types are skipped (forward compatibility).
            _ => {}
        }
    }

    let stop_reason = match value.get("stop_reason") {
        Some(Value::String(s)) => match s.as_str() {
            "end_turn" => ModelStopReason::EndTurn,
            "tool_use" => ModelStopReason::ToolUse,
            "max_tokens" => ModelStopReason::MaxTokens,
            "refusal" => ModelStopReason::Refusal,
            _ => ModelStopReason::EndTurn,
        },
        None | Some(Value::Null) => return Err(permanent("missing stop_reason")),
        Some(_) => return Err(permanent("stop_reason: expected a string")),
    };

    let usage = value
        .get("usage")
        .filter(|v| v.is_object())
        .map(crate::translation::usage::model_usage_from_anthropic);

    Ok(ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(texts.join("\n")),
            tool_calls,
        },
        usage,
        stop_reason,
        reasoning: if thinking_parts.is_empty() {
            None
        } else {
            Some(ReasoningPayload {
                text: thinking_parts.join("\n"),
                signature,
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

#[cfg(test)]
mod tests {
    use super::*;
    use causa_kernel::{
        ContextVersion, ConversationId, FrameId, FrameScope, MediaRef, ModelContext, ModelUsage,
        RoundId, ToolDefinition, TurnId,
    };

    use crate::translation::media::{MediaPayload, MediaSet};
    use crate::translation::test_support::{
        call, frame, media_part, parts, result, result_with_media, text, text_part,
    };

    fn render(frame: &ContextFrame) -> Value {
        render_anthropic_messages(
            frame,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
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
        assert_eq!(v["system"], json!("be terse"));
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
        // adjacent same-role texts join into one block (shared policy)
        assert_eq!(
            msgs[3]["content"],
            json!([{"type": "text", "text": "injected\nbare inject\nunknown tag"}])
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
        // is provider_call_id when present, kernel call_id otherwise
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
            json!({"type": "tool_use", "id": "kc2", "name": "list", "input": {}})
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
                "tool_use_id": "kc2",
                "content": "{\"error\":\"boom\"}",
                "is_error": true,
            })
        );
    }

    #[test]
    fn unpaired_tool_result_falls_back_to_kernel_call_id() {
        let f = frame(vec![result(
            0,
            "orphan",
            ToolResultStatus::Succeeded,
            json!("x"),
        )]);
        let v = render(&f);
        assert_eq!(
            v["messages"][0]["content"][0]["tool_use_id"],
            json!("orphan")
        );
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
            max_tokens: None,
            ..GenerationOptions::default()
        };
        let v = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &surface,
            &generation,
            &ModelRef::new("claude-test"),
            CacheDirective::None,
        )
        .unwrap();
        assert_eq!(v["model"], json!("claude-test"));
        assert_eq!(v["temperature"], json!(0.5));
        // Anthropic requires max_tokens; None renders as the default
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
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&v2).unwrap(),
            serde_json::to_string(&again).unwrap()
        );
    }

    #[test]
    fn conversation_scope_renders_same_body_as_turn_scope() {
        let blocks = vec![text(0, "hi", Some("user")), text(1, "hello", None)];
        let turn_scope = FrameScope::Turn {
            turn_id: TurnId::new("t1"),
            source_version: ContextVersion(1),
        };
        let conv_scope = FrameScope::Conversation {
            conversation_id: ConversationId("c1".into()),
            active_turn_id: TurnId::new("t2"),
            source_version: ContextVersion(2),
        };
        let turn = ContextFrame {
            frame_id: FrameId::from_scope(&turn_scope, RoundId(1)),
            scope: turn_scope,
            round_id: RoundId(1),
            model_context: ModelContext {
                blocks: blocks.clone(),
            },
        };
        let conv = ContextFrame {
            frame_id: FrameId::from_scope(&conv_scope, RoundId(1)),
            scope: conv_scope,
            round_id: RoundId(1),
            model_context: ModelContext { blocks },
        };
        assert_eq!(render(&turn), render(&conv));
    }

    #[test]
    fn empty_frame_is_invalid_request() {
        let f = frame(vec![]);
        let e = render_anthropic_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("m"),
            CacheDirective::None,
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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::StablePrefix,
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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::StablePrefix,
        )
        .unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert!(msgs[0]["content"][0].get("cache_control").is_some());
    }

    // --- media (Slice 6.5) --------------------------------------------------

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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
        )
        .unwrap();
        // the system parameter is text-only; the walk degraded it
        assert_eq!(v["system"], json!("[media: image/png asset-1]"));
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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
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
        let f = frame(vec![result(
            0,
            "kc1",
            ToolResultStatus::Succeeded,
            json!("ok"),
        )]);
        let v = render_anthropic_messages(
            &f,
            &resolved("unused"),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::None,
        )
        .unwrap();
        assert_eq!(v["messages"][0]["content"][0]["content"], json!("ok"));
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
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
            CacheDirective::StablePrefix,
        )
        .unwrap();
        // the second block's first text joins the seam (block boundaries
        // are envelope-only), so both user blocks render as one message:
        // [text "hi\nsee", image]
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(
            msgs[0]["content"][0],
            json!({"type": "text", "text": "hi\nsee"})
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
        assert_eq!(out.response.text.0, "Let me check.");
        assert_eq!(out.response.tool_calls.len(), 1);
        let tc = &out.response.tool_calls[0];
        assert_eq!(tc.tool_name, "read");
        assert_eq!(tc.arguments, json!({"path": "a"}));
        assert_eq!(tc.provider_call_id.as_deref(), Some("toolu_9"));
        assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
        let reasoning = out.reasoning.unwrap();
        assert_eq!(reasoning.text, "hmm");
        assert_eq!(reasoning.signature.as_deref(), Some("sig1"));
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
    fn stop_reason_mapping_full_and_unknown_degrades_to_end_turn() {
        for (wire, expected) in [
            ("end_turn", ModelStopReason::EndTurn),
            ("tool_use", ModelStopReason::ToolUse),
            ("max_tokens", ModelStopReason::MaxTokens),
            ("refusal", ModelStopReason::Refusal),
            // unknown values degrade to EndTurn, not a fabricated interrupt
            ("model_context_overflow", ModelStopReason::EndTurn),
        ] {
            let v = json!({"content": [], "stop_reason": wire});
            let out = parse_anthropic_response(&v).unwrap();
            assert_eq!(
                std::mem::discriminant(&out.stop_reason),
                std::mem::discriminant(&expected),
                "wire stop_reason = {wire}"
            );
        }
    }

    #[test]
    fn missing_stop_reason_is_permanent_error() {
        let v = json!({"content": []});
        let e = parse_anthropic_response(&v).unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Permanent));
        assert!(e.message.contains("stop_reason"));
    }

    #[test]
    fn multiple_texts_join_and_exotic_blocks_are_skipped() {
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
        assert_eq!(out.response.text.0, "a\nb");
        assert!(out.response.tool_calls.is_empty());
        assert!(out.reasoning.is_none());
    }

    #[test]
    fn malformed_tool_use_is_permanent_error() {
        let v = json!({"content": [{"type": "tool_use", "id": "x"}], "stop_reason": "tool_use"});
        let e = parse_anthropic_response(&v).unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Permanent));
        assert!(e.message.contains("name"));
    }
}
