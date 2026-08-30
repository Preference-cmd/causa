//! Anthropic Messages translation for the context kernel.
//!
//! This is the kernel-native translation face (Slice 3): pure functions
//! from [`reimagine_context_kernel::ContextFrame`] to an Anthropic
//! Messages request body, and from an Anthropic Messages response body
//! back to [`reimagine_context_kernel::ModelOutput`]. Transport-free —
//! the reqwest adapter in `reimagine-agent-provider` owns HTTP.
//!
//! # `BlockMeta::source` vocabulary
//!
//! The kernel records `source` verbatim and never interprets it; this
//! module is its interpreter. The vocabulary is open — unknown tags fall
//! back to the user role rather than failing, because renderers must be
//! total over frames that outlive any single vocabulary revision.
//!
//! | `BlockMeta::source`      | Rendering semantics                                  |
//! |--------------------------|------------------------------------------------------|
//! | `None`                   | assistant — the model door never stamps `source`      |
//! | `"assistant"`            | assistant — host-prepared model turn (history replay) |
//! | `"system"`               | top-level `system` parameter (joined with `\n`)       |
//! | `"user"`                 | user role                                             |
//! | `"inject"` / `"inject:…"`| user role (rendering policy for injected context)     |
//! | anything else            | user role (documented open-vocabulary fallback)       |
//!
//! # Structural rules
//!
//! - Consecutive blocks with the same wire role merge into one message;
//!   Anthropic requires strictly alternating `user` / `assistant` roles.
//! - Tool call blocks render as assistant `tool_use` content blocks. The
//!   wire id is `meta.provider_call_id`, falling back to the kernel
//!   `call_id` for synthetic calls the provider never named.
//! - Tool result blocks render as `tool_result` content blocks in the
//!   following user message, with `tool_use_id` resolved through the
//!   `call_id → provider id` map built from the frame's call blocks. An
//!   unpaired result (its call block was dropped, e.g. by compaction)
//!   falls back to the kernel `call_id`; the provider rejects the orphan
//!   at HTTP time, which is the loud failure path. Any status other than
//!   `Succeeded` sets `is_error: true`. Result `content` passes through
//!   as a string when the observation is a JSON string; any other JSON
//!   value is serialized to a string.
//! - `GenerationOptions::max_tokens` is required by Anthropic; a `None`
//!   renders as [`DEFAULT_MAX_TOKENS`].
//! - `reasoning` is parsed as a wire envelope only. The kernel does not
//!   persist reasoning as facts, so cross-turn thinking replay is out of
//!   scope here (the adapter sees fact-layer blocks each round).
//! - `redacted_thinking` and unknown content block types are skipped for
//!   forward compatibility with provider extensions.

use std::collections::HashMap;

use serde_json::{Value, json};

use reimagine_context_kernel::{
    BlockContent, ContextFrame, GenerationOptions, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRef, ModelResponse, ModelStopReason, ReasoningPayload, TextPayload,
    ToolCallDraft, ToolResultStatus, ToolSurface,
};

/// Anthropic requires `max_tokens`; this is the documented default when
/// `GenerationOptions::max_tokens` is `None`.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Render a [`ContextFrame`] into an Anthropic Messages request body.
///
/// The body is complete: `model`, `max_tokens`, `messages`, plus `system`,
/// `temperature`, and `tools` when the inputs call for them. Rendering is
/// deterministic — the same frame, surface, generation, and model always
/// produce byte-identical JSON.
pub fn render_anthropic_messages(
    frame: &ContextFrame,
    tool_surface: &ToolSurface,
    generation: &GenerationOptions,
    model: &ModelRef,
) -> Result<Value, ModelInvokeError> {
    let blocks = &frame.model_context.blocks;

    // Pairing map: kernel call_id -> the id the provider saw on the
    // tool_use block. Synthetic calls (no provider id) render with the
    // kernel call_id, so the map's fallback is the same rule applied at
    // result-render time.
    let mut provider_ids: HashMap<String, String> = HashMap::new();
    let mut system_parts: Vec<String> = Vec::new();
    for block in blocks {
        match &block.content {
            BlockContent::ToolCall(call) => {
                provider_ids.insert(
                    call.call_id.0.clone(),
                    block
                        .meta
                        .provider_call_id
                        .clone()
                        .unwrap_or_else(|| call.call_id.0.clone()),
                );
            }
            BlockContent::Text(text) if block.meta.source.as_deref() == Some("system") => {
                system_parts.push(text.0.clone());
            }
            _ => {}
        }
    }

    // Assemble messages: consecutive same-role blocks merge into one
    // message (Anthropic requires strictly alternating roles).
    let mut messages: Vec<(&'static str, Vec<Value>)> = Vec::new();
    for block in blocks {
        match &block.content {
            // Already extracted into the top-level system parameter.
            BlockContent::Text(_) if block.meta.source.as_deref() == Some("system") => {}
            BlockContent::Text(text) => {
                let role = match block.meta.source.as_deref() {
                    None | Some("assistant") => "assistant",
                    // "user", "inject[:detail]", unknown open-vocabulary
                    // tags — all render as user.
                    Some(_) => "user",
                };
                append_message(&mut messages, role, json!({"type": "text", "text": text.0}));
            }
            BlockContent::ToolCall(call) => {
                let id = block
                    .meta
                    .provider_call_id
                    .clone()
                    .unwrap_or_else(|| call.call_id.0.clone());
                append_message(
                    &mut messages,
                    "assistant",
                    json!({
                        "type": "tool_use",
                        "id": id,
                        "name": call.tool_name,
                        "input": call.arguments,
                    }),
                );
            }
            BlockContent::ToolResult(result) => {
                let tool_use_id = provider_ids
                    .get(&result.call_id.0)
                    .cloned()
                    .unwrap_or_else(|| result.call_id.0.clone());
                let content = match &result.output.content {
                    Value::String(s) => Value::String(s.clone()),
                    other => Value::String(other.to_string()),
                };
                let mut block_json = json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                });
                if result.status != ToolResultStatus::Succeeded {
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
        body["system"] = json!(system_parts.join("\n"));
    }
    if !tool_surface.definitions.is_empty() {
        body["tools"] = json!(
            tool_surface
                .definitions
                .iter()
                .map(|d| json!({
                    "name": d.name,
                    "description": d.description,
                    "input_schema": d.parameters,
                }))
                .collect::<Vec<_>>()
        );
    }
    Ok(body)
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

    let usage = value.get("usage").and_then(Value::as_object).map(|u| {
        reimagine_context_kernel::ModelUsage {
            input_tokens: u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0) as usize,
            output_tokens: u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0) as usize,
            cache_read_tokens: u
                .get("cache_read_input_tokens")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            cache_write_tokens: u
                .get("cache_creation_input_tokens")
                .and_then(Value::as_u64)
                .map(|v| v as usize),
            // Anthropic does not expose reasoning token counts.
            reasoning_tokens: None,
        }
    });

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
    use reimagine_context_kernel::{
        BlockId, BlockMeta, BlockSequence, ContextBlock, ContextVersion, ConversationId, FrameId,
        FrameScope, ModelContext, ModelUsage, RoundId, ToolCallId, ToolCallPayload, ToolDefinition,
        ToolOutput, ToolResultPayload, TurnId,
    };

    fn block(
        seq: u64,
        content: BlockContent,
        source: Option<&str>,
        provider_call_id: Option<&str>,
    ) -> ContextBlock {
        ContextBlock {
            id: BlockId {
                turn_id: TurnId::new("t1"),
                sequence: BlockSequence(seq),
            },
            sequence: BlockSequence(seq),
            content,
            meta: BlockMeta {
                provider_call_id: provider_call_id.map(String::from),
                source: source.map(String::from),
            },
        }
    }

    fn text(seq: u64, text: &str, source: Option<&str>) -> ContextBlock {
        block(
            seq,
            BlockContent::Text(TextPayload::new(text)),
            source,
            None,
        )
    }

    fn call(
        seq: u64,
        call_id: &str,
        provider: Option<&str>,
        name: &str,
        arguments: Value,
    ) -> ContextBlock {
        block(
            seq,
            BlockContent::ToolCall(ToolCallPayload {
                call_id: ToolCallId::new(call_id),
                tool_name: name.into(),
                arguments,
            }),
            None,
            provider,
        )
    }

    fn result(seq: u64, call_id: &str, status: ToolResultStatus, content: Value) -> ContextBlock {
        block(
            seq,
            BlockContent::ToolResult(ToolResultPayload {
                call_id: ToolCallId::new(call_id),
                status,
                output: ToolOutput::new(content),
            }),
            None,
            None,
        )
    }

    fn turn_frame(blocks: Vec<ContextBlock>) -> ContextFrame {
        let scope = FrameScope::Turn {
            turn_id: TurnId::new("t1"),
            source_version: ContextVersion(3),
        };
        ContextFrame {
            frame_id: FrameId::from_scope(&scope, RoundId(0)),
            scope,
            round_id: RoundId(0),
            model_context: ModelContext { blocks },
        }
    }

    fn render(frame: &ContextFrame) -> Value {
        render_anthropic_messages(
            frame,
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("claude-test"),
        )
        .unwrap()
    }

    // --- rendering ---------------------------------------------------------

    #[test]
    fn source_vocabulary_full_branch_coverage() {
        let f = turn_frame(vec![
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
    fn tool_round_trip_pairing_and_consecutive_merge() {
        let f = turn_frame(vec![
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
        let f = turn_frame(vec![result(
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
        let f = turn_frame(vec![text(0, "hi", Some("user"))]);

        let generation = GenerationOptions {
            temperature: Some(0.5),
            max_tokens: None,
        };
        let v = render_anthropic_messages(&f, &surface, &generation, &ModelRef::new("claude-test"))
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
        };
        let v2 =
            render_anthropic_messages(&f, &surface, &generation, &ModelRef::new("claude-test"))
                .unwrap();
        assert_eq!(v2["max_tokens"], json!(100));
        // byte determinism over the full body
        let again =
            render_anthropic_messages(&f, &surface, &generation, &ModelRef::new("claude-test"))
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
        let f = turn_frame(vec![]);
        let e = render_anthropic_messages(
            &f,
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("m"),
        )
        .unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::InvalidRequest));
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
