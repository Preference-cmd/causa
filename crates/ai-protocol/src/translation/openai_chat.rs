//! OpenAI Chat Completions translation for the context kernel.
//!
//! Kernel-native translation face (Slice 3): pure functions from
//! [`reimagine_context_kernel::ContextFrame`] to a Chat Completions request
//! body and back to [`reimagine_context_kernel::ModelOutput`].
//! Transport-free; the reqwest adapter in `reimagine-agent-provider` owns
//! HTTP.
//!
//! The renderer-independent policy (source vocabulary, empty-text skip,
//! text joining, tool id pairing, observation stringification) lives in
//! [`super::context_frame`]; this module only shapes the chat wire.
//!
//! # Chat-specific structural rules
//!
//! - `system` text segments stay in frame position as `role: "system"`
//!   messages (OpenAI carries system inline, unlike Anthropic's
//!   top-level parameter).
//! - A run of assistant text segments and tool call segments coalesces
//!   into ONE assistant message. The wire carries assistant content as a
//!   single string, so text around tool calls joins in frame order; a
//!   run with no text omits `content`, a run with no calls omits
//!   `tool_calls`.
//! - Each tool result segment renders as its own `role: "tool"` message
//!   — `tool_call_id` pairing requires one message per call. There is no
//!   `is_error` flag on this wire; error information travels in the
//!   content.
//! - Tool call arguments are a JSON *string* on the wire (encoded by the
//!   emitter, decoded by the shared [`super::context_frame`] codec);
//!   parsing rejects a non-JSON arguments string as `Permanent`.
//! - `max_tokens` is optional for OpenAI (no default injected).

use serde_json::{Value, json};

use reimagine_context_kernel::{
    ContextFrame, GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRef,
    ModelResponse, ModelStopReason, ReasoningPayload, TextPayload, ToolCallDraft, ToolSurface,
};

use super::context_frame::{self, Role, Segment};

/// A run of assistant text + tool call segments coalescing into one
/// assistant message.
struct AssistantRun {
    text: String,
    calls: Vec<Value>,
}

/// Render a [`ContextFrame`] into an OpenAI Chat Completions request body.
/// Deterministic: identical inputs produce byte-identical JSON.
pub fn render_openai_chat_messages(
    frame: &ContextFrame,
    tool_surface: &ToolSurface,
    generation: &GenerationOptions,
    model: &ModelRef,
) -> Result<Value, ModelInvokeError> {
    let normalized = context_frame::normalize(frame);

    // A run of assistant text + tool call segments coalesces into one
    // assistant message; any other segment closes the open run.
    let mut messages: Vec<Value> = Vec::new();
    let mut run: Option<AssistantRun> = None;
    for segment in &normalized.segments {
        let closes_run = !matches!(
            segment,
            Segment::Text {
                role: Role::Assistant,
                ..
            } | Segment::ToolCall(_)
        );
        if closes_run && let Some(finished) = run.take() {
            messages.push(assistant_message(finished));
        }
        match segment {
            Segment::Text {
                role: Role::System,
                text,
            } => {
                messages.push(json!({"role": "system", "content": text}));
            }
            Segment::Text {
                role: Role::User,
                text,
            } => {
                messages.push(json!({"role": "user", "content": text}));
            }
            Segment::Text {
                role: Role::Assistant,
                text,
            } => match &mut run {
                Some(state) => {
                    state.text.push('\n');
                    state.text.push_str(text);
                }
                None => {
                    run = Some(AssistantRun {
                        text: text.clone(),
                        calls: Vec::new(),
                    })
                }
            },
            Segment::ToolCall(call) => {
                let wire_call = json!({
                    "id": call.wire_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments.to_string(),
                    },
                });
                match &mut run {
                    Some(state) => state.calls.push(wire_call),
                    None => {
                        run = Some(AssistantRun {
                            text: String::new(),
                            calls: vec![wire_call],
                        })
                    }
                }
            }
            Segment::ToolResult {
                wire_id, content, ..
            } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": wire_id,
                    "content": content,
                }));
            }
        }
    }
    if let Some(finished) = run.take() {
        messages.push(assistant_message(finished));
    }

    if messages.is_empty() {
        return Err(ModelInvokeError::new(
            ModelInvokeErrorKind::InvalidRequest,
            "frame rendered to zero messages; nothing to send",
        ));
    }

    let mut body = json!({
        "model": model.0,
        "messages": messages,
    });
    if let Some(temperature) = generation.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = generation.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    if !tool_surface.definitions.is_empty() {
        body["tools"] = json!(
            tool_surface
                .definitions
                .iter()
                .map(|d| json!({
                    "type": "function",
                    "function": {
                        "name": d.name,
                        "description": d.description,
                        "parameters": d.parameters,
                    },
                }))
                .collect::<Vec<_>>()
        );
    }
    Ok(body)
}

/// Materialize an assistant run: empty text is omitted (a calls-only
/// message carries no `content`), a text-only message carries no
/// `tool_calls` key.
fn assistant_message(run: AssistantRun) -> Value {
    let mut message = json!({"role": "assistant"});
    if !run.text.is_empty() {
        message["content"] = json!(run.text);
    }
    if !run.calls.is_empty() {
        message["tool_calls"] = json!(run.calls);
    }
    message
}

/// Parse an OpenAI Chat Completions response body into a kernel
/// [`ModelOutput`].
///
/// Unknown `finish_reason` values degrade to [`ModelStopReason::EndTurn`]
/// rather than inventing an interruption the provider did not report;
/// `reasoning_content` (the OpenAI-compatible reasoning extension) maps to
/// the reasoning envelope with no signature.
pub fn parse_openai_chat_response(value: &Value) -> Result<ModelOutput, ModelInvokeError> {
    fn permanent(message: impl Into<String>) -> ModelInvokeError {
        ModelInvokeError::new(ModelInvokeErrorKind::Permanent, message)
    }

    let choice = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or_else(|| permanent("missing choices[0]"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| permanent("missing choices[0].message"))?;

    let text = match message.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(_) => return Err(permanent("choices[0].message.content: expected a string")),
    };

    let mut tool_calls = Vec::new();
    if let Some(wire_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in wire_calls {
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| permanent("tool_calls[]: missing `id` string"))?;
            let function = call
                .get("function")
                .ok_or_else(|| permanent("tool_calls[]: missing `function`"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| permanent("tool_calls[].function: missing `name` string"))?;
            // The wire carries arguments as a JSON string; an empty string
            // degrades to an empty object (shared OpenAI-family codec).
            let arguments = context_frame::decode_wire_arguments(function.get("arguments"))?;
            tool_calls.push(ToolCallDraft {
                tool_name: name.to_string(),
                arguments,
                provider_call_id: Some(id.to_string()),
            });
        }
    }

    let stop_reason = match choice.get("finish_reason") {
        Some(Value::String(s)) => match s.as_str() {
            "stop" => ModelStopReason::EndTurn,
            "tool_calls" => ModelStopReason::ToolUse,
            "length" => ModelStopReason::MaxTokens,
            "content_filter" => ModelStopReason::Refusal,
            _ => ModelStopReason::EndTurn,
        },
        None | Some(Value::Null) => return Err(permanent("missing choices[0].finish_reason")),
        Some(_) => return Err(permanent("choices[0].finish_reason: expected a string")),
    };

    let reasoning = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .map(|text| ReasoningPayload {
            text: text.to_string(),
            signature: None,
        });

    let usage = value
        .get("usage")
        .filter(|v| v.is_object())
        .map(crate::translation::usage::model_usage_from_openai_chat);

    Ok(ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls,
        },
        usage,
        stop_reason,
        reasoning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_context_kernel::{
        BlockContent, BlockId, BlockMeta, BlockSequence, ContextBlock, ContextVersion, FrameId,
        FrameScope, ModelContext, ModelUsage, RoundId, ToolCallId, ToolCallPayload, ToolDefinition,
        ToolOutput, ToolResultPayload, ToolResultStatus, TurnId,
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

    fn frame(blocks: Vec<ContextBlock>) -> ContextFrame {
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
        render_openai_chat_messages(
            frame,
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
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
            text(5, "unknown tag", Some("provider:gpt-x")),
        ]);
        let v = render(&f);
        let msgs = v["messages"].as_array().unwrap();
        // system stays inline as a message; assistant(None) / user /
        // assistant("assistant") / user("inject:note" + unknown merged)
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0], json!({"role": "system", "content": "be terse"}));
        assert_eq!(
            msgs[1],
            json!({"role": "assistant", "content": "model said"})
        );
        assert_eq!(msgs[2], json!({"role": "user", "content": "user said"}));
        assert_eq!(msgs[3], json!({"role": "assistant", "content": "replayed"}));
        assert_eq!(
            msgs[4],
            json!({"role": "user", "content": "injected\nunknown tag"})
        );
    }

    #[test]
    fn tool_round_trip_pairing_and_assistant_merge() {
        let f = frame(vec![
            text(0, "reading now", None),
            call(1, "kc1", Some("toolu_a"), "read", json!({"path": "a"})),
            call(2, "kc2", None, "list", json!({})),
            result(3, "kc1", ToolResultStatus::Succeeded, json!("file-a")),
            result(4, "kc2", ToolResultStatus::Failed, json!({"error": "boom"})),
        ]);
        let v = render(&f);
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        // text + both calls merge into ONE assistant message; arguments
        // serialize to a JSON string; id falls back to the kernel call_id
        assert_eq!(
            msgs[0],
            json!({
                "role": "assistant",
                "content": "reading now",
                "tool_calls": [
                    {"id": "toolu_a", "type": "function",
                     "function": {"name": "read", "arguments": "{\"path\":\"a\"}"}},
                    {"id": "kc2", "type": "function",
                     "function": {"name": "list", "arguments": "{}"}},
                ],
            })
        );
        // each result is its own role:"tool" message with paired ids
        assert_eq!(
            msgs[1],
            json!({"role": "tool", "tool_call_id": "toolu_a", "content": "file-a"})
        );
        assert_eq!(
            msgs[2],
            json!({"role": "tool", "tool_call_id": "kc2", "content": "{\"error\":\"boom\"}"})
        );
    }

    #[test]
    fn assistant_tool_calls_only_omits_empty_content() {
        let f = frame(vec![call(0, "kc1", Some("toolu_a"), "read", json!({}))]);
        let v = render(&f);
        assert_eq!(
            v["messages"][0],
            json!({
                "role": "assistant",
                "tool_calls": [
                    {"id": "toolu_a", "type": "function",
                     "function": {"name": "read", "arguments": "{}"}},
                ],
            })
        );
        assert!(v["messages"][0].get("content").is_none());
    }

    #[test]
    fn adjacent_assistant_texts_join_into_one_message() {
        // P1-1: assistant text runs coalesce across calls too — the wire
        // carries assistant content as a single string.
        let f = frame(vec![
            text(0, "first", None),
            call(1, "kc1", Some("toolu_a"), "read", json!({})),
            text(2, "second", None),
        ]);
        let v = render(&f);
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"], json!("first\nsecond"));
        assert_eq!(msgs[0]["tool_calls"].as_array().unwrap().len(), 1);

        // no calls in between: still one message, text joined
        let f = frame(vec![text(0, "a", None), text(1, "b", None)]);
        let v = render(&f);
        assert_eq!(
            v["messages"],
            json!([{"role": "assistant", "content": "a\nb"}])
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
        assert_eq!(v["messages"][0]["tool_call_id"], json!("orphan"));
    }

    #[test]
    fn tools_generation_and_byte_determinism() {
        let surface = ToolSurface::from_definitions(vec![ToolDefinition {
            name: "read".into(),
            description: "read a file".into(),
            parameters: json!({"type": "object"}),
        }]);
        let f = frame(vec![text(0, "hi", Some("user"))]);

        let generation = GenerationOptions {
            temperature: Some(0.5),
            max_tokens: None,
        };
        let v = render_openai_chat_messages(&f, &surface, &generation, &ModelRef::new("gpt-test"))
            .unwrap();
        assert_eq!(v["model"], json!("gpt-test"));
        assert_eq!(v["temperature"], json!(0.5));
        // OpenAI does not require max_tokens; None stays absent
        assert!(v.get("max_tokens").is_none());
        assert_eq!(
            v["tools"],
            json!([{
                "type": "function",
                "function": {"name": "read", "description": "read a file", "parameters": {"type": "object"}},
            }])
        );

        let generation = GenerationOptions {
            temperature: Some(0.5),
            max_tokens: Some(100),
        };
        let v2 = render_openai_chat_messages(&f, &surface, &generation, &ModelRef::new("gpt-test"))
            .unwrap();
        assert_eq!(v2["max_tokens"], json!(100));
        let again =
            render_openai_chat_messages(&f, &surface, &generation, &ModelRef::new("gpt-test"))
                .unwrap();
        assert_eq!(
            serde_json::to_string(&v2).unwrap(),
            serde_json::to_string(&again).unwrap()
        );
    }

    #[test]
    fn empty_frame_is_invalid_request() {
        let e = render_openai_chat_messages(
            &frame(vec![]),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("m"),
        )
        .unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::InvalidRequest));
    }

    // --- parsing -----------------------------------------------------------

    #[test]
    fn parse_full_message_shape_with_provider_id_passthrough() {
        let v = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{\"path\": \"a\"}"},
                    }],
                    "reasoning_content": "hmm",
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 11},
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        });
        let out = parse_openai_chat_response(&v).unwrap();
        assert_eq!(out.response.text.0, "Let me check.");
        assert_eq!(out.response.tool_calls.len(), 1);
        let tc = &out.response.tool_calls[0];
        assert_eq!(tc.tool_name, "read");
        assert_eq!(tc.arguments, json!({"path": "a"}));
        assert_eq!(tc.provider_call_id.as_deref(), Some("call_9"));
        assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
        let reasoning = out.reasoning.unwrap();
        assert_eq!(reasoning.text, "hmm");
        assert_eq!(reasoning.signature, None);
        let u: ModelUsage = out.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (100, 20));
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.cache_write_tokens, None);
        assert_eq!(u.reasoning_tokens, Some(5));
    }

    #[test]
    fn finish_reason_mapping_full_and_unknown_degrades_to_end_turn() {
        for (wire, expected) in [
            ("stop", ModelStopReason::EndTurn),
            ("tool_calls", ModelStopReason::ToolUse),
            ("length", ModelStopReason::MaxTokens),
            ("content_filter", ModelStopReason::Refusal),
            ("some_new_reason", ModelStopReason::EndTurn),
        ] {
            let v = json!({"choices": [{"message": {"content": ""}, "finish_reason": wire}]});
            let out = parse_openai_chat_response(&v).unwrap();
            assert_eq!(
                std::mem::discriminant(&out.stop_reason),
                std::mem::discriminant(&expected),
                "finish_reason = {wire}"
            );
        }
    }

    #[test]
    fn missing_choices_or_finish_reason_is_permanent_error() {
        let e = parse_openai_chat_response(&json!({"choices": []})).unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Permanent));
        let e = parse_openai_chat_response(&json!({"choices": [{"message": {}}]})).unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Permanent));
        assert!(e.message.contains("finish_reason"));
    }

    #[test]
    fn malformed_arguments_string_is_permanent_error() {
        let v = json!({
            "choices": [{
                "message": {"content": null, "tool_calls": [
                    {"id": "c1", "function": {"name": "read", "arguments": "{not json"}},
                ]},
                "finish_reason": "tool_calls",
            }],
        });
        let e = parse_openai_chat_response(&v).unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Permanent));
        assert!(e.message.contains("arguments"));
    }

    #[test]
    fn empty_arguments_string_degrades_to_empty_object() {
        let v = json!({
            "choices": [{
                "message": {"tool_calls": [
                    {"id": "c1", "function": {"name": "list", "arguments": ""}},
                ]},
                "finish_reason": "tool_calls",
            }],
        });
        let out = parse_openai_chat_response(&v).unwrap();
        assert_eq!(out.response.tool_calls[0].arguments, json!({}));
    }
}
