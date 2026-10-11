//! OpenAI Chat Completions translation for the context kernel.
//!
//! Pure functions from [`causa_kernel::ContextFrame`] to a Chat
//! Completions request body and back to [`causa_kernel::ModelOutput`].
//! Transport-free; the reqwest adapter in `causa-provider` owns HTTP.
//!
//! The renderer-independent policy (source vocabulary, empty-text skip,
//! no cross-block text join, tool id pairing, observation stringification) lives in
//! `super::context_frame`; this module only shapes the chat wire.
//!
//! # Chat-specific structural rules
//!
//! - `system` text segments stay in frame position as `role: "system"`
//!   messages (OpenAI carries system inline, unlike Anthropic's
//!   top-level parameter).
//! - A run of assistant text segments, tool call segments and own-format
//!   sealed parts coalesces into ONE assistant message (the exception to one
//!   item per block). The wire carries assistant content as a single string,
//!   so the run's texts concatenate in frame order with no separator. The
//!   message starts from the run's first sealed object (later ones are
//!   ignored), then `role`, `content` (only when non-empty) and
//!   `tool_calls` (only when non-empty) are written over it. A tool call
//!   with matching `wire` starts from its `fields` and writes `id`,
//!   `function.name` and `function.arguments`; the provider's arguments
//!   string is kept while it parses to the fact's value.
//! - Parsing a message emits, in order: one sealed item holding every
//!   message key other than `role`, `tool_calls` and a non-empty string
//!   `content` (`reasoning_content`, a null or empty `content`, unknown
//!   keys), when any remain; one text item for non-empty `content`; one
//!   tool-call item per call, with `wire.fields` the call minus `id` and
//!   `function.name`.
//! - Each tool result segment renders as its own `role: "tool"` message
//!   — `tool_call_id` pairing requires one message per call. There is no
//!   `is_error` flag on this wire; error information travels in the
//!   content.
//! - Tool-result media **hoists**: a tool message cannot carry images,
//!   so each result's attachments render as a user message immediately
//!   after the result group — one provenance text part naming the call, then the
//!   images (data-URL `image_url` parts); placeholders degrade to text
//!   parts in the same message.
//! - Media parts in user position render as `image_url` content-array
//!   messages (data URLs); assistant- and system-position media degrade
//!   to the shared placeholder text (the walk decides).
//! - Tool call arguments are a JSON *string* on the wire (encoded by the
//!   emitter, decoded by the shared `super::context_frame` codec);
//!   parsing rejects a non-JSON arguments string as `Permanent`.
//! - Sealed `reasoning_content` is replayed by default: the run's sealed
//!   object becomes the assistant message, so a thinking-mode tool loop under
//!   an API that requires the field back (DeepSeek) keeps working.
//!   [`ChatRenderOptions::replay_reasoning_content`] turns that off; with it
//!   off the renderer removes the `reasoning_content` key from its own sealed
//!   items — skipping an item left empty, like one of another format — and
//!   changes nothing else. Other sealed message fields always replay.
//! - Output limits use `max_completion_tokens`, including reasoning tokens;
//!   no default limit is injected. Effort renders as `reasoning_effort`.
//!   Active effort combined with temperature is rejected before transport.
//!   Model-specific effort and tool support must be validated by the host.

use serde_json::{Value, json};

use causa_kernel::{
    CacheDirective, ContextFrame, GenerationOptions, ModelInvokeError, ModelInvokeErrorKind,
    ModelItem, ModelOutput, ModelRef, ModelResponse, ReasoningPayload, ToolCallDraft, ToolSurface,
};

use super::context_frame::{self, ResolvedMedia, Role, Segment};
use super::media::MediaSet;
use super::render_options::ChatRenderOptions;
use crate::Protocol;

/// A run of assistant text + tool call segments coalescing into one
/// assistant message.
#[derive(Default)]
struct AssistantRun {
    text: String,
    calls: Vec<Value>,
    /// The first sealed message-level object of the run; later ones are
    /// ignored.
    sealed: Option<Value>,
}

/// Render a [`ContextFrame`] into an OpenAI Chat Completions request body.
/// Deterministic: identical inputs produce byte-identical JSON.
///
/// `cache` is accepted for interface parity and ignored: OpenAI-family
/// providers cache server-side, so the directive has no wire
/// representation on this protocol. `GenerationOptions::output_schema`
/// renders as `response_format` (structured output); schema validation
/// and corrective retry stay host-side. `options` carries the Chat-only
/// `reasoning_content` replay choice; see [`ChatRenderOptions`].
pub fn render_openai_chat_messages(
    frame: &ContextFrame,
    media: &MediaSet,
    tool_surface: &ToolSurface,
    generation: &GenerationOptions,
    model: &ModelRef,
    _cache: CacheDirective,
    options: &ChatRenderOptions,
) -> Result<Value, ModelInvokeError> {
    super::generation::validate_openai(generation)?;
    super::tool_history::validate_frame(frame)?;
    let normalized = context_frame::normalize(frame, media, Protocol::OpenAiChatCompletions);

    // A run of assistant text + tool call segments coalesces into one
    // assistant message; any other segment closes the open run.
    let mut messages: Vec<Value> = Vec::new();
    let mut run: Option<AssistantRun> = None;
    let mut result_media = Vec::new();
    for segment in &normalized.segments {
        // A whitespace-only assistant text part is omitted from the request
        // (as \`pi-ai\` does); the fact stays in the Context. This is a listed
        // exception to the round-trip property.
        if let Segment::Text {
            role: Role::Assistant,
            text,
            ..
        } = segment
            && context_frame::is_blank_text(text)
        {
            continue;
        }
        if !matches!(segment, Segment::ToolResult { .. }) {
            messages.append(&mut result_media);
        }
        let closes_run = !matches!(
            segment,
            Segment::Text {
                role: Role::Assistant,
                ..
            } | Segment::ToolCall(_)
                | Segment::Sealed(_)
        );
        if closes_run && let Some(finished) = run.take() {
            messages.extend(assistant_message(finished));
        }
        match segment {
            Segment::Text {
                role: Role::System,
                text,
                ..
            } => {
                messages.push(json!({"role": "system", "content": text}));
            }
            Segment::Text {
                role: Role::User,
                text,
                ..
            } => {
                messages.push(json!({"role": "user", "content": text}));
            }
            Segment::Media { role, media } => match (role, media) {
                (Role::System, ResolvedMedia::Placeholder(text)) => {
                    messages.push(json!({"role": "system", "content": text}));
                }
                (Role::Assistant, ResolvedMedia::Placeholder(text)) => {
                    // closes the open run, then a degraded assistant text
                    if let Some(finished) = run.take() {
                        messages.extend(assistant_message(finished));
                    }
                    messages.push(json!({"role": "assistant", "content": text}));
                }
                (
                    _,
                    ResolvedMedia::Image {
                        media_type,
                        data_base64,
                    },
                ) => {
                    messages.push(json!({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": data_url(media_type, data_base64)}}],
                    }));
                }
                (_, ResolvedMedia::Placeholder(text)) => {
                    messages.push(json!({"role": "user", "content": text}));
                }
            },
            Segment::Text {
                role: Role::Assistant,
                text,
                ..
            } => run
                .get_or_insert_with(AssistantRun::default)
                .text
                .push_str(text),
            Segment::Sealed(item) => {
                // With replay off, only `reasoning_content` is removed; other
                // sealed message fields still replay, and an item left empty
                // is skipped like one of another format.
                let item = if options.replay_reasoning_content {
                    Some(item.clone())
                } else {
                    without_reasoning_content(item)
                };
                if let Some(item) = item {
                    let state = run.get_or_insert_with(AssistantRun::default);
                    if state.sealed.is_none() {
                        state.sealed = Some(item);
                    }
                }
            }
            Segment::ToolCall(call) => {
                let mut wire_call = context_frame::residue_object(call.wire.as_ref());
                wire_call.entry("type").or_insert_with(|| json!("function"));
                wire_call.insert("id".into(), json!(call.wire_id));
                let mut function = match wire_call.remove("function") {
                    Some(Value::Object(function)) => function,
                    _ => serde_json::Map::new(),
                };
                function.insert("name".into(), json!(call.name));
                let arguments = context_frame::wire_arguments_string(
                    function.get("arguments"),
                    &call.arguments,
                );
                function.insert("arguments".into(), json!(arguments));
                wire_call.insert("function".into(), Value::Object(function));
                run.get_or_insert_with(AssistantRun::default)
                    .calls
                    .push(Value::Object(wire_call));
            }
            Segment::ToolResult {
                wire_id,
                content,
                notes,
                media,
                ..
            } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": wire_id,
                    "content": context_frame::result_text_with_notes(content, notes),
                }));
                // A tool message cannot carry images: hoist the result's
                // media into a user message after the result group, provenance
                // first, one part per attachment in payload order.
                if !media.is_empty() {
                    let mut parts = vec![json!({
                        "type": "text",
                        "text": format!("[tool result media for call {wire_id}]"),
                    })];
                    parts.extend(media.iter().map(|m| match m {
                        ResolvedMedia::Image {
                            media_type,
                            data_base64,
                        } => json!({
                            "type": "image_url",
                            "image_url": {"url": data_url(media_type, data_base64)},
                        }),
                        ResolvedMedia::Placeholder(text) => {
                            json!({"type": "text", "text": text})
                        }
                    }));
                    result_media.push(json!({"role": "user", "content": parts}));
                }
            }
        }
    }
    if let Some(finished) = run.take() {
        messages.extend(assistant_message(finished));
    }

    messages.append(&mut result_media);
    super::tool_history::validate_chat(&messages)?;

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
        body["max_completion_tokens"] = json!(max_tokens);
    }
    if let Some(effort) = generation.reasoning_effort {
        body["reasoning_effort"] = json!(effort);
    }
    if let Some(schema) = &generation.output_schema {
        body["response_format"] = json!({
            "type": "json_schema",
            "json_schema": {
                "name": super::OUTPUT_SCHEMA_NAME,
                "schema": schema,
                "strict": false,
            },
        });
    }
    if !tool_surface.definitions.is_empty() {
        body["tools"] = json!(context_frame::tool_definitions(
            tool_surface,
            context_frame::ToolShape::OpenAiChat
        ));
    }
    Ok(body)
}

/// Materialize an assistant run. The message starts from the run's sealed
/// message-level object when it has one; `role`, `content` (the run's texts
/// concatenated with no separator, only when non-empty) and `tool_calls`
/// (only when non-empty) are written over it. Without a sealed object, empty
/// text is omitted (a calls-only message carries no `content`) and a
/// text-only message carries no `tool_calls` key.
///
/// `None` when the result would carry neither a `content` key nor
/// `tool_calls`: a run whose only text was whitespace (omitted above) leaves
/// just the sealed residue, and an assistant message with neither is not a
/// valid Chat message, so the whole message is omitted like the text was.
fn assistant_message(run: AssistantRun) -> Option<Value> {
    let mut message = context_frame::residue_object(run.sealed.as_ref());
    message.insert("role".into(), json!("assistant"));
    if !run.text.is_empty() {
        message.insert("content".into(), json!(run.text));
    }
    if !run.calls.is_empty() {
        message.insert("tool_calls".into(), json!(run.calls));
    }
    (message.contains_key("content") || message.contains_key("tool_calls"))
        .then_some(Value::Object(message))
}

/// The sealed Chat object without its `reasoning_content` key; `None` when
/// that leaves it empty. A non-object sealed item is returned unchanged.
fn without_reasoning_content(item: &Value) -> Option<Value> {
    let Value::Object(map) = item else {
        return Some(item.clone());
    };
    let mut map = map.clone();
    map.remove("reasoning_content");
    (!map.is_empty()).then_some(Value::Object(map))
}

/// The inline data URL for a resolved image payload (`image_url` shape).
fn data_url(media_type: &str, data_base64: &str) -> String {
    format!("data:{media_type};base64,{data_base64}")
}

/// Parse an OpenAI Chat Completions response body into a kernel
/// [`ModelOutput`].
///
/// A `finish_reason` the kernel has no meaning for maps to
/// [`causa_kernel::ModelStopReason::Unrecognized`] rather than being guessed, and the raw
/// wire value is preserved on [`ModelOutput::raw_stop_reason`]. The mapping
/// table is shared with the streaming decoder, so batch and streaming cannot
/// disagree. `reasoning_content` (the OpenAI-compatible reasoning extension)
/// is kept as a sealed message-level item — so the renderer can send it back
/// — and additionally maps to the reasoning observation, which is
/// display-only.
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

    let mut items: Vec<ModelItem> = Vec::new();
    // Message-level keys the facts do not hold (`reasoning_content`, a null
    // or empty `content`, and anything unknown) travel as one sealed item,
    // first, so the renderer can rebuild the message around it.
    let mut residue = message.as_object().cloned().unwrap_or_default();
    residue.remove("role");
    residue.remove("tool_calls");
    if !text.is_empty() {
        residue.remove("content");
    }
    if !residue.is_empty() {
        items.push(context_frame::sealed_item(
            Protocol::OpenAiChatCompletions,
            Value::Object(residue),
        ));
    }
    if !text.is_empty() {
        items.push(ModelItem::text(text));
    }

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
            // Completed calls require a complete JSON-encoded argument value.
            let arguments = context_frame::decode_wire_arguments(function.get("arguments"))?;
            // The entry minus `id` and `function.name`: `type`, `index` and
            // the raw `function.arguments` string stay.
            let mut fields = context_frame::without_keys(call, &["id"]);
            if let Some(function) = fields.get_mut("function").and_then(Value::as_object_mut) {
                function.remove("name");
            }
            items.push(ModelItem {
                wire: Some(context_frame::wire_meta(
                    Protocol::OpenAiChatCompletions,
                    fields,
                )),
                ..ModelItem::tool_call(ToolCallDraft {
                    tool_name: name.to_string(),
                    arguments,
                    provider_call_id: Some(id.to_string()),
                })
            });
        }
    }

    let raw_stop_reason = match crate::translation::stop_reason::chat_raw(choice) {
        Some(raw) => raw.to_owned(),
        None if choice.get("finish_reason").is_some_and(|v| !v.is_null()) => {
            return Err(permanent("choices[0].finish_reason: expected a string"));
        }
        None => return Err(permanent("missing choices[0].finish_reason")),
    };
    let stop_reason = crate::translation::stop_reason::chat(&raw_stop_reason);

    let reasoning = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .map(|text| ReasoningPayload {
            text: text.to_string(),
        });

    let usage = value
        .get("usage")
        .filter(|v| v.is_object())
        .map(crate::translation::usage::model_usage_from_openai_chat);

    Ok(ModelOutput {
        response: ModelResponse { items },
        usage,
        stop_reason,
        raw_stop_reason: Some(raw_stop_reason),
        reasoning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use causa_kernel::{
        ModelItemContent, ModelStopReason, ModelUsage, ToolDefinition, ToolResultStatus,
    };

    use crate::translation::media::{MediaPayload, MediaSet};
    use crate::translation::test_support::{
        call, frame, label_id, media_part, parts, result, result_with_media, text, text_part,
    };

    fn render(frame: &ContextFrame) -> Value {
        render_openai_chat_messages(
            frame,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
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
        // assistant("assistant") / user("inject:note") / user(unknown):
        // user texts are never joined across block seams
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[0], json!({"role": "system", "content": "be terse"}));
        assert_eq!(
            msgs[1],
            json!({"role": "assistant", "content": "model said"})
        );
        assert_eq!(msgs[2], json!({"role": "user", "content": "user said"}));
        assert_eq!(msgs[3], json!({"role": "assistant", "content": "replayed"}));
        assert_eq!(msgs[4], json!({"role": "user", "content": "injected"}));
        assert_eq!(msgs[5], json!({"role": "user", "content": "unknown tag"}));
    }

    #[test]
    fn tool_round_trip_pairing_and_assistant_merge() {
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
        assert_eq!(msgs.len(), 3);
        // text + both calls merge into ONE assistant message; arguments
        // serialize to a JSON string; an unnamed call falls back to its
        // declaration BlockId UUID.
        assert_eq!(
            msgs[0],
            json!({
                "role": "assistant",
                "content": "reading now",
                "tool_calls": [
                    {"id": "toolu_a", "type": "function",
                     "function": {"name": "read", "arguments": "{\"path\":\"a\"}"}},
                    {"id": fallback_id, "type": "function",
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
            json!({"role": "tool", "tool_call_id": fallback_id, "content": "{\"error\":\"boom\"}"})
        );
    }

    #[test]
    fn assistant_tool_calls_only_omits_empty_content() {
        let f = frame(vec![
            call(0, "kc1", Some("toolu_a"), "read", json!({})),
            result(1, "kc1", ToolResultStatus::Succeeded, json!("ok")),
        ]);
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
    fn adjacent_assistant_texts_concatenate_into_one_message() {
        // Assistant text runs coalesce across calls too — the wire
        // carries assistant content as a single string.
        let f = frame(vec![
            text(0, "first", None),
            call(1, "kc1", Some("toolu_a"), "read", json!({})),
            text(2, "second", None),
            result(3, "kc1", ToolResultStatus::Succeeded, json!("ok")),
        ]);
        let v = render(&f);
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"], json!("firstsecond"));
        assert_eq!(msgs[0]["tool_calls"].as_array().unwrap().len(), 1);

        // no calls in between: still one message, texts concatenated with
        // no separator
        let f = frame(vec![text(0, "a", None), text(1, "b", None)]);
        let v = render(&f);
        assert_eq!(
            v["messages"],
            json!([{"role": "assistant", "content": "ab"}])
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
        let error = render_openai_chat_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
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
            ..GenerationOptions::default()
        };
        let v = render_openai_chat_messages(
            &f,
            &MediaSet::new(),
            &surface,
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
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
            ..GenerationOptions::default()
        };
        let v2 = render_openai_chat_messages(
            &f,
            &MediaSet::new(),
            &surface,
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(v2["max_completion_tokens"], json!(100));
        let again = render_openai_chat_messages(
            &f,
            &MediaSet::new(),
            &surface,
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&v2).unwrap(),
            serde_json::to_string(&again).unwrap()
        );
    }

    #[test]
    fn output_schema_maps_and_cache_directive_is_a_noop() {
        let f = frame(vec![text(0, "hi", Some("user"))]);
        let generation = GenerationOptions {
            output_schema: Some(json!({"type": "object"})),
            ..GenerationOptions::default()
        };
        let v = render_openai_chat_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            v["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {"type": "object"},
                    "strict": false,
                },
            })
        );
        // the cache directive has no wire representation on this protocol
        let cached = render_openai_chat_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::StablePrefix,
            &ChatRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(v, cached);
    }

    #[test]
    fn empty_frame_is_invalid_request() {
        let e = render_openai_chat_messages(
            &frame(vec![]),
            &MediaSet::new(),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("m"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::InvalidRequest));
    }

    // --- media -------------------------------------------------------------

    use causa_kernel::MediaRef;

    fn resolved(reference: &str) -> MediaSet {
        let mut m = MediaSet::new();
        m.insert(reference, MediaPayload::new("image/png", "AAAA"));
        m
    }

    #[test]
    fn image_parts_render_as_image_url_messages() {
        let f = frame(vec![parts(
            0,
            vec![
                text_part("caption"),
                media_part("image/png", "asset-1"),
                text_part("after"),
            ],
            Some("user"),
        )]);
        let v = render_openai_chat_messages(
            &f,
            &resolved("asset-1"),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap();
        // part boundaries preserved: each text part is its own message,
        // the image rides an array-content user message with a data URL
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs[0], json!({"role": "user", "content": "caption"}));
        assert_eq!(
            msgs[1],
            json!({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]})
        );
        assert_eq!(msgs[2], json!({"role": "user", "content": "after"}));
    }

    #[test]
    fn unresolved_media_degrades_to_the_deterministic_placeholder() {
        let f = frame(vec![parts(
            0,
            vec![media_part("image/png", "missing-asset")],
            Some("user"),
        )]);
        let v = render_openai_chat_messages(
            &f,
            &MediaSet::new(),
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            v["messages"][0],
            json!({"role": "user", "content": "[media: image/png missing-asset]"})
        );
    }

    #[test]
    fn tool_result_media_hoists_into_a_user_message_with_provenance() {
        let f = frame(vec![
            call(0, "kc1", Some("call_1"), "render", json!({})),
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
        media.insert("asset-2", MediaPayload::new("image/jpeg", "BBBB"));
        let v = render_openai_chat_messages(
            &f,
            &media,
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap();
        let msgs = v["messages"].as_array().unwrap();
        // tool message stays text-only; media hoists right after it
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(
            msgs[2]["content"],
            json!([
                {"type": "text", "text": "[tool result media for call call_1]"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBBB"}},
            ])
        );
    }

    #[test]
    fn hoisted_placeholders_stay_in_payload_order() {
        let fallback_id = label_id("kc1").0.to_string();
        let f = frame(vec![
            call(0, "kc1", None, "render", json!({})),
            result_with_media(
                1,
                "kc1",
                ToolResultStatus::Succeeded,
                json!("mixed"),
                vec![
                    MediaRef::new("image/png", "good"),
                    MediaRef::new("image/png", "missing"),
                    MediaRef::new("application/pdf", "doc"),
                ],
            ),
        ]);
        let mut media = MediaSet::new();
        media.insert("good", MediaPayload::new("image/png", "AAAA"));
        media.insert("doc", MediaPayload::new("application/pdf", "BBBB"));
        let v = render_openai_chat_messages(
            &f,
            &media,
            &ToolSurface::empty(),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ChatRenderOptions::default(),
        )
        .unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(
            msgs[2]["content"],
            json!([
                {"type": "text", "text": format!("[tool result media for call {fallback_id}]")},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "[media: image/png missing]"},
                // resolved but non-image: degraded, never inlined
                {"type": "text", "text": "[media: application/pdf doc]"},
            ])
        );
    }

    // --- parsing -----------------------------------------------------------

    /// The stored value of every sealed part of a response, in item order.
    fn sealed_items(out: &ModelOutput) -> Vec<Value> {
        out.response
            .items
            .iter()
            .flat_map(|item| match &item.content {
                ModelItemContent::Parts(parts) => parts
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
        assert_eq!(out.response.text(), "Let me check.");
        assert_eq!(out.response.tool_calls().count(), 1);
        let tc = &out.response.tool_calls().next().unwrap();
        assert_eq!(tc.tool_name, "read");
        assert_eq!(tc.arguments, json!({"path": "a"}));
        assert_eq!(tc.provider_call_id.as_deref(), Some("call_9"));
        assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
        // The message residue, `reasoning_content` included, is sealed material
        // so the renderer can send it back verbatim.
        assert_eq!(
            sealed_items(&out),
            vec![json!({"reasoning_content": "hmm"})]
        );
        let reasoning = out.reasoning.unwrap();
        assert_eq!(reasoning.text, "hmm");
        let u: ModelUsage = out.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (100, 20));
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.cache_write_tokens, None);
        assert_eq!(u.reasoning_tokens, Some(5));
    }

    #[test]
    fn finish_reason_mapping_full_and_unknown_is_unrecognized() {
        for (wire, expected) in [
            ("stop", ModelStopReason::EndTurn),
            ("end", ModelStopReason::EndTurn),
            ("tool_calls", ModelStopReason::ToolUse),
            ("function_call", ModelStopReason::ToolUse),
            ("length", ModelStopReason::MaxTokens),
            ("content_filter", ModelStopReason::Refusal),
            // An unknown value is reported as what it is, never guessed.
            ("some_new_reason", ModelStopReason::Unrecognized),
        ] {
            let v = json!({"choices": [{"message": {"content": ""}, "finish_reason": wire}]});
            let out = parse_openai_chat_response(&v).unwrap();
            assert_eq!(
                std::mem::discriminant(&out.stop_reason),
                std::mem::discriminant(&expected),
                "finish_reason = {wire}"
            );
            assert_eq!(out.raw_stop_reason.as_deref(), Some(wire));
        }
    }

    #[test]
    fn missing_choices_or_finish_reason_is_permanent_error() {
        let e = parse_openai_chat_response(&json!({"choices": []})).unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Permanent));
        let e = parse_openai_chat_response(&json!({"choices": [{"message": {}}]})).unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Permanent));
        assert!(e.message.contains("finish_reason"));
        // Present but not a string is reported as such, not as missing.
        let e =
            parse_openai_chat_response(&json!({"choices": [{"message": {}, "finish_reason": 7}]}))
                .unwrap_err();
        assert!(matches!(e.kind(), ModelInvokeErrorKind::Permanent));
        assert!(e.message.contains("expected a string"));
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
    fn empty_arguments_string_is_permanent_error() {
        let v = json!({
            "choices": [{
                "message": {"tool_calls": [
                    {"id": "c1", "function": {"name": "list", "arguments": ""}},
                ]},
                "finish_reason": "tool_calls",
            }],
        });
        let error = parse_openai_chat_response(&v).unwrap_err();
        assert_eq!(error.kind, ModelInvokeErrorKind::Permanent);
    }
}
