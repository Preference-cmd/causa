//! OpenAI Responses API translation for the context kernel.
//!
//! Pure functions from [`causa_kernel::ContextFrame`] to a Responses API
//! request body and back to [`causa_kernel::ModelOutput`]. Transport-free;
//! the reqwest adapter in `causa-provider` owns HTTP.
//!
//! The renderer-independent policy (role mapping, empty-text skip,
//! no cross-block text join, tool id pairing, observation stringification) lives in
//! `super::context_frame`; this module only shapes the Responses wire.
//!
//! # Responses-specific structural rules
//!
//! - The leading run of `system` text segments moves to the top-level
//!   `instructions` parameter (joined with `\n`); a `system` block after
//!   conversation content renders at its position as an input item, an
//!   append like any other.
//! - One user `Message` block renders as one user message item whose
//!   content holds its parts in order: `input_text` for text and
//!   `input_image` (data URL, `detail: "auto"`) for a resolved image, with a
//!   degraded attachment as `input_text`. Adjacent user blocks stay
//!   separate items: the block boundary is a fact boundary. Assistant text
//!   segments become `output_text` message items; assistant- and
//!   system-position media degrade to the shared placeholder text (the walk
//!   decides).
//! - Each tool call renders as a flat `function_call` item and each tool
//!   result as a flat `function_call_output` item — the Responses wire
//!   pairs through `call_id`s, not message roles. There is no `is_error`
//!   flag on this wire; error information travels in the content.
//! - Tool-result media **embeds**: a result without media keeps the string
//!   `output`; with media, `output` is an array whose first item is the
//!   result text (with its notes) as `input_text`, followed by one item per
//!   attachment in payload order — `input_image` as above, or the
//!   placeholder as `input_text`.
//! - Function call `arguments` are a JSON *string* on the wire (encoded
//!   by the emitter, decoded by the shared `super::context_frame`
//!   codec).
//! - Stop-reason derivation: the Responses wire has no `finish_reason`.
//!   A refusal output item wins, then `incomplete` + `max_output_tokens`
//!   (or the Chat word `length`) → MaxTokens, then any `function_call` output → ToolUse, else
//!   EndTurn. `incomplete/content_filter` maps to Refusal; an unknown
//!   incomplete reason is Unrecognized with the output intact, while
//!   UnknownOutcome is reserved for a non-terminal or unknown status. Failed
//!   responses return a classified error.
//! - Wire fidelity. Each `message` output item parses to one `Parts` item (one
//!   text part per `output_text` part) whose `wire.fields` is the item with
//!   the `text` of each `output_text` part removed; each `function_call` item
//!   parses to a tool-call item whose `wire.fields` is the item minus
//!   `call_id` and `name`, keeping `id` and the raw `arguments` string. A
//!   block with matching `wire` renders as one item built from `fields`
//!   (message: each text goes back into its `output_text` slot, falling back
//!   to facts when the counts differ; call: `call_id`, `name` and, unless the
//!   raw string still parses to the fact's value, `arguments` are written
//!   over it). Own-format sealed parts are emitted verbatim as input items.
//! - Reasoning output items are kept whole as sealed items in place — `id`,
//!   `summary`, `encrypted_content` and every other field exactly as returned
//!   — and additionally contribute their summary texts to the reasoning
//!   observation, which is display-only; an item without summary text
//!   contributes its raw `reasoning_text` content parts instead. Every other
//!   output item type this adapter has no vocabulary for is kept the same
//!   way, so the turn can be echoed back as returned.
//! - The request sends `include: ["reasoning.encrypted_content"]` and
//!   `store: false` by default (as Codex and `pi-ai` do: encrypted reasoning
//!   is what makes a sealed reasoning item replayable under `store: false` or
//!   zero data retention). [`ResponsesRenderOptions`] changes either.
//! - Tools: by default `tools` carries the frame's current tool state, the
//!   fold of its `ToolChange` blocks, so a change rewrites it. With
//!   [`ResponsesRenderOptions::additional_tools`], `tools` stays at the
//!   frame's first change (the declaration) and each later change renders at
//!   its position as an `additional_tools` input item with
//!   `role: "developer"`, as Codex and `pi-ai` send it. `additional_tools`
//!   only adds, so a frame in which a later change removes a tool or
//!   defines a name a second time renders as by default.
//! - `reasoning_effort` renders as `reasoning.effort`; output limits use
//!   `max_output_tokens`, including reasoning tokens. The effort is written
//!   exactly as given and beside any temperature; which values a model
//!   supports is the host's knowledge.

use serde_json::{Value, json};

use causa_kernel::{
    CacheDirective, ContentPart, ContextFrame, GenerationOptions, ModelInvokeError,
    ModelInvokeErrorKind, ModelItem, ModelItemContent, ModelOutput, ModelRef, ModelResponse,
    ModelStopReason, ReasoningPayload, TextPayload, ToolCallDraft,
};

use super::context_frame::{self, ResolvedMedia, Role, Segment};
use super::media::MediaSet;
use super::render_options::ResponsesRenderOptions;
use crate::Protocol;

/// The `include` value this adapter sends for encrypted reasoning, the one
/// that makes a reasoning item replayable under `store: false`.
const ENCRYPTED_REASONING_INCLUDE: &str = "reasoning.encrypted_content";

/// Render a [`ContextFrame`] into an OpenAI Responses API request body.
/// Deterministic: identical inputs produce byte-identical JSON.
///
/// `cache` is accepted for interface parity and ignored: OpenAI-family
/// providers cache server-side, so the directive has no wire
/// representation on this protocol. `GenerationOptions::output_schema`
/// renders as `text.format` (structured output); schema validation and
/// corrective retry stay host-side. `options` carries the Responses-only
/// `include`/`store` choices; see [`ResponsesRenderOptions`].
pub fn render_openai_responses_input(
    frame: &ContextFrame,
    media: &MediaSet,
    generation: &GenerationOptions,
    model: &ModelRef,
    _cache: CacheDirective,
    options: &ResponsesRenderOptions,
) -> Result<Value, ModelInvokeError> {
    super::generation::validate_openai(generation)?;
    super::tool_history::validate_frame(frame)?;
    let tool_plan = context_frame::plan_tools(
        frame,
        if options.additional_tools {
            context_frame::ToolPlacement::ResponsesAdditive
        } else {
            context_frame::ToolPlacement::Fold
        },
    );
    let normalized =
        context_frame::normalize(frame, media, Protocol::OpenAiResponses, tool_plan.in_place);

    let mut instructions: Vec<String> = Vec::new();
    let mut items: Vec<Value> = Vec::new();
    let segments = &normalized.segments;
    let leading = context_frame::leading_system_len(segments);
    let mut next = 0;
    while next < segments.len() {
        let segment = &segments[next];
        next += 1;
        match segment {
            // Only the leading system run belongs in `instructions`; a system
            // block appended after conversation content renders at its position
            // as an input item, exactly where the fact sits.
            Segment::Text {
                role: Role::System,
                text,
                ..
            } if next - 1 < leading => instructions.push(text.clone()),
            Segment::Text {
                role: Role::System,
                text,
                ..
            } => items.push(json!({
                "role": "system",
                "content": [{"type": "input_text", "text": text}],
            })),
            // One user block renders as one message item, its parts in order.
            Segment::Text {
                role: Role::User, ..
            }
            | Segment::Media {
                role: Role::User, ..
            } => {
                let parts = context_frame::user_block_run(segments, next - 1);
                items.push(json!({
                    "role": "user",
                    "content": segments[next - 1..next - 1 + parts]
                        .iter()
                        .filter_map(|segment| match segment {
                            Segment::Text { text, .. } => {
                                Some(json!({"type": "input_text", "text": text}))
                            }
                            Segment::Media { media, .. } => Some(input_media(media)),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                }));
                next += parts - 1;
            }
            Segment::Text {
                role: Role::Assistant,
                text,
                origin,
            } => {
                let run = context_frame::same_block_run(segments, next - 1);
                let message = origin
                    .wire
                    .as_ref()
                    .filter(|_| run == origin.parts)
                    .and_then(|fields| {
                        message_from_wire(fields, segments[next - 1..next - 1 + run].iter())
                    });
                if let Some(message) = message {
                    items.push(message);
                    next += run - 1;
                } else {
                    items.push(json!({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }));
                }
            }
            Segment::Sealed(item) => items.push(item.clone()),
            Segment::Media {
                role,
                media: ResolvedMedia::Placeholder(text),
                ..
            } => match role {
                Role::System if next - 1 < leading => instructions.push(text.clone()),
                Role::System => items.push(json!({
                    "role": "system",
                    "content": [{"type": "input_text", "text": text}],
                })),
                _ => items.push(json!({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                })),
            },
            Segment::Media {
                media: ResolvedMedia::Image { .. },
                ..
            } => unreachable!("the walk renders images only in user position"),
            Segment::ToolCall(call) => {
                let mut item = context_frame::residue_object(call.wire.as_ref());
                item.entry("type").or_insert_with(|| json!("function_call"));
                item.insert("call_id".into(), json!(call.wire_id));
                item.insert("name".into(), json!(call.name));
                let arguments =
                    context_frame::wire_arguments_string(item.get("arguments"), &call.arguments);
                item.insert("arguments".into(), json!(arguments));
                items.push(Value::Object(item));
            }
            Segment::ToolResult {
                wire_id,
                content,
                notes,
                media,
                ..
            } => {
                // A result without media keeps the string form; with media,
                // `output` is an array: the text first, then one item per
                // attachment in payload order.
                let text = context_frame::result_text_with_notes(content, notes);
                let output = if media.is_empty() {
                    json!(text)
                } else {
                    let mut parts = vec![json!({"type": "input_text", "text": text})];
                    parts.extend(media.iter().map(input_media));
                    json!(parts)
                };
                items.push(json!({
                    "type": "function_call_output",
                    "call_id": wire_id,
                    "output": output,
                }));
            }
            // Additive placement: the change only adds tools, each under a
            // name not offered before.
            Segment::ToolChange(change) => {
                if !change.added.is_empty() {
                    items.push(json!({
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": context_frame::tool_definitions(
                            &change.added,
                            context_frame::ToolShape::OpenAiResponses,
                        ),
                    }));
                }
            }
        }
    }

    if items.is_empty() {
        return Err(ModelInvokeError::new(
            ModelInvokeErrorKind::InvalidRequest,
            "frame rendered to zero input items; nothing to send",
        ));
    }

    let mut body = json!({
        "model": model.0,
        "input": items,
    });
    if options.include_encrypted_reasoning {
        body["include"] = json!([ENCRYPTED_REASONING_INCLUDE]);
    }
    if let Some(store) = options.store {
        body["store"] = json!(store);
    }
    if !instructions.is_empty() {
        body["instructions"] = json!(instructions.join("\n"));
    }
    if let Some(temperature) = generation.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = generation.max_tokens {
        body["max_output_tokens"] = json!(max_tokens);
    }
    if let Some(effort) = &generation.reasoning_effort {
        body["reasoning"] = json!({"effort": effort});
    }
    if let Some(schema) = &generation.output_schema {
        body["text"] = json!({
            "format": {
                "type": "json_schema",
                "name": super::OUTPUT_SCHEMA_NAME,
                "schema": schema,
                "strict": false,
            },
        });
    }
    if !tool_plan.tools.is_empty() {
        body["tools"] = json!(context_frame::tool_definitions(
            &tool_plan.tools,
            context_frame::ToolShape::OpenAiResponses
        ));
    }
    super::generation::merge_extra_body(&mut body, generation);
    // The schema is the host's own value, not an object the adapter wrote:
    // the merge may add keys beside it and must not reach into it.
    if let Some(schema) = &generation.output_schema {
        body["text"]["format"]["schema"] = schema.clone();
    }
    Ok(body)
}

/// Rebuild a recorded `message` item around its text facts: each text goes
/// back into the `output_text` slot at the same position. `None` when the
/// recorded item does not have exactly one slot per text, in which case the
/// caller renders from facts.
fn message_from_wire<'a>(
    fields: &Value,
    texts: impl Iterator<Item = &'a Segment>,
) -> Option<Value> {
    let mut item = context_frame::residue_object(Some(fields));
    let slots = item.get_mut("content")?.as_array_mut()?;
    let mut slots: Vec<&mut Value> = slots
        .iter_mut()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .collect();
    let texts: Vec<&String> = texts
        .filter_map(|segment| match segment {
            Segment::Text { text, .. } => Some(text),
            _ => None,
        })
        .collect();
    if slots.len() != texts.len() {
        return None;
    }
    for (slot, text) in slots.iter_mut().zip(texts) {
        slot["text"] = json!(text);
    }
    item.insert("role".into(), json!("assistant"));
    Some(Value::Object(item))
}

/// One media attachment as an input content item: an `input_image` with a
/// data URL and `detail: "auto"`, or the placeholder as `input_text`.
fn input_media(media: &ResolvedMedia) -> Value {
    match media {
        ResolvedMedia::Image {
            media_type,
            data_base64,
        } => json!({
            "type": "input_image",
            "image_url": format!("data:{media_type};base64,{data_base64}"),
            "detail": "auto",
        }),
        ResolvedMedia::Placeholder(text) => json!({"type": "input_text", "text": text}),
    }
}

/// Parse an OpenAI Responses API response body into a kernel
/// [`ModelOutput`]. See the module docs for the stop-reason derivation.
pub fn parse_openai_responses_output(value: &Value) -> Result<ModelOutput, ModelInvokeError> {
    fn undecodable(message: impl Into<String>) -> ModelInvokeError {
        ModelInvokeError::new(ModelInvokeErrorKind::Undecodable, message)
    }

    let status = value
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| undecodable("missing status"))?;
    let incomplete_reason = match status {
        "completed" => None,
        "failed" => {
            return Err(crate::translation::streaming::common::provider_error(
                value.get("error").unwrap_or(&Value::Null),
            ));
        }
        // An incomplete response carries valid content; a reason the kernel
        // has no meaning for is reported as `Unrecognized` with the wire value
        // preserved, not turned into a parse failure. An absent or non-string
        // reason stays `None`: the wire carried no reason, and the raw stop
        // value must not invent one.
        "incomplete" => value
            .get("incomplete_details")
            .and_then(|d| d.get("reason"))
            .and_then(Value::as_str),
        _ => {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::UnknownOutcome,
                match status {
                    "queued" => "Responses non-terminal status: queued",
                    "in_progress" => "Responses non-terminal status: in_progress",
                    "cancelled" => "Responses status without an outcome: cancelled",
                    _ => "Responses unrecognized status",
                },
            ));
        }
    };

    let empty = Vec::new();
    let output = match value.get("output") {
        None | Some(Value::Null) => &empty,
        Some(Value::Array(items)) => items,
        Some(_) => return Err(undecodable("output: expected an array")),
    };

    let mut items: Vec<ModelItem> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut refusal = false;
    let mapped = crate::translation::stop_reason::responses(status, incomplete_reason);
    for (position, item) in output.iter().enumerate() {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let content = match item.get("content") {
                    None | Some(Value::Null) => continue,
                    Some(Value::Array(items)) => items,
                    Some(_) => {
                        return Err(undecodable("output[].message.content: expected an array"));
                    }
                };
                let mut parts = Vec::new();
                for part in content {
                    match part.get("type").and_then(Value::as_str) {
                        Some("output_text") => {
                            let text = part
                                .get("text")
                                .and_then(Value::as_str)
                                .ok_or_else(|| undecodable("output_text: missing `text` string"))?;
                            parts.push(ContentPart::Text(TextPayload::new(text)));
                        }
                        // Refusal content inside a message marks the stop.
                        Some("refusal") => refusal = true,
                        _ => {}
                    }
                }
                // The item minus the text it holds: every `output_text` part
                // keeps its other keys (`annotations`, `logprobs`, ...) in
                // place, and other part types stay untouched.
                let mut fields = item.clone();
                if let Some(Value::Array(slots)) = fields.get_mut("content") {
                    for slot in slots
                        .iter_mut()
                        .filter(|p| p.get("type").and_then(Value::as_str) == Some("output_text"))
                    {
                        if let Some(map) = slot.as_object_mut() {
                            map.remove("text");
                        }
                    }
                }
                items.push({
                    let mut message = ModelItem::new(ModelItemContent::Parts(parts));
                    message.wire =
                        Some(context_frame::wire_meta(Protocol::OpenAiResponses, fields));
                    message
                });
            }
            Some("function_call") => {
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| undecodable("function_call: missing `call_id` string"))?;
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| undecodable("function_call: missing `name` string"))?;
                // The token limit can stop the response inside the last item.
                let cut_off = mapped == ModelStopReason::MaxTokens && position + 1 == output.len();
                let Some(arguments) =
                    context_frame::decode_wire_arguments(item.get("arguments"), cut_off)?
                else {
                    continue;
                };
                items.push({
                    let mut call_item = ModelItem::tool_call(
                        ToolCallDraft::new(name.to_string(), arguments)
                            .with_provider_call_id(call_id.to_string()),
                    );
                    call_item.wire = Some(context_frame::wire_meta(
                        Protocol::OpenAiResponses,
                        context_frame::without_keys(item, &["call_id", "name"]),
                    ));
                    call_item
                });
            }
            Some("reasoning") => {
                let texts = |key: &str| -> Vec<String> {
                    item.get(key)
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter(|part| {
                            key == "summary"
                                || part.get("type").and_then(Value::as_str)
                                    == Some("reasoning_text")
                        })
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .map(str::to_string)
                        .collect()
                };
                // Raw reasoning is shown only when the server gave no summary.
                let mut shown = texts("summary");
                if shown.is_empty() {
                    shown = texts("content");
                }
                reasoning_parts.extend(shown);
                // The whole item — `id`, `summary`, `encrypted_content` and
                // any unknown field — is sealed material in place; the
                // texts above are the display-only observation.
                items.push(context_frame::sealed_item(
                    Protocol::OpenAiResponses,
                    item.clone(),
                ));
            }
            // A top-level refusal output item.
            Some("refusal") => {
                refusal = true;
                if let Some(text) = item.get("refusal").and_then(Value::as_str) {
                    items.push(ModelItem::text(text));
                }
            }
            // Every other output item type is kept as a sealed item in
            // place, so the turn can be echoed back as returned.
            _ => items.push(context_frame::sealed_item(
                Protocol::OpenAiResponses,
                item.clone(),
            )),
        }
    }

    let has_tool_call = items
        .iter()
        .any(|item| matches!(item.content, ModelItemContent::ToolCall(_)));
    // The shared table classifies every terminal status. A top-level refusal
    // item is a refusal whatever the status said, and Responses has no
    // tool-call stop value, so a terminal response carrying a \`function_call\`
    // item is a tool use (as \`pi-ai\`).
    let stop_reason = if refusal {
        ModelStopReason::Refusal
    } else if mapped == ModelStopReason::EndTurn && has_tool_call {
        ModelStopReason::ToolUse
    } else {
        mapped
    };
    let raw_stop_reason =
        crate::translation::stop_reason::responses_raw(value, status, incomplete_reason);

    let usage = value
        .get("usage")
        .filter(|v| v.is_object())
        .map(crate::translation::usage::model_usage_from_openai_responses);

    Ok({
        let mut output = ModelOutput::new(ModelResponse { items }, stop_reason);
        output.usage = usage;
        output.raw_stop_reason = Some(raw_stop_reason);
        output.reasoning = if reasoning_parts.is_empty() {
            None
        } else {
            Some(ReasoningPayload::new(reasoning_parts.join("\n")))
        };
        output
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use causa_kernel::{ModelStopReason, ModelUsage, ToolDefinition, ToolResultStatus};

    use crate::translation::media::{MediaPayload, MediaSet};
    use crate::translation::test_support::{
        call, frame, media_part, parts, result, result_with_media, text, text_part,
        text_with_source, with_tools,
    };

    fn render(frame: &ContextFrame) -> Value {
        render_openai_responses_input(
            frame,
            &MediaSet::new(),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
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
        assert_eq!(v["instructions"], json!("be terse"));
        let input = v["input"].as_array().unwrap();
        // assistant / user / assistant / user / user: user texts are never joined across block seams
        assert_eq!(input.len(), 5);
        assert_eq!(
            input[0],
            json!({"role": "assistant", "content": [{"type": "output_text", "text": "model said"}]})
        );
        assert_eq!(
            input[1],
            json!({"role": "user", "content": [{"type": "input_text", "text": "user said"}]})
        );
        assert_eq!(
            input[2],
            json!({"role": "assistant", "content": [{"type": "output_text", "text": "replayed"}]})
        );
        assert_eq!(
            input[3],
            json!({"role": "user", "content": [{"type": "input_text", "text": "user override"}]})
        );
        assert_eq!(
            input[4],
            json!({"role": "user", "content": [{"type": "input_text", "text": "user no source"}]})
        );
    }

    #[test]
    fn tool_round_trip_pairing_with_flat_call_ids() {
        let fallback_id = call(2, "kc2", None, "list", json!({})).id().0.to_string();
        let f = frame(vec![
            text(0, "reading now", Role::Assistant),
            call(1, "kc1", Some("toolu_a"), "read", json!({"path": "a"})),
            call(2, "kc2", None, "list", json!({})),
            result(3, "kc1", ToolResultStatus::Succeeded, json!("file-a")),
            result(4, "kc2", ToolResultStatus::Failed, json!({"error": "boom"})),
        ]);
        let v = render(&f);
        let input = v["input"].as_array().unwrap();
        assert_eq!(input.len(), 5);
        assert_eq!(
            input[1],
            json!({
                "type": "function_call",
                "call_id": "toolu_a",
                "name": "read",
                "arguments": "{\"path\":\"a\"}",
            })
        );
        assert_eq!(
            input[2],
            json!({
                "type": "function_call",
                "call_id": fallback_id,
                "name": "list",
                "arguments": "{}",
            })
        );
        assert_eq!(
            input[3],
            json!({"type": "function_call_output", "call_id": "toolu_a", "output": "file-a"})
        );
        assert_eq!(
            input[4],
            json!({
                "type": "function_call_output",
                "call_id": fallback_id,
                "output": "{\"error\":\"boom\"}",
            })
        );
    }

    #[test]
    fn tools_generation_and_byte_determinism() {
        let surface = vec![ToolDefinition::new(
            "read",
            "read a file",
            json!({"type": "object"}),
        )];
        let f = frame(vec![text(0, "hi", Role::User)]);

        let generation = GenerationOptions::default()
            .with_temperature(0.5)
            .with_max_tokens(256);
        let v = render_openai_responses_input(
            &with_tools(&f, &surface),
            &MediaSet::new(),
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(v["model"], json!("gpt-test"));
        assert_eq!(v["temperature"], json!(0.5));
        assert_eq!(v["max_output_tokens"], json!(256));
        // Responses tools are flat, not nested under "function"
        assert_eq!(
            v["tools"],
            json!([{
                "type": "function",
                "name": "read",
                "description": "read a file",
                "parameters": {"type": "object"},
            }])
        );
        let again = render_openai_responses_input(
            &with_tools(&f, &surface),
            &MediaSet::new(),
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&v).unwrap(),
            serde_json::to_string(&again).unwrap()
        );
    }

    #[test]
    fn output_schema_maps_and_cache_directive_is_a_noop() {
        let f = frame(vec![text(0, "hi", Role::User)]);
        let generation = GenerationOptions::default().with_output_schema(json!({"type": "object"}));
        let v = render_openai_responses_input(
            &f,
            &MediaSet::new(),
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            v["text"],
            json!({
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": {"type": "object"},
                    "strict": false,
                },
            })
        );
        // the cache directive has no wire representation on this protocol
        let cached = render_openai_responses_input(
            &f,
            &MediaSet::new(),
            &generation,
            &ModelRef::new("gpt-test"),
            CacheDirective::StablePrefix,
            &ResponsesRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(v, cached);
    }

    #[test]
    fn empty_frame_is_invalid_request() {
        let e = render_openai_responses_input(
            &frame(vec![]),
            &MediaSet::new(),
            &GenerationOptions::default(),
            &ModelRef::new("m"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(e.kind, ModelInvokeErrorKind::InvalidRequest));
    }

    // --- media -------------------------------------------------------------

    use causa_kernel::MediaRef;

    fn resolved(reference: &str) -> MediaSet {
        let mut m = MediaSet::new();
        m.insert(reference, MediaPayload::inline("image/png", "AAAA"));
        m
    }

    #[test]
    fn one_user_block_renders_as_one_message_item() {
        let f = frame(vec![
            parts(
                0,
                vec![text_part("caption"), media_part("image/png", "asset-1")],
                Role::User,
            ),
            parts(1, vec![text_part("first"), text_part("second")], Role::User),
            text(2, "alone", Role::User),
        ]);
        let v = render_openai_responses_input(
            &f,
            &resolved("asset-1"),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        )
        .unwrap();
        // part boundaries are content items inside the block's one item;
        // adjacent user blocks stay separate items
        assert_eq!(
            v["input"],
            json!([
                {"role": "user", "content": [
                    {"type": "input_text", "text": "caption"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "auto"},
                ]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "first"},
                    {"type": "input_text", "text": "second"},
                ]},
                {"role": "user", "content": [{"type": "input_text", "text": "alone"}]},
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
        let v = render_openai_responses_input(
            &f,
            &MediaSet::new(),
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(
            v["input"][0],
            json!({"role": "user", "content": [
                {"type": "input_text", "text": "[media: image/png missing-asset]"}
            ]})
        );
    }

    #[test]
    fn tool_result_media_embeds_in_the_function_call_output() {
        let f = frame(vec![
            call(0, "kc1", Some("call_1"), "render", json!({})),
            call(1, "kc2", Some("call_2"), "render", json!({})),
            result_with_media(
                2,
                "kc1",
                ToolResultStatus::Succeeded,
                json!("chart ready"),
                vec![
                    MediaRef::new("image/png", "asset-1"),
                    MediaRef::new("image/png", "asset-2"),
                ],
            ),
            result(3, "kc2", ToolResultStatus::Succeeded, json!("no chart")),
        ]);
        let mut media = MediaSet::new();
        media.insert("asset-1", MediaPayload::inline("image/png", "AAAA"));
        // asset-2 unresolved -> placeholder item in the same output array
        let v = render_openai_responses_input(
            &f,
            &media,
            &GenerationOptions::default(),
            &ModelRef::new("gpt-test"),
            CacheDirective::None,
            &ResponsesRenderOptions::default(),
        )
        .unwrap();
        let items = v["input"].as_array().unwrap();
        assert_eq!(items.len(), 4, "no user item follows the results");
        assert_eq!(
            items[2],
            json!({"type": "function_call_output", "call_id": "call_1", "output": [
                {"type": "input_text", "text": "chart ready"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "auto"},
                {"type": "input_text", "text": "[media: image/png asset-2]"},
            ]})
        );
        // a result without media keeps the string form
        assert_eq!(
            items[3],
            json!({"type": "function_call_output", "call_id": "call_2", "output": "no chart"})
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
                        ContentPart::Sealed(sealed) => Some(sealed.item.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect()
    }

    #[test]
    fn parse_output_items_with_provider_id_passthrough() {
        let v = json!({
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hmm"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "Let me check."}]},
                {"type": "function_call", "call_id": "call_9", "name": "read",
                 "arguments": "{\"path\": \"a\"}"},
            ],
            "status": "completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 11},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        });
        let out = parse_openai_responses_output(&v).unwrap();
        assert_eq!(out.response.text(), "Let me check.");
        assert_eq!(out.response.tool_calls().count(), 1);
        let tc = &out.response.tool_calls().next().unwrap();
        assert_eq!(tc.tool_name, "read");
        assert_eq!(tc.arguments, json!({"path": "a"}));
        assert_eq!(tc.provider_call_id.as_deref(), Some("call_9"));
        assert!(matches!(out.stop_reason, ModelStopReason::ToolUse));
        // The whole reasoning item is sealed in place, not just its summary.
        assert_eq!(
            sealed_items(&out),
            vec![json!({
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "hmm"}],
            })]
        );
        let reasoning = out.reasoning.unwrap();
        assert_eq!(reasoning.text, "hmm");
        let u: ModelUsage = out.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (100, 20));
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.reasoning_tokens, Some(5));
    }

    #[test]
    fn status_derivation_rows() {
        let with_output = |output: Value, mut extra: Value| {
            let obj = extra.as_object_mut().unwrap();
            obj.insert("output".into(), output);
            extra
        };
        // completed, no calls → EndTurn
        let out =
            parse_openai_responses_output(&with_output(json!([]), json!({"status": "completed"})))
                .unwrap();
        assert!(matches!(out.stop_reason, ModelStopReason::EndTurn));
        // incomplete + max_output_tokens → MaxTokens
        let out = parse_openai_responses_output(&with_output(
            json!([]),
            json!({
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            }),
        ))
        .unwrap();
        assert!(matches!(out.stop_reason, ModelStopReason::MaxTokens));
        // A content-filter interruption is an explicit refusal.
        let out = parse_openai_responses_output(&with_output(
            json!([]),
            json!({
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            }),
        ))
        .unwrap();
        assert!(matches!(out.stop_reason, ModelStopReason::Refusal));
        // refusal output item wins
        let out = parse_openai_responses_output(&with_output(
            json!([{"type": "refusal", "refusal": "no"}]),
            json!({"status": "completed"}),
        ))
        .unwrap();
        assert!(matches!(out.stop_reason, ModelStopReason::Refusal));
        assert_eq!(out.response.text(), "no");
    }

    #[test]
    fn missing_status_is_undecodable_error() {
        let e = parse_openai_responses_output(&json!({"output": []})).unwrap_err();
        assert!(matches!(e.kind, ModelInvokeErrorKind::Undecodable));
        assert!(e.message.contains("status"));
    }

    #[test]
    fn malformed_function_call_arguments_is_undecodable_error() {
        let v = json!({
            "output": [{"type": "function_call", "call_id": "c1", "name": "read",
                        "arguments": "{not json"}],
            "status": "completed",
        });
        let e = parse_openai_responses_output(&v).unwrap_err();
        assert!(matches!(e.kind, ModelInvokeErrorKind::Undecodable));
        assert!(e.message.contains("arguments"));
    }

    #[test]
    fn unknown_output_items_are_skipped() {
        let v = json!({
            "output": [
                {"type": "web_search_call", "id": "ws_1"},
                {"type": "message", "content": [{"type": "output_text", "text": "a"}]},
            ],
            "status": "completed",
        });
        let out = parse_openai_responses_output(&v).unwrap();
        assert_eq!(out.response.text(), "a");
        assert!(out.response.tool_calls().next().is_none());
    }
}
