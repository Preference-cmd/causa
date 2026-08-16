//! Streaming delta translation for LLM providers (single implementation).
//!
//! This module owns the delta-level decoding of provider stream payloads
//! into [`AgentStreamEvent`]s — the only implementation in the
//! workspace. Concrete transports (currently `ReqwestSseStream` in
//! `agent-provider`) keep HTTP + SSE byte parsing and route parsed
//! events through the accumulators here.
//!
//! Three accumulators, one per wire protocol:
//!
//! - [`OpenAiStreamAccumulator`]: Chat Completions chunks, translated
//!   from the JSON payload directly (OpenAI sends plain `data:` lines
//!   with no `event:` field). The transport intercepts the literal
//!   `data: [DONE]` marker and emits the terminal `Done` with the
//!   last `finish_reason` (AC-01); this accumulator never emits `Done`
//!   itself.
//! - [`AnthropicStreamAccumulator`]: Messages events, discriminated by
//!   the SSE `event:` field (`message_start`, `content_block_*`,
//!   `message_delta`, `message_stop`). Emits the terminal `Done` on
//!   `message_stop`, carrying the `stop_reason` captured from
//!   `message_delta` (AC-01).
//! - [`ResponsesStreamAccumulator`]: OpenAI Responses events,
//!   discriminated by the SSE `event:` field. Argument deltas arrive
//!   base64-encoded and are decoded before accumulation. Emits the
//!   terminal `Done` on `response.completed` and a terminal `Error` on
//!   `response.failed` (the failure payload is surfaced, never a clean
//!   `Done`, R2-01).
//!
//! Decoding is permissive, matching the transport layer's live
//! behavior: unparseable chunks and unknown event types are silently
//! ignored (a provider blip must not fail a turn mid-stream). Tool-call
//! assembly drops partial entries whose `id`/`name` never arrived (with
//! a host-visible `Warning`, R2-04); argument fragments that do not
//! parse as JSON degrade to `Value::Null`. Provider-controlled
//! tool-call indices are capped at [`MAX_TOOL_CALL_SLOTS`] so a single
//! malformed event cannot trigger a giant allocation (R2-03).

use serde_json::Value;

use reimagine_agent_harness::{AgentStreamEvent, ToolCall, ToolCallId, Usage};

use crate::translation::usage::{openai_cached_tokens, openai_reasoning_tokens};

/// Upper bound on the number of tool-call slots an accumulator will
/// grow (R2-03). Provider-controlled `index` values are untrusted: a
/// single malformed/adversarial event must not drive a multi-GB
/// allocation via `while vec.len() <= index { push }`.
const MAX_TOOL_CALL_SLOTS: usize = 64;

/// OpenAI Chat Completions stream accumulator.
///
/// Holds partially-built tool calls by index plus the accumulated
/// assistant text (used by transports to decide whether an EOF without
/// a terminal marker still deserves a `Done`, AC-06). `ingest_chunk`
/// returns the events for one chunk, including the flush of complete
/// tool calls when the chunk carries `finish_reason: "tool_calls"`.
#[derive(Debug, Default)]
pub struct OpenAiStreamAccumulator {
    /// Accumulated assistant text (not emitted, only for content
    /// detection at EOF).
    text: String,
    /// Partially-built tool calls by index.
    tool_calls: Vec<OpenAiPartialToolCall>,
    /// Last `finish_reason` seen, emitted with the terminal `Done` by
    /// the transport (AC-01).
    finish_reason: Option<String>,
    /// Whether an out-of-range tool-call index has already been warned
    /// about (R2-03): warn once per stream, not per malformed event.
    slot_overflow_warned: bool,
}

#[derive(Debug, Default, Clone)]
struct OpenAiPartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl OpenAiStreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one OpenAI chat completion chunk, returning the events
    /// it produces (content/reasoning/tool-call deltas, a flush of
    /// complete tool calls on `finish_reason: "tool_calls"`, usage).
    pub fn ingest_chunk(&mut self, chunk: &Value) -> Vec<AgentStreamEvent> {
        let mut out = Vec::new();

        // Extract content delta.
        if let Some(delta_content) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|v| v.as_str())
            && !delta_content.is_empty()
        {
            self.text.push_str(delta_content);
            out.push(AgentStreamEvent::ContentDelta(delta_content.to_string()));
        }

        // Extract reasoning deltas (o-series / DeepSeek-style
        // `reasoning_content`). Display-only: never accumulated into
        // the assistant message.
        if let Some(reasoning) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("reasoning_content"))
            .and_then(|v| v.as_str())
            && !reasoning.is_empty()
        {
            out.push(AgentStreamEvent::ReasoningDelta(reasoning.to_string()));
        }

        // Extract tool call deltas.
        if let Some(tool_calls) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("tool_calls"))
            .and_then(|v| v.as_array())
        {
            for (i, call) in tool_calls.iter().enumerate() {
                let index = call
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .map(|i| i as usize)
                    .unwrap_or(i);
                // R2-03: provider-controlled index is untrusted; beyond
                // the slot cap the event is ignored (warn once).
                if index >= MAX_TOOL_CALL_SLOTS {
                    self.warn_slot_overflow(&mut out, index);
                    break;
                }
                while self.tool_calls.len() <= index {
                    self.tool_calls.push(OpenAiPartialToolCall::default());
                }
                let entry = &mut self.tool_calls[index];
                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                    entry.id = Some(id.to_string());
                }
                if let Some(name) = call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                {
                    entry.name = Some(name.to_string());
                }
                if let Some(args) = call
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                {
                    entry.arguments.push_str(args);
                }
            }
        }

        // Extract finish reason — flush complete tool calls and remember
        // the reason for the terminal `Done` (AC-01).
        if let Some(finish_reason) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
        {
            self.finish_reason = Some(finish_reason.to_string());
            if finish_reason == "tool_calls" {
                self.flush_tool_calls(&mut out);
            }
        }

        // Extract usage.
        if let Some(usage) = chunk.get("usage") {
            let input = usage.get("prompt_tokens").and_then(|v| v.as_u64());
            let output = usage.get("completion_tokens").and_then(|v| v.as_u64());
            let reasoning = openai_reasoning_tokens(usage);
            let cached = openai_cached_tokens(usage);
            out.push(AgentStreamEvent::Usage(
                Usage::new(input, output)
                    .with_reasoning_tokens(reasoning)
                    .with_cache_read(cached),
            ));
        }

        out
    }

    /// Flush complete tool calls into `out`. Entries missing required
    /// fields are dropped — surfaced as a `Warning` so dropped calls are
    /// visible to hosts (R2-04).
    fn flush_tool_calls(&mut self, out: &mut Vec<AgentStreamEvent>) {
        let mut dropped = 0usize;
        for partial in self.tool_calls.drain(..) {
            if let (Some(id), Some(name)) = (partial.id, partial.name) {
                let arguments = if partial.arguments.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                };
                out.push(AgentStreamEvent::ToolCall(ToolCall::new(
                    ToolCallId::new(id),
                    name,
                    arguments,
                )));
            } else {
                dropped += 1;
            }
        }
        if dropped > 0 {
            out.push(AgentStreamEvent::Warning(format!(
                "dropped {dropped} incomplete tool call(s) missing id/name at flush"
            )));
        }
    }

    /// Warn once that a tool-call index exceeded the slot cap (R2-03).
    fn warn_slot_overflow(&mut self, out: &mut Vec<AgentStreamEvent>, index: usize) {
        if !self.slot_overflow_warned {
            self.slot_overflow_warned = true;
            out.push(AgentStreamEvent::Warning(format!(
                "tool-call index {index} exceeds the {MAX_TOOL_CALL_SLOTS}-slot cap; ignoring out-of-range tool-call events"
            )));
        }
    }

    /// Whether any content or tool-call state was accumulated. Used by
    /// transports to decide whether an EOF without a terminal marker
    /// still terminates with a `Done` (AC-06).
    pub fn has_content(&self) -> bool {
        !self.text.is_empty() || !self.tool_calls.is_empty()
    }

    /// Whether any partial tool-call entries remain unflushed. Used by
    /// transports to warn that an EOF dropped incomplete tool calls
    /// (D-5).
    pub fn has_partial_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// The last `finish_reason` seen, consumed for the terminal `Done`.
    pub fn take_finish_reason(&mut self) -> Option<String> {
        self.finish_reason.take()
    }
}

/// Anthropic Messages stream accumulator keyed by content-block index.
///
/// Captures input-side usage from `message_start` (input_tokens plus
/// cache fields) and merges it into the report emitted on
/// `message_delta` (which carries output_tokens). Emits the terminal
/// `Done` on `message_stop`, carrying the `stop_reason` from
/// `message_delta` (AC-01) — truncation ("max_tokens") must reach the
/// loop so incomplete responses are not reported as final.
#[derive(Debug, Default)]
pub struct AnthropicStreamAccumulator {
    /// Accumulated assistant text (not emitted, only for content
    /// detection at EOF).
    text: String,
    /// Partially-built tool calls by content-block index.
    tool_calls: Vec<AnthropicPartialToolCall>,
    /// `stop_reason` captured from `message_delta`, emitted with the
    /// terminal `Done` on `message_stop` (AC-01).
    stop_reason: Option<String>,
    /// Input-side counts captured from `message_start`; merged into the
    /// usage report emitted on `message_delta`.
    input: Option<u64>,
    cache_creation: Option<u64>,
    cache_read: Option<u64>,
    /// Whether an out-of-range content-block index has already been
    /// warned about (R2-03).
    slot_overflow_warned: bool,
}

#[derive(Debug, Default, Clone)]
struct AnthropicPartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl AnthropicStreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one Anthropic SSE event, returning the events it
    /// produces. `event_type` is the SSE `event:` field value (the
    /// discriminator the transport layer passes through); `data` is the
    /// parsed JSON payload.
    pub fn ingest_event(
        &mut self,
        event_type: Option<&str>,
        data: &Value,
    ) -> Vec<AgentStreamEvent> {
        let mut out = Vec::new();
        let Some(event_type) = event_type else {
            return out;
        };

        match event_type {
            "content_block_start" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    if index >= MAX_TOOL_CALL_SLOTS {
                        self.warn_slot_overflow(&mut out, index);
                        return out;
                    }
                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(AnthropicPartialToolCall::default());
                    }
                    if let Some(block) = data.get("content_block")
                        && block.get("type").and_then(|v| v.as_str()) == Some("tool_use")
                    {
                        if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                            self.tool_calls[index].id = Some(id.to_string());
                        }
                        if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                            self.tool_calls[index].name = Some(name.to_string());
                        }
                    }
                }
            }
            "content_block_delta" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    if let Some(delta) = data.get("delta") {
                        match delta.get("type").and_then(|v| v.as_str()) {
                            Some("text_delta") => {
                                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                    self.text.push_str(text);
                                    out.push(AgentStreamEvent::ContentDelta(text.to_string()));
                                }
                            }
                            Some("thinking_delta") => {
                                if let Some(text) = delta.get("thinking").and_then(|v| v.as_str()) {
                                    out.push(AgentStreamEvent::ReasoningDelta(text.to_string()));
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(partial) =
                                    delta.get("partial_json").and_then(|v| v.as_str())
                                {
                                    if index >= MAX_TOOL_CALL_SLOTS {
                                        self.warn_slot_overflow(&mut out, index);
                                        return out;
                                    }
                                    while self.tool_calls.len() <= index {
                                        self.tool_calls.push(AnthropicPartialToolCall::default());
                                    }
                                    self.tool_calls[index].arguments.push_str(partial);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "content_block_stop" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    if let Some(partial) = self.tool_calls.get_mut(index)
                        && let (Some(id), Some(name)) = (partial.id.clone(), partial.name.clone())
                    {
                        let arguments = if partial.arguments.is_empty() {
                            Value::Null
                        } else {
                            serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                        };
                        *partial = AnthropicPartialToolCall::default();
                        out.push(AgentStreamEvent::ToolCall(ToolCall::new(
                            ToolCallId::new(id),
                            name,
                            arguments,
                        )));
                    }
                }
            }
            "message_start" => {
                // `message_start` carries input-side counts
                // (input_tokens, cache_creation/read); `message_delta`
                // carries output_tokens. Capture here, merge below.
                if let Some(usage) = data.get("message").and_then(|m| m.get("usage")) {
                    self.input = usage.get("input_tokens").and_then(|v| v.as_u64());
                    self.cache_creation = usage
                        .get("cache_creation_input_tokens")
                        .and_then(|v| v.as_u64());
                    self.cache_read = usage
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_u64());
                }
            }
            "message_delta" => {
                if let Some(delta) = data.get("delta")
                    && let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str())
                {
                    // Stash the provider stop_reason; it is emitted with
                    // the terminal `Done` on `message_stop` (AC-01).
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(usage) = data.get("usage") {
                    let input = self
                        .input
                        .or_else(|| usage.get("input_tokens").and_then(|v| v.as_u64()));
                    let output = usage.get("output_tokens").and_then(|v| v.as_u64());
                    let cache_creation = self.cache_creation.or_else(|| {
                        usage
                            .get("cache_creation_input_tokens")
                            .and_then(|v| v.as_u64())
                    });
                    let cache_read = self.cache_read.or_else(|| {
                        usage
                            .get("cache_read_input_tokens")
                            .and_then(|v| v.as_u64())
                    });
                    out.push(AgentStreamEvent::Usage(
                        Usage::new(input, output)
                            .with_cache_creation(cache_creation)
                            .with_cache_read(cache_read),
                    ));
                }
            }
            "message_stop" => {
                out.push(AgentStreamEvent::Done {
                    stop_reason: self.stop_reason.take(),
                });
            }
            _ => {}
        }

        out
    }

    /// Whether any content or tool-call state was accumulated. Used by
    /// transports to decide whether an EOF without a terminal marker
    /// still terminates with a `Done` (AC-06).
    pub fn has_content(&self) -> bool {
        !self.text.is_empty() || !self.tool_calls.is_empty()
    }

    /// Whether any partial tool-call entries remain unflushed. Used by
    /// transports to warn that an EOF dropped incomplete tool calls
    /// (D-5).
    pub fn has_partial_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// The `stop_reason` captured from `message_delta`, consumed for
    /// the terminal `Done` (AC-01).
    pub fn take_stop_reason(&mut self) -> Option<String> {
        self.stop_reason.take()
    }

    /// Warn once that a content-block index exceeded the slot cap
    /// (R2-03).
    fn warn_slot_overflow(&mut self, out: &mut Vec<AgentStreamEvent>, index: usize) {
        if !self.slot_overflow_warned {
            self.slot_overflow_warned = true;
            out.push(AgentStreamEvent::Warning(format!(
                "content-block index {index} exceeds the {MAX_TOOL_CALL_SLOTS}-slot cap; ignoring out-of-range tool-call events"
            )));
        }
    }
}

/// OpenAI Responses stream accumulator.
///
/// V1 handles the event families the Agent loop consumes:
/// `response.output_text.delta` (content), the `response.function_call*`
/// family (tool calls), `response.completed` (usage + done), and
/// `response.failed` (terminal error, R2-01 — the failure payload is
/// surfaced as an `Error` event, never a clean `Done`). `response.created`,
/// `response.in_progress`, reasoning deltas, and item bookkeeping
/// events are ignored. Argument deltas arrive base64-encoded and are
/// decoded before accumulation.
#[derive(Debug, Default)]
pub struct ResponsesStreamAccumulator {
    /// Accumulated assistant text (not emitted, only for content
    /// detection at EOF, R2-04).
    text: String,
    /// Partially-built tool calls by `output_index`. Entries are reset
    /// once the complete item arrives in `response.output_item.done`.
    tool_calls: Vec<ResponsesPartialToolCall>,
    /// Whether an out-of-range output index has already been warned
    /// about (R2-03).
    slot_overflow_warned: bool,
}

/// Partial tool call for the OpenAI Responses path. The Responses API
/// streams only base64-encoded argument fragments in
/// `response.function_call_arguments.delta`; id and name arrive with the
/// complete item in `response.output_item.done`.
#[derive(Debug, Default, Clone)]
struct ResponsesPartialToolCall {
    arguments: String,
}

impl ResponsesStreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one OpenAI Responses SSE event, returning the events it
    /// produces. `event_type` is the SSE `event:` field value; `data` is
    /// the parsed JSON payload.
    pub fn ingest_event(
        &mut self,
        event_type: Option<&str>,
        data: &Value,
    ) -> Vec<AgentStreamEvent> {
        let mut out = Vec::new();
        let Some(event_type) = event_type else {
            return out;
        };

        match event_type {
            "response.output_text.delta" => {
                if let Some(text) = data.get("delta").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    self.text.push_str(text);
                    out.push(AgentStreamEvent::ContentDelta(text.to_string()));
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(text) = data.get("delta").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    out.push(AgentStreamEvent::ReasoningDelta(text.to_string()));
                }
            }
            "response.function_call_arguments.delta" => {
                let Some(delta) = data.get("delta").and_then(|v| v.as_str()) else {
                    return out;
                };
                let index = data
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|i| i as usize)
                    .unwrap_or(0);
                if index >= MAX_TOOL_CALL_SLOTS {
                    self.warn_slot_overflow(&mut out, index);
                    return out;
                }
                while self.tool_calls.len() <= index {
                    self.tool_calls.push(ResponsesPartialToolCall::default());
                }
                // Arguments deltas are base64-encoded JSON fragments;
                // decode before accumulating. Tolerate plain fragments.
                let decoded = decode_base64(delta)
                    .and_then(|bytes| String::from_utf8(bytes).ok())
                    .unwrap_or_else(|| delta.to_string());
                self.tool_calls[index].arguments.push_str(&decoded);
            }
            "response.function_call_arguments.done" => {
                // The provider may deliver the full arguments here; when
                // the accumulated deltas are empty, use this payload.
                let index = data
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|i| i as usize)
                    .unwrap_or(0);
                if let Some(arguments) = data.get("arguments").and_then(|v| v.as_str())
                    && !arguments.is_empty()
                {
                    if index >= MAX_TOOL_CALL_SLOTS {
                        self.warn_slot_overflow(&mut out, index);
                        return out;
                    }
                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(ResponsesPartialToolCall::default());
                    }
                    if self.tool_calls[index].arguments.is_empty() {
                        self.tool_calls[index].arguments = arguments.to_string();
                    }
                }
            }
            "response.output_item.done" => {
                let Some(item) = data.get("item") else {
                    return out;
                };
                if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
                    return out;
                }
                let Some(name) = item.get("name").and_then(|v| v.as_str()) else {
                    return out;
                };
                let output_index = data
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|i| i as usize)
                    .unwrap_or(0);
                // The streamed item carries the full arguments; prefer the
                // complete payload over accumulated deltas.
                let arguments = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        self.tool_calls
                            .get(output_index)
                            .map(|partial| partial.arguments.clone())
                    })
                    .unwrap_or_else(|| "{}".to_string());
                // The Responses API uses `call_id` as the stable id that
                // tool-result messages reference; fall back to `id`.
                let id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("id").and_then(|v| v.as_str()))
                    .unwrap_or("");
                let arguments_value = serde_json::from_str(&arguments).unwrap_or(Value::Null);
                out.push(AgentStreamEvent::ToolCall(ToolCall::new(
                    ToolCallId::new(id),
                    name.to_string(),
                    arguments_value,
                )));
                // The complete item has been delivered; reset the partial
                // entry so an EOF cannot report a false "incomplete tool
                // call" (R2-04).
                if let Some(partial) = self.tool_calls.get_mut(output_index) {
                    *partial = ResponsesPartialToolCall::default();
                }
            }
            "response.compaction" => {
                // Server-side compaction (PV-01b reserved channel,
                // consumed in CM-V2e): the provider replaced earlier
                // items with an opaque compaction item. Informational
                // for the runtime.
                if let Some(item_id) = data.get("item_id").and_then(|v| v.as_str()) {
                    out.push(AgentStreamEvent::Compacted {
                        item_id: item_id.to_string(),
                    });
                }
            }
            "response.completed" => {
                if let Some(usage) = data.get("response").and_then(|r| r.get("usage")) {
                    let input = usage.get("input_tokens").and_then(|v| v.as_u64());
                    let output = usage.get("output_tokens").and_then(|v| v.as_u64());
                    let reasoning = usage
                        .get("output_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                        .and_then(|v| v.as_u64());
                    let cached = openai_cached_tokens(usage);
                    out.push(AgentStreamEvent::Usage(
                        Usage::new(input, output)
                            .with_reasoning_tokens(reasoning)
                            .with_cache_read(cached),
                    ));
                }
                out.push(AgentStreamEvent::Done { stop_reason: None });
            }
            "response.failed" => {
                // Terminal failure: surface the error payload as an
                // `Error` event instead of a clean `Done` so a failed
                // stream can never be reported as a successful empty
                // turn (R2-01).
                let message = data
                    .get("response")
                    .and_then(|r| r.get("error"))
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        data.get("response")
                            .and_then(|r| r.get("status"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_else(|| "response failed".to_string());
                out.push(AgentStreamEvent::Error(format!(
                    "provider stream failed: {message}"
                )));
            }
            _ => {}
        }

        out
    }

    /// Whether any content or tool-call state was accumulated. Used by
    /// transports to decide whether an EOF without a terminal marker
    /// still terminates with a `Done` (AC-06). Text received via
    /// `output_text.delta` counts; completed tool calls no longer do
    /// (their partial entries are reset on `output_item.done`, R2-04).
    pub fn has_content(&self) -> bool {
        !self.text.is_empty() || self.tool_calls.iter().any(|t| !t.arguments.is_empty())
    }

    /// Whether any partial tool-call entries remain unflushed. Used by
    /// transports to warn that an EOF dropped incomplete tool calls
    /// (D-5). Completed calls are reset in `output_item.done` (R2-04),
    /// so a fully delivered stream never reports a false warning.
    pub fn has_partial_tool_calls(&self) -> bool {
        self.tool_calls.iter().any(|t| !t.arguments.is_empty())
    }

    /// Warn once that an output index exceeded the slot cap (R2-03).
    fn warn_slot_overflow(&mut self, out: &mut Vec<AgentStreamEvent>, index: usize) {
        if !self.slot_overflow_warned {
            self.slot_overflow_warned = true;
            out.push(AgentStreamEvent::Warning(format!(
                "output index {index} exceeds the {MAX_TOOL_CALL_SLOTS}-slot cap; ignoring out-of-range tool-call events"
            )));
        }
    }
}

/// Decode a base64 string (standard alphabet with `=` padding) into raw
/// bytes.
///
/// Mirrors the strict behavior of the `base64` crate's STANDARD engine
/// (already a workspace dependency, but `ai-protocol` must not grow new
/// dependencies): input length must be a multiple of 4, padding only
/// appears as a trailing run of at most two `=`, and every non-padding
/// character must be in the standard alphabet. Invalid input yields
/// `None` so callers can fall back to treating the fragment as plain
/// text (some proxies stream unencoded fragments).
fn decode_base64(input: &str) -> Option<Vec<u8>> {
    let bytes = input.as_bytes();
    if bytes.is_empty() {
        return Some(Vec::new());
    }
    if bytes.len() % 4 != 0 {
        return None;
    }
    // Locate the trailing padding run, if any.
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b'=' {
        end -= 1;
    }
    if bytes.len() - end > 2 {
        return None;
    }
    let mut out = Vec::with_capacity(end / 4 * 3);
    let mut i = 0;
    while i + 4 <= end {
        let a = base64_digit(bytes[i])?;
        let b = base64_digit(bytes[i + 1])?;
        let c = base64_digit(bytes[i + 2])?;
        let d = base64_digit(bytes[i + 3])?;
        out.push((a << 2) | (b >> 4));
        out.push(((b & 0x0F) << 4) | (c >> 2));
        out.push(((c & 0x03) << 6) | d);
        i += 4;
    }
    // Trailing partial group (2 or 3 data chars, padded with `=`).
    let remaining = end - i;
    if remaining == 2 {
        let a = base64_digit(bytes[i])?;
        let b = base64_digit(bytes[i + 1])?;
        out.push((a << 2) | (b >> 4));
    } else if remaining == 3 {
        let a = base64_digit(bytes[i])?;
        let b = base64_digit(bytes[i + 1])?;
        let c = base64_digit(bytes[i + 2])?;
        out.push((a << 2) | (b >> 4));
        out.push(((b & 0x0F) << 4) | (c >> 2));
    }
    Some(out)
}

/// Map one base64 alphabet character to its 6-bit value.
fn base64_digit(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decode_base64_standard_vectors() {
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("eyJ4Ijo=").unwrap(), b"{\"x\":");
        assert_eq!(decode_base64("ICA0Mn0=").unwrap(), b"  42}");
        assert_eq!(decode_base64("").unwrap(), b"");
        // 3 bytes -> no padding; 1 byte -> single padding.
        assert_eq!(decode_base64("YWJj").unwrap(), b"abc");
        assert_eq!(decode_base64("YQ==").unwrap(), b"a");
        // 2 bytes -> one padding char.
        assert_eq!(decode_base64("YWI=").unwrap(), b"ab");
    }

    #[test]
    fn decode_base64_rejects_invalid_input() {
        // Bad alphabet character.
        assert!(decode_base64("aGVsbG8*").is_none());
        // Length not a multiple of 4 (canonical padding required,
        // matching the base64 crate's STANDARD engine).
        assert!(decode_base64("aGVsbG8").is_none());
        assert!(decode_base64("YQ").is_none());
        // More than two padding chars.
        assert!(decode_base64("YQ======").is_none());
        // Padding in the middle.
        assert!(decode_base64("aG=sbA==").is_none());
    }

    #[test]
    fn openai_accumulator_flushes_tool_calls_on_tool_calls_finish() {
        let mut acc = OpenAiStreamAccumulator::new();
        let mut events = acc.ingest_chunk(&json!({
            "choices": [{
                "delta": { "role": "assistant", "content": "He" }
            }]
        }));
        assert!(matches!(events[0], AgentStreamEvent::ContentDelta(_)));
        events = acc.ingest_chunk(&json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "c1",
                        "function": { "name": "echo", "arguments": "{\"x\":" }
                    }]
                }
            }]
        }));
        assert!(events.is_empty(), "partial tool calls emit no events");
        assert!(
            acc.has_partial_tool_calls(),
            "partial entries are tracked until the flush"
        );
        events = acc.ingest_chunk(&json!({
            "choices": [{
                "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": "1}" } }] },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 4 }
        }));
        assert_eq!(events.len(), 2, "flush + usage");
        let AgentStreamEvent::ToolCall(call) = &events[0] else {
            panic!("expected ToolCall, got {:?}", events[0]);
        };
        assert_eq!(call.id().as_str(), "c1");
        assert_eq!(call.name(), "echo");
        assert_eq!(call.arguments(), &json!({"x": 1}));
        assert_eq!(acc.take_finish_reason().as_deref(), Some("tool_calls"));
        assert!(acc.has_content());
        assert!(
            !acc.has_partial_tool_calls(),
            "the flush drained all partial entries"
        );
    }

    #[test]
    fn anthropic_accumulator_tracks_partial_tool_calls() {
        let mut acc = AnthropicStreamAccumulator::new();
        assert!(!acc.has_partial_tool_calls());
        // A tool_use block starts but only argument fragments arrive
        // (no id/name): the entry stays partial.
        acc.ingest_event(
            Some("content_block_start"),
            &json!({"index": 0, "content_block": {"type": "tool_use"}}),
        );
        acc.ingest_event(
            Some("content_block_delta"),
            &json!({"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"x\":"}}),
        );
        assert!(acc.has_partial_tool_calls());
        // content_block_stop without id/name emits nothing and the
        // partial entry remains tracked (D-5).
        let events = acc.ingest_event(Some("content_block_stop"), &json!({"index": 0}));
        assert!(events.is_empty(), "incomplete call is not flushed");
        assert!(acc.has_partial_tool_calls());
    }

    #[test]
    fn responses_accumulator_tracks_partial_tool_calls() {
        let mut acc = ResponsesStreamAccumulator::new();
        assert!(!acc.has_partial_tool_calls());
        acc.ingest_event(
            Some("response.function_call_arguments.delta"),
            &json!({"output_index": 0, "delta": "eyJ4IjogMX0="}),
        );
        assert!(acc.has_partial_tool_calls());
        // A completed item resets the partial entry (R2-04): the stream
        // no longer reports a false "incomplete tool call" at EOF, and
        // content detection drops the delivered call.
        let events = acc.ingest_event(
            Some("response.output_item.done"),
            &json!({"output_index": 0, "item": {"type": "function_call", "name": "echo", "arguments": "{}", "call_id": "c1"}}),
        );
        assert!(
            matches!(events.first(), Some(AgentStreamEvent::ToolCall(_))),
            "completed item emits the ToolCall"
        );
        assert!(
            !acc.has_partial_tool_calls(),
            "completed call is reset (R2-04)"
        );
        assert!(
            !acc.has_content(),
            "a delivered call alone is not pending content"
        );
        // Text deltas count as content (R2-04).
        acc.ingest_event(
            Some("response.output_text.delta"),
            &json!({"delta": "hello"}),
        );
        assert!(acc.has_content());
        assert!(!acc.has_partial_tool_calls());
    }

    #[test]
    fn responses_failed_surfaces_error_instead_of_done() {
        let mut acc = ResponsesStreamAccumulator::new();
        let events = acc.ingest_event(
            Some("response.failed"),
            &json!({"response": {"status": "failed", "error": {"code": "server_error", "message": "upstream exploded"}}}),
        );
        let Some(AgentStreamEvent::Error(message)) = events.first() else {
            panic!("response.failed must emit a terminal Error, got {events:?}");
        };
        assert!(message.contains("upstream exploded"));
        // Fallback when no error payload is present.
        let events =
            acc.ingest_event(Some("response.failed"), &json!({"response": {"status": "failed"}}));
        assert!(matches!(events.first(), Some(AgentStreamEvent::Error(m)) if m.contains("failed")));
    }

    #[test]
    fn out_of_range_indices_are_capped_not_allocated() {
        // R2-03: a single malformed event with a giant index must not
        // trigger a multi-GB allocation; it is ignored with a Warning.
        let mut openai = OpenAiStreamAccumulator::new();
        let events = openai.ingest_chunk(&json!({
            "choices": [{"delta": {"tool_calls": [{"index": 2_000_000_000, "id": "t1", "function": {"name": "echo", "arguments": "{}"}}]}}]
        }));
        assert!(matches!(events.first(), Some(AgentStreamEvent::Warning(_))));
        assert!(!openai.has_partial_tool_calls());

        let mut anthropic = AnthropicStreamAccumulator::new();
        let events = anthropic.ingest_event(
            Some("content_block_start"),
            &json!({"index": 2_000_000_000, "content_block": {"type": "tool_use", "id": "t1", "name": "echo"}}),
        );
        assert!(matches!(events.first(), Some(AgentStreamEvent::Warning(_))));
        assert!(!anthropic.has_partial_tool_calls());

        let mut responses = ResponsesStreamAccumulator::new();
        let events = responses.ingest_event(
            Some("response.function_call_arguments.delta"),
            &json!({"output_index": 2_000_000_000, "delta": "e30="}),
        );
        assert!(matches!(events.first(), Some(AgentStreamEvent::Warning(_))));
        assert!(!responses.has_partial_tool_calls());
    }

    #[test]
    fn openai_flush_warns_on_dropped_incomplete_calls() {
        // R2-04: entries missing id/name are dropped at flush and the
        // drop is visible to hosts.
        let mut acc = OpenAiStreamAccumulator::new();
        acc.ingest_chunk(&json!({
            "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"x\":1}"}}]}}]
        }));
        assert!(acc.has_partial_tool_calls());
        let events = acc.ingest_chunk(&json!({
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}]
        }));
        assert!(
            events.iter().any(|e| matches!(e, AgentStreamEvent::Warning(m) if m.contains("dropped"))),
            "dropped incomplete calls surface a Warning, got {events:?}"
        );
        assert!(!acc.has_partial_tool_calls());
    }
}
