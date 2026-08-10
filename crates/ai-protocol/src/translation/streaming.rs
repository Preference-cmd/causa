use serde_json::Value;

use reimagine_agent_harness::{AgentStreamEvent, ToolCall, ToolCallId, Usage};

use crate::error::ProviderAdapterError;

/// OpenAI stream delta accumulator. Holds partially-built tool calls by index.
#[derive(Debug, Default)]
pub struct OpenAiStreamAccumulator {
    pub calls: Vec<PartialToolCall>,
    pub stop_reason: Option<String>,
    pub usage: Option<Usage>,
    pub done: bool,
}

#[derive(Debug, Default, Clone)]
pub struct PartialToolCall {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

impl OpenAiStreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one OpenAI chat completion chunk. Returns events for this chunk,
    /// excluding complete tool-call flushes.
    pub fn ingest_chunk(
        &mut self,
        chunk: &Value,
    ) -> Result<Vec<AgentStreamEvent>, ProviderAdapterError> {
        let mut out = Vec::new();
        if let Some(delta_content) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|v| v.as_str())
            && !delta_content.is_empty()
        {
            out.push(AgentStreamEvent::ContentDelta(delta_content.to_string()));
        }
        // Reasoning content (OpenAI o-series `reasoning_content` delta,
        // DeepSeek-style servers). Display-only: hosts surface it as
        // progress, the runtime never replays it into history.
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
                while self.calls.len() <= index {
                    self.calls.push(PartialToolCall::default());
                }
                let entry = &mut self.calls[index];
                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                    entry.id = Some(id.to_string());
                    out.push(AgentStreamEvent::ToolCallDelta {
                        index: index as u32,
                        id: Some(ToolCallId::new(id.to_string())),
                        name: None,
                        arguments_delta: None,
                    });
                }
                if let Some(name) = call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                {
                    entry.name = Some(name.to_string());
                    out.push(AgentStreamEvent::ToolCallDelta {
                        index: index as u32,
                        id: None,
                        name: Some(name.to_string()),
                        arguments_delta: None,
                    });
                }
                if let Some(args) = call
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                {
                    entry.arguments.push_str(args);
                    out.push(AgentStreamEvent::ToolCallDelta {
                        index: index as u32,
                        id: None,
                        name: None,
                        arguments_delta: Some(args.to_string()),
                    });
                }
            }
        }
        if let Some(reason) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
        {
            self.stop_reason = Some(reason.to_string());
        }
        if let Some(usage) = chunk.get("usage") {
            let input = usage.get("prompt_tokens").and_then(|v| v.as_u64());
            let output = usage.get("completion_tokens").and_then(|v| v.as_u64());
            let reasoning = crate::translation::usage::openai_reasoning_tokens(usage);
            // Cache hits (prompt_tokens_details on Chat Completions,
            // input_tokens_details on Responses) map onto cache_read.
            let cached = crate::translation::usage::openai_cached_tokens(usage);
            let usage_report = Usage::new(input, output)
                .with_reasoning_tokens(reasoning)
                .with_cache_read(cached);
            self.usage = Some(usage_report.clone());
            out.push(AgentStreamEvent::Usage(usage_report));
        }
        Ok(out)
    }

    /// Flush complete tool calls. Entries missing required fields are dropped.
    pub fn flush_complete_tool_calls(&mut self) -> Vec<AgentStreamEvent> {
        let mut out = Vec::new();
        for partial in self.calls.drain(..) {
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
            }
        }
        out
    }

    pub fn finalize(mut self) -> AgentStreamEvent {
        self.done = true;
        AgentStreamEvent::Done {
            stop_reason: self.stop_reason,
        }
    }
}

/// Anthropic stream accumulator keyed by content-block index.
#[derive(Debug, Default)]
pub struct AnthropicStreamAccumulator {
    pub text: String,
    pub tool_calls: Vec<PartialToolCall>,
    pub stop_reason: Option<String>,
    pub usage: Option<Usage>,
    pub done: bool,
    /// Input-side counts captured from `message_start` (which carries
    /// `input_tokens` and the cache fields); merged into the usage report
    /// emitted on `message_delta` (which carries `output_tokens`).
    input: Option<u64>,
    cache_creation: Option<u64>,
    cache_read: Option<u64>,
}

impl AnthropicStreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest_event(
        &mut self,
        event: &Value,
    ) -> Result<Vec<AgentStreamEvent>, ProviderAdapterError> {
        let mut out = Vec::new();
        let event_type = event.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
            ProviderAdapterError::serialization("anthropic stream event missing type")
        })?;
        match event_type {
            "content_block_start" => {
                let index = event.get("index").and_then(|v| v.as_u64()).ok_or_else(|| {
                    ProviderAdapterError::serialization("content_block_start missing index")
                })? as usize;
                let block = event.get("content_block");
                while self.tool_calls.len() <= index {
                    self.tool_calls.push(PartialToolCall::default());
                }
                if let Some(b) = block
                    && b.get("type").and_then(|v| v.as_str()) == Some("tool_use")
                {
                    if let Some(id) = b.get("id").and_then(|v| v.as_str()) {
                        self.tool_calls[index].id = Some(id.to_string());
                        out.push(AgentStreamEvent::ToolCallDelta {
                            index: index as u32,
                            id: Some(ToolCallId::new(id.to_string())),
                            name: None,
                            arguments_delta: None,
                        });
                    }
                    if let Some(name) = b.get("name").and_then(|v| v.as_str()) {
                        self.tool_calls[index].name = Some(name.to_string());
                        out.push(AgentStreamEvent::ToolCallDelta {
                            index: index as u32,
                            id: None,
                            name: Some(name.to_string()),
                            arguments_delta: None,
                        });
                    }
                }
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(|v| v.as_u64()).ok_or_else(|| {
                    ProviderAdapterError::serialization("content_block_delta missing index")
                })? as usize;
                let delta = event.get("delta");
                if let Some(d) = delta {
                    match d.get("type").and_then(|v| v.as_str()) {
                        Some("text_delta") => {
                            if let Some(t) = d.get("text").and_then(|v| v.as_str()) {
                                self.text.push_str(t);
                                out.push(AgentStreamEvent::ContentDelta(t.to_string()));
                            }
                        }
                        Some("thinking_delta") => {
                            if let Some(t) = d.get("thinking").and_then(|v| v.as_str()) {
                                out.push(AgentStreamEvent::ReasoningDelta(t.to_string()));
                            }
                        }
                        Some("input_json_delta") => {
                            if let Some(partial) = d.get("partial_json").and_then(|v| v.as_str()) {
                                while self.tool_calls.len() <= index {
                                    self.tool_calls.push(PartialToolCall::default());
                                }
                                self.tool_calls[index].arguments.push_str(partial);
                                out.push(AgentStreamEvent::ToolCallDelta {
                                    index: index as u32,
                                    id: None,
                                    name: None,
                                    arguments_delta: Some(partial.to_string()),
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            "content_block_stop" => {
                let index = event.get("index").and_then(|v| v.as_u64()).ok_or_else(|| {
                    ProviderAdapterError::serialization("content_block_stop missing index")
                })? as usize;
                if let Some(partial) = self.tool_calls.get_mut(index)
                    && let (Some(id), Some(name)) = (partial.id.clone(), partial.name.clone())
                {
                    let arguments = if partial.arguments.is_empty() {
                        Value::Null
                    } else {
                        serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                    };
                    *partial = PartialToolCall::default();
                    out.push(AgentStreamEvent::ToolCall(ToolCall::new(
                        ToolCallId::new(id),
                        name,
                        arguments,
                    )));
                }
            }
            "message_start" => {
                // `message_start` carries input-side counts
                // (input_tokens, cache_creation/read); `message_delta`
                // carries output_tokens. Capture here, merge below.
                if let Some(usage) = event
                    .get("message")
                    .and_then(|m| m.get("usage"))
                {
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
                if let Some(delta) = event.get("delta")
                    && let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str())
                {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(usage) = event.get("usage") {
                    let input = self.input.or_else(|| {
                        usage.get("input_tokens").and_then(|v| v.as_u64())
                    });
                    let output = usage.get("output_tokens").and_then(|v| v.as_u64());
                    let cache_creation = self.cache_creation.or_else(|| {
                        usage.get("cache_creation_input_tokens").and_then(|v| v.as_u64())
                    });
                    let cache_read = self.cache_read.or_else(|| {
                        usage.get("cache_read_input_tokens").and_then(|v| v.as_u64())
                    });
                    let report = Usage::new(input, output)
                        .with_cache_creation(cache_creation)
                        .with_cache_read(cache_read);
                    self.usage = Some(report.clone());
                    out.push(AgentStreamEvent::Usage(report));
                }
            }
            "message_stop" => {
                self.done = true;
                out.push(AgentStreamEvent::Done {
                    stop_reason: self.stop_reason.take(),
                });
            }
            _ => {}
        }
        Ok(out)
    }
}
