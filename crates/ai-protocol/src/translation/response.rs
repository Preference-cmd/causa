use serde_json::Value;

use reimagine_agent_harness::{AgentResponse, Message, ToolCall, ToolCallId, Usage};

use crate::error::ProviderAdapterError;

/// Translate an OpenAI-compatible chat completion response JSON into an
/// [`AgentResponse`].
pub fn from_openai_response(value: &Value) -> Result<AgentResponse, ProviderAdapterError> {
    let choice0 = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| ProviderAdapterError::serialization("missing choices[0]"))?;
    let message = choice0
        .get("message")
        .ok_or_else(|| ProviderAdapterError::serialization("missing choices[0].message"))?;
    let finish_reason = choice0
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let tool_calls = parse_openai_tool_calls(message.get("tool_calls"))?;

    let message = if tool_calls.is_empty() {
        Message::assistant(content)
    } else {
        Message::assistant_with_tool_calls(content, tool_calls)
    };

    let mut resp = AgentResponse::new(message);
    if let Some(reason) = finish_reason {
        resp = resp.with_stop_reason(reason);
    }
    if let Some(usage) = parse_openai_usage(value.get("usage"))? {
        resp = resp.with_usage(usage);
    }
    Ok(resp)
}

fn parse_openai_tool_calls(value: Option<&Value>) -> Result<Vec<ToolCall>, ProviderAdapterError> {
    let mut out = Vec::new();
    let Some(arr) = value.and_then(|v| v.as_array()) else {
        return Ok(out);
    };
    for (i, call) in arr.iter().enumerate() {
        let id = call
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ProviderAdapterError::serialization(format!("tool_calls[{i}].id missing"))
            })?
            .to_string();
        let name = call
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ProviderAdapterError::serialization(format!(
                    "tool_calls[{i}].function.name missing"
                ))
            })?
            .to_string();
        let args_str = call
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
            .unwrap_or("{}");
        let arguments = serde_json::from_str(args_str).map_err(|e| {
            ProviderAdapterError::serialization(format!("tool_calls[{i}].function.arguments: {e}"))
        })?;
        out.push(ToolCall::new(ToolCallId::new(id), name, arguments));
    }
    Ok(out)
}

fn parse_openai_usage(value: Option<&Value>) -> Result<Option<Usage>, ProviderAdapterError> {
    let Some(usage) = value else { return Ok(None) };
    let input = usage.get("prompt_tokens").and_then(|v| v.as_u64());
    let output = usage.get("completion_tokens").and_then(|v| v.as_u64());
    // Cache hits (prompt_tokens_details.cached_tokens on Chat Completions,
    // input_tokens_details on Responses) map onto the cache_read slot of
    // the shared budget convention.
    let cached = crate::translation::usage::openai_cached_tokens(usage);
    let reasoning = crate::translation::usage::openai_reasoning_tokens(usage);
    Ok(Some(
        Usage::new(input, output)
            .with_reasoning_tokens(reasoning)
            .with_cache_read(cached),
    ))
}

/// Translate an Anthropic messages response JSON into an [`AgentResponse`].
pub fn from_anthropic_response(value: &Value) -> Result<AgentResponse, ProviderAdapterError> {
    let content_arr = value
        .get("content")
        .and_then(|c| c.as_array())
        .ok_or_else(|| ProviderAdapterError::serialization("missing content array"))?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for (i, block) in content_arr.iter().enumerate() {
        match block.get("type").and_then(|v| v.as_str()) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ProviderAdapterError::serialization(format!("content[{i}].id missing"))
                    })?
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ProviderAdapterError::serialization(format!("content[{i}].name missing"))
                    })?
                    .to_string();
                let arguments = block.get("input").cloned().unwrap_or(Value::Null);
                tool_calls.push(ToolCall::new(ToolCallId::new(id), name, arguments));
            }
            _ => {}
        }
    }
    let message = if tool_calls.is_empty() {
        Message::assistant(text)
    } else {
        Message::assistant_with_tool_calls(text, tool_calls)
    };
    let mut resp = AgentResponse::new(message);
    if let Some(reason) = value.get("stop_reason").and_then(|v| v.as_str()) {
        resp = resp.with_stop_reason(reason.to_string());
    }
    if let Some(usage) = value.get("usage") {
        let input = usage.get("input_tokens").and_then(|v| v.as_u64());
        let output = usage.get("output_tokens").and_then(|v| v.as_u64());
        let cache_creation = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64());
        let cache_read = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64());
        resp = resp.with_usage(
            Usage::new(input, output)
                .with_cache_creation(cache_creation)
                .with_cache_read(cache_read),
        );
    }
    Ok(resp)
}

/// Translate an OpenAI Responses API response JSON into an
/// [`AgentResponse`]. `output` items of type `message` contribute
/// `output_text` content blocks; items of type `function_call` become
/// tool calls (their `arguments` is a JSON string).
pub fn from_responses_response(value: &Value) -> Result<AgentResponse, ProviderAdapterError> {
    let output = value
        .get("output")
        .and_then(|o| o.as_array())
        .ok_or_else(|| ProviderAdapterError::serialization("missing output array"))?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for (i, item) in output.iter().enumerate() {
        match item.get("type").and_then(|v| v.as_str()) {
            Some("message") => {
                if let Some(blocks) = item.get("content").and_then(|c| c.as_array()) {
                    for block in blocks {
                        if block.get("type").and_then(|v| v.as_str()) == Some("output_text")
                            && let Some(t) = block.get("text").and_then(|v| v.as_str())
                        {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                    }
                }
            }
            Some("function_call") => {
                let id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ProviderAdapterError::serialization(format!("output[{i}].call_id missing"))
                    })?
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ProviderAdapterError::serialization(format!("output[{i}].name missing"))
                    })?
                    .to_string();
                let args_str = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let arguments = serde_json::from_str(args_str).map_err(|e| {
                    ProviderAdapterError::serialization(format!("output[{i}].arguments: {e}"))
                })?;
                tool_calls.push(ToolCall::new(ToolCallId::new(id), name, arguments));
            }
            _ => {}
        }
    }
    let message = if tool_calls.is_empty() {
        Message::assistant(text)
    } else {
        Message::assistant_with_tool_calls(text, tool_calls)
    };
    let mut resp = AgentResponse::new(message);
    if let Some(usage) = value.get("usage") {
        let input = usage.get("input_tokens").and_then(|v| v.as_u64());
        let output = usage.get("output_tokens").and_then(|v| v.as_u64());
        let reasoning = usage
            .get("output_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64());
        let cached = crate::translation::usage::openai_cached_tokens(usage);
        resp = resp.with_usage(
            Usage::new(input, output)
                .with_reasoning_tokens(reasoning)
                .with_cache_read(cached),
        );
    }
    Ok(resp)
}
