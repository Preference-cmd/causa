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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- OpenAI chat completions ----

    #[test]
    fn openai_happy_path_text_message_with_finish_reason_and_usage() {
        let value = json!({
            "choices": [{
                "message": { "role": "assistant", "content": "Hello!" },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 12, "completion_tokens": 7 },
        });
        let resp = from_openai_response(&value).expect("ok");
        assert_eq!(resp.message(), &Message::assistant("Hello!"));
        assert_eq!(resp.stop_reason(), Some("stop"));
        assert_eq!(resp.usage(), Some(&Usage::new(Some(12), Some(7))));
    }

    #[test]
    fn openai_tool_calls_translate_to_harness_tool_calls() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "generate_image",
                            "arguments": "{\"prompt\": \"sunset\"}",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        });
        let resp = from_openai_response(&value).expect("ok");
        let call = ToolCall::new(
            ToolCallId::new("call_1"),
            "generate_image",
            json!({"prompt": "sunset"}),
        );
        assert_eq!(
            resp.message(),
            &Message::assistant_with_tool_calls("", vec![call])
        );
        assert_eq!(resp.stop_reason(), Some("tool_calls"));
    }

    #[test]
    fn openai_usage_extracts_input_output_cache_and_reasoning_tokens() {
        let value = json!({
            "choices": [{
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": { "cached_tokens": 40 },
                "completion_tokens_details": { "reasoning_tokens": 10 },
            },
        });
        let resp = from_openai_response(&value).expect("ok");
        assert_eq!(
            resp.usage(),
            Some(&Usage::new(Some(100), Some(50))
                .with_reasoning_tokens(Some(10))
                .with_cache_read(Some(40)))
        );
    }

    #[test]
    fn openai_missing_usage_and_finish_reason_are_omitted() {
        let value = json!({
            "choices": [{ "message": { "role": "assistant", "content": "hi" } }],
        });
        let resp = from_openai_response(&value).expect("ok");
        assert_eq!(resp.usage(), None);
        assert_eq!(resp.stop_reason(), None);
    }

    #[test]
    fn openai_missing_choices_surfaces_serialization_error() {
        for value in [
            json!({}),
            json!({ "choices": [] }),
            json!({ "choices": { "not": "an array" } }),
        ] {
            assert_eq!(
                from_openai_response(&value),
                Err(ProviderAdapterError::Serialization(
                    "missing choices[0]".to_string()
                ))
            );
        }
    }

    #[test]
    fn openai_missing_choice_message_surfaces_serialization_error() {
        let value = json!({ "choices": [{}] });
        assert_eq!(
            from_openai_response(&value),
            Err(ProviderAdapterError::Serialization(
                "missing choices[0].message".to_string()
            ))
        );
    }

    #[test]
    fn openai_tool_call_missing_id_surfaces_serialization_error() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "function": { "name": "f", "arguments": "{}" },
                    }],
                }
            }],
        });
        assert_eq!(
            from_openai_response(&value),
            Err(ProviderAdapterError::Serialization(
                "tool_calls[0].id missing".to_string()
            ))
        );
    }

    #[test]
    fn openai_tool_call_missing_function_name_surfaces_serialization_error() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "arguments": "{}" },
                    }],
                }
            }],
        });
        assert_eq!(
            from_openai_response(&value),
            Err(ProviderAdapterError::Serialization(
                "tool_calls[0].function.name missing".to_string()
            ))
        );
    }

    #[test]
    fn openai_malformed_arguments_surfaces_serialization_error() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "f", "arguments": "{not json" },
                    }],
                }
            }],
        });
        let err = from_openai_response(&value).unwrap_err();
        assert!(matches!(
            err,
            ProviderAdapterError::Serialization(m)
                if m.starts_with("tool_calls[0].function.arguments:")
        ));
    }

    // Current behavior: non-string `content` (e.g. the block-array shape
    // some OpenAI-compatible servers emit) is silently coerced to "" — the
    // model's text is dropped rather than surfaced as a serialization
    // error. Documented as-is; AC-12 flags this for a follow-up decision.
    #[test]
    fn openai_non_string_content_coerces_to_empty_text() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": [{ "type": "text", "text": "dropped" }],
                },
            }],
        });
        let resp = from_openai_response(&value).expect("ok");
        assert_eq!(resp.message(), &Message::assistant(""));
    }

    // Current behavior: a `tool_calls` value that is not an array parses as
    // no tool calls rather than an error, and a tool call without
    // `arguments` defaults to `{}`.
    #[test]
    fn openai_tolerates_non_array_tool_calls_and_missing_arguments() {
        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": { "not": "an array" },
                }
            }],
        });
        let resp = from_openai_response(&value).expect("ok");
        assert_eq!(resp.message(), &Message::assistant(""));

        let value = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": { "name": "f" },
                    }],
                }
            }],
        });
        let resp = from_openai_response(&value).expect("ok");
        assert_eq!(
            resp.message(),
            &Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(ToolCallId::new("call_1"), "f", json!({}))],
            )
        );
    }

    // ---- Anthropic messages ----

    #[test]
    fn anthropic_happy_path_text_blocks_joined_with_newline() {
        let value = json!({
            "content": [
                { "type": "text", "text": "first" },
                { "type": "text", "text": "second" },
            ],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 5, "output_tokens": 9 },
        });
        let resp = from_anthropic_response(&value).expect("ok");
        assert_eq!(resp.message(), &Message::assistant("first\nsecond"));
        assert_eq!(resp.stop_reason(), Some("end_turn"));
        assert_eq!(resp.usage(), Some(&Usage::new(Some(5), Some(9))));
    }

    #[test]
    fn anthropic_tool_use_blocks_become_tool_calls() {
        let value = json!({
            "content": [
                { "type": "text", "text": "Calling a tool." },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "render",
                    "input": { "seed": 42 },
                },
            ],
            "stop_reason": "tool_use",
        });
        let resp = from_anthropic_response(&value).expect("ok");
        let call = ToolCall::new(ToolCallId::new("toolu_1"), "render", json!({ "seed": 42 }));
        assert_eq!(
            resp.message(),
            &Message::assistant_with_tool_calls("Calling a tool.", vec![call])
        );
        assert_eq!(resp.stop_reason(), Some("tool_use"));
    }

    #[test]
    fn anthropic_usage_extracts_input_output_and_cache_tokens() {
        let value = json!({
            "content": [{ "type": "text", "text": "hi" }],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 60,
                "cache_read_input_tokens": 30,
            },
        });
        let resp = from_anthropic_response(&value).expect("ok");
        assert_eq!(
            resp.usage(),
            Some(&Usage::new(Some(100), Some(50))
                .with_cache_creation(Some(60))
                .with_cache_read(Some(30)))
        );
    }

    #[test]
    fn anthropic_empty_content_yields_empty_assistant_message() {
        let resp = from_anthropic_response(&json!({ "content": [] })).expect("ok");
        assert_eq!(resp.message(), &Message::assistant(""));
        assert_eq!(resp.stop_reason(), None);
        assert_eq!(resp.usage(), None);
    }

    #[test]
    fn anthropic_missing_content_array_surfaces_serialization_error() {
        for value in [json!({}), json!({ "content": { "not": "an array" } })] {
            assert_eq!(
                from_anthropic_response(&value),
                Err(ProviderAdapterError::Serialization(
                    "missing content array".to_string()
                ))
            );
        }
    }

    #[test]
    fn anthropic_tool_use_missing_id_surfaces_serialization_error() {
        let value = json!({
            "content": [{ "type": "tool_use", "name": "f", "input": {} }],
        });
        assert_eq!(
            from_anthropic_response(&value),
            Err(ProviderAdapterError::Serialization(
                "content[0].id missing".to_string()
            ))
        );
    }

    #[test]
    fn anthropic_tool_use_missing_name_surfaces_serialization_error() {
        let value = json!({
            "content": [{ "type": "tool_use", "id": "toolu_1", "input": {} }],
        });
        assert_eq!(
            from_anthropic_response(&value),
            Err(ProviderAdapterError::Serialization(
                "content[0].name missing".to_string()
            ))
        );
    }

    // Current behavior: a `tool_use` block without `input` yields `Null`
    // arguments instead of an error — the harness keeps the call and the
    // tool receives `null`. Documented as-is.
    #[test]
    fn anthropic_tool_use_missing_input_becomes_null_arguments() {
        let value = json!({
            "content": [{ "type": "tool_use", "id": "toolu_1", "name": "f" }],
        });
        let resp = from_anthropic_response(&value).expect("ok");
        assert_eq!(
            resp.message(),
            &Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(ToolCallId::new("toolu_1"), "f", Value::Null)],
            )
        );
    }

    // Text blocks without a string `text` and unknown block types (e.g.
    // `thinking`, `tool_result`) are skipped silently.
    #[test]
    fn anthropic_unknown_or_emptied_blocks_are_ignored() {
        let value = json!({
            "content": [
                { "type": "text" },
                { "type": "thinking", "thinking": "not surfaced" },
                { "type": "tool_result", "tool_use_id": "x" },
            ],
        });
        let resp = from_anthropic_response(&value).expect("ok");
        assert_eq!(resp.message(), &Message::assistant(""));
        assert_eq!(resp.usage(), None);
    }

    // ---- OpenAI Responses API ----

    #[test]
    fn responses_happy_path_output_text_message() {
        let value = json!({
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [
                    { "type": "output_text", "text": "Hi there" },
                    { "type": "output_text", "text": " again" },
                ],
            }],
            "usage": { "input_tokens": 4, "output_tokens": 6 },
        });
        let resp = from_responses_response(&value).expect("ok");
        assert_eq!(resp.message(), &Message::assistant("Hi there\n again"));
        assert_eq!(resp.usage(), Some(&Usage::new(Some(4), Some(6))));
    }

    #[test]
    fn responses_function_call_items_become_tool_calls() {
        let value = json!({
            "output": [{
                "type": "function_call",
                "call_id": "fc_1",
                "name": "lookup",
                "arguments": "{\"key\": \"abc\"}",
            }],
        });
        let resp = from_responses_response(&value).expect("ok");
        assert_eq!(
            resp.message(),
            &Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(
                    ToolCallId::new("fc_1"),
                    "lookup",
                    json!({"key": "abc"})
                )],
            )
        );
    }

    #[test]
    fn responses_mixed_output_items_combine_text_and_tool_calls() {
        let value = json!({
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "Checking..." }],
                },
                {
                    "type": "function_call",
                    "call_id": "fc_2",
                    "name": "render",
                    "arguments": "{}",
                },
            ],
        });
        let resp = from_responses_response(&value).expect("ok");
        assert_eq!(
            resp.message(),
            &Message::assistant_with_tool_calls(
                "Checking...",
                vec![ToolCall::new(ToolCallId::new("fc_2"), "render", json!({}))],
            )
        );
    }

    #[test]
    fn responses_usage_extracts_input_output_reasoning_and_cache() {
        let value = json!({
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "hi" }],
            }],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "output_tokens_details": { "reasoning_tokens": 10 },
                "input_tokens_details": { "cached_tokens": 40 },
            },
        });
        let resp = from_responses_response(&value).expect("ok");
        assert_eq!(
            resp.usage(),
            Some(&Usage::new(Some(100), Some(50))
                .with_reasoning_tokens(Some(10))
                .with_cache_read(Some(40)))
        );
    }

    #[test]
    fn responses_missing_output_array_surfaces_serialization_error() {
        for value in [json!({}), json!({ "output": { "not": "an array" } })] {
            assert_eq!(
                from_responses_response(&value),
                Err(ProviderAdapterError::Serialization(
                    "missing output array".to_string()
                ))
            );
        }
    }

    #[test]
    fn responses_function_call_missing_call_id_surfaces_serialization_error() {
        let value = json!({
            "output": [{ "type": "function_call", "name": "f", "arguments": "{}" }],
        });
        assert_eq!(
            from_responses_response(&value),
            Err(ProviderAdapterError::Serialization(
                "output[0].call_id missing".to_string()
            ))
        );
    }

    #[test]
    fn responses_function_call_missing_name_surfaces_serialization_error() {
        let value = json!({
            "output": [{ "type": "function_call", "call_id": "fc_1", "arguments": "{}" }],
        });
        assert_eq!(
            from_responses_response(&value),
            Err(ProviderAdapterError::Serialization(
                "output[0].name missing".to_string()
            ))
        );
    }

    #[test]
    fn responses_malformed_arguments_surfaces_serialization_error() {
        let value = json!({
            "output": [{
                "type": "function_call",
                "call_id": "fc_1",
                "name": "f",
                "arguments": "not json",
            }],
        });
        let err = from_responses_response(&value).unwrap_err();
        assert!(matches!(
            err,
            ProviderAdapterError::Serialization(m) if m.starts_with("output[0].arguments:")
        ));
    }

    // `message` items contribute only `output_text` blocks; other content
    // block types (e.g. `refusal`) are skipped, and items with unknown
    // types (e.g. `reasoning`) are ignored entirely.
    #[test]
    fn responses_non_output_text_and_unknown_items_are_ignored() {
        let value = json!({
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        { "type": "refusal", "refusal": "nope" },
                        { "type": "output_text", "text": "kept" },
                    ],
                },
                { "type": "reasoning", "summary": [] },
            ],
        });
        let resp = from_responses_response(&value).expect("ok");
        assert_eq!(resp.message(), &Message::assistant("kept"));
        assert_eq!(resp.usage(), None);
    }

    // A `message` item without a `content` array is skipped silently.
    #[test]
    fn responses_message_item_without_content_yields_empty_text() {
        let value = json!({
            "output": [{ "type": "message", "role": "assistant" }],
        });
        let resp = from_responses_response(&value).expect("ok");
        assert_eq!(resp.message(), &Message::assistant(""));
    }

    // Current behavior: `arguments` defaults to `{}` when missing. Also,
    // the Responses API carries a `status` field rather than a
    // `stop_reason`; the translator ignores it, so `stop_reason` is always
    // `None` for Responses responses.
    #[test]
    fn responses_missing_arguments_defaults_to_empty_object_and_status_is_ignored() {
        let value = json!({
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "fc_1",
                "name": "f",
            }],
        });
        let resp = from_responses_response(&value).expect("ok");
        assert_eq!(
            resp.message(),
            &Message::assistant_with_tool_calls(
                "",
                vec![ToolCall::new(ToolCallId::new("fc_1"), "f", json!({}))],
            )
        );
        assert_eq!(resp.stop_reason(), None);
    }
}
