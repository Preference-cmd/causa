use serde_json::{Value, json};

use reimagine_agent_harness::{ContentBlock, FileContentBlock, Message};

use crate::error::ProviderAdapterError;

/// `true` when the message carries at least one file block.
fn has_file_blocks(m: &Message) -> bool {
    m.blocks()
        .iter()
        .any(|b| matches!(b, ContentBlock::File(_)))
}

/// Validate a file block for wire translation and return its
/// `(media_type, base64)` pair.
///
/// Only `image/*` media types with an inline base64 source are
/// representable on any supported wire. Url sources must have been
/// resolved to inline base64 by the adapter first (remote downloads are
/// not supported in V2). Anything else is rejected explicitly — file
/// blocks are never silently dropped.
fn image_data(file: &FileContentBlock) -> Result<(&str, &str), ProviderAdapterError> {
    if !file.media_type().starts_with("image/") {
        return Err(ProviderAdapterError::configuration(format!(
            "file block with media type `{}` cannot be sent to providers; \
             only image/* file blocks are supported",
            file.media_type()
        )));
    }
    let base64 = file.source().base64().ok_or_else(|| {
        ProviderAdapterError::configuration(format!(
            "file block url source `{}` must be resolved to inline base64 before \
             wire translation; remote URLs are not supported in V2",
            file.source().url().unwrap_or_default()
        ))
    })?;
    Ok((file.media_type(), base64))
}

/// Reject file blocks outside user messages: no supported wire format
/// carries images on system, assistant, or tool messages.
fn reject_file_blocks_outside_user(messages: &[Message]) -> Result<(), ProviderAdapterError> {
    for m in messages {
        if m.role() != "user" && has_file_blocks(m) {
            return Err(ProviderAdapterError::configuration(format!(
                "file blocks are only supported on user messages, got role `{}`",
                m.role()
            )));
        }
    }
    Ok(())
}

/// Build the OpenAI chat-completions `content` array for a user message
/// carrying file blocks: text parts plus `image_url` parts for image
/// file blocks.
fn openai_content_parts(m: &Message) -> Result<Vec<Value>, ProviderAdapterError> {
    let mut parts = Vec::with_capacity(m.blocks().len());
    for block in m.blocks() {
        match block {
            ContentBlock::Text(text) => parts.push(json!({ "type": "text", "text": text })),
            ContentBlock::File(file) => {
                let (media_type, base64) = image_data(file)?;
                parts.push(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{media_type};base64,{base64}"),
                    }
                }));
            }
        }
    }
    Ok(parts)
}

/// Build the Anthropic `content` block array for a user message
/// carrying file blocks: text blocks plus `image` source blocks.
fn anthropic_content_blocks(m: &Message) -> Result<Vec<Value>, ProviderAdapterError> {
    let mut blocks = Vec::with_capacity(m.blocks().len());
    for block in m.blocks() {
        match block {
            ContentBlock::Text(text) => blocks.push(json!({ "type": "text", "text": text })),
            ContentBlock::File(file) => {
                let (media_type, base64) = image_data(file)?;
                blocks.push(json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64,
                    }
                }));
            }
        }
    }
    Ok(blocks)
}

/// Build the OpenAI Responses `content` array for a user message item
/// carrying file blocks: `input_text` plus `input_image` parts.
fn responses_content_parts(m: &Message) -> Result<Vec<Value>, ProviderAdapterError> {
    let mut parts = Vec::with_capacity(m.blocks().len());
    for block in m.blocks() {
        match block {
            ContentBlock::Text(text) => parts.push(json!({ "type": "input_text", "text": text })),
            ContentBlock::File(file) => {
                let (media_type, base64) = image_data(file)?;
                parts.push(json!({
                    "type": "input_image",
                    "image_url": format!("data:{media_type};base64,{base64}"),
                }));
            }
        }
    }
    Ok(parts)
}

/// Build the `messages` array for an OpenAI-compatible chat completion request
/// from a slice of [`Message`]. Tool messages are mapped to role `"tool"` with
/// `tool_call_id` attached. Assistant messages that contain tool calls are
/// mapped to role `"assistant"` with a `tool_calls` array.
///
/// User messages that carry file blocks become content arrays: text parts
/// plus `image_url` parts (`data:<media_type>;base64,<base64>`). Text-only
/// messages keep the plain string shape. File blocks on other roles, and
/// file blocks that are not inline `image/*` base64, are rejected.
pub fn to_openai_messages(messages: &[Message]) -> Result<Vec<Value>, ProviderAdapterError> {
    reject_file_blocks_outside_user(messages)?;
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m.role();
        match role {
            "system" | "user" => {
                if has_file_blocks(m) {
                    out.push(json!({ "role": role, "content": openai_content_parts(m)? }));
                } else {
                    out.push(json!({ "role": role, "content": m.content() }));
                }
            }
            "assistant" => {
                if m.tool_calls().is_empty() {
                    out.push(json!({ "role": "assistant", "content": m.content() }));
                } else {
                    let calls: Vec<Value> = m
                        .tool_calls()
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id().as_str(),
                                "type": "function",
                                "function": {
                                    "name": c.name(),
                                    "arguments": c.arguments().to_string(),
                                }
                            })
                        })
                        .collect();
                    let mut obj = json!({ "role": "assistant", "tool_calls": calls });
                    if !m.content().is_empty() {
                        obj["content"] = json!(m.content());
                    } else {
                        obj["content"] = Value::Null;
                    }
                    out.push(obj);
                }
            }
            "tool" => {
                let id = m
                    .tool_call_id()
                    .map(|i| i.as_str().to_string())
                    .unwrap_or_default();
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": m.content(),
                }));
            }
            other => {
                out.push(
                    json!({ "role": "user", "content": format!("[{other}] {}", m.content()) }),
                );
            }
        }
    }
    Ok(out)
}

/// Build the `messages` array for an Anthropic messages API call. System
/// content is returned as a separate `system` field; the caller is responsible
/// for putting it on the request envelope. Assistant tool calls become
/// `tool_use` content blocks; tool messages become `tool_result` content
/// blocks.
///
/// User messages that carry file blocks become content block arrays with
/// `image` source blocks; text-only messages keep the plain string shape.
///
/// With `cache_control: true`, prompt-caching breakpoints are placed at the
/// three static-prefix anchors (PV-05): the end of the system prompt, the
/// end of the tool definitions, and the end of the conversation history.
/// System content is returned as a block array so its last block can carry
/// the `cache_control` marker.
pub fn to_anthropic_messages(
    messages: &[Message],
    cache_control: bool,
) -> Result<(Option<Value>, Vec<Value>), ProviderAdapterError> {
    reject_file_blocks_outside_user(messages)?;
    let mut system: Option<String> = None;
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role() {
            "system" => {
                system = Some(match system {
                    Some(existing) => format!("{existing}\n{}", m.content()),
                    None => m.content().to_string(),
                });
            }
            "user" => {
                if has_file_blocks(m) {
                    out.push(json!({ "role": "user", "content": anthropic_content_blocks(m)? }));
                } else {
                    out.push(json!({ "role": "user", "content": m.content() }));
                }
            }
            "assistant" => {
                if m.tool_calls().is_empty() {
                    out.push(json!({ "role": "assistant", "content": m.content() }));
                } else {
                    let blocks: Vec<Value> = m
                        .tool_calls()
                        .iter()
                        .map(|c| {
                            json!({
                                "type": "tool_use",
                                "id": c.id().as_str(),
                                "name": c.name(),
                                "input": c.arguments(),
                            })
                        })
                        .collect();
                    let mut content: Vec<Value> = Vec::new();
                    if !m.content().is_empty() {
                        content.push(json!({ "type": "text", "text": m.content() }));
                    }
                    content.extend(blocks);
                    out.push(json!({ "role": "assistant", "content": content }));
                }
            }
            "tool" => {
                let id = m
                    .tool_call_id()
                    .map(|i| i.as_str().to_string())
                    .unwrap_or_default();
                out.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": m.content(),
                    }],
                }));
            }
            other => {
                out.push(
                    json!({ "role": "user", "content": format!("[{other}] {}", m.content()) }),
                );
            }
        }
    }
    if cache_control {
        // Conversation-history breakpoint: the last message is the longest
        // stable prefix for cache reads across turns. The marker lives on a
        // content *block* (never on the message object), so string content
        // is promoted to a text-block array. Image blocks never carry the
        // marker (Anthropic restricts cache_control to text blocks), so
        // mixed arrays are marked on their last text block.
        if let Some(last) = out.last_mut() {
            let last = last.as_object_mut().expect("messages are objects");
            let content = last.get_mut("content");
            match content {
                // Empty content carries nothing into the cache prefix;
                // skip the marker entirely.
                Some(Value::String(text)) if !text.is_empty() => {
                    let text = text.clone();
                    last.insert(
                        "content".into(),
                        json!([{
                            "type": "text",
                            "text": text,
                            "cache_control": { "type": "ephemeral" },
                        }]),
                    );
                }
                Some(Value::String(_)) => {} // empty — no marker
                Some(Value::Array(blocks)) => {
                    // Mark the last text block when present (tool_use /
                    // tool_result blocks change per turn and are excluded;
                    // image blocks cannot carry the marker).
                    if let Some(text_block) = blocks
                        .iter_mut()
                        .rev()
                        .find(|b| b.get("type").and_then(|v| v.as_str()) == Some("text"))
                    {
                        text_block
                            .as_object_mut()
                            .expect("blocks are objects")
                            .insert("cache_control".into(), json!({ "type": "ephemeral" }));
                    }
                }
                _ => {}
            }
        }
    }
    let system = system.map(|text| {
        if cache_control {
            // System breakpoint: last system block carries the marker.
            json!([{ "type": "text", "text": text, "cache_control": { "type": "ephemeral" } }])
        } else {
            json!(text)
        }
    });
    Ok((system, out))
}

/// Collect system messages into a single `instructions` string for the
/// OpenAI Responses API. System content is concatenated with newlines,
/// mirroring `to_anthropic_messages`; returns `None` when no system
/// message is present.
pub fn to_responses_instructions(messages: &[Message]) -> Option<String> {
    let mut system: Option<String> = None;
    for m in messages {
        if m.role() == "system" {
            system = Some(match system {
                Some(existing) => format!("{existing}\n{}", m.content()),
                None => m.content().to_string(),
            });
        }
    }
    system
}

/// Build the `input` array for an OpenAI Responses API request. System
/// messages are skipped here (the caller puts them in the `instructions`
/// field). User and assistant messages become `{role, content:[...]}`
/// items; assistant tool calls become `function_call` items; tool
/// results become `function_call_output` items.
///
/// User items that carry file blocks get `input_image` parts in their
/// content array; text-only user items keep the single `input_text`
/// shape. File blocks on other roles, and file blocks that are not
/// inline `image/*` base64, are rejected.
pub fn to_responses_input(
    messages: &[Message],
    system: Option<&str>,
) -> Result<Vec<Value>, ProviderAdapterError> {
    reject_file_blocks_outside_user(messages)?;
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        match m.role() {
            "system" => {
                debug_assert!(
                    system.is_some(),
                    "system messages must be extracted into the `system` argument"
                );
            }
            "user" => {
                if has_file_blocks(m) {
                    out.push(json!({
                        "role": "user",
                        "content": responses_content_parts(m)?,
                    }));
                } else {
                    out.push(json!({
                        "role": "user",
                        "content": [{ "type": "input_text", "text": m.content() }],
                    }));
                }
            }
            "assistant" => {
                if !m.content().is_empty() {
                    out.push(json!({
                        "role": "assistant",
                        "content": [{ "type": "output_text", "text": m.content() }],
                    }));
                }
                for c in m.tool_calls() {
                    out.push(json!({
                        "type": "function_call",
                        "call_id": c.id().as_str(),
                        "name": c.name(),
                        "arguments": c.arguments().to_string(),
                    }));
                }
            }
            "tool" => {
                let id = m
                    .tool_call_id()
                    .map(|i| i.as_str().to_string())
                    .unwrap_or_default();
                out.push(json!({
                    "type": "function_call_output",
                    "call_id": id,
                    "output": m.content(),
                }));
            }
            other => {
                out.push(json!({
                    "role": "user",
                    "content": [{ "type": "input_text", "text": format!("[{other}] {}", m.content()) }],
                }));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent_harness::{FileContentBlock, ToolCall, ToolCallId};
    use serde_json::json;

    #[test]
    fn openai_messages_user_and_system() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let v = to_openai_messages(&msgs).expect("ok");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0]["role"], "system");
        assert_eq!(v[1]["role"], "user");
    }

    #[test]
    fn openai_messages_assistant_with_tool_calls_uses_function_shape() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let v = to_openai_messages(&msgs).expect("ok");
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["tool_calls"][0]["function"]["name"], "echo");
        assert_eq!(v[0]["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn openai_messages_tool_role_carries_tool_call_id() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let v = to_openai_messages(&msgs).expect("ok");
        assert_eq!(v[0]["role"], "tool");
        assert_eq!(v[0]["tool_call_id"], "c1");
        assert_eq!(v[0]["content"], "ok");
    }

    #[test]
    fn openai_messages_pure_text_keeps_string_shape() {
        let msgs = vec![Message::user("hi")];
        let v = to_openai_messages(&msgs).expect("ok");
        assert_eq!(v[0]["content"], "hi");
        assert!(v[0]["content"].is_string());
    }

    #[test]
    fn openai_messages_file_blocks_become_image_url_parts() {
        let msgs = vec![Message::user_with_blocks(vec![
            ContentBlock::Text("what is this?".into()),
            ContentBlock::File(FileContentBlock::data("image/png", "iVBORw0KGgo=")),
        ])];
        let v = to_openai_messages(&msgs).expect("ok");
        let content = v[0]["content"].as_array().expect("array");
        assert_eq!(content.len(), 2);
        assert_eq!(
            content[0],
            json!({ "type": "text", "text": "what is this?" })
        );
        assert_eq!(
            content[1],
            json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,iVBORw0KGgo=" },
            })
        );
    }

    #[test]
    fn openai_messages_image_only_user_message_has_no_text_part() {
        let msgs = vec![Message::user_with_blocks(vec![ContentBlock::File(
            FileContentBlock::data("image/jpeg", "AAAA"),
        )])];
        let v = to_openai_messages(&msgs).expect("ok");
        let content = v[0]["content"].as_array().expect("array");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(
            content[0]["image_url"]["url"],
            "data:image/jpeg;base64,AAAA"
        );
    }

    #[test]
    fn openai_messages_non_image_file_block_is_rejected() {
        let msgs = vec![Message::user_with_blocks(vec![ContentBlock::File(
            FileContentBlock::data("audio/mpeg", "AAAA"),
        )])];
        let err = to_openai_messages(&msgs).expect_err("must reject");
        assert!(err.to_string().contains("audio/mpeg"), "{err}");
        assert!(err.to_string().contains("image/*"), "{err}");
    }

    #[test]
    fn openai_messages_url_source_file_block_is_rejected() {
        let msgs = vec![Message::user_with_blocks(vec![ContentBlock::File(
            FileContentBlock::url("image/png", "refs/pic.png"),
        )])];
        let err = to_openai_messages(&msgs).expect_err("must reject");
        assert!(err.to_string().contains("inline base64"), "{err}");
    }

    #[test]
    fn openai_messages_file_block_on_assistant_role_is_rejected() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let assistant = Message::assistant_with_tool_calls("", vec![call]).with_blocks(vec![
            ContentBlock::File(FileContentBlock::data("image/png", "AAAA")),
        ]);
        let err = to_openai_messages(&[assistant]).expect_err("must reject");
        assert!(err.to_string().contains("user messages"), "{err}");
    }

    #[test]
    fn anthropic_messages_splits_system_out() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let (system, v) = to_anthropic_messages(&msgs, false).expect("ok");
        assert_eq!(system, Some(json!("sys")));
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["role"], "user");
    }

    #[test]
    fn anthropic_messages_with_cache_control_uses_system_blocks() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let (system, v) = to_anthropic_messages(&msgs, true).expect("ok");
        assert_eq!(
            system,
            Some(json!([{
                "type": "text",
                "text": "sys",
                "cache_control": { "type": "ephemeral" }
            }]))
        );
        // Last user message is promoted to a text block carrying the
        // conversation-history breakpoint.
        assert_eq!(
            v[0]["content"],
            json!([{
                "type": "text",
                "text": "hi",
                "cache_control": { "type": "ephemeral" }
            }])
        );
    }

    #[test]
    fn anthropic_messages_cache_control_marks_last_text_block_in_arrays() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![
            Message::assistant_with_tool_calls("thinking", vec![call]),
            Message::tool_result(ToolCallId::new("c1"), "ok"),
        ];
        let (_, v) = to_anthropic_messages(&msgs, true).expect("ok");
        let last = &v[v.len() - 1];
        assert_eq!(last["role"], "user");
        let blocks = last["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_result");
        // Tool-result blocks never carry the marker; nothing to mark here
        // means no cache_control anywhere in this message.
        for block in blocks {
            assert!(block.get("cache_control").is_none());
        }
    }

    #[test]
    fn anthropic_messages_cache_control_with_image_marks_last_text_block() {
        let msgs = vec![Message::user_with_blocks(vec![
            ContentBlock::Text("describe".into()),
            ContentBlock::File(FileContentBlock::data("image/png", "AAAA")),
        ])];
        let (_, v) = to_anthropic_messages(&msgs, true).expect("ok");
        let blocks = v[0]["content"].as_array().expect("array");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["cache_control"], json!({ "type": "ephemeral" }));
        // Image blocks never carry the marker.
        assert_eq!(blocks[1]["type"], "image");
        assert!(blocks[1].get("cache_control").is_none());
    }

    #[test]
    fn anthropic_messages_assistant_tool_call_uses_tool_use_block() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let (_, v) = to_anthropic_messages(&msgs, false).expect("ok");
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["content"][0]["type"], "tool_use");
        assert_eq!(v[0]["content"][0]["id"], "c1");
        assert_eq!(v[0]["content"][0]["name"], "echo");
        assert_eq!(v[0]["content"][0]["input"]["x"], 1);
    }

    #[test]
    fn anthropic_messages_tool_role_becomes_tool_result_block() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let (_, v) = to_anthropic_messages(&msgs, false).expect("ok");
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[0]["content"][0]["type"], "tool_result");
        assert_eq!(v[0]["content"][0]["tool_use_id"], "c1");
        assert_eq!(v[0]["content"][0]["content"], "ok");
    }

    #[test]
    fn anthropic_messages_file_blocks_become_image_blocks() {
        let msgs = vec![Message::user_with_blocks(vec![
            ContentBlock::Text("describe".into()),
            ContentBlock::File(FileContentBlock::data("image/jpeg", "AAAA")),
        ])];
        let (system, v) = to_anthropic_messages(&msgs, false).expect("ok");
        assert_eq!(system, None);
        let content = v[0]["content"].as_array().expect("array");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0], json!({ "type": "text", "text": "describe" }));
        assert_eq!(
            content[1],
            json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "AAAA",
                },
            })
        );
    }

    #[test]
    fn anthropic_messages_non_image_file_block_is_rejected() {
        let msgs = vec![Message::user_with_blocks(vec![ContentBlock::File(
            FileContentBlock::data("video/mp4", "AAAA"),
        )])];
        let err = to_anthropic_messages(&msgs, false).expect_err("must reject");
        assert!(err.to_string().contains("video/mp4"), "{err}");
    }

    #[test]
    fn responses_instructions_collects_system_messages() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        assert_eq!(to_responses_instructions(&msgs).as_deref(), Some("sys"));
        assert_eq!(to_responses_instructions(&[]), None);
    }

    #[test]
    fn responses_input_user_and_assistant_items() {
        let msgs = vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant("yo"),
        ];
        let v = to_responses_input(&msgs, Some("sys")).expect("ok");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[0]["content"][0]["type"], "input_text");
        assert_eq!(v[0]["content"][0]["text"], "hi");
        assert_eq!(v[1]["role"], "assistant");
        assert_eq!(v[1]["content"][0]["type"], "output_text");
        assert_eq!(v[1]["content"][0]["text"], "yo");
    }

    #[test]
    fn responses_input_skips_system_messages() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let v = to_responses_input(&msgs, Some("sys")).expect("ok");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["role"], "user");
    }

    #[test]
    fn responses_input_assistant_tool_call_becomes_function_call_item() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let v = to_responses_input(&msgs, None).expect("ok");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["type"], "function_call");
        assert_eq!(v[0]["call_id"], "c1");
        assert_eq!(v[0]["name"], "echo");
        assert_eq!(v[0]["arguments"], "{\"x\":1}");
    }

    #[test]
    fn responses_input_tool_result_becomes_function_call_output_item() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let v = to_responses_input(&msgs, None).expect("ok");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["type"], "function_call_output");
        assert_eq!(v[0]["call_id"], "c1");
        assert_eq!(v[0]["output"], "ok");
    }

    #[test]
    fn responses_input_file_blocks_become_input_image_parts() {
        let msgs = vec![Message::user_with_blocks(vec![
            ContentBlock::Text("what is this?".into()),
            ContentBlock::File(FileContentBlock::data("image/png", "iVBORw0KGgo=")),
        ])];
        let v = to_responses_input(&msgs, None).expect("ok");
        let content = v[0]["content"].as_array().expect("array");
        assert_eq!(content.len(), 2);
        assert_eq!(
            content[0],
            json!({ "type": "input_text", "text": "what is this?" })
        );
        assert_eq!(
            content[1],
            json!({ "type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=" })
        );
    }

    #[test]
    fn responses_input_pure_text_user_item_keeps_single_input_text() {
        let msgs = vec![Message::user("hi")];
        let v = to_responses_input(&msgs, None).expect("ok");
        let content = v[0]["content"].as_array().expect("array");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "input_text");
    }

    #[test]
    fn responses_input_non_image_file_block_is_rejected() {
        let msgs = vec![Message::user_with_blocks(vec![ContentBlock::File(
            FileContentBlock::data("application/pdf", "AAAA"),
        )])];
        let err = to_responses_input(&msgs, None).expect_err("must reject");
        assert!(err.to_string().contains("application/pdf"), "{err}");
    }
}
