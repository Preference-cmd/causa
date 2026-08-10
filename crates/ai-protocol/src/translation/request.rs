use serde_json::{Value, json};

use reimagine_agent_harness::Message;

/// Build the `messages` array for an OpenAI-compatible chat completion request
/// from a slice of [`Message`]. Tool messages are mapped to role `"tool"` with
/// `tool_call_id` attached. Assistant messages that contain tool calls are
/// mapped to role `"assistant"` with a `tool_calls` array.
pub fn to_openai_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m.role();
        match role {
            "system" | "user" => {
                out.push(json!({ "role": role, "content": m.content() }));
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
    out
}

/// Build the `messages` array for an Anthropic messages API call. System
/// content is returned as a separate `system` field; the caller is responsible
/// for putting it on the request envelope. Assistant tool calls become
/// `tool_use` content blocks; tool messages become `tool_result` content
/// blocks.
pub fn to_anthropic_messages(messages: &[Message]) -> (Option<String>, Vec<Value>) {
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
                out.push(json!({ "role": "user", "content": m.content() }));
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
    (system, out)
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
pub fn to_responses_input(messages: &[Message], system: Option<&str>) -> Vec<Value> {
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
                out.push(json!({
                    "role": "user",
                    "content": [{ "type": "input_text", "text": m.content() }],
                }));
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_agent_harness::{ToolCall, ToolCallId};
    use serde_json::json;

    #[test]
    fn openai_messages_user_and_system() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let v = to_openai_messages(&msgs);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0]["role"], "system");
        assert_eq!(v[1]["role"], "user");
    }

    #[test]
    fn openai_messages_assistant_with_tool_calls_uses_function_shape() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let v = to_openai_messages(&msgs);
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["tool_calls"][0]["function"]["name"], "echo");
        assert_eq!(v[0]["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn openai_messages_tool_role_carries_tool_call_id() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let v = to_openai_messages(&msgs);
        assert_eq!(v[0]["role"], "tool");
        assert_eq!(v[0]["tool_call_id"], "c1");
        assert_eq!(v[0]["content"], "ok");
    }

    #[test]
    fn anthropic_messages_splits_system_out() {
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let (system, v) = to_anthropic_messages(&msgs);
        assert_eq!(system.as_deref(), Some("sys"));
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["role"], "user");
    }

    #[test]
    fn anthropic_messages_assistant_tool_call_uses_tool_use_block() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let (_, v) = to_anthropic_messages(&msgs);
        assert_eq!(v[0]["role"], "assistant");
        assert_eq!(v[0]["content"][0]["type"], "tool_use");
        assert_eq!(v[0]["content"][0]["id"], "c1");
        assert_eq!(v[0]["content"][0]["name"], "echo");
        assert_eq!(v[0]["content"][0]["input"]["x"], 1);
    }

    #[test]
    fn anthropic_messages_tool_role_becomes_tool_result_block() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let (_, v) = to_anthropic_messages(&msgs);
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[0]["content"][0]["type"], "tool_result");
        assert_eq!(v[0]["content"][0]["tool_use_id"], "c1");
        assert_eq!(v[0]["content"][0]["content"], "ok");
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
        let v = to_responses_input(&msgs, Some("sys"));
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
        let v = to_responses_input(&msgs, Some("sys"));
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["role"], "user");
    }

    #[test]
    fn responses_input_assistant_tool_call_becomes_function_call_item() {
        let call = ToolCall::new(ToolCallId::new("c1"), "echo", json!({"x": 1}));
        let msgs = vec![Message::assistant_with_tool_calls("", vec![call])];
        let v = to_responses_input(&msgs, None);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["type"], "function_call");
        assert_eq!(v[0]["call_id"], "c1");
        assert_eq!(v[0]["name"], "echo");
        assert_eq!(v[0]["arguments"], "{\"x\":1}");
    }

    #[test]
    fn responses_input_tool_result_becomes_function_call_output_item() {
        let msgs = vec![Message::tool_result(ToolCallId::new("c1"), "ok")];
        let v = to_responses_input(&msgs, None);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["type"], "function_call_output");
        assert_eq!(v[0]["call_id"], "c1");
        assert_eq!(v[0]["output"], "ok");
    }
}
