use reimagine_agent_harness::{AgentRequest, AgentToolDefinition, Message, ModelName};
use reimagine_ai_protocol::translation;
use serde_json::json;

#[test]
fn responses_input_translation_messages_tool_call_and_tool_result() {
    let req = AgentRequest::new(
        ModelName::new("gpt-5-test"),
        vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant_with_tool_calls(
                "",
                vec![reimagine_agent_harness::ToolCall::new(
                    reimagine_agent_harness::ToolCallId::new("c1"),
                    "echo",
                    json!({"x": 1}),
                )],
            ),
            Message::tool_result(reimagine_agent_harness::ToolCallId::new("c1"), "ok"),
        ],
    );
    let instructions = translation::request::to_responses_instructions(req.messages());
    let input = translation::request::to_responses_input(req.messages(), instructions.as_deref())
        .expect("ok");
    assert_eq!(instructions.as_deref(), Some("sys"));
    assert_eq!(input.len(), 3);
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[0]["content"][0]["type"], "input_text");
    assert_eq!(input[1]["type"], "function_call");
    assert_eq!(input[1]["call_id"], "c1");
    assert_eq!(input[1]["name"], "echo");
    assert_eq!(input[1]["arguments"], "{\"x\":1}");
    assert_eq!(input[2]["type"], "function_call_output");
    assert_eq!(input[2]["call_id"], "c1");
    assert_eq!(input[2]["output"], "ok");
}

#[test]
fn responses_tools_translation_matches_chat_completions_shape() {
    let req = AgentRequest::new(ModelName::new("gpt-5-test"), vec![Message::user("hi")])
        .with_tools(vec![AgentToolDefinition::new(
            "echo",
            "echo something",
            json!({"type": "object", "properties": {"x": {"type": "number"}}}),
        )]);
    let tools = translation::tools::to_responses_tools(req.tools());
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "echo");
    assert_eq!(tools[0]["function"]["description"], "echo something");
    assert_eq!(
        tools[0]["function"]["parameters"]["properties"]["x"]["type"],
        "number"
    );
}

#[test]
fn responses_response_translation_assistant_text() {
    let payload = json!({
        "id": "resp_1",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": "hello" }]
        }],
        "usage": { "input_tokens": 7, "output_tokens": 11, "total_tokens": 18 }
    });
    let resp = translation::response::from_responses_response(&payload).unwrap();
    assert_eq!(resp.message().content(), "hello");
    assert_eq!(resp.message().tool_calls().len(), 0);
    let usage = resp.usage().unwrap();
    assert_eq!(usage.input_tokens(), Some(7));
    assert_eq!(usage.output_tokens(), Some(11));
}

#[test]
fn responses_response_translation_function_call_item() {
    let payload = json!({
        "id": "resp_1",
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": "{\"x\": 42}"
        }]
    });
    let resp = translation::response::from_responses_response(&payload).unwrap();
    assert_eq!(resp.message().content(), "");
    let calls = resp.message().tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id().as_str(), "call_1");
    assert_eq!(calls[0].name(), "echo");
    assert_eq!(calls[0].arguments(), &json!({"x": 42}));
}

#[test]
fn responses_response_translation_missing_output_is_serialization_error() {
    let payload = json!({});
    let err = translation::response::from_responses_response(&payload).unwrap_err();
    assert!(matches!(
        err,
        reimagine_ai_protocol::ProviderAdapterError::Serialization(_)
    ));
}

#[test]
fn responses_response_translation_malformed_arguments_is_serialization_error() {
    let payload = json!({
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": "not-json"
        }]
    });
    let err = translation::response::from_responses_response(&payload).unwrap_err();
    assert!(matches!(
        err,
        reimagine_ai_protocol::ProviderAdapterError::Serialization(_)
    ));
}
