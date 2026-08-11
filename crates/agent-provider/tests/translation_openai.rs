use reimagine_agent_harness::{AgentRequest, AgentToolDefinition, Message, ModelName};
use reimagine_ai_protocol::translation;
use serde_json::json;

#[test]
fn openai_request_translation_user_message_and_tool_definition() {
    let req =
        AgentRequest::new(ModelName::new("gpt-test"), vec![Message::user("hi")]).with_tools(vec![
            AgentToolDefinition::new(
                "echo",
                "echo something",
                json!({"type": "object", "properties": {"x": {"type": "number"}}}),
            ),
        ]);
    let messages = translation::request::to_openai_messages(req.messages()).expect("ok");
    let tools = translation::tools::to_openai_tools(req.tools());
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "echo");
    assert_eq!(
        tools[0]["function"]["parameters"]["properties"]["x"]["type"],
        "number"
    );
}

#[test]
fn openai_response_translation_assistant_text() {
    let payload = json!({
        "choices": [
            { "message": { "role": "assistant", "content": "hello" }, "finish_reason": "stop" }
        ],
        "usage": { "prompt_tokens": 7, "completion_tokens": 11 }
    });
    let resp = translation::response::from_openai_response(&payload).unwrap();
    assert_eq!(resp.message().content(), "hello");
    assert_eq!(resp.message().tool_calls().len(), 0);
    assert_eq!(resp.stop_reason(), Some("stop"));
    let usage = resp.usage().unwrap();
    assert_eq!(usage.input_tokens(), Some(7));
    assert_eq!(usage.output_tokens(), Some(11));
}

#[test]
fn openai_response_translation_tool_call_with_stringified_arguments() {
    let payload = json!({
        "choices": [
            { "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": { "name": "echo", "arguments": "{\"x\":1}" }
                }]
            }, "finish_reason": "tool_calls" }
        ]
    });
    let resp = translation::response::from_openai_response(&payload).unwrap();
    assert_eq!(resp.message().tool_calls().len(), 1);
    let call = &resp.message().tool_calls()[0];
    assert_eq!(call.id().as_str(), "c1");
    assert_eq!(call.name(), "echo");
    assert_eq!(call.arguments(), &json!({"x": 1}));
    assert_eq!(resp.stop_reason(), Some("tool_calls"));
}

#[test]
fn openai_response_translation_missing_choices_is_serialization_error() {
    let payload = json!({});
    let err = translation::response::from_openai_response(&payload).unwrap_err();
    assert!(matches!(
        err,
        reimagine_ai_protocol::ProviderAdapterError::Serialization(_)
    ));
}

#[test]
fn openai_response_translation_malformed_tool_call_arguments_is_serialization_error() {
    let payload = json!({
        "choices": [
            { "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": { "name": "echo", "arguments": "not-json" }
                }]
            }, "finish_reason": "tool_calls" }
        ]
    });
    let err = translation::response::from_openai_response(&payload).unwrap_err();
    assert!(matches!(
        err,
        reimagine_ai_protocol::ProviderAdapterError::Serialization(_)
    ));
}
