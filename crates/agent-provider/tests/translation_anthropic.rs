use reimagine_ai_protocol::translation;
use serde_json::json;

#[test]
fn anthropic_response_translation_text_only() {
    let payload = json!({
        "content": [{ "type": "text", "text": "hi back" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 5, "output_tokens": 8 }
    });
    let resp = translation::response::from_anthropic_response(&payload).unwrap();
    assert_eq!(resp.message().content(), "hi back");
    assert!(resp.message().tool_calls().is_empty());
    assert_eq!(resp.stop_reason(), Some("end_turn"));
    assert_eq!(resp.usage().unwrap().output_tokens(), Some(8));
}

#[test]
fn anthropic_response_translation_text_plus_tool_use() {
    let payload = json!({
        "content": [
            { "type": "text", "text": "calling tool" },
            { "type": "tool_use", "id": "c1", "name": "echo", "input": { "x": 1 } }
        ],
        "stop_reason": "tool_use"
    });
    let resp = translation::response::from_anthropic_response(&payload).unwrap();
    assert_eq!(resp.message().content(), "calling tool");
    assert_eq!(resp.message().tool_calls().len(), 1);
    let call = &resp.message().tool_calls()[0];
    assert_eq!(call.id().as_str(), "c1");
    assert_eq!(call.name(), "echo");
    assert_eq!(call.arguments(), &json!({"x": 1}));
}

#[test]
fn anthropic_response_translation_missing_content_array_is_serialization_error() {
    let payload = json!({});
    let err = translation::response::from_anthropic_response(&payload).unwrap_err();
    assert!(matches!(
        err,
        reimagine_ai_protocol::ProviderAdapterError::Serialization(_)
    ));
}

#[test]
fn anthropic_request_translation_tool_definitions() {
    let defs = vec![reimagine_agent_harness::AgentToolDefinition::new(
        "echo",
        "echo something",
        json!({"type": "object"}),
    )];
    let v = translation::tools::to_anthropic_tools(&defs);
    assert_eq!(v[0]["name"], "echo");
    assert_eq!(v[0]["description"], "echo something");
    assert_eq!(v[0]["input_schema"]["type"], "object");
}
