use serde_json::{Value, json};

use reimagine_agent_harness::AgentToolDefinition;

/// Translate a slice of `AgentToolDefinition` into the OpenAI `tools` array
/// format. Each entry is a function tool with a JSON-Schema `parameters`
/// object.
pub fn to_openai_tools(defs: &[AgentToolDefinition]) -> Vec<Value> {
    defs.iter()
        .map(|d| {
            json!({
                "type": "function",
                "function": {
                    "name": d.name(),
                    "description": d.description(),
                    "parameters": d.parameters(),
                }
            })
        })
        .collect()
}

/// Translate a slice of `AgentToolDefinition` into the Anthropic `tools` array
/// format. Anthropic's shape is `{ name, description, input_schema }` per tool.
///
/// With `cache_control: true` (PV-05), the tool-definitions breakpoint lands
/// on the last tool — the tools array is part of the static request prefix.
pub fn to_anthropic_tools(defs: &[AgentToolDefinition], cache_control: bool) -> Vec<Value> {
    let mut out: Vec<Value> = defs
        .iter()
        .map(|d| {
            json!({
                "name": d.name(),
                "description": d.description(),
                "input_schema": d.parameters(),
            })
        })
        .collect();
    if cache_control
        && let Some(last) = out.last_mut()
    {
        last.as_object_mut()
            .expect("tools are objects")
            .insert("cache_control".into(), json!({ "type": "ephemeral" }));
    }
    out
}

/// Translate a slice of `AgentToolDefinition` into the OpenAI Responses
/// API `tools` array format. Same function-tool shape as
/// [`to_openai_tools`]; the Responses API accepts it verbatim.
pub fn to_responses_tools(defs: &[AgentToolDefinition]) -> Vec<Value> {
    to_openai_tools(defs)
}
