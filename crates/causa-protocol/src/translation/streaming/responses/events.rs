use super::state::{Item, Responses};
use crate::translation::streaming::common::{done, index, permanent, provider_error, string};
use crate::translation::{
    openai_responses::parse_openai_responses_output, sse_parser::SseEvent,
    usage::model_usage_from_openai_responses,
};
use causa_kernel::{ModelInvokeError, StreamDelta};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

impl Responses {
    pub fn event(&mut self, event: &SseEvent) -> Result<Vec<StreamDelta>, ModelInvokeError> {
        let value: Value = serde_json::from_str(&event.data)
            .map_err(|_| permanent("Responses event contains invalid JSON"))?;
        let kind = string(&value, "type")?;
        if let Some(name) = &event.event
            && name != kind
            && name != "message"
        {
            return Err(permanent("Responses SSE event/type mismatch"));
        }
        let mut deltas = Vec::new();
        match kind {
            "error" => return Err(provider_error(value.get("error").unwrap_or(&value))),
            "response.created" | "response.in_progress" => {
                let response = value
                    .get("response")
                    .ok_or_else(|| permanent("Responses missing response"))?;
                self.response_id(response)?;
                let status = string(response, "status")?;
                if status != "in_progress" && status != "queued" {
                    return Err(permanent("Responses invalid initial status"));
                }
            }
            "response.output_item.added" => {
                let i = index(&value, "output_index")?;
                self.budget.item(self.items.len())?;
                if i != self.items.len() {
                    return Err(permanent(
                        "Responses output index must follow declaration order",
                    ));
                }
                let wire = value
                    .get("item")
                    .ok_or_else(|| permanent("Responses missing item"))?;
                let id = string(wire, "id")?;
                let item_type = string(wire, "type")?;
                if self.items.values().any(|item| item.id == id) {
                    return Err(permanent("Responses duplicate item id"));
                }
                if id.is_empty() {
                    return Err(permanent("Responses empty item id"));
                }
                if item_type == "message" && !wire.get("content").is_some_and(Value::is_array) {
                    return Err(permanent("Responses message content must be an array"));
                }
                if item_type == "reasoning" && !wire.get("summary").is_some_and(Value::is_array) {
                    return Err(permanent("Responses reasoning summary must be an array"));
                }
                self.budget.add(id.len() + item_type.len())?;
                let mut item = Item {
                    id: id.into(),
                    kind: item_type.into(),
                    call_id: None,
                    name: None,
                    call_index: None,
                    arguments: String::new(),
                    arguments_done: false,
                    closed: false,
                    parts: BTreeMap::new(),
                    closed_parts: BTreeSet::new(),
                    completed_parts: BTreeSet::new(),
                    summaries: BTreeSet::new(),
                    closed_summaries: BTreeSet::new(),
                    completed_summaries: BTreeSet::new(),
                };
                if item_type == "function_call" {
                    let call_id = string(wire, "call_id")?;
                    let name = string(wire, "name")?;
                    if call_id.is_empty() || name.is_empty() {
                        return Err(permanent("Responses empty tool identity"));
                    }
                    if self
                        .items
                        .values()
                        .any(|item| item.call_id.as_deref() == Some(call_id))
                    {
                        return Err(permanent("Responses duplicate call id"));
                    }
                    // A call opens with no arguments yet; some servers omit
                    // the empty string instead of sending it.
                    let arguments = match wire.get("arguments") {
                        None | Some(Value::Null) => "",
                        Some(_) => string(wire, "arguments")?,
                    };
                    self.budget
                        .add(call_id.len() + name.len() + arguments.len())?;
                    item.call_id = Some(call_id.into());
                    item.name = Some(name.into());
                    item.arguments = arguments.into();
                    item.call_index = Some(self.tools);
                    self.tools += 1;
                    deltas.push(StreamDelta::ToolCallDelta {
                        call_index: item.call_index.expect("tool index"),
                        provider_call_id: item.call_id.clone(),
                        name_delta: item.name.clone(),
                        arguments_delta: if arguments.is_empty() {
                            None
                        } else {
                            Some(arguments.into())
                        },
                    });
                }
                self.items.insert(i, item);
            }
            "response.content_part.added"
            | "response.reasoning_summary_part.added"
            | "response.output_text.delta"
            | "response.refusal.delta"
            | "response.reasoning_summary_text.delta"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done"
            | "response.output_text.done"
            | "response.refusal.done"
            | "response.reasoning_summary_text.done"
            | "response.content_part.done"
            | "response.reasoning_summary_part.done" => {
                deltas.extend(self.content_event(kind, &value)?);
            }
            "response.output_item.done" => {
                let i = index(&value, "output_index")?;
                let item = self
                    .items
                    .get_mut(&i)
                    .ok_or_else(|| permanent("Responses done for undeclared item"))?;
                if item.closed {
                    return Err(permanent("Responses duplicate item done"));
                }
                let wire = value
                    .get("item")
                    .ok_or_else(|| permanent("Responses missing completed item"))?;
                Self::identity(item, wire)?;
                if item.kind == "function_call" {
                    Self::arguments(string(wire, "arguments")?)?;
                    item.arguments.clear();
                    item.arguments_done = true;
                }
                self.budget.snapshot(wire)?;
                item.closed = true;
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                let response = value
                    .get("response")
                    .ok_or_else(|| permanent("Responses missing terminal response"))?;
                self.response_id(response)?;
                let expected = kind.strip_prefix("response.").expect("response prefix");
                if string(response, "status")? != expected {
                    return Err(permanent("Responses terminal event/status mismatch"));
                }
                self.budget.snapshot(response)?;
                if expected == "failed" {
                    return Err(provider_error(
                        response.get("error").unwrap_or(&Value::Null),
                    ));
                }
                let output = response
                    .get("output")
                    .and_then(Value::as_array)
                    .ok_or_else(|| permanent("Responses terminal output must be an array"))?;
                if output.len() > self.budget.limits.max_items {
                    return Err(permanent("Responses terminal exceeds max_items"));
                }
                // Terminal snapshot may replace advisory text and arguments,
                // while every declared output identity and tool ordinal stays fixed.
                if output.len() != self.items.len() {
                    return Err(permanent("Responses terminal output declarations differ"));
                }
                let mut tool_index = 0;
                for (i, wire) in output.iter().enumerate() {
                    let item = self
                        .items
                        .get(&i)
                        .ok_or_else(|| permanent("Responses terminal missing item declaration"))?;
                    Self::identity(item, wire)?;
                    if item.kind == "function_call" {
                        if item.call_index != Some(tool_index) {
                            return Err(permanent("Responses terminal tool order changed"));
                        }
                        Self::arguments(string(wire, "arguments")?)?;
                        tool_index += 1;
                    }
                    if let Some(parts) = wire
                        .get("content")
                        .or_else(|| wire.get("summary"))
                        .and_then(Value::as_array)
                        && parts.len() > self.budget.limits.max_items
                    {
                        return Err(permanent("Responses terminal parts exceed max_items"));
                    }
                }
                let output = parse_openai_responses_output(response)?;
                if let Some(usage) = response.get("usage").filter(|v| v.is_object()) {
                    deltas.push(StreamDelta::Usage(model_usage_from_openai_responses(usage)));
                }
                deltas.push(done(output));
            }
            _ => {}
        }
        Ok(deltas)
    }
}
