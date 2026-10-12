use std::collections::BTreeMap;

use causa_kernel::{ModelInvokeError, StreamDelta};
use serde_json::{Value, json};

use super::{
    StreamLimits,
    common::{Budget, done, index, merge_usage, provider_error, string, undecodable},
};
use crate::translation::{
    openai_chat::parse_openai_chat_response, sse_parser::SseEvent,
    usage::model_usage_from_openai_chat,
};

#[derive(Default)]
struct Tool {
    id: Option<String>,
    name: String,
    arguments: String,
}

pub(super) struct Chat {
    budget: Budget,
    id: Option<String>,
    text: String,
    reasoning: String,
    tools: BTreeMap<usize, Tool>,
    finish: Option<String>,
    usage: Value,
    has_usage: bool,
}

impl Chat {
    pub fn new(limits: StreamLimits) -> Self {
        Self {
            budget: Budget::new(limits),
            id: None,
            text: String::new(),
            reasoning: String::new(),
            tools: BTreeMap::new(),
            finish: None,
            usage: json!({}),
            has_usage: false,
        }
    }
    pub fn event(&mut self, event: &SseEvent) -> Result<Vec<StreamDelta>, ModelInvokeError> {
        if event.data == "[DONE]" {
            let finish = self
                .finish
                .as_ref()
                .ok_or_else(|| undecodable("Chat [DONE] without finish_reason"))?;
            let mut calls = Vec::with_capacity(self.tools.len());
            for tool in self.tools.values() {
                let id = tool
                    .id
                    .as_ref()
                    .ok_or_else(|| undecodable("Chat tool missing id"))?;
                if tool.name.is_empty() {
                    return Err(undecodable("Chat tool missing name"));
                }
                serde_json::from_str::<Value>(&tool.arguments)
                    .map_err(|_| undecodable("Chat malformed or missing tool arguments"))?;
                calls.push(json!({"id":id,"type":"function","function":{"name":tool.name,"arguments":tool.arguments}}));
            }
            // The assembled message mirrors what a batch response carries, so
            // the batch parser derives the same items (including the sealed
            // message-level item for `reasoning_content`). Fields the stream
            // never delivers are not invented: `content` is absent when no
            // text arrived.
            let mut message = json!({"role": "assistant", "tool_calls": calls});
            if !self.text.is_empty() {
                message["content"] = json!(self.text);
            }
            if !self.reasoning.is_empty() {
                message["reasoning_content"] = json!(self.reasoning);
            }
            let mut wire = json!({"choices":[{"message":message,"finish_reason":finish}]});
            if self.has_usage {
                wire["usage"] = std::mem::take(&mut self.usage);
            }
            return Ok(vec![done(parse_openai_chat_response(&wire)?)]);
        }
        let value: Value = serde_json::from_str(&event.data)
            .map_err(|_| undecodable("Chat event contains invalid JSON"))?;
        if let Some(error) = value.get("error") {
            return Err(provider_error(error));
        }
        if let Some(id) = value.get("id") {
            let id = id
                .as_str()
                .ok_or_else(|| undecodable("Chat invalid response id"))?;
            if let Some(previous) = &self.id {
                if previous != id {
                    return Err(undecodable("Chat response id changed"));
                }
            } else {
                self.budget.add(id.len())?;
                self.id = Some(id.into());
            }
        }
        let choices = value
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| undecodable("Chat missing choices array"))?;
        if choices.len() > 1 {
            return Err(undecodable("Chat streaming supports one choice"));
        }
        let mut deltas = Vec::new();
        if let Some(choice) = choices.first() {
            if index(choice, "index")? != 0 {
                return Err(undecodable("Chat streaming supports choice index zero"));
            }
            let delta = choice
                .get("delta")
                .and_then(Value::as_object)
                .ok_or_else(|| undecodable("Chat missing delta object"))?;
            if self.finish.is_some() {
                return Err(undecodable("Chat choice follows finish_reason"));
            }
            if let Some(content) = delta.get("content").filter(|v| !v.is_null()) {
                let text = content
                    .as_str()
                    .ok_or_else(|| undecodable("Chat content delta must be a string"))?;
                self.budget.add(text.len())?;
                self.text.push_str(text);
                if !text.is_empty() {
                    deltas.push(StreamDelta::TextDelta { delta: text.into() });
                }
            }
            if let Some(reasoning) = delta.get("reasoning_content").filter(|v| !v.is_null()) {
                let reasoning = reasoning
                    .as_str()
                    .ok_or_else(|| undecodable("Chat reasoning_content delta must be a string"))?;
                self.budget.add(reasoning.len())?;
                self.reasoning.push_str(reasoning);
            }
            if delta.contains_key("function_call") {
                return Err(undecodable("Chat legacy function_call is unsupported"));
            }
            if let Some(calls) = delta.get("tool_calls") {
                let calls = calls
                    .as_array()
                    .ok_or_else(|| undecodable("Chat tool_calls delta must be an array"))?;
                for call in calls {
                    let i = index(call, "index")?;
                    if !self.tools.contains_key(&i) {
                        self.budget.item(self.tools.len())?;
                        if i != self.tools.len() {
                            return Err(undecodable(
                                "Chat tool index must follow declaration order",
                            ));
                        }
                        self.tools.insert(i, Tool::default());
                    }
                    if let Some(kind) = call.get("type")
                        && kind != "function"
                    {
                        return Err(undecodable("Chat unsupported tool type"));
                    }
                    let incoming_id = call.get("id").map(|_| string(call, "id")).transpose()?;
                    if let Some(id) = incoming_id {
                        if id.is_empty() {
                            return Err(undecodable("Chat empty tool id"));
                        }
                        if self
                            .tools
                            .iter()
                            .any(|(other, tool)| *other != i && tool.id.as_deref() == Some(id))
                        {
                            return Err(undecodable("Chat duplicate tool id"));
                        }
                    }
                    let tool = self.tools.get_mut(&i).expect("declared tool");
                    if let Some(id) = incoming_id {
                        if let Some(previous) = &tool.id {
                            if previous != id {
                                return Err(undecodable("Chat tool id changed"));
                            }
                        } else {
                            self.budget.add(id.len())?;
                            tool.id = Some(id.into());
                        }
                    }
                    let function = call
                        .get("function")
                        .map(|v| {
                            v.as_object()
                                .ok_or_else(|| undecodable("Chat invalid function delta"))
                        })
                        .transpose()?;
                    let name = function
                        .and_then(|f| f.get("name"))
                        .map(|v| {
                            v.as_str()
                                .ok_or_else(|| undecodable("Chat invalid name delta"))
                        })
                        .transpose()?;
                    let args = function
                        .and_then(|f| f.get("arguments"))
                        .map(|v| {
                            v.as_str()
                                .ok_or_else(|| undecodable("Chat invalid arguments delta"))
                        })
                        .transpose()?;
                    if let Some(name) = name {
                        self.budget.add(name.len())?;
                        tool.name.push_str(name);
                    }
                    if let Some(args) = args {
                        self.budget.add(args.len())?;
                        tool.arguments.push_str(args);
                    }
                    deltas.push(StreamDelta::ToolCallDelta {
                        call_index: i,
                        provider_call_id: incoming_id.map(Into::into),
                        name_delta: name.map(Into::into),
                        arguments_delta: args.map(Into::into),
                    });
                    if deltas.len() > self.budget.limits.max_pending_deltas {
                        return Err(undecodable("Chat exceeds max_pending_deltas"));
                    }
                }
            }
            if let Some(reason) = choice.get("finish_reason").filter(|v| !v.is_null()) {
                let reason = reason
                    .as_str()
                    .ok_or_else(|| undecodable("Chat invalid finish_reason"))?;
                // Accepted verbatim: the shared mapping table classifies it, so
                // an unknown value becomes Unrecognized rather than an error.
                self.finish = Some(reason.into());
            }
        }
        if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
            merge_usage(&mut self.usage, usage, &mut self.budget)?;
            self.has_usage = true;
            deltas.push(StreamDelta::Usage(model_usage_from_openai_chat(
                &self.usage,
            )));
        }
        Ok(deltas)
    }
}
