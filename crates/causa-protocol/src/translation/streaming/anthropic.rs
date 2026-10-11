use std::collections::BTreeMap;

use causa_kernel::{ModelInvokeError, StreamDelta};
use serde_json::{Value, json};

use super::{
    StreamLimits,
    common::{Budget, done, index, merge_usage, permanent, provider_error, string},
};
use crate::translation::{
    anthropic::parse_anthropic_response, sse_parser::SseEvent, usage::model_usage_from_anthropic,
};

struct Block {
    wire: Value,
    arguments: String,
    saw_arguments: bool,
    closed: bool,
    call_index: Option<usize>,
}

pub(super) struct Anthropic {
    budget: Budget,
    started: bool,
    stopped: bool,
    blocks: BTreeMap<usize, Block>,
    tools: usize,
    stop_reason: Option<String>,
    usage: Value,
    has_usage: bool,
}

impl Anthropic {
    pub fn new(limits: StreamLimits) -> Self {
        Self {
            budget: Budget::new(limits),
            started: false,
            stopped: false,
            blocks: BTreeMap::new(),
            tools: 0,
            stop_reason: None,
            usage: json!({}),
            has_usage: false,
        }
    }
    pub fn event(&mut self, event: &SseEvent) -> Result<Vec<StreamDelta>, ModelInvokeError> {
        let value: Value = serde_json::from_str(&event.data)
            .map_err(|_| permanent("Anthropic event contains invalid JSON"))?;
        let kind = string(&value, "type")?;
        if let Some(name) = &event.event
            && name != kind
            && name != "message"
        {
            return Err(permanent("Anthropic SSE event/type mismatch"));
        }
        if kind == "error" {
            return Err(provider_error(
                value
                    .get("error")
                    .ok_or_else(|| permanent("Anthropic missing error"))?,
            ));
        }
        if kind == "ping" {
            return Ok(Vec::new());
        }
        let mut deltas = Vec::new();
        match kind {
            "message_start" => {
                if self.started {
                    return Err(permanent("Anthropic duplicate message_start"));
                }
                let message = value
                    .get("message")
                    .and_then(Value::as_object)
                    .ok_or_else(|| permanent("Anthropic missing message"))?;
                if message
                    .get("content")
                    .and_then(Value::as_array)
                    .is_none_or(|c| !c.is_empty())
                {
                    return Err(permanent("Anthropic message_start must have empty content"));
                }
                if message.get("stop_reason").is_some_and(|r| !r.is_null()) {
                    return Err(permanent("Anthropic message_start contains stop_reason"));
                }
                self.started = true;
                if let Some(usage) = message.get("usage").filter(|u| !u.is_null()) {
                    merge_usage(&mut self.usage, usage, &mut self.budget)?;
                    self.has_usage = true;
                    deltas.push(StreamDelta::Usage(model_usage_from_anthropic(&self.usage)));
                }
            }
            "content_block_start" => {
                self.require_active()?;
                if self.stopped {
                    return Err(permanent("Anthropic block after message_delta"));
                }
                self.budget.item(self.blocks.len())?;
                let i = index(&value, "index")?;
                if i != self.blocks.len() {
                    return Err(permanent(
                        "Anthropic block index must follow declaration order",
                    ));
                }
                let wire = value
                    .get("content_block")
                    .cloned()
                    .ok_or_else(|| permanent("Anthropic missing content_block"))?;
                let block_type = string(&wire, "type")?.to_owned();
                self.budget.add(block_type.len())?;
                let mut call_index = None;
                match block_type.as_str() {
                    "text" => {
                        let text = string(&wire, "text")?;
                        self.budget.add(text.len())?;
                        if !text.is_empty() {
                            deltas.push(StreamDelta::TextDelta { delta: text.into() });
                        }
                    }
                    "thinking" => {
                        let text = string(&wire, "thinking")?;
                        self.budget.add(text.len())?;
                        if let Some(sig) = wire.get("signature") {
                            self.budget.add(
                                sig.as_str()
                                    .ok_or_else(|| permanent("Anthropic invalid signature"))?
                                    .len(),
                            )?;
                        }
                        if !text.is_empty() {
                            deltas.push(StreamDelta::ReasoningDelta { delta: text.into() });
                        }
                    }
                    "tool_use" => {
                        let id = string(&wire, "id")?;
                        let name = string(&wire, "name")?;
                        if id.is_empty() || name.is_empty() {
                            return Err(permanent("Anthropic empty tool identity"));
                        }
                        if self
                            .blocks
                            .values()
                            .any(|b| b.wire.get("id").and_then(Value::as_str) == Some(id))
                        {
                            return Err(permanent("Anthropic duplicate tool id"));
                        }
                        if !wire.get("input").is_some_and(Value::is_object) {
                            return Err(permanent("Anthropic tool input must be an object"));
                        }
                        self.budget.add(id.len() + name.len())?;
                        call_index = Some(self.tools);
                        self.tools += 1;
                        deltas.push(StreamDelta::ToolCallDelta {
                            call_index: call_index.expect("tool index"),
                            provider_call_id: Some(id.into()),
                            name_delta: Some(name.into()),
                            arguments_delta: None,
                        });
                    }
                    _ => {}
                }
                // Every block is kept as the complete parser stores it:
                // `text` and `tool_use` residue, `thinking` with its
                // signature, `redacted_thinking` and unknown block types as
                // sealed material. Nothing is stripped; every field the
                // stream delivers is budgeted like any output.
                let object = wire
                    .as_object()
                    .ok_or_else(|| permanent("Anthropic block must be an object"))?;
                let known: &[&str] = match block_type.as_str() {
                    "text" => &["type", "text"],
                    "tool_use" => &["type", "id", "name", "input"],
                    _ => &[],
                };
                for (key, value) in object.iter() {
                    if !known.contains(&key.as_str()) {
                        self.budget.add(
                            key.len()
                                + serde_json::to_vec(value)
                                    .map_err(|_| permanent("Anthropic invalid field"))?
                                    .len(),
                        )?;
                    }
                }
                if block_type == "tool_use" {
                    self.budget.add(
                        serde_json::to_vec(&wire["input"])
                            .map_err(|_| permanent("Anthropic invalid input"))?
                            .len(),
                    )?;
                }
                self.blocks.insert(
                    i,
                    Block {
                        wire,
                        arguments: String::new(),
                        saw_arguments: false,
                        closed: false,
                        call_index,
                    },
                );
            }
            "content_block_delta" => {
                self.require_active()?;
                if self.stopped {
                    return Err(permanent("Anthropic delta after message_delta"));
                }
                let i = index(&value, "index")?;
                let block = self
                    .blocks
                    .get_mut(&i)
                    .ok_or_else(|| permanent("Anthropic delta for undeclared block"))?;
                if block.closed {
                    return Err(permanent("Anthropic delta for closed block"));
                }
                let delta = value
                    .get("delta")
                    .ok_or_else(|| permanent("Anthropic missing delta"))?;
                let delta_type = string(delta, "type")?;
                let block_type = string(&block.wire, "type")?;
                let (field, delta_field) = match delta_type {
                    "text_delta" if block_type == "text" => ("text", "text"),
                    "thinking_delta" if block_type == "thinking" => ("thinking", "thinking"),
                    "signature_delta" if block_type == "thinking" => ("signature", "signature"),
                    "input_json_delta" if block_type == "tool_use" => {
                        let text = string(delta, "partial_json")?;
                        self.budget.add(text.len())?;
                        block.arguments.push_str(text);
                        block.saw_arguments = true;
                        deltas.push(StreamDelta::ToolCallDelta {
                            call_index: block.call_index.expect("tool index"),
                            provider_call_id: None,
                            name_delta: None,
                            arguments_delta: Some(text.into()),
                        });
                        return Ok(deltas);
                    }
                    "citations_delta" if block_type == "text" => {
                        // Citations arrive as deltas and belong to the block's
                        // wire residue, appended in delivery order.
                        let citation = delta
                            .get("citation")
                            .ok_or_else(|| permanent("Anthropic missing citation"))?;
                        self.budget.add(
                            serde_json::to_vec(citation)
                                .map_err(|_| permanent("Anthropic invalid citation"))?
                                .len(),
                        )?;
                        let object = block.wire.as_object_mut().expect("validated block");
                        let citations = object.entry("citations").or_insert_with(|| json!([]));
                        if !citations.is_array() {
                            *citations = json!([]);
                        }
                        citations
                            .as_array_mut()
                            .expect("citations array")
                            .push(citation.clone());
                        return Ok(deltas);
                    }
                    // Any other block that declared an object `input` (a
                    // server-side tool call, kept as sealed material) receives
                    // it the same way. The fragments rebuild the block so it
                    // matches the batch form; they are not client tool-call
                    // observations.
                    "input_json_delta"
                        if !matches!(block_type, "text" | "thinking")
                            && block.wire.get("input").is_some_and(Value::is_object) =>
                    {
                        let text = string(delta, "partial_json")?;
                        self.budget.add(text.len())?;
                        block.arguments.push_str(text);
                        block.saw_arguments = true;
                        return Ok(deltas);
                    }
                    "text_delta" | "thinking_delta" | "signature_delta" | "input_json_delta"
                    | "citations_delta" => {
                        return Err(permanent("Anthropic delta/block type conflict"));
                    }
                    // A delta this decoder has no rule for cannot be applied,
                    // and a block missing part of its content must not become
                    // material that is replayed verbatim. The diagnostic is
                    // static: the wire value is provider-controlled.
                    _ => return Err(permanent("Anthropic unsupported content block delta type")),
                };
                let text = string(delta, delta_field)?;
                self.budget.add(text.len())?;
                let target = block
                    .wire
                    .as_object_mut()
                    .expect("validated block")
                    .entry(field)
                    .or_insert_with(|| Value::String(String::new()));
                if !target.is_string() {
                    return Err(permanent("Anthropic invalid block field"));
                }
                // serde_json strings have no mutable string accessor; take the
                // existing value to append without copying its accumulated prefix.
                let mut accumulated = match std::mem::replace(&mut block.wire[field], Value::Null) {
                    Value::String(s) => s,
                    _ => unreachable!("validated string"),
                };
                accumulated.push_str(text);
                block.wire[field] = Value::String(accumulated);
                if field == "text" {
                    deltas.push(StreamDelta::TextDelta { delta: text.into() });
                }
                if field == "thinking" {
                    deltas.push(StreamDelta::ReasoningDelta { delta: text.into() });
                }
            }
            "content_block_stop" => {
                self.require_active()?;
                if self.stopped {
                    return Err(permanent("Anthropic block stop after message_delta"));
                }
                let block = self
                    .blocks
                    .get_mut(&index(&value, "index")?)
                    .ok_or_else(|| permanent("Anthropic stop for undeclared block"))?;
                if block.closed {
                    return Err(permanent("Anthropic duplicate block stop"));
                }
                if block.saw_arguments {
                    let input: Value = serde_json::from_str(&block.arguments)
                        .map_err(|_| permanent("Anthropic malformed tool arguments"))?;
                    if !input.is_object() {
                        return Err(permanent("Anthropic tool arguments must be an object"));
                    }
                    block.wire["input"] = input;
                    block.arguments.clear();
                }
                block.closed = true;
            }
            "message_delta" => {
                self.require_active()?;
                if self.blocks.values().any(|b| !b.closed) {
                    return Err(permanent("Anthropic message_delta before block closure"));
                }
                let delta = value
                    .get("delta")
                    .ok_or_else(|| permanent("Anthropic missing message delta"))?;
                if let Some(reason) = delta.get("stop_reason").filter(|r| !r.is_null()) {
                    let reason = reason
                        .as_str()
                        .ok_or_else(|| permanent("Anthropic invalid stop_reason"))?;
                    // The value is recorded verbatim and classified by the same
                    // mapping table the batch parser uses. An unknown value is
                    // not an error here: it becomes Unrecognized downstream.
                    if self.stop_reason.as_deref().is_some_and(|old| old != reason) {
                        return Err(permanent("Anthropic stop_reason changed"));
                    }
                    self.stop_reason = Some(reason.into());
                    self.stopped = true;
                }
                if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
                    merge_usage(&mut self.usage, usage, &mut self.budget)?;
                    self.has_usage = true;
                    deltas.push(StreamDelta::Usage(model_usage_from_anthropic(&self.usage)));
                }
            }
            "message_stop" => {
                self.require_active()?;
                if self.blocks.values().any(|b| !b.closed) {
                    return Err(permanent("Anthropic message_stop with open blocks"));
                }
                let reason = self
                    .stop_reason
                    .as_ref()
                    .ok_or_else(|| permanent("Anthropic message_stop without stop_reason"))?;
                let content: Vec<Value> = std::mem::take(&mut self.blocks)
                    .into_values()
                    .map(|b| b.wire)
                    .collect();
                let mut wire = json!({"content":content,"stop_reason":reason});
                if self.has_usage {
                    wire["usage"] = std::mem::take(&mut self.usage);
                }
                deltas.push(done(parse_anthropic_response(&wire)?));
            }
            _ => {}
        }
        Ok(deltas)
    }
    fn require_active(&self) -> Result<(), ModelInvokeError> {
        if !self.started {
            return Err(permanent("Anthropic event before message_start"));
        }
        Ok(())
    }
}
