use super::state::Responses;
use crate::translation::streaming::common::{index, permanent, string};
use causa_kernel::{ModelInvokeError, StreamDelta};
use serde_json::Value;

impl Responses {
    pub(super) fn content_event(
        &mut self,
        kind: &str,
        value: &Value,
    ) -> Result<Vec<StreamDelta>, ModelInvokeError> {
        let mut deltas = Vec::new();
        match kind {
            "response.content_part.added" | "response.reasoning_summary_part.added" => {
                self.budget.item(self.parts)?;
                let item = Self::item(&mut self.items, value)?;
                let part = value
                    .get("part")
                    .ok_or_else(|| permanent("Responses missing part"))?;
                let part_type = string(part, "type")?;
                if matches!(part_type, "output_text" | "summary_text" | "refusal") {
                    let initial = string(
                        part,
                        if part_type == "refusal" {
                            "refusal"
                        } else {
                            "text"
                        },
                    )?;
                    self.budget.add(initial.len())?;
                    if !initial.is_empty() && part_type == "output_text" {
                        deltas.push(StreamDelta::TextDelta {
                            delta: initial.into(),
                        });
                    }
                    if !initial.is_empty() && part_type == "summary_text" {
                        deltas.push(StreamDelta::ReasoningDelta {
                            delta: initial.into(),
                        });
                    }
                }
                if kind == "response.content_part.added" {
                    if item.kind != "message" {
                        return Err(permanent("Responses content part for non-message"));
                    }
                    let i = index(value, "content_index")?;
                    if i != item.parts.len() {
                        return Err(permanent("Responses invalid content part order"));
                    }
                    self.budget.add(part_type.len())?;
                    item.parts.insert(i, part_type.into());
                } else {
                    if item.kind != "reasoning" || part_type != "summary_text" {
                        return Err(permanent("Responses invalid reasoning summary part"));
                    }
                    let i = index(value, "summary_index")?;
                    if i != item.summaries.len() {
                        return Err(permanent("Responses invalid summary part order"));
                    }
                    item.summaries.insert(i);
                }
                self.parts += 1;
            }
            "response.output_text.delta"
            | "response.refusal.delta"
            | "response.reasoning_summary_text.delta"
            | "response.function_call_arguments.delta" => {
                let item = Self::item(&mut self.items, value)?;
                let delta = string(value, "delta")?;
                self.budget.add(delta.len())?;
                match kind {
                    "response.function_call_arguments.delta" => {
                        if item.kind != "function_call" || item.arguments_done {
                            return Err(permanent("Responses arguments delta outside open tool"));
                        }
                        item.arguments.push_str(delta);
                        deltas.push(StreamDelta::ToolCallDelta {
                            call_index: item.call_index.expect("tool index"),
                            provider_call_id: None,
                            name_delta: None,
                            arguments_delta: Some(delta.into()),
                        });
                    }
                    "response.reasoning_summary_text.delta" => {
                        let i = index(value, "summary_index")?;
                        if item.kind != "reasoning"
                            || !item.summaries.contains(&i)
                            || item.closed_summaries.contains(&i)
                        {
                            return Err(permanent(
                                "Responses reasoning delta outside open summary",
                            ));
                        }
                        deltas.push(StreamDelta::ReasoningDelta {
                            delta: delta.into(),
                        });
                    }
                    _ => {
                        let i = index(value, "content_index")?;
                        let expected = if kind == "response.output_text.delta" {
                            "output_text"
                        } else {
                            "refusal"
                        };
                        if item.kind != "message"
                            || item.parts.get(&i).map(String::as_str) != Some(expected)
                            || item.closed_parts.contains(&i)
                        {
                            return Err(permanent(
                                "Responses text delta outside matching open part",
                            ));
                        }
                        if expected == "output_text" {
                            deltas.push(StreamDelta::TextDelta {
                                delta: delta.into(),
                            });
                        }
                    }
                }
            }
            "response.function_call_arguments.done" => {
                let item = Self::item(&mut self.items, value)?;
                if item.kind != "function_call" || item.arguments_done {
                    return Err(permanent("Responses invalid arguments done order"));
                }
                if let Some(name) = value.get("name")
                    && name.as_str() != item.name.as_deref()
                {
                    return Err(permanent("Responses tool name changed"));
                }
                let args = string(value, "arguments")?;
                Self::arguments(args)?;
                self.budget.add(args.len())?;
                item.arguments.clear();
                item.arguments_done = true;
            }
            "response.output_text.done"
            | "response.refusal.done"
            | "response.reasoning_summary_text.done" => {
                let item = Self::item(&mut self.items, value)?;
                if kind == "response.reasoning_summary_text.done" {
                    let i = index(value, "summary_index")?;
                    if !item.summaries.contains(&i) || !item.closed_summaries.insert(i) {
                        return Err(permanent("Responses invalid summary done order"));
                    }
                    self.budget.add(string(value, "text")?.len())?;
                } else {
                    let i = index(value, "content_index")?;
                    let expected = if kind == "response.output_text.done" {
                        "output_text"
                    } else {
                        "refusal"
                    };
                    if item.parts.get(&i).map(String::as_str) != Some(expected)
                        || !item.closed_parts.insert(i)
                    {
                        return Err(permanent("Responses invalid text done order"));
                    }
                    self.budget.add(
                        string(
                            value,
                            if expected == "output_text" {
                                "text"
                            } else {
                                "refusal"
                            },
                        )?
                        .len(),
                    )?;
                }
            }
            "response.content_part.done" | "response.reasoning_summary_part.done" => {
                let item = Self::item(&mut self.items, value)?;
                let part = value
                    .get("part")
                    .ok_or_else(|| permanent("Responses missing completed part"))?;
                if kind == "response.content_part.done" {
                    let i = index(value, "content_index")?;
                    if item.parts.get(&i).map(String::as_str) != Some(string(part, "type")?) {
                        return Err(permanent("Responses completed part identity changed"));
                    }
                    if !item.completed_parts.insert(i) {
                        return Err(permanent("Responses duplicate completed content part"));
                    }
                    let part_type = string(part, "type")?;
                    if matches!(part_type, "output_text" | "refusal") {
                        string(
                            part,
                            if part_type == "output_text" {
                                "text"
                            } else {
                                "refusal"
                            },
                        )?;
                    }
                    item.closed_parts.insert(i);
                } else {
                    let i = index(value, "summary_index")?;
                    if !item.summaries.contains(&i) || string(part, "type")? != "summary_text" {
                        return Err(permanent("Responses invalid completed summary"));
                    }
                    string(part, "text")?;
                    if !item.completed_summaries.insert(i) {
                        return Err(permanent("Responses duplicate completed summary part"));
                    }
                    item.closed_summaries.insert(i);
                }
                self.budget.snapshot(part)?;
            }
            _ => unreachable!("content event dispatch"),
        }
        Ok(deltas)
    }
}
