//! Request-only tool exchange validation. General Context edits remain local.

use std::collections::HashSet;

use causa_kernel::{BlockContent, ContextFrame, ModelInvokeError, ModelInvokeErrorKind};
use serde_json::Value;

fn invalid(message: impl Into<String>) -> ModelInvokeError {
    ModelInvokeError::new(ModelInvokeErrorKind::InvalidRequest, message)
}

/// These adapters send a self-contained frame, without remote conversation IDs.
pub(super) fn validate_frame(frame: &ContextFrame) -> Result<(), ModelInvokeError> {
    let mut declarations = HashSet::new();
    let mut wire_ids = HashSet::new();
    let mut completed = HashSet::new();
    for block in &frame.blocks {
        match block.content() {
            BlockContent::ToolCall(call) => {
                if call.tool_name.trim().is_empty() || !call.arguments.is_object() {
                    return Err(invalid(
                        "tool declarations require a name and object arguments",
                    ));
                }
                let wire_id = block
                    .meta()
                    .provider_call_id
                    .clone()
                    .unwrap_or_else(|| block.id().0.to_string());
                if wire_id.is_empty() || !wire_ids.insert(wire_id) {
                    return Err(invalid(
                        "tool declaration wire IDs must be nonempty and unique in the request",
                    ));
                }
                if !declarations.insert(block.id()) {
                    return Err(invalid("duplicate tool declaration identity in request"));
                }
            }
            BlockContent::ToolResult(result) => {
                if !declarations.contains(&result.call_block_id) {
                    return Err(invalid(
                        "tool result requires an earlier declaration in this request",
                    ));
                }
                if !completed.insert(result.call_block_id) {
                    return Err(invalid(
                        "duplicate tool result for one declaration in request",
                    ));
                }
            }
            BlockContent::Parts(_) => {}
        }
    }
    // Preserve input order for deterministic first-error selection.
    for block in &frame.blocks {
        if matches!(block.content(), BlockContent::ToolCall(_)) && !completed.contains(&block.id())
        {
            return Err(invalid(format!(
                "tool declaration {:?} has no result in request",
                block.id()
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_chat(messages: &[Value]) -> Result<(), ModelInvokeError> {
    let mut pending = HashSet::new();
    for message in messages {
        if message["role"] == "tool" {
            if !pending.remove(message["tool_call_id"].as_str().unwrap_or_default()) {
                return Err(invalid(
                    "Chat tool result does not answer the preceding assistant tool calls",
                ));
            }
        } else {
            if !pending.is_empty() {
                return Err(invalid(
                    "Chat tool calls must be followed by their tool results before another message",
                ));
            }
            if let Some(calls) = message["tool_calls"].as_array() {
                pending.extend(calls.iter().filter_map(|call| call["id"].as_str()));
            }
        }
    }
    if !pending.is_empty() {
        return Err(invalid("Chat request ends with unanswered tool calls"));
    }
    Ok(())
}

pub(super) fn validate_anthropic(messages: &[(&str, Vec<Value>)]) -> Result<(), ModelInvokeError> {
    let mut pending = HashSet::new();
    for (role, content) in messages {
        if !pending.is_empty() && *role != "user" {
            return Err(invalid(
                "Anthropic tool_use must be answered by the next user message",
            ));
        }
        let mut saw_other = false;
        for block in content {
            if block["type"] == "tool_result" {
                if *role != "user"
                    || saw_other
                    || !pending.remove(block["tool_use_id"].as_str().unwrap_or_default())
                {
                    return Err(invalid(
                        "Anthropic tool results must begin the next user message and match its preceding tool_use blocks",
                    ));
                }
            } else {
                saw_other = true;
                if *role == "user" && !pending.is_empty() {
                    return Err(invalid(
                        "Anthropic user content cannot precede outstanding tool results",
                    ));
                }
                if block["type"] == "tool_use" {
                    pending.insert(block["id"].as_str().unwrap_or_default());
                }
            }
        }
        if *role == "user" && !pending.is_empty() {
            return Err(invalid("Anthropic user message is missing tool results"));
        }
    }
    if !pending.is_empty() {
        return Err(invalid(
            "Anthropic request ends with unanswered tool_use blocks",
        ));
    }
    Ok(())
}
