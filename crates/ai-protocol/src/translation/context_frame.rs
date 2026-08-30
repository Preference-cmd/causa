//! Shared policy walk over kernel `ContextFrame`s (crate-internal).
//!
//! The three protocol renderers (`anthropic`, `openai_chat`,
//! `openai_responses`) share one normalization pass; this module owns it
//! so the policy cannot drift between protocols. The walk applies every
//! renderer-independent decision exactly once:
//!
//! - Wire-role assignment from the `BlockMeta::source` vocabulary (the
//!   table lives in the [`super`] module docs — it is public contract).
//! - Empty text blocks are skipped, mirroring the kernel model door's
//!   own commit policy (the host door can commit them).
//! - Adjacent same-role text blocks join into one segment with `\n` —
//!   the kernel attaches no meaning to block boundaries between
//!   same-role texts, and one shared rule is the anti-drift guarantee.
//! - Tool call ids come from `meta.provider_call_id`, falling back to
//!   the kernel `call_id` for synthetic calls the provider never named;
//!   the same rule resolves tool result ids through the frame's
//!   `call_id → wire id` map (pre-pass, so order never matters). An
//!   unpaired result falls back to its own kernel `call_id`; the
//!   provider rejects the orphan at HTTP time — the loud failure path.
//! - Non-string tool observations serialize to a string.
//!
//! Emitters then map the ordered [`Segment`] list to per-protocol wire
//! shapes (message grouping, content block shapes, argument encoding,
//! error flags). Rendering stays deterministic: the walk is
//! single-pass over an ordered list and no hash-map iteration reaches
//! the output.

use std::collections::HashMap;

use serde_json::Value;

use reimagine_context_kernel::{
    BlockContent, ContextFrame, ModelInvokeError, ModelInvokeErrorKind, ToolResultStatus,
};

/// The wire role a text block renders as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    System,
    User,
    Assistant,
}

pub(crate) fn text_role(source: Option<&str>) -> Role {
    match source {
        None | Some("assistant") => Role::Assistant,
        Some("system") => Role::System,
        // "user", "inject[:detail]", unknown open-vocabulary tags
        Some(_) => Role::User,
    }
}

/// A tool call prepared for the wire: id resolution already applied.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedCall {
    pub wire_id: String,
    pub name: String,
    pub arguments: Value,
}

/// One normalized, frame-order-preserving piece of the conversation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Segment {
    /// Adjacent same-role text joined; empty texts skipped.
    Text {
        role: Role,
        text: String,
    },
    ToolCall(PreparedCall),
    ToolResult {
        wire_id: String,
        status: ToolResultStatus,
        content: String,
    },
}

pub(crate) struct NormalizedFrame {
    pub segments: Vec<Segment>,
}

/// Run the shared policy walk over a frame.
pub(crate) fn normalize(frame: &ContextFrame) -> NormalizedFrame {
    let blocks = &frame.model_context.blocks;

    // Pairing map: kernel call_id -> the id the provider saw on the tool
    // call. Pre-pass over the whole frame, so a result never depends on
    // where its call block sits.
    let mut provider_ids: HashMap<String, String> = HashMap::new();
    for block in blocks {
        if let BlockContent::ToolCall(call) = &block.content {
            provider_ids.insert(
                call.call_id.0.clone(),
                block
                    .meta
                    .provider_call_id
                    .clone()
                    .unwrap_or_else(|| call.call_id.0.clone()),
            );
        }
    }

    let mut segments: Vec<Segment> = Vec::new();
    for block in blocks {
        match &block.content {
            BlockContent::Text(text) => {
                // Mirrors the kernel model door: empty texts are not
                // conversation content.
                if text.0.is_empty() {
                    continue;
                }
                let role = text_role(block.meta.source.as_deref());
                match segments.last_mut() {
                    Some(Segment::Text {
                        role: last_role,
                        text: last_text,
                    }) if *last_role == role => {
                        last_text.push('\n');
                        last_text.push_str(&text.0);
                    }
                    _ => segments.push(Segment::Text {
                        role,
                        text: text.0.clone(),
                    }),
                }
            }
            BlockContent::ToolCall(call) => segments.push(Segment::ToolCall(PreparedCall {
                wire_id: block
                    .meta
                    .provider_call_id
                    .clone()
                    .unwrap_or_else(|| call.call_id.0.clone()),
                name: call.tool_name.clone(),
                arguments: call.arguments.clone(),
            })),
            BlockContent::ToolResult(result) => {
                let content = match &result.output.content {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                segments.push(Segment::ToolResult {
                    wire_id: provider_ids
                        .get(&result.call_id.0)
                        .cloned()
                        .unwrap_or_else(|| result.call_id.0.clone()),
                    status: result.status.clone(),
                    content,
                });
            }
        }
    }

    NormalizedFrame { segments }
}

/// OpenAI-family function-arguments wire codec: arguments travel as a
/// JSON string; an empty or absent string degrades to an empty object;
/// anything else the provider sent inline passes through.
pub(crate) fn decode_wire_arguments(raw: Option<&Value>) -> Result<Value, ModelInvokeError> {
    match raw {
        None | Some(Value::Null) => Ok(serde_json::json!({})),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(serde_json::json!({})),
        Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| {
            ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                format!("function arguments: {e}"),
            )
        }),
        Some(other) => Ok(other.clone()),
    }
}
