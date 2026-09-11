//! Shared policy walk over kernel `ContextFrame`s (crate-internal).
//!
//! The three renderers (`anthropic`, `openai_chat`, `openai_responses`)
//! share one normalization pass; this module owns it so the policy cannot
//! drift between protocols. The walk applies every renderer-independent
//! decision exactly once:
//!
//! - Wire-role assignment from the `BlockMeta::source` vocabulary (the
//!   table in the [`super`] docs is public contract).
//! - Empty text is not conversation content: an empty `Text` part is
//!   dropped, and a Parts block left with nothing renders nothing
//!   (mirroring the kernel model door's commit policy; the host door can
//!   commit them).
//! - Two-tier adjacency. Part boundaries inside one Parts block are
//!   preserved to the wire (adjacent `Text` parts render as separate
//!   content blocks). Block boundaries are envelope-only: same-role text
//!   meeting across a block seam joins with `\n`, and only at the seam —
//!   the join is attempted for a block's FIRST text emission only.
//! - Media: each [`MediaRef`] resolves through the caller's
//!   [`MediaSet`]. A resolved `image/*` payload renders inline, but only
//!   in user position (no protocol accepts assistant- or system-authored
//!   input images); anything else — missing resolution, non-image type,
//!   empty payload, non-user role — degrades to the deterministic text
//!   placeholder `[media: {type} {reference}]`. The decision is made
//!   once, here, never per-renderer.
//! - Tool call ids come from `meta.provider_call_id`, falling back to
//!   the kernel `call_id` for synthetic calls the provider never named;
//!   tool result ids resolve through the frame's `(turn_id, call_id) →
//!   wire id` map (pre-pass, so order never matters). The turn id scopes
//!   the key because `ToolCallId` is unique only within one turn — two
//!   turns may reuse an id, and each turn's result keeps its own wire id.
//!   An unpaired result falls back to its own kernel `call_id`; the
//!   provider rejects the orphan at HTTP time (the loud failure path).
//! - Tool result media travels on the result segment in result order;
//!   whether it embeds (Anthropic) or hoists (OpenAI-family) is the
//!   emitter's call. Non-string tool observations serialize to a string.
//!
//! Emitters then map the ordered [`Segment`] list to per-protocol wire
//! shapes (message grouping, content block shapes, argument encoding,
//! error flags). Rendering stays deterministic: the walk is single-pass
//! over an ordered list and no hash-map iteration reaches the output.

use std::collections::HashMap;

use serde_json::Value;

use causa_kernel::{
    BlockContent, ContentPart, ContextFrame, MediaRef, ModelInvokeError, ModelInvokeErrorKind,
    ToolResultStatus, ToolSurface,
};

use super::media::MediaSet;

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

/// One media attachment as the emitters consume it: the walk's single
/// render-or-degrade decision, applied identically in all protocols.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ResolvedMedia {
    /// A resolved `image/*` payload — renderable inline everywhere.
    Image {
        media_type: String,
        data_base64: String,
    },
    /// Deterministic degradation: unresolved reference, non-image type,
    /// or empty payload. Carries the final placeholder text.
    Placeholder(String),
}

fn placeholder_text(r: &MediaRef) -> String {
    format!("[media: {} {}]", r.media_type, r.reference)
}

fn resolve_media(r: &MediaRef, role: Role, media: &MediaSet) -> ResolvedMedia {
    // Input images render inline only in user position — no protocol
    // accepts model-authored (assistant) or system-position input
    // images. Anything else degrades before the emitters see it.
    if role != Role::User {
        return ResolvedMedia::Placeholder(placeholder_text(r));
    }
    match media.get(r) {
        Some(p) if p.media_type.starts_with("image/") && !p.data_base64.is_empty() => {
            ResolvedMedia::Image {
                media_type: p.media_type.clone(),
                data_base64: p.data_base64.clone(),
            }
        }
        // resolved but not a renderable image, or not resolved at all
        _ => ResolvedMedia::Placeholder(placeholder_text(r)),
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
    /// Adjacent same-role text joined across block seams; empty texts
    /// skipped.
    Text {
        role: Role,
        text: String,
    },
    /// A media part from a Parts block, resolved or degraded.
    Media {
        role: Role,
        media: ResolvedMedia,
    },
    ToolCall(PreparedCall),
    ToolResult {
        wire_id: String,
        status: ToolResultStatus,
        content: String,
        /// The result's media attachments, resolved or degraded, in
        /// payload order.
        media: Vec<ResolvedMedia>,
    },
}

pub(crate) struct NormalizedFrame {
    pub segments: Vec<Segment>,
}

/// Run the shared policy walk over a frame, resolving media through
/// `media`.
pub(crate) fn normalize(frame: &ContextFrame, media: &MediaSet) -> NormalizedFrame {
    let blocks = &frame.model_context.blocks;

    // Pairing map: (owning turn id, kernel call_id) -> the id the
    // provider saw on the tool call. The turn scopes the key because a
    // `ToolCallId` is unique only within its turn; a result block
    // always pairs with a call block of the same turn (kernel-enforced),
    // so this resolves across a whole conversation frame without
    // cross-turn collisions. Pre-pass over the frame, so a result never
    // depends on where its call block sits.
    let mut provider_ids: HashMap<(String, String), String> = HashMap::new();
    for block in blocks {
        if let BlockContent::ToolCall(call) = &block.content {
            provider_ids.insert(
                (block.id.turn_id.0.clone(), call.call_id.0.clone()),
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
        // Whether this block has emitted any segment yet. The cross-block
        // text join is attempted only for a block's first emission, so
        // part boundaries INSIDE one block stay preserved while block
        // seams stay envelope-only.
        let mut block_emitted = false;
        match &block.content {
            BlockContent::Parts(parts) => {
                let role = text_role(block.meta.source.as_deref());
                for part in parts {
                    match part {
                        ContentPart::Text(t) => {
                            // Empty text is not conversation content.
                            if t.0.is_empty() {
                                continue;
                            }
                            if !block_emitted
                                && let Some(Segment::Text {
                                    role: last_role,
                                    text: last_text,
                                }) = segments.last_mut()
                                && *last_role == role
                            {
                                last_text.push('\n');
                                last_text.push_str(&t.0);
                            } else {
                                segments.push(Segment::Text {
                                    role,
                                    text: t.0.clone(),
                                });
                            }
                            block_emitted = true;
                        }
                        ContentPart::Media(r) => {
                            segments.push(Segment::Media {
                                role,
                                media: resolve_media(r, role, media),
                            });
                            block_emitted = true;
                        }
                    }
                }
            }
            BlockContent::ToolCall(call) => {
                segments.push(Segment::ToolCall(PreparedCall {
                    wire_id: block
                        .meta
                        .provider_call_id
                        .clone()
                        .unwrap_or_else(|| call.call_id.0.clone()),
                    name: call.tool_name.clone(),
                    arguments: call.arguments.clone(),
                }));
            }
            BlockContent::ToolResult(result) => {
                let content = match &result.output.content {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                // Tool-result media renders in user position on every
                // protocol (embedded or hoisted), so it resolves as
                // renderable regardless of the result block's role.
                let resolved = result
                    .media
                    .iter()
                    .map(|r| resolve_media(r, Role::User, media))
                    .collect();
                segments.push(Segment::ToolResult {
                    wire_id: provider_ids
                        .get(&(block.id.turn_id.0.clone(), result.call_id.0.clone()))
                        .cloned()
                        .unwrap_or_else(|| result.call_id.0.clone()),
                    status: result.status.clone(),
                    content,
                    media: resolved,
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

/// The per-wire envelope for a tool definition. The field mapping is
/// identical across all three protocols; only nesting and the schema key
/// differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolShape {
    /// Anthropic Messages: flat entry, schema under `input_schema`.
    Anthropic,
    /// Chat Completions: schema under `parameters`, wrapped in a
    /// `function` object.
    OpenAiChat,
    /// Responses API: flat entry, schema under `parameters`.
    OpenAiResponses,
}

/// Render a [`ToolSurface`] into a protocol's `tools` array entries —
/// the single home of the name/description/parameters mapping, so the
/// three renderers cannot drift.
pub(crate) fn tool_definitions(tool_surface: &ToolSurface, shape: ToolShape) -> Vec<Value> {
    tool_surface
        .definitions
        .iter()
        .map(|d| match shape {
            ToolShape::Anthropic => serde_json::json!({
                "name": d.name,
                "description": d.description,
                "input_schema": d.parameters,
            }),
            ToolShape::OpenAiChat => serde_json::json!({
                "type": "function",
                "function": {
                    "name": d.name,
                    "description": d.description,
                    "parameters": d.parameters,
                },
            }),
            ToolShape::OpenAiResponses => serde_json::json!({
                "type": "function",
                "name": d.name,
                "description": d.description,
                "parameters": d.parameters,
            }),
        })
        .collect()
}
