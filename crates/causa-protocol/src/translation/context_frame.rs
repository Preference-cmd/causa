//! Shared policy walk over kernel `ContextFrame`s (crate-internal).
//!
//! The three renderers (`anthropic`, `openai_chat`, `openai_responses`)
//! share one normalization pass; this module owns it so the policy cannot
//! drift between protocols. The walk applies every renderer-independent
//! decision exactly once:
//!
//! - Wire-role assignment directly from `BlockContent::Message::role`.
//! - Empty text is not conversation content: an empty `Text` part is
//!   dropped, and a Message block left with nothing renders nothing
//!   (mirroring the kernel model door's commit policy, which drops only
//!   zero-length text; the host door can commit them). Whitespace-only text
//!   is content and is kept verbatim.
//! - No text is joined across block seams, for any role: one part renders
//!   as one content block, and no renderer inserts a character the model
//!   did not produce. Where a wire format carries one string per message
//!   (Chat assistant content), the emitter concatenates the group's texts
//!   with no separator.
//! - Wire fidelity. Each segment carries the index of the frame block it
//!   came from, and that block's `wire.fields` only when `wire.format`
//!   equals the rendering protocol's name; a block of another format
//!   renders from facts exactly like a block without `wire`. A
//!   [`ContentPart::Sealed`] part of the rendering protocol's format
//!   becomes a [`Segment::Sealed`] the emitter writes verbatim; one of
//!   another format is skipped silently. A sealed part outside assistant
//!   position is a malformed frame, rejected by frame validation before
//!   the walk runs.
//! - Media: each [`MediaRef`] resolves through the caller's
//!   [`MediaSet`]. A resolved `image/*` payload renders inline, but only
//!   in user position (no protocol accepts assistant- or system-authored
//!   input images); anything else — missing resolution, non-image type,
//!   empty payload, non-user role — degrades to the deterministic text
//!   placeholder `[media: {type} {reference}]`. The decision is made
//!   once, here, never per-renderer.
//! - Tool call ids come from `meta.provider_call_id`, falling back to
//!   the declaration block UUID. Tool results resolve directly through
//!   their declaration `BlockId`. Renderers first reject incomplete or
//!   ambiguous exchanges; normalization never invents missing declarations.
//! - Tool result media travels on the result segment in result order;
//!   whether it embeds (Anthropic and Responses) or hoists (Chat) is the
//!   emitter's call. Non-string tool observations serialize to a string.
//! - Text and media segments carry the index of their source block, so an
//!   emitter can group one user `Message` block into one wire message
//!   ([`user_block_run`]). Grouping across blocks is the emitter's call:
//!   Anthropic merges same-role neighbours because its roles must
//!   alternate; the OpenAI-family emitters keep each block its own message.
//!
//! - Tool changes: the first [`ToolChange`] block of a frame declares the
//!   request-level tool list and never becomes a segment; a later one
//!   becomes a [`Segment::ToolChange`] only when the renderer places changes
//!   in place (see [`plan_tools`]), and renders nothing otherwise.
//!
//! Emitters then map the ordered [`Segment`] list to per-protocol wire
//! shapes (message grouping, content block shapes, argument encoding,
//! error flags). Rendering stays deterministic: the walk is single-pass
//! over an ordered list and no hash-map iteration reaches the output.

use std::collections::HashMap;

use serde_json::Value;

use causa_kernel::{
    BlockContent, ContentPart, ContextFrame, MediaRef, ModelInvokeError, ModelInvokeErrorKind,
    ModelItem, ModelItemContent, SealedPart, ToolChange, ToolDefinition, ToolResultStatus,
    WireMeta, current_tools,
};

pub(crate) use causa_kernel::Role;

use super::media::MediaSet;
use crate::Protocol;

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
    /// The block's `wire.fields`, present only when the block's wire format
    /// is the rendering protocol's.
    pub wire: Option<Value>,
}

/// Where a text segment came from, so an emitter can put a block's wire
/// fields back around its facts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Origin {
    /// Index of the source block in the frame.
    pub block: usize,
    /// Number of parts the source block holds, including parts the walk
    /// dropped or turned into other segments.
    pub parts: usize,
    /// The block's `wire.fields`, present only when the block's wire format
    /// is the rendering protocol's.
    pub wire: Option<Value>,
}

/// The number of leading segments that render into the top-level system
/// parameter.
///
/// A system block that precedes all conversation content renders there — that
/// is its position, and the API rejects a `system` message as the first entry
/// of `messages`. A system block appended after conversation content is an
/// append like any other and renders at its position as an inline system
/// message; a renderer never hoists it. Only text and degraded system media
/// can serve as leading system content, and any non-system segment ends the
/// run.
pub(crate) fn leading_system_len(segments: &[Segment]) -> usize {
    segments
        .iter()
        .take_while(|segment| {
            matches!(
                segment,
                Segment::Text {
                    role: Role::System,
                    ..
                } | Segment::Media {
                    role: Role::System,
                    media: ResolvedMedia::Placeholder(_),
                    ..
                }
            )
        })
        .count()
}

/// Whether a text segment is blank for rendering purposes: it carries no
/// visible character, so a renderer may omit it from the request. The fact
/// stays in the `Context`; only the rendered request differs.
pub(crate) fn is_blank_text(text: &str) -> bool {
    text.trim().is_empty()
}

/// Length of the run of assistant text segments from `from` that came from
/// the same block as `segments[from]`.
pub(crate) fn same_block_run(segments: &[Segment], from: usize) -> usize {
    let Some(Segment::Text { origin, .. }) = segments.get(from) else {
        return 0;
    };
    segments[from..]
        .iter()
        .take_while(|segment| {
            matches!(
                segment,
                Segment::Text { role: Role::Assistant, origin: o, .. } if o.block == origin.block
            )
        })
        .count()
}

/// The frame block a user-position text or media segment came from; `None`
/// for any other segment.
fn user_part_block(segment: &Segment) -> Option<usize> {
    match segment {
        Segment::Text {
            role: Role::User,
            origin,
            ..
        } => Some(origin.block),
        Segment::Media {
            role: Role::User,
            block,
            ..
        } => Some(*block),
        _ => None,
    }
}

/// Length of the run of user text and media segments from `from` that came
/// from the same block as `segments[from]`: the parts one user message
/// carries on a wire that groups by block. `0` when `segments[from]` is not
/// a user part.
pub(crate) fn user_block_run(segments: &[Segment], from: usize) -> usize {
    let Some(block) = segments.get(from).and_then(user_part_block) else {
        return 0;
    };
    segments[from..]
        .iter()
        .take_while(|segment| user_part_block(segment) == Some(block))
        .count()
}

/// A copy of a JSON object without the named keys (any other value is
/// copied unchanged). Parsers use it to split a wire item into the facts they
/// read and the residue they keep.
pub(crate) fn without_keys(value: &Value, keys: &[&str]) -> Value {
    let mut copy = value.clone();
    if let Value::Object(map) = &mut copy {
        for key in keys {
            map.remove(*key);
        }
    }
    copy
}

/// The residue of a wire item as block wire metadata for `protocol`.
pub(crate) fn wire_meta(protocol: Protocol, fields: Value) -> WireMeta {
    WireMeta::new(protocol.as_str().to_owned(), fields)
}

/// One `ModelItem` holding a single sealed part of `protocol`'s format. The
/// value is carried uninterpreted; the matching emitter writes it back
/// verbatim, and every other emitter skips it.
pub(crate) fn sealed_item(protocol: Protocol, item: Value) -> ModelItem {
    ModelItem::new(ModelItemContent::Parts(vec![ContentPart::Sealed(
        SealedPart::new(protocol.as_str().to_owned(), item),
    )]))
}

/// The object a wire item is rebuilt on: the stored residue when it is an
/// object, otherwise an empty one.
pub(crate) fn residue_object(fields: Option<&Value>) -> serde_json::Map<String, Value> {
    match fields {
        Some(Value::Object(map)) => map.clone(),
        _ => serde_json::Map::new(),
    }
}

/// The `arguments` string to send for a tool call. The provider's original
/// string is sent unchanged when it still parses to the fact's value;
/// otherwise (edited arguments, or no original) the facts are encoded.
pub(crate) fn wire_arguments_string(raw: Option<&Value>, facts: &Value) -> String {
    if let Some(raw) = raw.and_then(Value::as_str)
        && serde_json::from_str::<Value>(raw).is_ok_and(|parsed| parsed == *facts)
    {
        return raw.to_owned();
    }
    facts.to_string()
}

/// One normalized, frame-order-preserving piece of the conversation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Segment {
    /// One non-empty text part. Never joined with its neighbours.
    Text {
        role: Role,
        text: String,
        origin: Origin,
    },
    /// A sealed part of the rendering protocol's own format, in assistant
    /// position; the emitter writes it verbatim.
    Sealed(Value),
    /// A media part from a Message block, resolved or degraded.
    Media {
        role: Role,
        media: ResolvedMedia,
        /// Index of the source block in the frame.
        block: usize,
    },
    ToolCall(PreparedCall),
    ToolResult {
        wire_id: String,
        status: ToolResultStatus,
        content: String,
        /// Ordered model-visible notes stored with this result.
        notes: Vec<String>,
        /// The result's media attachments, resolved or degraded, in
        /// payload order.
        media: Vec<ResolvedMedia>,
    },
    /// A tool change after the frame's declaration, emitted only when the
    /// renderer places changes in place.
    ToolChange(ToolChange),
}

pub(crate) struct NormalizedFrame {
    pub segments: Vec<Segment>,
}

/// Add stored notes to a tool result's model-visible text while preserving
/// the output payload as an independent fact.
pub(crate) fn result_text_with_notes(content: &str, notes: &[String]) -> String {
    if notes.is_empty() {
        return content.to_owned();
    }
    let mut text = format!("{content}\n\nNotes:");
    for note in notes {
        text.push_str("\n- ");
        text.push_str(note);
    }
    text
}

/// Run the shared policy walk over a frame for `protocol`, resolving media
/// through `media`. `in_place_changes` is [`ToolPlan::in_place`]: whether
/// tool changes after the declaration become segments.
pub(crate) fn normalize(
    frame: &ContextFrame,
    media: &MediaSet,
    protocol: Protocol,
    in_place_changes: bool,
) -> NormalizedFrame {
    let blocks = &frame.blocks;

    // Block identity is stable across turns and directly joins declarations
    // with results in a merged conversation frame.
    let mut provider_ids: HashMap<causa_kernel::BlockId, String> = HashMap::new();
    for block in blocks {
        if let BlockContent::ToolCall(_) = block.content() {
            provider_ids.insert(
                block.id(),
                block
                    .meta()
                    .provider_call_id
                    .clone()
                    .unwrap_or_else(|| block.id().0.to_string()),
            );
        }
    }

    let mut segments: Vec<Segment> = Vec::new();
    let mut declared = false;
    for (block_index, block) in blocks.iter().enumerate() {
        // The block's wire fields are visible to this renderer only when
        // they were recorded for its own format.
        let own_wire = || {
            block
                .meta()
                .wire
                .as_ref()
                .filter(|wire| wire.format == protocol.as_str())
                .map(|wire| wire.fields.clone())
        };
        match block.content() {
            BlockContent::Message { role, parts } => {
                let role = *role;
                for part in parts {
                    match part {
                        ContentPart::Text(t) => {
                            // Empty text is not conversation content.
                            if t.0.is_empty() {
                                continue;
                            }
                            segments.push(Segment::Text {
                                role,
                                text: t.0.clone(),
                                origin: Origin {
                                    block: block_index,
                                    parts: parts.len(),
                                    wire: if role == Role::Assistant {
                                        own_wire()
                                    } else {
                                        None
                                    },
                                },
                            });
                        }
                        ContentPart::Media(r) => {
                            segments.push(Segment::Media {
                                role,
                                media: resolve_media(r, role, media),
                                block: block_index,
                            });
                        }
                        ContentPart::Sealed(sealed) => {
                            // Frame validation rejects sealed parts outside
                            // assistant position; another format's item is
                            // not ours to write.
                            if role == Role::Assistant && sealed.format == protocol.as_str() {
                                segments.push(Segment::Sealed(sealed.item.clone()));
                            }
                        }
                    }
                }
            }
            BlockContent::ToolCall(call) => {
                segments.push(Segment::ToolCall(PreparedCall {
                    wire_id: block
                        .meta()
                        .provider_call_id
                        .clone()
                        .unwrap_or_else(|| block.id().0.to_string()),
                    name: call.tool_name.clone(),
                    arguments: call.arguments.clone(),
                    wire: own_wire(),
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
                        .get(&result.call_block_id)
                        .cloned()
                        .expect("renderer validated tool declaration"),
                    status: result.status.clone(),
                    content,
                    notes: result.notes.iter().map(|note| note.0.clone()).collect(),
                    media: resolved,
                });
            }
            BlockContent::ToolChange(change) => {
                // The first change is the declaration: it renders as the
                // request-level tool list, never at its position.
                if declared && in_place_changes {
                    segments.push(Segment::ToolChange(change.clone()));
                }
                declared = true;
            }
        }
    }

    NormalizedFrame { segments }
}

/// OpenAI-family function-arguments wire codec. A completed call must carry
/// a JSON string containing a complete value; zero-parameter calls carry
/// `"{}"`. Missing, empty and inline values are unusable provider responses.
///
/// `cut_off` marks the one call that may be incomplete: the last one in a
/// response the token limit stopped. Its arguments end wherever the limit
/// fell, so when they do not parse the result is `Ok(None)` and the caller
/// leaves the call out; nothing could execute it, and the stop reason says
/// what happened. A call anywhere else must be complete.
pub(crate) fn decode_wire_arguments(
    raw: Option<&Value>,
    cut_off: bool,
) -> Result<Option<Value>, ModelInvokeError> {
    let parsed = raw
        .and_then(Value::as_str)
        .ok_or("function arguments: expected a JSON string")
        .and_then(|raw| {
            serde_json::from_str(raw).map_err(|_| "function arguments: invalid or incomplete JSON")
        });
    match parsed {
        Ok(arguments) => Ok(Some(arguments)),
        Err(_) if cut_off => Ok(None),
        Err(message) => Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Undecodable,
            message,
        )),
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

/// Render tool definitions into a protocol's tool entries — the single home
/// of the name/description/parameters mapping, so the three renderers cannot
/// drift.
pub(crate) fn tool_definitions(definitions: &[ToolDefinition], shape: ToolShape) -> Vec<Value> {
    definitions
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

/// How a renderer may place a frame's tool changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolPlacement {
    /// Every change folds into the request-level tool list.
    Fold,
    /// Anthropic in-place changes: needs a non-empty declaration.
    AnthropicInPlace,
    /// Responses `additional_tools`: only additions of new names.
    ResponsesAdditive,
}

/// The request-level tool list of one render, and whether later tool
/// changes render at their position.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolPlan {
    /// Definitions rendered as the request-level tool list.
    pub tools: Vec<ToolDefinition>,
    /// Whether changes after the declaration render in place; when `false`,
    /// `tools` already holds the frame's current tool state.
    pub in_place: bool,
}

/// Decide the request-level tool list for `placement`. In-place placement
/// keeps the list at the declaration, the frame's first [`ToolChange`], so it
/// stays value-equal while the set changes; it falls back to the current
/// tool state when the protocol cannot express the frame's changes in place.
pub(crate) fn plan_tools(frame: &ContextFrame, placement: ToolPlacement) -> ToolPlan {
    let fold = || ToolPlan {
        tools: current_tools(&frame.blocks),
        in_place: false,
    };
    let mut changes = frame
        .blocks
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolChange(change) => Some(change),
            _ => None,
        });
    let Some(first) = changes.next() else {
        return fold();
    };
    let mut declaration = Vec::new();
    first.apply_to(&mut declaration);
    let expressible = match placement {
        ToolPlacement::Fold => false,
        // Anthropic rejects a tool list in which every tool is deferred, and
        // in-place mode adds a deferred placeholder.
        ToolPlacement::AnthropicInPlace => !declaration.is_empty(),
        // `additional_tools` only adds: a removal or a second definition of
        // a name needs the request-level list rewritten.
        ToolPlacement::ResponsesAdditive => {
            let mut names: Vec<&str> = declaration.iter().map(|d| d.name.as_str()).collect();
            changes.all(|change| {
                change.removed.is_empty()
                    && change.added.iter().all(|definition| {
                        let new = !names.contains(&definition.name.as_str());
                        names.push(&definition.name);
                        new
                    })
            })
        }
    };
    if expressible {
        ToolPlan {
            tools: declaration,
            in_place: true,
        }
    } else {
        fold()
    }
}
