//! Block facts -- the kernel-side vocabulary of a conversation.
//!
//! A ContextBlock is a typed fact with three orthogonal axes:
//! identity (id), content (BlockContent), and envelope
//! provenance (BlockMeta). Provider-specific role assignment (system /
//! user / assistant / tool) is the renderer's job, not the kernel's.
//!
//! Content vocabulary is **Parts**: one
//! logical message's mixed content commits as one block of ordered
//! [`ContentPart`]s — the block is the fact atom (identity, one version
//! bump, pairing invariant), the part is the content
//! atom (ordered, identity-free, shares the envelope). Media enters as
//! a [`MediaRef`] — a cheap durable reference; bytes never enter facts,
//! they appear only in resolved render payloads (provider side).

use serde::{Deserialize, Serialize};

use crate::context::ids::BlockId;
use crate::context::tool_data::ToolResultPayload;

/// Envelope provenance. Fields are serde-additive so legacy snapshots
/// without them still deserialize.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockMeta {
    /// Provider-issued identifier (e.g. upstream call id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_call_id: Option<String>,

    /// Origin tag (e.g. "user", "provider:gpt-4o", "host"). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// A single string of text. Serializes transparently as the inner string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TextPayload(pub String);
impl TextPayload {
    /// Wrap any string-like value as a text payload.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// A model-issued tool call. Its containing block ID is the declaration
/// identity; provider-issued identifiers ride on BlockMeta::provider_call_id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallPayload {
    /// Name of the invoked tool — the key the executor dispatches on.
    pub tool_name: String,
    /// The call's arguments as a JSON value.
    pub arguments: serde_json::Value,
}

/// A typed fact. Three axes: identity, content, envelope provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextBlock {
    /// Identity axis: the UUID assigned to this fact block.
    id: BlockId,
    /// Content axis: the typed fact payload.
    content: BlockContent,
    /// Envelope provenance axis; serde-additive, defaulting when absent.
    meta: BlockMeta,
}

impl ContextBlock {
    /// Creates a context fact with an explicit identity, content, and metadata.
    ///
    /// Callers that change a block's content or metadata must supply a new
    /// identity. Construction does not validate identity reuse against any
    /// other block or context.
    pub fn new(id: BlockId, content: BlockContent, meta: BlockMeta) -> Self {
        Self { id, content, meta }
    }

    /// Returns this block's stable identity.
    pub const fn id(&self) -> BlockId {
        self.id
    }

    /// Returns this block's content by shared reference.
    pub const fn content(&self) -> &BlockContent {
        &self.content
    }

    /// Returns this block's metadata by shared reference.
    pub const fn meta(&self) -> &BlockMeta {
        &self.meta
    }
}

/// A media fact: a cheap, durable reference. Bytes never enter facts —
/// they appear only in resolved render payloads (provider side).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaRef {
    /// IANA media type; only `image/*` renders to the wire, anything else
    /// degrades to a deterministic text placeholder.
    pub media_type: String,
    /// Host-side asset reference (workspace asset id / content-addressed
    /// hash). The kernel never interprets it; the host's resolver does.
    pub reference: String,
}

impl MediaRef {
    /// Wraps a media type and a host-side asset reference.
    pub fn new(media_type: impl Into<String>, reference: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            reference: reference.into(),
        }
    }
}

/// The content shape of a block. Three shapes: one message's ordered
/// parts (any role), a tool call, a tool result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", content = "value", rename_all = "snake_case")]
pub enum BlockContent {
    /// One logical message's ordered content parts (text and media
    /// references) — atomic at the block: identity, one
    /// version bump, and the pairing invariant live only here.
    Parts(Vec<ContentPart>),
    /// A model-issued tool invocation.
    ToolCall(ToolCallPayload),
    /// The recorded outcome of a prior call, paired by
    /// `ToolResultPayload::call_block_id`.
    ToolResult(ToolResultPayload),
}

/// One content atom inside a [`BlockContent::Parts`] block: ordered,
/// identity-free, sharing the block's envelope. New modalities arrive
/// as new variants (additive); media bytes stay out of facts forever.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "part", content = "value", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text; provider role assignment stays the renderer's job.
    Text(TextPayload),
    /// A media reference; bytes never enter facts.
    Media(MediaRef),
}
