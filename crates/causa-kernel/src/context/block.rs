//! Block facts -- the kernel-side vocabulary of a conversation.
//!
//! A ContextBlock is a typed fact with three orthogonal axes:
//! identity (id), content (BlockContent), and envelope
//! provenance (BlockMeta). Who produced a message (system, user or
//! assistant) is a kernel fact recorded on message content as a [`Role`];
//! `BlockMeta::source` is provenance only and does not affect rendering.
//!
//! Content vocabulary is **Message**: one
//! logical message's mixed content commits as one block of ordered
//! [`ContentPart`]s with its [`Role`] — the block is the fact atom (identity and envelope),
//! and the part is the content
//! atom (ordered, identity-free, shares the envelope). Media enters as
//! a [`MediaRef`] — a cheap durable reference; bytes never enter facts,
//! they appear only in resolved render payloads (provider side).
//! Model-issued wire items the framework cannot read travel as
//! [`SealedPart`]s, and the wire fields of a readable item ride in
//! [`BlockMeta::wire`].

use serde::{Deserialize, Serialize};

use crate::context::ids::BlockId;
use crate::context::tool_data::ToolResultPayload;
use crate::context::tools::ToolChange;

/// The speaker of a conversation message.
///
/// Exhaustive on purpose: a renderer must be total over the role vocabulary, so
/// a new variant is a breaking change that should stop downstream code from
/// compiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Instructions or background context provided to the model.
    System,
    /// Input from the human user or external caller.
    User,
    /// Output produced by the model.
    Assistant,
}

/// Envelope provenance. Fields are serde-additive so older records
/// without them still deserialize.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BlockMeta {
    /// Provider-issued identifier (e.g. upstream call id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_call_id: Option<String>,

    /// Origin tag (e.g. "user", "provider:gpt-4o", "host"). Optional provenance
    /// only: no renderer, validator, or runtime reads it and it does not affect
    /// rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// What the provider's item carried beyond the facts in this block.
    /// Written by the model door, read only by the protocol renderer whose
    /// format matches; the kernel never interprets it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wire: Option<WireMeta>,
}
impl BlockMeta {
    /// Metadata naming only where the block came from, such as `"user"`.
    /// Provenance only: no renderer, validator, or runtime reads it and it does
    /// not affect rendering.
    pub fn from_source(source: impl Into<String>) -> Self {
        Self {
            source: Some(source.into()),
            ..Self::default()
        }
    }

    /// Sets the provider-issued call identifier.
    pub fn with_provider_call_id(mut self, provider_call_id: impl Into<String>) -> Self {
        self.provider_call_id = Some(provider_call_id.into());
        self
    }

    /// Sets the origin tag. Provenance only: no renderer, validator, or runtime
    /// reads it and it does not affect rendering.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Sets the wire-level residue of the item.
    pub fn with_wire(mut self, wire: WireMeta) -> Self {
        self.wire = Some(wire);
        self
    }
}

/// Wire-level residue of one provider item: every field the translator did
/// not turn into block content, kept uninterpreted so a renderer of the same
/// format can send the item back as returned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WireMeta {
    /// Wire format the fields belong to (the protocol's stable name, for
    /// example `openai_responses`).
    pub format: String,
    /// The item as returned, minus the fields the block's content holds.
    pub fields: serde_json::Value,
}
impl WireMeta {
    /// Wire residue in the named format.
    pub fn new(format: impl Into<String>, fields: serde_json::Value) -> Self {
        Self {
            format: format.into(),
            fields,
        }
    }
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
#[non_exhaustive]
pub struct ToolCallPayload {
    /// Name of the invoked tool — the key the executor dispatches on.
    pub tool_name: String,
    /// The call's arguments as a JSON value.
    pub arguments: serde_json::Value,
}
impl ToolCallPayload {
    /// A declaration of one call.
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
        }
    }
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

    /// Constructs a text message block with the given identity, role, and text content,
    /// using default metadata.
    pub fn text(id: BlockId, role: Role, text: impl Into<String>) -> Self {
        Self::new(id, BlockContent::text(role, text), BlockMeta::default())
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
#[non_exhaustive]
pub struct MediaRef {
    /// IANA media type; only `image/*` renders to the wire, anything else
    /// degrades to a deterministic text placeholder.
    pub media_type: String,
    /// Host-side asset reference (workspace asset id / content-addressed
    /// hash). The kernel never interprets it; the host's resolver does.
    pub reference: String,
    /// The model-visible file name; images do not use it; when `None` the
    /// serialized form is the same as a [`MediaRef`] without the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl MediaRef {
    /// Wraps a media type and a host-side asset reference.
    pub fn new(media_type: impl Into<String>, reference: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            reference: reference.into(),
            name: None,
        }
    }

    /// Sets the model-visible file name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// The content shape of a block. Four shapes: one message's ordered
/// parts (any role), a tool call, a tool result, a change to the offered
/// tools.
///
/// Exhaustive on purpose: a renderer must handle every shape, so a new one is
/// a breaking change that should stop downstream code from compiling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", content = "value", rename_all = "snake_case")]
pub enum BlockContent {
    /// One logical message's ordered content parts (text, media
    /// references, or sealed wire items) and the role that produced them —
    /// atomic at the block: one identity, one content identity; execution
    /// pairing is enforced by specialized append operations rather than by
    /// importing arbitrary facts.
    Message {
        /// Speaker role for this message.
        role: Role,
        /// Ordered content parts.
        parts: Vec<ContentPart>,
    },
    /// A model-issued tool invocation.
    ToolCall(ToolCallPayload),
    /// The recorded outcome of a prior call, paired by
    /// `ToolResultPayload::call_block_id`.
    ToolResult(ToolResultPayload),
    /// The tools offered to the model change from this position on. The
    /// first one in a context declares the initial set.
    ToolChange(ToolChange),
}

impl BlockContent {
    /// Constructs a [`BlockContent::Message`] containing a single [`ContentPart::Text`].
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self::Message {
            role,
            parts: vec![ContentPart::Text(TextPayload::new(text))],
        }
    }
}

/// One content atom inside a [`BlockContent::Message`] block: ordered,
/// identity-free, sharing the block's envelope. New modalities arrive
/// as new variants (additive); media bytes stay out of facts forever.
///
/// Exhaustive on purpose, like [`BlockContent`]. A new part is additive only
/// in the serde sense: stored material written before it still reads. In Rust
/// it is a breaking change, because a wildcard arm in a renderer would drop
/// the new part without a trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "part", content = "value", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text; provider role assignment stays the renderer's job.
    Text(TextPayload),
    /// A media reference; bytes never enter facts.
    Media(MediaRef),
    /// A model-issued wire item the kernel carries and never interprets.
    Sealed(SealedPart),
}

/// A model-issued wire item carried as content the kernel cannot read.
///
/// Only a renderer of the matching `format` writes it back; every other
/// renderer skips it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SealedPart {
    /// Wire format of `item`; interpreted only by the matching renderer.
    pub format: String,
    /// The item exactly as the provider returned it.
    pub item: serde_json::Value,
}
impl SealedPart {
    /// A sealed item in the named wire format.
    pub fn new(format: impl Into<String>, item: serde_json::Value) -> Self {
        Self {
            format: format.into(),
            item,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn named_media_ref_round_trips_through_serde() {
        let media = MediaRef::new("application/pdf", "asset://doc").with_name("memo.pdf");
        let serialized = serde_json::to_value(&media).expect("serialization succeeds");
        assert_eq!(
            serialized,
            json!({
                "media_type": "application/pdf",
                "reference": "asset://doc",
                "name": "memo.pdf",
            })
        );
        let deserialized: MediaRef =
            serde_json::from_value(serialized).expect("deserialization succeeds");
        assert_eq!(deserialized, media);
        assert_eq!(deserialized.name.as_deref(), Some("memo.pdf"));
    }

    #[test]
    fn unnamed_media_ref_serializes_with_no_name_key() {
        let media = MediaRef::new("image/png", "asset://img");
        let serialized = serde_json::to_value(&media).expect("serialization succeeds");
        assert_eq!(
            serialized,
            json!({
                "media_type": "image/png",
                "reference": "asset://img",
            })
        );
        assert!(!serialized.as_object().unwrap().contains_key("name"));
        let deserialized: MediaRef =
            serde_json::from_value(serialized).expect("deserialization succeeds");
        assert_eq!(deserialized, media);
        assert_eq!(deserialized.name, None);
    }
}
