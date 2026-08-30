//! Block facts -- the kernel-side vocabulary of a conversation.
//!
//! A ContextBlock is a typed fact with three orthogonal axes:
//! identity (id, sequence), content (BlockContent), and envelope
//! provenance (BlockMeta). Provider-specific role assignment (system /
//! user / assistant / tool) is the renderer's job, not the kernel's.

use serde::{Deserialize, Serialize};

use crate::context::ids::{BlockId, BlockSequence};
use crate::context::tool_data::{ToolCallId, ToolResultPayload};

/// Envelope provenance. Fields are serde-additive so legacy snapshots
/// without them still deserialize.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockMeta {
    /// Provider-issued identifier (e.g. upstream call id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_call_id: Option<String>,

    /// Origin tag (e.g. "user", "provider:gpt-4o", "host"). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// A single string of text. Serializes transparently as the inner string.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TextPayload(pub String);
impl TextPayload {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// A model-issued tool call. call_id is the kernel-generated causal key;
/// any provider-issued identifier rides on BlockMeta::provider_call_id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallPayload {
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// A typed fact. Three axes: identity, content, envelope provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlock {
    pub id: BlockId,
    pub sequence: BlockSequence,
    pub content: BlockContent,
    pub meta: BlockMeta,
}

/// The content shape of a block. Three shapes:
/// text (any role), tool call, tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape", content = "value", rename_all = "snake_case")]
pub enum BlockContent {
    Text(TextPayload),
    ToolCall(ToolCallPayload),
    ToolResult(ToolResultPayload),
}
