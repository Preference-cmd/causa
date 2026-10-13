//! Tool-domain fact vocabulary — call ids, results, outputs, artifacts.
//!
//! This module holds only recorded facts: what the tool door persists.
//! Specialized append operations validate call/result pairing. Behavior and execution vocabulary
//! (the `Tool` trait, the `ArtifactStore` port, definitions and dispatch
//! context) live in `crate::ports::tool`. Execution policies belong to the
//! runtime; this module does not depend on them.

use crate::context::ids::BlockId;
use serde::{Deserialize, Serialize};

/// Borrowed content key for comparing tool calls within a caller-selected
/// scope. Instance pairing always uses the declaration's [`BlockId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToolCallId<'a> {
    /// Tool name included in the complete content key.
    pub tool_name: &'a str,
    /// Arguments included with their `serde_json::Value` equality semantics.
    pub arguments: &'a serde_json::Value,
}
impl<'a> ToolCallId<'a> {
    /// Borrows the content key from a tool declaration.
    pub const fn new(tool_name: &'a str, arguments: &'a serde_json::Value) -> Self {
        Self {
            tool_name,
            arguments,
        }
    }
}

/// Terminal status of one tool invocation, recorded verbatim as a fact;
/// mapping from execution errors is the driver's job, not the kernel's.
///
/// Exhaustive on purpose: renderers and the runtime decide by status, so a new one is a breaking change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    /// The tool returned an outcome.
    Succeeded,
    /// The tool ran but reported failure (including panic isolation).
    Failed,
    /// The call never ran — unknown tool, denied by a processor, or otherwise
    /// refused before execution.
    Rejected,
    /// The invocation was cancelled before producing an outcome.
    Cancelled,
    /// The invocation exceeded its time budget before producing an outcome.
    TimedOut,
    /// No outcome was observed (e.g. the call-deadline backstop fired);
    /// whether the turn continues is the harness's unknown-outcome
    /// configuration, not tool vocabulary.
    UnknownOutcome,
}

/// Optional observability sidecar on a tool output; never part of pairing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolOutputMeta {
    /// Wall-clock duration of the execution, if measured.
    pub duration_ms: Option<u64>,
    /// Estimated token count of the output body before truncation, when known.
    /// This does not include result notes or media.
    pub original_tokens: Option<usize>,
    /// Free-form JSON for tool- or driver-specific annotations.
    pub extra: Option<serde_json::Value>,
}

/// Whether tool output content was shortened to fit the token limit.
///
/// Exhaustive on purpose: content was shortened or it was not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    /// The content is the tool's full output.
    None,
    /// The content was replaced by head + notice + tail; the full output
    /// spilled to an artifact when a store was available.
    Middle,
}

/// What a tool returned, as recorded by the tool door.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolOutput {
    /// The observation payload; after middle truncation this is a JSON
    /// string (head + notice + tail) regardless of the original shape.
    pub content: serde_json::Value,
    /// Whether and how the content was shortened.
    pub truncation: Truncation,
    /// Optional observability metadata.
    pub meta: Option<ToolOutputMeta>,
    /// Reference to the full output in an artifact store, if spilled.
    pub artifact: Option<ArtifactRef>,
}
impl ToolOutput {
    /// Creates an output with no truncation, metadata, or artifact.
    pub fn new(content: serde_json::Value) -> Self {
        Self {
            content,
            truncation: Truncation::None,
            meta: None,
            artifact: None,
        }
    }
    /// Returns whether the content was shortened (any marker other than
    /// [`Truncation::None`]).
    pub fn is_truncated(&self) -> bool {
        !matches!(self.truncation, Truncation::None)
    }
}

/// The result fact for one tool call: its declaration identity, terminal
/// status, output, media, and ordered notes. Pairing is kernel-enforced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolResultPayload {
    /// The declaration block this result answers.
    pub call_block_id: BlockId,
    /// Terminal status of the invocation.
    pub status: ToolResultStatus,
    /// The tool's output, possibly truncated with an artifact spill.
    pub output: ToolOutput,
    /// Media artifacts attached to this result — references only; the
    /// bytes live in the host's asset store, resolved provider-side at
    /// render time. Serde-additive: absent media defaults to empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<crate::context::block::MediaRef>,
    /// Ordered result notes. `ToolBatch::resolve_at` places any pre-execution
    /// context notes before notes returned by the tool; tool implementations
    /// should return only notes they add.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<crate::context::block::TextPayload>,
}
impl ToolResultPayload {
    /// A result for one declaration, with no media and no notes.
    pub fn new(call_block_id: BlockId, status: ToolResultStatus, output: ToolOutput) -> Self {
        Self {
            call_block_id,
            status,
            output,
            media: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Replaces the media artifacts attached to this result.
    pub fn with_media(mut self, media: Vec<crate::context::block::MediaRef>) -> Self {
        self.media = media;
        self
    }

    /// Replaces the ordered result notes.
    pub fn with_notes(mut self, notes: Vec<crate::context::block::TextPayload>) -> Self {
        self.notes = notes;
        self
    }
}

/// Pointer to a tool output persisted out-of-band (e.g. a spilled oversized
/// observation), kept on the result instead of the full content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ArtifactRef {
    /// Store-assigned handle for retrieving the artifact.
    pub id: String,
    /// Serialized size of the artifact in bytes.
    pub size_bytes: usize,
    /// What the artifact holds.
    pub kind: ArtifactKind,
    /// Whether the artifact is durably persisted in a store.
    pub persisted: bool,
}
impl ArtifactRef {
    /// A reference to one stored artifact.
    pub fn new(
        id: impl Into<String>,
        size_bytes: usize,
        kind: ArtifactKind,
        persisted: bool,
    ) -> Self {
        Self {
            id: id.into(),
            size_bytes,
            kind,
            persisted,
        }
    }
}

/// What an artifact holds. A label for the host; nothing in the library
/// decides by it, so further kinds may be added.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArtifactKind {
    /// The complete, untruncated tool output.
    FullOutput,
    /// Cached intermediate output (e.g. streamed pipe data).
    PipeCache,
    /// Raw binary payload.
    Binary,
}
