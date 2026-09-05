//! Tool-domain fact vocabulary — call ids, results, outputs, artifacts.
//!
//! This module holds only recorded facts: what the tool door persists and
//! what the pairing invariant validates. Behavior and execution vocabulary
//! (the `Tool` trait, the `ArtifactStore` port, definitions, limits, outcome
//! policies, dispatch context) live in `crate::ports::tool`; canonical
//! modules must depend on this module, never on the executor-sized behavior
//! module.

use serde::{Deserialize, Serialize};

/// Canonical causal key pairing a tool call with its result. Kernel-generated
/// (see `generate`), unique within a single turn context; never the
/// provider-issued call id, which rides on the envelope metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCallId(pub String);
impl ToolCallId {
    /// Wraps an arbitrary string as a tool call id.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    /// `tool_name + blake3(round_id + tool_name + arguments_json)[..8] + position`。
    /// `round_id` enters the hash preimage so the same `(tool, arguments,
    /// position)` yields a different id in every ModelRound — a model re-sending
    /// the same call in a later round (the legitimate dedup-recovery path) must
    /// not collide with a historical call_id. Uniqueness scope is a single
    /// TurnContext.
    pub fn generate(
        round_id: crate::context::ids::RoundId,
        tool_name: &str,
        arguments: &serde_json::Value,
        position: usize,
    ) -> Self {
        let json = serde_json::to_string(arguments)
            .unwrap_or_else(|_| "<unserializable-arguments>".to_string());
        let preimage = format!("{}|{}|{}", round_id.0, tool_name, json);
        let hash = blake3::hash(preimage.as_bytes());
        let hex = hash.to_hex();
        Self(format!("{}:{}:{}", tool_name, &hex[..8], position))
    }
}

/// Terminal status of one tool invocation, recorded verbatim as a fact;
/// mapping from execution errors is the driver's job, not the kernel's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolResultStatus {
    /// The tool returned an outcome.
    Succeeded,
    /// The tool ran but reported failure (including panic isolation).
    Failed,
    /// The call never ran — unknown tool, denied by a filter, or otherwise
    /// refused before execution.
    Rejected,
    /// The invocation was cancelled before producing an outcome.
    Cancelled,
    /// The invocation exceeded its time budget before producing an outcome.
    TimedOut,
    /// No outcome was observed (e.g. the call-deadline backstop fired);
    /// turn treatment is governed by the tool's `UnknownOutcomePolicy`.
    UnknownOutcome,
}

/// Optional observability sidecar on a tool output; never part of pairing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutputMeta {
    /// Wall-clock duration of the execution, if measured.
    pub duration_ms: Option<u64>,
    /// Estimated token count of the output before truncation, when known.
    pub original_tokens: Option<usize>,
    /// Free-form JSON for tool- or driver-specific annotations.
    pub extra: Option<serde_json::Value>,
}

/// Whether tool output content was shortened to fit the token limit.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// The result fact for one tool call: the paired call id, terminal status,
/// and the output. Committed through the tool door; pairing against the
/// committed call block is kernel-enforced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultPayload {
    /// The kernel-generated id of the call this result answers.
    pub call_id: ToolCallId,
    /// Terminal status of the invocation.
    pub status: ToolResultStatus,
    /// The tool's output, possibly truncated with an artifact spill.
    pub output: ToolOutput,
    /// Media artifacts attached to this result — references only; the
    /// bytes live in the host's asset store, resolved provider-side at
    /// render time. Serde-additive: snapshots from before Slice 6.5
    /// default to empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<crate::context::block::MediaRef>,
}

/// Pointer to a tool output persisted out-of-band (e.g. a spilled oversized
/// observation), kept on the result instead of the full content.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// What an artifact holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactKind {
    /// The complete, untruncated tool output.
    FullOutput,
    /// Cached intermediate output (e.g. streamed pipe data).
    PipeCache,
    /// Raw binary payload.
    Binary,
}
