//! Tool behavior — the `Tool` trait and the `ArtifactStore` port, plus the
//! execution vocabulary they share with drivers: the bound surface and call
//! context. Recorded facts (results, outputs, artifacts) live in
//! `crate::context::tool_data`; batch dispatch lives in `causa-runtime`'s
//! executor. A tool returns its recorded result and nothing else —
//! handling unknown outcomes and retaining output belong to execution
//! consumers, not tool declarations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::block::{TextPayload, ToolCallPayload};
use crate::context::ids::BlockId;
use crate::context::tool_data::{ArtifactKind, ArtifactRef, ToolResultPayload};
use crate::context::tools::ToolDefinition;
use crate::ports::control::CallControl;

/// The tool definitions bound for one invocation: what the executor can
/// dispatch this round. A driver records a difference from the context's
/// tool state as a [`crate::ToolChange`] block, and renderers read the tools
/// from the frame, never from this value.
///
/// Serializable for logs and tests; the serialized form is not part of the
/// stable set and may change in any release.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolSurface {
    /// Model-facing [`ToolDefinition`]s, in offer order.
    pub definitions: Vec<ToolDefinition>,
}

impl ToolSurface {
    /// A surface offering no tools.
    pub fn empty() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }

    /// Builds a surface from the given definitions.
    pub fn from_definitions(definitions: Vec<ToolDefinition>) -> Self {
        Self { definitions }
    }
}

/// Identity of one tool dispatch, handed to [`Tool::execute`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolCallContext {
    /// The declaration block whose result this dispatch produces.
    pub call_block_id: BlockId,
    /// The effective input after any pre-processing edits.
    pub input: ToolCallPayload,
    /// Notes added before execution. `ToolBatch::resolve_at` prepends these
    /// to the returned result's notes; tool implementations should return
    /// only notes they add and must not copy this field into the result.
    pub result_notes: Vec<TextPayload>,
}
impl ToolCallContext {
    /// A call context with no result notes.
    pub fn new(call_block_id: BlockId, input: ToolCallPayload) -> Self {
        Self {
            call_block_id,
            input,
            result_notes: Vec::new(),
        }
    }
}

impl ToolCallContext {
    /// Builds execution material from a committed declaration.
    pub fn from_declaration(call_block_id: BlockId, declaration: &ToolCallPayload) -> Self {
        Self {
            call_block_id,
            input: declaration.clone(),
            result_notes: Vec::new(),
        }
    }
}

/// Provenance a caller attaches to bytes handed to [`ArtifactStore::persist`],
/// so the store can label what it receives.
#[non_exhaustive]
pub struct ArtifactHint {
    /// The tool whose output produced the bytes.
    pub tool_name: String,
    /// The tool call the bytes belong to.
    pub call_block_id: BlockId,
    /// The [`ArtifactKind`] classification of the bytes.
    pub kind: ArtifactKind,
}
impl ArtifactHint {
    /// A hint for the artifact one call produced.
    pub fn new(tool_name: impl Into<String>, call_block_id: BlockId, kind: ArtifactKind) -> Self {
        Self {
            tool_name: tool_name.into(),
            call_block_id,
            kind,
        }
    }
}

/// Error surface of the [`ArtifactStore`] port.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StoreError {
    /// Persisting the artifact failed; carries the human-readable cause.
    #[error("persist failed: {0}")]
    Persist(String),
    /// Reading an artifact back failed; carries the human-readable cause.
    #[error("read failed: {0}")]
    Read(String),
}

/// Persistence port for tool artifacts — where oversized tool outputs are
/// spilled and read back. Callers persist raw bytes plus an [`ArtifactHint`]
/// and receive an [`ArtifactRef`] to embed in the result; concrete stores
/// are host concerns.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    /// Stores `data`, tagged with `hint`, and returns the [`ArtifactRef`]
    /// describing the persisted artifact.
    async fn persist(&self, data: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError>;
    /// Reads back the bytes stored under `id`; an optional byte range bounds
    /// the read.
    async fn read(
        &self,
        id: &str,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError>;
}

/// The tool port: one callable tool the model can invoke. Implementations
/// live outside the kernel; the driver's executor dispatches them. A tool
/// returns only its recorded result. Handling unknown outcomes and retaining
/// output belong to the execution consumer, not tool declarations.
///
/// # Requirements on `execute`
///
/// An executor may poll every call of one batch on a single task, with
/// nothing spawned. An implementation therefore:
///
/// - must not block the thread. Blocking or CPU-heavy work runs on a
///   blocking thread (for example `tokio::task::spawn_blocking`) and the
///   tool awaits it;
/// - must return promptly once its [`CallControl`] is cancelled or past its
///   deadline.
///
/// For a tool that meets both, a stop takes effect at its next yield point.
/// A tool that holds the thread also holds its sibling calls and the
/// executor's own stop and timeout checks, so nothing is guaranteed for it:
/// not cancellation, not the deadline, not a per-call timeout.
///
/// Calls of one batch are concurrent, not parallel, and their number is not
/// bounded. A tool that needs a limit shares a semaphore across its calls.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The model-facing [`ToolDefinition`] for this tool.
    fn definition(&self) -> ToolDefinition;
    /// Runs one call under the call's [`CallControl`], returning the
    /// recorded result. `store` is the host's [`ArtifactStore`] when one is
    /// configured: a tool that produces media or oversized output persists
    /// the bytes there and returns a reference; a tool that does not need
    /// storage ignores it.
    async fn execute(
        &self,
        ctx: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload;
}
