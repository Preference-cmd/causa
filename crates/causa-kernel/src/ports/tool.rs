//! Tool behavior — the `Tool` trait and the `ArtifactStore` port, plus the
//! execution vocabulary they share with drivers: definitions and call
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
use crate::ports::control::CallControl;

/// The model-facing description of one callable tool; renderers map it onto
/// the protocol's tool entries (name, description, parameter schema).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The name the model uses to invoke the tool.
    pub name: String,
    /// Short summary of what the tool does, shown to the model.
    pub description: String,
    /// Schema of the arguments the tool accepts, embedded verbatim in the
    /// rendered request.
    pub parameters: serde_json::Value,
}

/// Identity of one tool dispatch, handed to [`Tool::execute`].
#[derive(Debug, Clone)]
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
pub struct ArtifactHint {
    /// The tool whose output produced the bytes.
    pub tool_name: String,
    /// The tool call the bytes belong to.
    pub call_block_id: BlockId,
    /// The [`ArtifactKind`] classification of the bytes.
    pub kind: ArtifactKind,
}

/// Error surface of the [`ArtifactStore`] port.
#[derive(Debug, thiserror::Error)]
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
#[async_trait]
pub trait Tool: Send + Sync {
    /// The model-facing [`ToolDefinition`] for this tool.
    fn definition(&self) -> ToolDefinition;
    /// Runs one call under the call's [`CallControl`], returning the
    /// recorded result.
    async fn execute(&self, ctx: &ToolCallContext, control: &CallControl) -> ToolResultPayload;
    /// Extension point for tools that persist artifacts: receives the host's
    /// [`ArtifactStore`] when one is configured (`None` otherwise). The
    /// default ignores the store and delegates to [`Tool::execute`].
    async fn execute_with_store(
        &self,
        ctx: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        let _ = store;
        self.execute(ctx, control).await
    }
}
