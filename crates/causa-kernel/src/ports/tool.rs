//! Tool behavior — the `Tool` trait and the `ArtifactStore` port, plus the
//! execution vocabulary they share with drivers: definitions and call
//! context. Recorded facts (results, outputs, artifacts) live in
//! `crate::context::tool_data`; batch dispatch lives in `causa-runtime`'s
//! executor. A tool returns its recorded result and nothing else —
//! unknown-outcome continuation and output retention are reference-harness
//! configuration, not tool declarations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::tool_data::{ArtifactKind, ArtifactRef, ToolCallId, ToolResultPayload};
use crate::ports::control::CallControl;

/// The model-facing description of one callable tool; renderers map it onto
/// the protocol's tool entries (name, description, parameter schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// The stable [`ToolCallId`] this call's result must pair against.
    pub call_id: ToolCallId,
    /// Name of the tool being invoked, as the model called it.
    pub tool_name: String,
    /// The model-emitted arguments, as a raw JSON value.
    pub arguments: serde_json::Value,
}

/// Provenance a caller attaches to bytes handed to [`ArtifactStore::persist`],
/// so the store can label what it receives.
pub struct ArtifactHint {
    /// The tool whose output produced the bytes.
    pub tool_name: String,
    /// The tool call the bytes belong to.
    pub call_id: ToolCallId,
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
/// returns only its recorded result — whether an `UnknownOutcome` result
/// may continue the turn and how output is retained are reference-harness
/// configuration (`causa_runtime`), not tool declarations.
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
