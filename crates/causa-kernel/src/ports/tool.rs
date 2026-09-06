//! Tool behavior — the `Tool` trait and the `ArtifactStore` port, plus the
//! execution vocabulary that drivers and they consume: definitions, call
//! context, outcome policies, output limits. Recorded facts (results,
//! outputs, artifacts) live in `crate::context::tool_data`; batch dispatch
//! lives in `causa-runtime`'s executor.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::tool_data::{ArtifactKind, ArtifactRef, ToolCallId};
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

/// How a tool wants its `UnknownOutcome` result treated — a declaration the
/// driver obeys, not a fact the kernel interprets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownOutcomePolicy {
    /// Treat the unknown outcome as unsafe: the turn interrupts rather than
    /// continue on an unverifiable result (the default).
    Stop,
    /// Keep the turn alive; the `UnknownOutcome` result still lands in the
    /// transcript.
    Continue,
}

/// A tool door's result: the recorded [`ToolResultPayload`] fact plus the
/// [`UnknownOutcomePolicy`] the tool declares for it. Serde-additive: the
/// outcome rides the wire inside the runtime's continuation checkpoint
/// (Slice 6.5), so the derives are part of the contract now.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionOutcome {
    /// The recorded result (pairing id, status, output).
    pub result: crate::context::tool_data::ToolResultPayload,
    /// How the driver should treat this result if its status is
    /// `UnknownOutcome`.
    pub policy: UnknownOutcomePolicy,
}
impl ToolExecutionOutcome {
    /// Creates an outcome under the default [`UnknownOutcomePolicy::Stop`].
    pub fn new(result: crate::context::tool_data::ToolResultPayload) -> Self {
        Self {
            result,
            policy: UnknownOutcomePolicy::Stop,
        }
    }
    /// Builder: overrides the outcome's [`UnknownOutcomePolicy`].
    pub fn with_policy(mut self, policy: UnknownOutcomePolicy) -> Self {
        self.policy = policy;
        self
    }
}

/// Truncation thresholds applied when a tool result exceeds its token
/// estimate — the explicit truncation effect callers opt into.
#[derive(Debug, Clone)]
pub struct ToolOutputLimits {
    /// Maximum estimated tokens a tool result may carry before it gets
    /// truncated; [`usize::MAX`] (the default) disables truncation.
    pub max_tokens: usize,
}
impl Default for ToolOutputLimits {
    /// No limit by default — callers opt in to truncation.
    ///
    /// Specific limits are set by the caller (e.g. per-tool
    /// `Tool::output_limits()` or host configuration). The kernel
    /// defaults to `usize::MAX` so truncation is an *explicit*
    /// effect, never the absence of configuration.
    fn default() -> Self {
        Self {
            max_tokens: usize::MAX,
        }
    }
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
/// live outside the kernel; the driver's executor dispatches them.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The model-facing [`ToolDefinition`] for this tool.
    fn definition(&self) -> ToolDefinition;
    /// Per-tool truncation limits; `None` (the default) defers to the
    /// host's global limits.
    fn output_limits(&self) -> Option<ToolOutputLimits> {
        None
    }
    /// How an `UnknownOutcome` result from this tool is treated; defaults
    /// to [`UnknownOutcomePolicy::Stop`].
    fn unknown_outcome_policy(&self) -> UnknownOutcomePolicy {
        UnknownOutcomePolicy::Stop
    }
    /// Runs one call under the call's [`CallControl`], returning the
    /// recorded outcome.
    async fn execute(&self, ctx: &ToolCallContext, control: &CallControl) -> ToolExecutionOutcome;
    /// Extension point for tools that persist artifacts: receives the host's
    /// [`ArtifactStore`] when one is configured (`None` otherwise). The
    /// default ignores the store and delegates to [`Tool::execute`].
    async fn execute_with_store(
        &self,
        ctx: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolExecutionOutcome {
        let _ = store;
        self.execute(ctx, control).await
    }
}
