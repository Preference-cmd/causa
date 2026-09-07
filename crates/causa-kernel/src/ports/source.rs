//! `DynamicToolSource` port — the seam external tool catalogs (MCP
//! servers, plugin hosts) fill in so their tools can enter the model's
//! `ToolSurface` next to local Rust tools (Slice 10).
//!
//! The port stays in the kernel by the Slice 12 criterion: it is the
//! contract surface third parties implement against the facts crate alone
//! (`causa-extension` implements it with only this crate as dependency,
//! exactly as `causa-provider` implements `ModelGateway`). The aggregation
//! and dispatch logic that *consumes* the port lives in
//! `causa_runtime`'s executor.

use async_trait::async_trait;

use crate::context::block::ToolCallPayload;
use crate::context::tool_data::ToolResultPayload;
use crate::ports::control::CallControl;
use crate::ports::tool::ToolDefinition;

/// One external tool catalog. Implementors own their connection, their
/// naming (see the `mcp_{server_id}_{tool}` namespace convention in
/// `causa-extension`), and their change notifications.
#[async_trait]
pub trait DynamicToolSource: Send + Sync {
    /// Stable catalog identity — also the dispatch namespace key.
    fn id(&self) -> &str;

    /// Change signal for executor-side caching: bump whenever the tool
    /// listing may have changed (e.g. MCP `tools/list_changed`). The
    /// executor caches the listing keyed by this value — a source whose
    /// observed version matches the cache costs no re-list, and a source
    /// without change notifications (the `0` default) is therefore listed
    /// once and served from the cache until it bumps.
    fn version(&self) -> u64 {
        0
    }

    /// The current tool listing. Fails when the catalog is unreachable —
    /// the executor then keeps serving the source's last good listing
    /// (with a warning) and proceeds; a source with no cached listing yet
    /// is skipped. Either way the turn is not interrupted.
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError>;

    /// Execute one call against the catalog. `call.tool_name` arrives in
    /// the source's namespace; implementors de-namespace it (and reject
    /// names outside their namespace). Sources return the recorded result
    /// only — what an `UnknownOutcome` result does next is harness
    /// configuration, not source vocabulary.
    async fn invoke(
        &self,
        call: &ToolCallPayload,
        control: &CallControl,
    ) -> Result<ToolResultPayload, ToolExecutionError>;

    /// Execute one call with the host's artifact store available for
    /// media ingest: a source whose tools produce images can persist the
    /// bytes via `store` and attach [`crate::context::block::MediaRef`]s
    /// to the result's `ToolResultPayload.media` instead of degrading
    /// them to text. Additive with a delegating default, so sources that
    /// never produce media keep their existing `invoke` only.
    async fn invoke_with_store(
        &self,
        call: &ToolCallPayload,
        control: &CallControl,
        store: Option<&dyn crate::ports::tool::ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        let _ = store;
        Self::invoke(self, call, control).await
    }
}

/// Listing failure — the catalog as a whole is unusable right now.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SourceError {
    /// Connection failure or server offline; retryable (transient).
    #[error("tool source unavailable: {0}")]
    Unavailable(String),
    /// Protocol-level failure; not retryable (permanent).
    #[error("tool source protocol error: {0}")]
    Protocol(String),
}

/// Single-call failure inside a source. The executor's tool adapter
/// (`ToolBridge` in `causa_runtime`) turns every variant into a structured
/// tool result the model reads; the status mapping lives there, not here.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolExecutionError {
    /// The call may or may not have run server-side — the outcome is
    /// unknown, not failed (the executor marks it accordingly).
    #[error("tool call timed out")]
    TimedOut,
    /// Cancelled before completion; the outcome is unknown.
    #[error("tool call cancelled")]
    Cancelled,
    /// The name is not in this source's namespace.
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    /// Connection failure or server offline (transient).
    #[error("tool source unavailable: {0}")]
    Unavailable(String),
    /// Protocol-level failure (permanent).
    #[error("tool source protocol error: {0}")]
    Protocol(String),
}
