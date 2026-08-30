//! Tool behavior — the `Tool` trait and the `ArtifactStore` port. Value
//! types live in `crate::context::tool_data`; batch dispatch lives in the staged
//! `internal::executor`.

use async_trait::async_trait;

use crate::context::tool_data::{
    ArtifactKind, ArtifactRef, ToolCallContext, ToolCallId, ToolExecutionOutcome, ToolOutputLimits,
    UnknownOutcomePolicy,
};
use crate::ports::control::CallControl;

#[derive(Debug, Clone, Copy)]
pub enum IsolationLevel {
    Task,
    Subprocess,
}
impl Default for IsolationLevel {
    fn default() -> Self {
        Self::Task
    }
}

pub struct ArtifactHint {
    pub tool_name: String,
    pub call_id: ToolCallId,
    pub kind: ArtifactKind,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("persist failed: {0}")]
    Persist(String),
    #[error("read failed: {0}")]
    Read(String),
}

#[async_trait]
pub trait ArtifactStore: Send + Sync {
    async fn persist(&self, data: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError>;
    async fn read(
        &self,
        id: &str,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError>;
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> crate::context::tool_data::ToolDefinition;
    fn output_limits(&self) -> Option<ToolOutputLimits> {
        None
    }
    fn isolation_level(&self) -> IsolationLevel {
        IsolationLevel::Task
    }
    fn unknown_outcome_policy(&self) -> UnknownOutcomePolicy {
        UnknownOutcomePolicy::Stop
    }
    async fn execute(&self, ctx: &ToolCallContext, control: &CallControl) -> ToolExecutionOutcome;
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
