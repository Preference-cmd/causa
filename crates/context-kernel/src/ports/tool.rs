//! Tool behavior — the `Tool` trait and the `ArtifactStore` port, plus the
//! execution vocabulary that only they and the staged executor consume:
//! definitions, call context, outcome policies, output limits. Recorded
//! facts (results, outputs, artifacts) live in `crate::context::tool_data`;
//! batch dispatch lives in the staged `internal::executor`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::tool_data::{ArtifactKind, ArtifactRef, ToolCallId};
use crate::ports::control::CallControl;

#[derive(Debug, Clone, Copy, Default)]
pub enum IsolationLevel {
    #[default]
    Task,
    Subprocess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolCallContext {
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// How a tool wants its `UnknownOutcome` result treated — a declaration the
/// driver obeys, not a fact the kernel interprets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownOutcomePolicy {
    Stop,
    Continue,
}

#[derive(Debug, Clone)]
pub struct ToolExecutionOutcome {
    pub result: crate::context::tool_data::ToolResultPayload,
    pub policy: UnknownOutcomePolicy,
}
impl ToolExecutionOutcome {
    pub fn new(result: crate::context::tool_data::ToolResultPayload) -> Self {
        Self {
            result,
            policy: UnknownOutcomePolicy::Stop,
        }
    }
    pub fn with_policy(mut self, policy: UnknownOutcomePolicy) -> Self {
        self.policy = policy;
        self
    }
}

#[derive(Debug, Clone)]
pub struct ToolOutputLimits {
    pub max_tokens: usize,
}
impl Default for ToolOutputLimits {
    fn default() -> Self {
        Self { max_tokens: 7_500 }
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
    fn definition(&self) -> ToolDefinition;
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
