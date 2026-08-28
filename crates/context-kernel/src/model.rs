use serde::{Deserialize, Serialize};

use crate::block::TextPayload;
use crate::tool::ToolDefinition;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef(pub String);
impl ModelRef {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolSurface {
    pub definitions: Vec<ToolDefinition>,
}
impl ToolSurface {
    pub fn empty() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }
    pub fn from_definitions(definitions: Vec<ToolDefinition>) -> Self {
        Self { definitions }
    }
    /// Spec §4 constructor `ToolSurface::from_tools(&[Box<dyn Tool>])`.
    /// Kept as alias to `from_definitions` for API compatibility.
    pub fn from_tools(tools: &[Box<dyn crate::tool::Tool>]) -> Self {
        Self {
            definitions: tools.iter().map(|t| t.definition()).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDraft {
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Refusal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantPayload {
    pub text: TextPayload,
    pub tool_calls: Vec<ToolCallDraft>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    pub assistant: AssistantPayload,
    pub usage: Option<ModelUsage>,
    pub stop_reason: ModelStopReason,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelInvokeErrorKind {
    Transient,
    TimedOut,
    Cancelled,
    Permanent,
    InvalidRequest,
    UnknownOutcome,
}

#[derive(Debug)]
pub struct ModelInvokeError {
    pub kind: ModelInvokeErrorKind,
    pub message: String,
}
impl ModelInvokeError {
    pub fn new(kind: ModelInvokeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub fn kind(&self) -> &ModelInvokeErrorKind {
        &self.kind
    }
    pub fn is_retryable(&self, policy: &RetryPolicy) -> bool {
        match self.kind {
            ModelInvokeErrorKind::Transient => true,
            ModelInvokeErrorKind::TimedOut => policy.retry_timeouts,
            _ => false,
        }
    }
}
impl std::fmt::Display for ModelInvokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for ModelInvokeError {}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub retry_timeouts: bool,
}
impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 0,
            retry_timeouts: false,
        }
    }
}
