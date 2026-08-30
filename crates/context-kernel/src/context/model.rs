use serde::{Deserialize, Serialize};

use crate::context::block::TextPayload;
use crate::context::tool_data::ToolDefinition;

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
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    /// Provider-reported prompt-cache read (hit) tokens, if disclosed.
    #[serde(default)]
    pub cache_read_tokens: Option<usize>,
    /// Provider-reported prompt-cache write (population) tokens, if disclosed.
    #[serde(default)]
    pub cache_write_tokens: Option<usize>,
    /// Provider-reported reasoning/thinking tokens, if disclosed.
    #[serde(default)]
    pub reasoning_tokens: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallDraft {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    /// Provider-issued identifier for this tool call, if the upstream model
    /// API assigned one. The kernel records it verbatim and pairs tool
    /// results by the kernel-generated `ToolCallId`, never by this field.
    #[serde(default)]
    pub provider_call_id: Option<String>,
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
/// The model door's fact shape: what the model said (text) and asked (tool
/// calls). Named for the participant, not for any wire role.
pub struct ModelResponse {
    pub text: TextPayload,
    pub tool_calls: Vec<ToolCallDraft>,
}

/// Structured reasoning content: the model's thinking text plus the optional
/// provider signature some APIs attach so it can be replayed on later turns.
/// Recorded as-is; the kernel does not interpret it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningPayload {
    pub text: String,
    #[serde(default)]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    pub response: ModelResponse,
    pub usage: Option<ModelUsage>,
    pub stop_reason: ModelStopReason,
    pub reasoning: Option<ReasoningPayload>,
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
}
impl std::fmt::Display for ModelInvokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for ModelInvokeError {}
