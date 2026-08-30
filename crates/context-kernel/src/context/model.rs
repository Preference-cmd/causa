//! Fact vocabulary of the model door — what the fact machine validates and
//! records. The gateway envelope (`ModelOutput`), the transport error, and
//! request parameters (`ModelRef`, `GenerationOptions`, `ToolSurface`) are
//! port vocabulary and live in `crate::ports::gateway`.

use serde::{Deserialize, Serialize};

use crate::context::block::TextPayload;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallDraft {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    /// Provider-issued identifier for this tool call, if the upstream model
    /// API assigned one. The kernel records it verbatim (on the envelope
    /// `BlockMeta`) and pairs tool results by the kernel-generated
    /// `ToolCallId`, never by this field.
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

/// The model door's fact shape: what the model said (text) and asked (tool
/// calls). Named for the participant, not for any wire role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub text: TextPayload,
    pub tool_calls: Vec<ToolCallDraft>,
}
