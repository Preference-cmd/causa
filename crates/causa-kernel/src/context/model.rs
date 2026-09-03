//! Fact vocabulary of the model door — what the fact machine validates and
//! records. The gateway envelope (`ModelOutput`), the transport error, and
//! request parameters (`ModelRef`, `GenerationOptions`, `ToolSurface`) are
//! port vocabulary and live in `crate::ports::gateway`.

use serde::{Deserialize, Serialize};

use crate::context::block::TextPayload;

/// A tool call as the model proposed it, before the kernel assigns the
/// canonical `ToolCallId`. Exactly what the gateway parsed off the wire.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallDraft {
    /// Name of the tool the model asked to invoke.
    pub tool_name: String,
    /// Tool arguments as a JSON object; validated by the kernel before commit.
    pub arguments: serde_json::Value,
    /// Provider-issued identifier for this tool call, if the upstream model
    /// API assigned one. The kernel records it verbatim (on the envelope
    /// `BlockMeta`) and pairs tool results by the kernel-generated
    /// `ToolCallId`, never by this field.
    #[serde(default)]
    pub provider_call_id: Option<String>,
}

/// Why the model stopped producing output for this round, as reported by
/// the gateway. The kernel validates only the EndTurn/ToolUse structural
/// pairing; interpreting terminal reasons is driver policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStopReason {
    /// The model ended its turn without requesting tools; `tool_calls` must
    /// be empty.
    EndTurn,
    /// The model requested tool invocations; `tool_calls` must be non-empty.
    ToolUse,
    /// Output was cut off at the token limit.
    MaxTokens,
    /// The model (or a provider content filter) refused to produce output.
    Refusal,
}

/// The model door's fact shape: what the model said (text) and asked (tool
/// calls). Named for the participant, not for any wire role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// What the model said. Blank text produces no fact block on commit.
    pub text: TextPayload,
    /// What the model asked for, in its draft order; ids are assigned at
    /// commit.
    pub tool_calls: Vec<ToolCallDraft>,
}
