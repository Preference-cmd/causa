use serde::{Deserialize, Serialize};

use crate::ids::{BlockId, BlockSequence};
use crate::tool_data::ToolResultPayload;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockMeta {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPayload(pub String);
impl TextPayload {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInjectPayload {
    pub text: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallPayload {
    pub call_id: crate::tool_data::ToolCallId,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    /// Provider-issued identifier carried through from the model draft, if
    /// the upstream API assigned one. Recorded as-is; pairing stays on the
    /// kernel-generated `call_id`.
    #[serde(default)]
    pub provider_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlock {
    pub id: BlockId,
    pub sequence: BlockSequence,
    pub meta: BlockMeta,
    pub payload: BlockPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "payload")]
pub enum BlockPayload {
    #[serde(rename = "instruction.system")]
    InstructionSystem(TextPayload),
    #[serde(rename = "context.inject")]
    ContextInject(ContextInjectPayload),
    #[serde(rename = "request.user")]
    RequestUser(TextPayload),
    #[serde(rename = "response.assistant")]
    ResponseAssistant(TextPayload),
    #[serde(rename = "tool.call")]
    ToolCall(ToolCallPayload),
    #[serde(rename = "tool.result")]
    ToolResult(ToolResultPayload),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputPayload {
    InstructionSystem(TextPayload),
    ContextInject(ContextInjectPayload),
    RequestUser(TextPayload),
}
