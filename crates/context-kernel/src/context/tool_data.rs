//! Tool-domain fact vocabulary — call ids, results, outputs, artifacts.
//!
//! This module holds only recorded facts: what the tool door persists and
//! what the pairing invariant validates. Behavior and execution vocabulary
//! (the `Tool` trait, the `ArtifactStore` port, definitions, limits, outcome
//! policies, dispatch context) live in `crate::ports::tool`; canonical
//! modules must depend on this module, never on the executor-sized behavior
//! module.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolCallId(pub String);
impl ToolCallId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    /// `tool_name + blake3(round_id + tool_name + arguments_json)[..8] + position`。
    /// round_id 进入哈希前像，使同一 `(tool, arguments, position)` 在不同 ModelRound
    /// 生成不同 id——模型跨 round 重发同一调用（去重后的合法恢复路径）不得与
    /// 历史 call_id 碰撞。唯一性范围是单个 TurnContext。
    pub fn generate(
        round_id: crate::context::ids::RoundId,
        tool_name: &str,
        arguments: &serde_json::Value,
        position: usize,
    ) -> Self {
        let json = serde_json::to_string(arguments)
            .unwrap_or_else(|_| "<unserializable-arguments>".to_string());
        let preimage = format!("{}|{}|{}", round_id.0, tool_name, json);
        let hash = blake3::hash(preimage.as_bytes());
        let hex = hash.to_hex();
        Self(format!("{}:{}:{}", tool_name, &hex[..8], position))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolResultStatus {
    Succeeded,
    Failed,
    Rejected,
    Cancelled,
    TimedOut,
    UnknownOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutputMeta {
    pub duration_ms: Option<u64>,
    pub original_tokens: Option<usize>,
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    None,
    Middle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: serde_json::Value,
    pub truncation: Truncation,
    pub meta: Option<ToolOutputMeta>,
    pub artifact: Option<ArtifactRef>,
}
impl ToolOutput {
    pub fn new(content: serde_json::Value) -> Self {
        Self {
            content,
            truncation: Truncation::None,
            meta: None,
            artifact: None,
        }
    }
    pub fn is_truncated(&self) -> bool {
        !matches!(self.truncation, Truncation::None)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultPayload {
    pub call_id: ToolCallId,
    pub status: ToolResultStatus,
    pub output: ToolOutput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: String,
    pub size_bytes: usize,
    pub kind: ArtifactKind,
    pub persisted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactKind {
    FullOutput,
    PipeCache,
    Binary,
}
