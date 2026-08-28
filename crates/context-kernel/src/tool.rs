use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::control::CallControl;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationLevel {
    Task,
    Subprocess,
}
impl Default for IsolationLevel {
    fn default() -> Self {
        Self::Task
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownOutcomePolicy {
    Stop,
    Continue,
}

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
        round_id: crate::ids::RoundId,
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
    pub fn position(&self) -> usize {
        self.0
            .rsplit(':')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallContext {
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub arguments: serde_json::Value,
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

#[derive(Debug, Clone)]
pub struct ToolExecutionOutcome {
    pub result: ToolResultPayload,
    pub policy: UnknownOutcomePolicy,
}
impl ToolExecutionOutcome {
    pub fn new(result: ToolResultPayload) -> Self {
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

#[derive(Debug, Clone)]
pub struct ToolOutputLimits {
    pub max_tokens: usize,
}
impl Default for ToolOutputLimits {
    fn default() -> Self {
        Self { max_tokens: 7_500 }
    }
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

// ---------------------------------------------------------------------------
// ToolExecutor — dedup-then-parallel dispatch + truncation + artifact + panic isolation
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;

pub struct ToolExecutor {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ToolExecutor {
    pub fn from_vec(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut map = HashMap::new();
        for t in tools {
            map.insert(t.definition().name.clone(), t);
        }
        Self { tools: map }
    }

    pub fn from_map(map: HashMap<String, Arc<dyn Tool>>) -> Self {
        Self { tools: map }
    }

    /// Execute a single ToolCallPayload with panic isolation, a call-deadline
    /// backstop, and token-limit truncation.
    pub async fn execute_with_limits(
        &self,
        payload: crate::block::ToolCallPayload,
        control: CallControl,
        store: Option<Arc<dyn ArtifactStore>>,
        token_counter: Option<Arc<dyn crate::turn::TokenCounter>>,
        global_limits: ToolOutputLimits,
    ) -> ToolExecutionOutcome {
        let tool_opt = self.tools.get(&payload.tool_name).cloned();
        let Some(tool) = tool_opt else {
            return ToolExecutionOutcome::new(ToolResultPayload {
                call_id: payload.call_id.clone(),
                status: ToolResultStatus::Rejected,
                output: ToolOutput::new(
                    serde_json::json!({"error": format!("unknown tool: {}", payload.tool_name)}),
                ),
            });
        };

        let ctx = ToolCallContext {
            call_id: payload.call_id.clone(),
            tool_name: payload.tool_name.clone(),
            arguments: payload.arguments.clone(),
        };
        let store_ref: Option<&dyn ArtifactStore> =
            store.as_deref().map(|s| s as &dyn ArtifactStore);

        // Panic isolation (Task level) plus call-deadline backstop: a tool that
        // neither returns nor observes CallControl still yields a structured
        // UnknownOutcome instead of hanging the turn.
        use futures_util::FutureExt;
        let fut = {
            let tool = tool.clone();
            let control = control.clone();
            std::panic::AssertUnwindSafe(async move {
                tool.execute_with_store(&ctx, &control, store_ref).await
            })
            .catch_unwind()
        };
        let mut outcome = match control.deadline() {
            Some(deadline) => match tokio::time::timeout_at(deadline.into(), fut).await {
                Ok(Ok(o)) => o,
                Ok(Err(_)) => Self::panicked_outcome(&payload),
                Err(_) => Self::deadline_backstop_outcome(&payload),
            },
            None => match fut.await {
                Ok(o) => o,
                Err(_) => Self::panicked_outcome(&payload),
            },
        };

        // UnknownOutcome policy always comes from the trusted tool declaration,
        // never from the outcome the tool produced itself.
        if matches!(outcome.result.status, ToolResultStatus::UnknownOutcome) {
            outcome.policy = tool.unknown_outcome_policy();
        }

        // Token-limit truncation (middle truncation + artifact spill)
        let effective_limit = tool.output_limits().unwrap_or(global_limits).max_tokens;

        let estimated = if let Some(counter) = &token_counter {
            counter.estimate_value(&outcome.result.output.content)
        } else {
            serde_json::to_string(&outcome.result.output.content)
                .map(|s| s.len() / 4)
                .unwrap_or(0)
        };

        if estimated > effective_limit {
            let content_str = serde_json::to_string(&outcome.result.output.content)
                .unwrap_or_else(|_| outcome.result.output.content.to_string());
            let data_bytes = serde_json::to_vec(&outcome.result.output.content)
                .unwrap_or_else(|_| content_str.clone().into_bytes());

            let artifact: Option<ArtifactRef> = if let Some(store_arc) = &store {
                let hint = ArtifactHint {
                    tool_name: payload.tool_name.clone(),
                    call_id: payload.call_id.clone(),
                    kind: ArtifactKind::FullOutput,
                };
                match store_arc.persist(&data_bytes, hint).await {
                    Ok(r) => Some(r),
                    Err(_) => None,
                }
            } else {
                None
            };

            let notice = if let Some(ref a) = artifact {
                format!(
                    "\n...[truncated: original {} tokens, artifact:{}]...\n",
                    estimated, a.id
                )
            } else {
                format!(
                    "\n...[truncated: original {} tokens, showing head+tail]...\n",
                    estimated
                )
            };

            // head 60% + notice + tail 40% — slice at char boundaries safely
            let total = content_str.len();
            let head_target = total * 3 / 5;
            let tail_target = total - head_target;
            let head_end = floor_char_boundary(&content_str, head_target);
            let tail_start = ceil_char_boundary(&content_str, total.saturating_sub(tail_target));
            let head = &content_str[..head_end];
            let tail = &content_str[tail_start..];
            let preview = format!("{}{}{}", head, notice, tail);

            outcome.result.output.content = serde_json::Value::String(preview);
            outcome.result.output.truncation = Truncation::Middle;
            outcome.result.output.artifact = artifact;
            let prev_meta = outcome.result.output.meta.take();
            outcome.result.output.meta = Some(ToolOutputMeta {
                duration_ms: prev_meta.as_ref().and_then(|m| m.duration_ms),
                original_tokens: Some(estimated),
                extra: prev_meta.as_ref().and_then(|m| m.extra.clone()),
            });
        }

        outcome
    }

    fn panicked_outcome(payload: &crate::block::ToolCallPayload) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: payload.call_id.clone(),
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(serde_json::json!({"error": "tool panicked"})),
        })
    }

    fn deadline_backstop_outcome(payload: &crate::block::ToolCallPayload) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: payload.call_id.clone(),
            status: ToolResultStatus::UnknownOutcome,
            output: ToolOutput {
                content: serde_json::json!({"error": "tool did not return before call deadline"}),
                truncation: Truncation::None,
                meta: Some(ToolOutputMeta {
                    duration_ms: None,
                    original_tokens: None,
                    extra: Some(serde_json::json!({"reason": "call_deadline_backstop"})),
                }),
                artifact: None,
            },
        })
    }
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}
fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}
