//! Frame-materialization ports — window budget, compaction seam, token
//! counter, and the canonical `FramePolicy` carrier.

use crate::context::block::ContextBlock;

#[derive(Debug, Clone, Copy)]
pub struct WindowBudget {
    pub model_window_limit: usize,
    pub compaction_trigger: usize,
}
impl WindowBudget {
    pub fn should_compact(&self, estimated_tokens: usize) -> bool {
        estimated_tokens >= self.compaction_trigger
    }
}
impl Default for WindowBudget {
    fn default() -> Self {
        Self {
            model_window_limit: usize::MAX,
            compaction_trigger: usize::MAX,
        }
    }
}

pub struct CompactionInput {
    pub blocks: Vec<ContextBlock>,
    pub budget: WindowBudget,
    pub estimated_tokens: usize,
}
/// `summary` 仅为 host 观测信息；若摘要需要模型可见，实现应自行并入 `blocks`
/// ——`frame` 只物化 `out.blocks`，不会追加 `summary`。
pub struct CompactionOutput {
    pub blocks: Vec<ContextBlock>,
    pub summary: Option<ContextBlock>,
    pub truncated: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    #[error("compaction failed: {0}")]
    Failed(String),
}

#[async_trait::async_trait]
pub trait Compaction: Send + Sync {
    async fn compact(&self, input: CompactionInput) -> Result<CompactionOutput, CompactionError>;
}

/// 纯同步估算接口；无需 async_trait。
pub trait TokenCounter: Send + Sync {
    fn estimate(&self, blocks: &[ContextBlock]) -> usize;
    fn estimate_value(&self, value: &serde_json::Value) -> usize;
}

/// Carrier of the frame-materialization policy: trigger budget, optional
/// compaction, optional token counter. A canonical value assembled from port
/// instances — drivers (staged) build and own it, `TurnContext::frame()`
/// (canonical) evaluates it. Placeholder semantics stay frame-local and
/// non-persisting; real conversation-level policy is Slice 5 territory.
#[derive(Clone, Default)]
pub struct FramePolicy {
    pub window_budget: WindowBudget,
    pub compaction: Option<std::sync::Arc<dyn Compaction>>,
    pub token_counter: Option<std::sync::Arc<dyn TokenCounter>>,
}
impl std::fmt::Debug for FramePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FramePolicy")
            .field("window_budget", &self.window_budget)
            .field("compaction", &self.compaction.is_some())
            .field("token_counter", &self.token_counter.is_some())
            .finish()
    }
}
impl FramePolicy {
    /// Token estimate for a block list: the supplied counter if any, else the
    /// placeholder JSON-length/4 heuristic.
    pub fn estimate(&self, blocks: &[ContextBlock]) -> usize {
        if let Some(counter) = &self.token_counter {
            counter.estimate(blocks)
        } else {
            blocks
                .iter()
                .map(|b| {
                    serde_json::to_string(&b.content)
                        .map(|s| s.len() / 4)
                        .unwrap_or(0)
                })
                .sum()
        }
    }
}
