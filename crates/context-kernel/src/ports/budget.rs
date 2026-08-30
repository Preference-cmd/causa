//! Frame-materialization ports — window budget, compaction seam, token
//! counter, and the canonical `FramePolicy` carrier. The policy orchestrates
//! materialization itself: the fact machine offers only the lossless
//! projection and never awaits behavior.

use crate::context::block::ContextBlock;
use crate::context::ids::RoundId;
use crate::context::turn::{ContextFrame, TurnContext};

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

/// Error of policy-driven frame materialization: the only fallible step is
/// the compaction port.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("compaction failed: {0}")]
    CompactionFailed(String),
}

/// The placeholder token heuristic shared by the noop counter, the frame
/// policy fallback, and the executor: serialized JSON length divided by four.
/// Slice 5 replaces it with a real tokenizer.
pub fn placeholder_token_estimate_value(value: &serde_json::Value) -> usize {
    serde_json::to_string(value)
        .map(|s| s.len() / 4)
        .unwrap_or(0)
}

pub fn placeholder_token_estimate(blocks: &[ContextBlock]) -> usize {
    blocks
        .iter()
        .map(|b| {
            serde_json::to_string(&b.content)
                .map(|s| s.len() / 4)
                .unwrap_or(0)
        })
        .sum()
}

/// Carrier of the frame-materialization policy: trigger budget, optional
/// compaction, optional token counter. A canonical value assembled from port
/// instances — drivers (staged) build and own it, and it orchestrates
/// materialization itself, using only the fact machine's public accessors.
/// Placeholder semantics stay frame-local and non-persisting; real
/// conversation-level policy is Slice 5 territory.
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
            placeholder_token_estimate(blocks)
        }
    }

    /// Materialize the model context for `round_id` under this policy.
    /// Trigger evaluation stays canonical — the same state and the same
    /// policy always yield the same frame. Compaction output is frame-local
    /// and never written back into the fact state; the fact machine itself
    /// only ever offers the lossless projection and never awaits behavior.
    pub async fn materialize(
        &self,
        ctx: &TurnContext,
        round_id: RoundId,
    ) -> Result<ContextFrame, FrameError> {
        let estimated = self.estimate(ctx.blocks());
        if self.window_budget.should_compact(estimated)
            && let Some(comp) = &self.compaction
        {
            let input = CompactionInput {
                blocks: ctx.blocks().to_vec(),
                budget: self.window_budget,
                estimated_tokens: estimated,
            };
            let out = comp
                .compact(input)
                .await
                .map_err(|e| FrameError::CompactionFailed(e.to_string()))?;
            return Ok(ctx.frame_with(round_id, out.blocks));
        }
        Ok(ctx.frame(round_id))
    }
}
