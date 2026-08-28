//! Window budget and compaction seam — internal to context-kernel.

use crate::block::ContextBlock;

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

pub struct NoopCompaction;
#[async_trait::async_trait]
impl Compaction for NoopCompaction {
    async fn compact(&self, input: CompactionInput) -> Result<CompactionOutput, CompactionError> {
        Ok(CompactionOutput {
            blocks: input.blocks,
            summary: None,
            truncated: false,
        })
    }
}

/// 纯同步估算接口；无需 async_trait。
pub trait TokenCounter: Send + Sync {
    fn estimate(&self, blocks: &[ContextBlock]) -> usize;
    fn estimate_value(&self, value: &serde_json::Value) -> usize;
}

pub struct NoopTokenCounter;
impl TokenCounter for NoopTokenCounter {
    fn estimate(&self, blocks: &[ContextBlock]) -> usize {
        blocks
            .iter()
            .map(|b| {
                serde_json::to_string(&b.payload)
                    .map(|s| s.len() / 4)
                    .unwrap_or(0)
            })
            .sum()
    }
    fn estimate_value(&self, value: &serde_json::Value) -> usize {
        serde_json::to_string(value)
            .map(|s| s.len() / 4)
            .unwrap_or(0)
    }
}
