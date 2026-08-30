//! Noop port defaults — trivial implementations for hosts that need a
//! placeholder wiring. The port traits stay canonical in `budget`; only these
//! default instances are staged.

use crate::ports::budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, TokenCounter,
    placeholder_token_estimate, placeholder_token_estimate_value,
};
use async_trait::async_trait;

pub struct NoopCompaction;
#[async_trait]
impl Compaction for NoopCompaction {
    async fn compact(&self, input: CompactionInput) -> Result<CompactionOutput, CompactionError> {
        Ok(CompactionOutput {
            blocks: input.blocks,
            summary: None,
            truncated: false,
        })
    }
}

pub struct NoopTokenCounter;
impl TokenCounter for NoopTokenCounter {
    fn estimate(&self, blocks: &[crate::context::block::ContextBlock]) -> usize {
        placeholder_token_estimate(blocks)
    }
    fn estimate_value(&self, value: &serde_json::Value) -> usize {
        placeholder_token_estimate_value(value)
    }
}
