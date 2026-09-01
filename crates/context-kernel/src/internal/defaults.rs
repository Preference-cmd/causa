//! Noop port defaults — trivial implementations for hosts that need a
//! placeholder wiring. The port traits stay canonical in `budget`; only these
//! default instances are staged.

use crate::ports::budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, TokenCounter,
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
    fn estimate(&self, _blocks: &[crate::context::block::ContextBlock]) -> usize {
        0
    }
    fn estimate_value(&self, _value: &serde_json::Value) -> usize {
        0
    }
}

/// Example heuristic: serialized JSON length divided by 4.
///
/// NOT the kernel's policy. Hosts that don't yet have a real
/// tokenizer can wire their own `TokenCounter` with this logic;
/// `FramePolicy::estimate` itself has no fallback beyond 0.
#[allow(dead_code)]
pub fn placeholder_token_estimate_value(value: &serde_json::Value) -> usize {
    serde_json::to_string(value)
        .map(|s| s.len() / 4)
        .unwrap_or(0)
}
#[allow(dead_code)]
pub fn placeholder_token_estimate(blocks: &[crate::context::block::ContextBlock]) -> usize {
    blocks
        .iter()
        .map(|b| {
            placeholder_token_estimate_value(&serde_json::to_value(&b.content).unwrap_or_default())
        })
        .sum()
}
