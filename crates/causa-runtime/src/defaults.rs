//! Noop port defaults and the driver's fallback opinions — trivial
//! implementations for hosts that need a placeholder wiring. The port traits
//! stay canonical in `budget`; only these default instances are staged.
//!
//! # Driver policy surface (7.6)
//!
//! Every default the driver bakes in is either a config/policy object
//! (`RetryPolicy`, `TurnPolicy`, `HookCtx`, `TokenCounter`,
//! `UnknownOutcomePolicy`, `ToolOutputLimits`) or documented here / at the
//! impl site as the driver's opinion. The token-estimate fallback below is
//! the single home of the chars/4 heuristic — no other copy exists.

use async_trait::async_trait;
use causa_kernel::{Compaction, CompactionError, CompactionInput, CompactionOutput, TokenCounter};

/// Noop [`Compaction`] default: returns the input blocks unchanged —
/// no summary, nothing truncated.
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

/// Noop [`TokenCounter`] default: every estimate is `0`. The driver's real
/// fallback when no counter is wired is [`placeholder_token_estimate_value`].
pub struct NoopTokenCounter;
impl TokenCounter for NoopTokenCounter {
    fn estimate(&self, _blocks: &[causa_kernel::ContextBlock]) -> usize {
        0
    }
    fn estimate_value(&self, _value: &serde_json::Value) -> usize {
        0
    }
}

/// The driver's token-estimate fallback opinion: serialized JSON length
/// divided by 4.
///
/// NOT the kernel's policy. `ToolExecutor` falls back to this when no
/// `TokenCounter` is wired; hosts that don't yet have a real tokenizer can
/// wire their own `TokenCounter` with the same logic. This function is the
/// heuristic's single home (7.4) — do not inline a copy elsewhere.
pub fn placeholder_token_estimate_value(value: &serde_json::Value) -> usize {
    serde_json::to_string(value)
        .map(|s| s.len() / 4)
        .unwrap_or(0)
}

/// Block-level convenience over [`placeholder_token_estimate_value`].
pub fn placeholder_token_estimate(blocks: &[causa_kernel::ContextBlock]) -> usize {
    blocks
        .iter()
        .map(|b| {
            placeholder_token_estimate_value(&serde_json::to_value(&b.content).unwrap_or_default())
        })
        .sum()
}
