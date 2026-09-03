//! Frame-materialization ports — window budget, compaction seam, token
//! counter, and the canonical `FramePolicy` carrier. The policy orchestrates
//! materialization itself: the fact machine offers only the lossless
//! projection and never awaits behavior.

use crate::context::block::ContextBlock;
use crate::context::ids::RoundId;
use crate::context::turn::{ContextFrame, TurnContext};

/// Trigger thresholds for frame materialization, in estimated tokens. A pure
/// value: the kernel's trigger check reads only `compaction_trigger`; the
/// full budget rides along to the compaction implementation via
/// [`CompactionInput`].
#[derive(Debug, Clone, Copy)]
pub struct WindowBudget {
    /// Upper bound of the model's context window, in estimated tokens.
    /// Consulted by compaction implementations, not by the trigger check.
    pub model_window_limit: usize,
    /// Estimated-token threshold at or above which [`WindowBudget::should_compact`]
    /// fires. The default `usize::MAX` means "never trigger".
    pub compaction_trigger: usize,
}
impl WindowBudget {
    /// Whether frame materialization should compact: true when
    /// `estimated_tokens` has reached the `compaction_trigger`.
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

/// Input handed to a [`Compaction`] implementation: the blocks to reduce,
/// the budget in force, and the estimate that tripped the trigger.
pub struct CompactionInput {
    /// The current lossless block projection to compact. Input only —
    /// fact state is never mutated by compaction.
    pub blocks: Vec<ContextBlock>,
    /// The [`WindowBudget`] in force, including `model_window_limit` for
    /// the implementation's use.
    pub budget: WindowBudget,
    /// The token estimate that tripped `should_compact`.
    pub estimated_tokens: usize,
}
/// `summary` is host-observation only; if a summary must be model-visible,
/// the implementation folds it into `blocks` itself — `frame` materializes
/// `out.blocks` and never appends `summary`.
pub struct CompactionOutput {
    /// The replacement block list that `frame` materializes. Frame-local —
    /// never written back into fact state.
    pub blocks: Vec<ContextBlock>,
    /// Host-observation-only summary block, if the implementation produced
    /// one. Never appended to the frame.
    pub summary: Option<ContextBlock>,
    /// True when the output is not a lossless projection of
    /// [`CompactionInput::blocks`] — content was dropped or condensed.
    pub truncated: bool,
}

/// Failure of a [`Compaction`] implementation — the only error it may report.
#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    /// Compaction failed; carries the implementation-defined reason.
    #[error("compaction failed: {0}")]
    Failed(String),
}

/// Compaction port: reduce a block list to fit the window budget. Async and
/// host-pluggable; the output is frame-local (see [`CompactionOutput`]) and
/// never written back into fact state.
#[async_trait::async_trait]
pub trait Compaction: Send + Sync {
    /// Compact `input.blocks` under `input.budget`, returning the
    /// replacement blocks plus an optional host-only summary.
    async fn compact(&self, input: CompactionInput) -> Result<CompactionOutput, CompactionError>;
}

/// Purely synchronous estimation interface; no async_trait needed.
pub trait TokenCounter: Send + Sync {
    /// Estimated token count for a block list.
    fn estimate(&self, blocks: &[ContextBlock]) -> usize;
    /// Estimated token count for a single JSON value; the executor uses it
    /// to size tool outputs before limit-based truncation.
    fn estimate_value(&self, value: &serde_json::Value) -> usize;
}

/// Error of policy-driven frame materialization: the only fallible step is
/// the compaction port.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// The wired [`Compaction`] port failed; carries its error message.
    #[error("compaction failed: {0}")]
    CompactionFailed(String),
}

// Note: prior revisions shipped a `placeholder_token_estimate`
// helper ("JSON length / 4"). It was a specific heuristic carried
// inside `ports/`, violating "invariants only". `FramePolicy::estimate`
// now returns 0 when no `TokenCounter` is wired, so `should_compact`
// computes as "no trigger". Hosts that need the heuristic can wire
// `agent_runtime::defaults::NoopTokenCounter` or their own `TokenCounter`.

/// Carrier of the frame-materialization policy: trigger budget, optional
/// compaction, optional token counter. A canonical value assembled from port
/// instances — drivers build and own it, and it orchestrates
/// materialization itself, using only the fact machine's public accessors.
/// Placeholder semantics stay frame-local and non-persisting; real
/// conversation-level policy is Slice 5 territory.
#[derive(Clone, Default)]
pub struct FramePolicy {
    /// Trigger thresholds; the all-`usize::MAX` default never trips
    /// compaction.
    pub window_budget: WindowBudget,
    /// Compaction port applied when the trigger fires; absent means the
    /// lossless frame is always used.
    pub compaction: Option<std::sync::Arc<dyn Compaction>>,
    /// Token counter backing [`FramePolicy::estimate`]; absent means a zero
    /// estimate, which never triggers compaction.
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
    /// Token estimate for a block list: the supplied counter if any,
    /// else 0. A zero estimate never trips `should_compact` (default
    /// `WindowBudget` thresholds are `usize::MAX`), so compaction is
    /// genuinely opt-in via a wired `TokenCounter`.
    pub fn estimate(&self, blocks: &[ContextBlock]) -> usize {
        if let Some(counter) = &self.token_counter {
            counter.estimate(blocks)
        } else {
            0
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
