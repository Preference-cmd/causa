//! `TurnInteraction` port — the single host↔driver interaction boundary
//! during a turn (batch decisions and steering injection). When it fires is
//! reference driver policy, so the contract lives with its only consumer
//! (the driver) rather than in the kernel.
//!
//! One port with default no-ops replaces per-entry callback channels:
//! entry signatures never grow interaction parameters, and new
//! interactions (progress, per-call approval) arrive as new default
//! methods without breaking third-party implementors. The driver consumes
//! exactly one of these per turn via `TurnRunOptions.interaction`
//! (`config`); the noop default is [`crate::config::NoopInteraction`],
//! which stays with the config axes.

use async_trait::async_trait;

use crate::hook::UnknownDecision;
use causa_kernel::{RoundId, StreamDelta, TextPayload, ToolCallPayload, ToolResultPayload};

/// Host observations and decisions while a turn is in flight.
#[async_trait]
pub trait TurnInteraction: Send + Sync {
    /// Streaming-delta observation. Called once per provider delta in the
    /// model phase, with the round the delta belongs to (round attribution
    /// is driver state — the port carries it so hosts never have to guess
    /// round boundaries). Deltas observed here are advisory: a retried
    /// attempt re-streams the same frame and the host decides how to
    /// present the partial-then-reset flow.
    async fn on_delta(&self, _round_id: RoundId, _delta: &StreamDelta) {}

    /// Batch decision gate: called after the tool-use hook has
    /// filtered the model-emitted batch and before executor dispatch.
    /// Default [`BatchDecision::Proceed`] — the literal absence of
    /// opinion. Returning [`BatchDecision::Pause`] suspends the turn
    /// ([`crate::driver::TurnResult::Paused`], context left open) until
    /// the host resumes it through [`crate::resume::resume_turn`] with
    /// the withheld [`crate::hook::HookOutcome`] — approve is
    /// passthrough, reject is all-rejected, rewrite is the edited batch;
    /// the HookOutcome constructors cover every decision, so no separate
    /// resume-decision enum exists.
    async fn decide_batch(&self, _calls: &[ToolCallPayload]) -> BatchDecision {
        BatchDecision::Proceed
    }

    /// Steering inputs, pulled by the driver at every round
    /// boundary before the frame materializes; each non-empty entry is
    /// appended to the active turn with the `user.steering` source label
    /// and the next model round sees it. Default: empty (zero-cost
    /// pull). The queue itself is host vocabulary — the runtime holds no
    /// channel type.
    async fn pending_inputs(&self) -> Vec<TextPayload> {
        Vec::new()
    }
}

/// The driver's action for a tool-use batch.
#[derive(Debug, Clone)]
pub enum BatchDecision {
    /// Dispatch the batch to the executor unchanged.
    Proceed,
    /// Skip execution entirely; the supplied results become the tool
    /// results (the host supplies the error copy). Result `call_id`s
    /// must pair with the batch — the fact machine rejects unpaired
    /// results. Despite the variant's name, any recorded status is
    /// accepted; a precomputed `UnknownOutcome` result takes its action
    /// from `unknown_decisions` (or the runner's unknown-outcome
    /// configuration when omitted).
    Reject {
        /// Results standing in for the skipped batch; their `call_id`s must
        /// pair with the batch.
        results: Vec<ToolResultPayload>,
        /// Explicit unknown-outcome actions for precomputed results whose
        /// status is `UnknownOutcome` (see [`UnknownDecision`]); entries
        /// may be omitted.
        unknown_decisions: Vec<UnknownDecision>,
    },
    /// Execute the rewritten payloads instead of the model-emitted ones.
    /// Call ids are expected to be preserved so results pair with the
    /// committed tool-call blocks.
    Rewrite(Vec<ToolCallPayload>),
    /// Suspend the turn: context stays open, the batch is neither
    /// executed nor rejected, and the outcome carries
    /// [`crate::driver::TurnResult::Paused`]. `deadline` is the advisory
    /// decision budget the host grants itself (remaining time;
    /// host-side anchoring).
    Pause {
        /// Advisory remaining decision budget the host grants itself
        /// (host-side anchoring).
        deadline: Option<std::time::Duration>,
    },
}
