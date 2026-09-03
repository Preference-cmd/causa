//! `TurnInteraction` port — the single host↔driver interaction boundary
//! during a turn (Slice 6, chartered by Slice 12 Decision 3; batch
//! decisions and steering injection added by Slice 7).
//!
//! One port with default no-ops replaces per-entry callback channels:
//! entry signatures never grow interaction parameters, and new
//! interactions (progress, per-call approval) arrive as new default
//! methods without breaking third-party implementors. The driver consumes
//! exactly one of these per turn via `TurnRunOptions.interaction`
//! (agent-runtime).
//!
//! The kernel defines the contract only; the noop default
//! (`NoopInteraction`) lives with the driver's config axes in
//! `agent-runtime`, mirroring where the hook policies live.

use async_trait::async_trait;

use crate::context::block::{TextPayload, ToolCallPayload};
use crate::context::ids::RoundId;
use crate::ports::gateway::StreamDelta;
use crate::ports::tool::ToolExecutionOutcome;

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

    /// Batch decision gate (Slice 7): called after the tool-use hook has
    /// filtered the model-emitted batch and before executor dispatch.
    /// Default [`BatchDecision::Proceed`] — the literal absence of
    /// opinion. Returning [`BatchDecision::Pause`] suspends the turn
    /// (`TurnResult::Paused`, context left open) until the host resumes
    /// it through `resume_turn` (agent-runtime) with the withheld
    /// `HookOutcome` — approve is passthrough, reject is all-rejected,
    /// rewrite is the edited batch; the HookOutcome constructors cover
    /// every decision, so no separate resume-decision enum exists.
    async fn decide_batch(&self, _calls: &[ToolCallPayload]) -> BatchDecision {
        BatchDecision::Proceed
    }

    /// Steering inputs (Slice 7), pulled by the driver at every round
    /// boundary before the frame materializes; each non-empty entry is
    /// appended to the active turn with the `user.steering` source label
    /// and the next model round sees it. Default: empty (zero-cost
    /// pull). The queue itself is host vocabulary — the kernel holds no
    /// channel type.
    async fn pending_inputs(&self) -> Vec<TextPayload> {
        Vec::new()
    }
}

/// The driver's action for a tool-use batch (Slice 7).
#[derive(Debug, Clone)]
pub enum BatchDecision {
    /// Dispatch the batch to the executor unchanged.
    Proceed,
    /// Skip execution entirely; the supplied outcomes become the tool
    /// results (the host supplies the error copy). Outcome `call_id`s
    /// must pair with the batch — the fact machine rejects unpaired
    /// results.
    Reject {
        /// Outcomes standing in for the skipped batch; their `call_id`s must
        /// pair with the batch.
        results: Vec<ToolExecutionOutcome>,
    },
    /// Execute the rewritten payloads instead of the model-emitted ones.
    /// Call ids are expected to be preserved so results pair with the
    /// committed tool-call blocks.
    Rewrite(Vec<ToolCallPayload>),
    /// Suspend the turn: context stays open, the batch is neither
    /// executed nor rejected, and the outcome carries
    /// `TurnResult::Paused` with the model-emitted calls as
    /// `pending_calls`. `deadline` is the advisory decision budget the
    /// host grants itself (remaining time; host-side anchoring).
    Pause {
        /// Advisory remaining decision budget the host grants itself
        /// (host-side anchoring).
        deadline: Option<std::time::Duration>,
    },
}
