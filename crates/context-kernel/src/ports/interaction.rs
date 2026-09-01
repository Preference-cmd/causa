//! `TurnInteraction` port — the single host↔driver interaction boundary
//! during a turn (Slice 6, chartered by Slice 12 Decision 3).
//!
//! One port with default no-ops replaces per-entry callback channels:
//! entry signatures never grow interaction parameters, and new
//! interactions (approval decisions, steering inputs — Slice 7; progress,
//! per-call approval) arrive as new default methods without breaking
//! third-party implementors. The driver consumes exactly one of these
//! per turn via `TurnRunOptions.interaction` (agent-runtime).
//!
//! The kernel defines the contract only; the noop default
//! (`NoopInteraction`) lives with the driver's config axes in
//! `agent-runtime`, mirroring where the hook policies live.

use async_trait::async_trait;

use crate::context::ids::RoundId;
use crate::ports::gateway::StreamDelta;

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
}
