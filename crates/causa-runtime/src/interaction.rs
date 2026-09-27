//! Host observations during a turn.
//!
//! Turn interaction retains streaming observation and caller-provided input
//! injection. Tool-call decisions belong to ordinary borrowed batch
//! processors in the runtime chain.

use async_trait::async_trait;

use causa_kernel::{RoundId, StreamDelta, TextPayload};

/// Host observations while a turn is in flight.
#[async_trait]
pub trait TurnInteraction: Send + Sync {
    /// Streaming-delta observation. Called once per provider delta in the
    /// model phase, with its round attribution.
    async fn on_delta(&self, _round_id: RoundId, _delta: &StreamDelta) {}

    /// Inputs pulled at each round boundary before the frame materializes.
    /// Non-empty entries are appended to the active turn with the
    /// `user.steering` source label. Default: empty.
    async fn pending_inputs(&self) -> Vec<TextPayload> {
        Vec::new()
    }
}
