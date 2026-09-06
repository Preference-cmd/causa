//! Interaction-port tests — the `TurnInteraction` contract moved here from
//! the kernel (Slice 13): the seam's only consumer is the reference driver,
//! so the contract rides with the runtime. Default methods are no-ops, and
//! a host implementation only overrides what it observes.

use async_trait::async_trait;
use causa_kernel::{RoundId, StreamDelta};
use causa_runtime::{BatchDecision, TurnInteraction};
use std::sync::{Arc, Mutex};

/// A host-side observer counting what it sees through the port.
struct CountingInteraction {
    text_deltas: Mutex<usize>,
}

#[async_trait]
impl TurnInteraction for CountingInteraction {
    async fn on_delta(&self, _round_id: RoundId, delta: &StreamDelta) {
        if matches!(delta, StreamDelta::TextDelta { .. }) {
            *self.text_deltas.lock().unwrap() += 1;
        }
    }
}

#[tokio::test]
async fn turn_interaction_default_methods_are_noop_and_implementable() {
    let interaction = Arc::new(CountingInteraction {
        text_deltas: Mutex::new(0),
    });
    // Default no-op: the trait's own methods are callable on Arc<dyn _>.
    let noop: Arc<dyn TurnInteraction> = interaction.clone();
    noop.on_delta(
        RoundId(0),
        &StreamDelta::ReasoningDelta {
            delta: "thinking".into(),
        },
    )
    .await;
    interaction
        .on_delta(
            RoundId(3),
            &StreamDelta::TextDelta {
                delta: "token".into(),
            },
        )
        .await;
    assert_eq!(*interaction.text_deltas.lock().unwrap(), 1);
    // The batch gate and steering pull default to the absence of opinion.
    let decision = noop.decide_batch(&[]).await;
    assert!(matches!(decision, BatchDecision::Proceed));
    assert!(noop.pending_inputs().await.is_empty());
}
