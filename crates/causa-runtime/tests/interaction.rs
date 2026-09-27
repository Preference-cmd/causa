//! Interaction-port tests — the `TurnInteraction` contract lives with the
//! runtime because the execution stack is its only consumer. Default
//! methods are no-ops; a host implementation overrides only what it observes.

use async_trait::async_trait;
use causa_kernel::{RoundId, StreamDelta};
use causa_runtime::TurnInteraction;
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
    // The steering pull defaults to no injected inputs.
    assert!(noop.pending_inputs().await.is_empty());
}

struct InputInteraction;

#[async_trait]
impl TurnInteraction for InputInteraction {
    async fn pending_inputs(&self) -> Vec<causa_kernel::TextPayload> {
        vec![causa_kernel::TextPayload::new("injected")]
    }
}

#[tokio::test]
async fn interaction_can_supply_round_boundary_inputs() {
    let inputs = InputInteraction.pending_inputs().await;
    assert_eq!(inputs, vec![causa_kernel::TextPayload::new("injected")]);
}
