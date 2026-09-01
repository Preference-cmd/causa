//! Serialization round-trip tests for outcome types (Slice 5A Phase A).
//!
//! Verifies that the kernel's terminal outcome values (TurnResult,
//! TurnOutcome, ConversationOutcome, and ConversationState) survive a
//! JSON round-trip. The in-memory TurnContext is serialized through its
//! snapshot projection - the wire shape never carries the live mutable
//! state machine.
//!
//! These tests pin the serde tag discipline: adding a new
//! TurnInterruption variant or a new ConversationState field must keep
//! the derive (Serialize, Deserialize) honest.

mod common;

use common::{commit_sealed, endturn_output, turn_id};
use reimagine_context_kernel::{
    ConversationId, ConversationState, ModelInvokeErrorKind, ModelStopReason, SealedResult,
    TextPayload, TurnContext, TurnId, TurnInterruption, TurnOutcome, TurnResult,
};
use serde_json::json;

#[test]
fn turn_result_completed_round_trip() {
    let original = TurnResult::Completed {
        final_output: endturn_output("hello"),
    };
    let value = serde_json::to_value(&original).expect("serialize");
    // Default serde external tagging: `{"Completed": { "final_output": ... }}`.
    assert!(value.get("Completed").is_some());
    assert!(
        value
            .get("Completed")
            .and_then(|c| c.get("final_output"))
            .is_some()
    );
    let restored: TurnResult = serde_json::from_value(value).expect("deserialize");
    assert_eq!(
        serde_json::to_value(&restored).expect("re-serialize"),
        serde_json::to_value(&original).expect("serialize original")
    );
}

#[test]
fn turn_result_interrupted_retry_exhausted_round_trip() {
    let original = TurnResult::Interrupted {
        cause: TurnInterruption::RetryExhausted {
            last_kind: ModelInvokeErrorKind::Transient,
            last_error: "boom".into(),
        },
    };
    let json = serde_json::to_string(&original).expect("serialize");
    // The wire format is serde's external default; the test pins the
    // round-trip equivalence rather than the exact shape (the shape is
    // covered by the explicit TurnInterruption JSON below).
    let restored: TurnResult = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        serde_json::to_string(&restored).expect("re-serialize"),
        json
    );
    // The inner TurnInterruption round-trips intact (its own serde tag
    // discipline is asserted by `turn_interruption_max_tokens_variant_round_trip`).
    match &restored {
        TurnResult::Interrupted {
            cause:
                TurnInterruption::RetryExhausted {
                    last_kind,
                    last_error,
                },
        } => {
            assert_eq!(*last_kind, ModelInvokeErrorKind::Transient);
            assert_eq!(last_error, "boom");
        }
        other => panic!("expected RetryExhausted, got {other:?}"),
    }
}

#[test]
fn turn_outcome_round_trip_preserves_snapshot() {
    let mut context = TurnContext::new(turn_id("t-out"));
    context
        .append_input(TextPayload::new("user said hi"), "user")
        .expect("append input");
    let outcome = TurnOutcome {
        context,
        result: TurnResult::Interrupted {
            cause: TurnInterruption::CompactionFailed {
                reason: "test reason".into(),
            },
        },
        trace: reimagine_context_kernel::TurnTrace::new(),
    };
    let json = serde_json::to_string(&outcome).expect("serialize");
    let restored: TurnOutcome = serde_json::from_str(&json).expect("deserialize");
    let restored_json = serde_json::to_string(&restored).expect("re-serialize");
    assert_eq!(json, restored_json);
    assert!(restored.context.is_sealed());
    assert_eq!(restored.context.turn_id(), turn_id("t-out"));
    assert_eq!(restored.context.blocks().len(), 1);
}

#[test]
fn conversation_outcome_round_trip_preserves_history() {
    let mut state = ConversationState::new(ConversationId("conv-rt".into()));
    commit_sealed(&mut state, "t1", SealedResult::Completed);

    let outcome = reimagine_context_kernel::ConversationOutcome {
        state,
        result: TurnResult::Completed {
            final_output: endturn_output("done"),
        },
        trace: reimagine_context_kernel::TurnTrace::new(),
    };
    let json = serde_json::to_string(&outcome).expect("serialize");
    let restored: reimagine_context_kernel::ConversationOutcome =
        serde_json::from_str(&json).expect("deserialize");
    let restored_json = serde_json::to_string(&restored).expect("re-serialize");
    assert_eq!(json, restored_json);
    assert_eq!(restored.state.snapshot_count(), 1);
    assert_eq!(
        restored.state.conversation_id(),
        &ConversationId("conv-rt".into())
    );
}

#[test]
fn conversation_state_round_trip_with_sealed_active() {
    let mut state = ConversationState::new(ConversationId("conv-active".into()));
    state.begin_turn(TurnId::new("a")).expect("begin");
    state
        .active_turn_mut()
        .expect("active")
        .append_input(TextPayload::new("payload"), "user")
        .expect("append");
    state
        .seal_turn(TurnId::new("a"), SealedResult::Interrupted)
        .expect("seal");

    let json = serde_json::to_string(&state).expect("serialize");
    let restored: ConversationState = serde_json::from_str(&json).expect("deserialize");
    let active = restored.active_turn().expect("active slot preserved");
    assert_eq!(active.turn_id(), TurnId::new("a"));
    assert!(active.is_sealed());
    assert_eq!(active.blocks().len(), 1);
}

#[test]
fn sealed_result_round_trip_variants() {
    for variant in [SealedResult::Completed, SealedResult::Interrupted] {
        let v = serde_json::to_value(variant).expect("serialize");
        let r: SealedResult = serde_json::from_value(v.clone()).expect("deserialize");
        assert_eq!(serde_json::to_value(r).expect("re-serialize"), v);
    }
}

#[test]
fn turn_interruption_max_tokens_variant_round_trip() {
    let cause = TurnInterruption::MaxModelRounds { limit: 7 };
    let value = serde_json::to_value(&cause).expect("serialize");
    assert_eq!(
        value.get("kind").and_then(|v| v.as_str()),
        Some("MaxModelRounds")
    );
    let restored: TurnInterruption = serde_json::from_value(value).expect("deserialize");
    assert_eq!(
        serde_json::to_value(&restored).expect("re-serialize"),
        json!({"kind":"MaxModelRounds","detail":{"limit":7}})
    );
}

#[test]
fn model_stop_reason_serialization_is_stable() {
    let r = ModelStopReason::EndTurn;
    let v = serde_json::to_value(r).expect("serialize");
    assert_eq!(v, json!("end_turn"));
    let restored: ModelStopReason = serde_json::from_value(v).expect("deserialize");
    assert_eq!(restored, ModelStopReason::EndTurn);
}
