//! Stable serialization boundaries: turn results and the validated
//! conversation fact aggregate.

mod common;

use causa_kernel::{BlockContent, ConversationId, ModelInvokeErrorKind, TextPayload, TurnId};
use causa_runtime::{ConversationState, SealedResult, TurnInterruption, TurnResult, new_block_id};
use common::{commit_sealed, endturn_output};
use serde_json::json;

#[test]
fn completed_turn_result_round_trips() {
    let original = TurnResult::Completed {
        final_output: endturn_output("hello"),
    };
    let value = serde_json::to_value(&original).expect("serialize");
    assert!(value.get("Completed").is_some());
    let restored: TurnResult = serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(serde_json::to_value(restored).unwrap(), value);
}

#[test]
fn interrupted_turn_result_round_trips_with_explicit_cause() {
    let original = TurnResult::Interrupted {
        cause: TurnInterruption::RetryExhausted {
            last_kind: ModelInvokeErrorKind::Transient,
            last_error: "gateway unavailable".into(),
        },
    };
    let value = serde_json::to_value(&original).unwrap();
    let restored: TurnResult = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(restored).unwrap(), value);
}

#[test]
fn interruption_tag_and_identity_are_serialized() {
    let id = new_block_id();
    let cause = TurnInterruption::UnsafeUnknownOutcome { call_block_id: id };
    let value = serde_json::to_value(&cause).unwrap();
    assert_eq!(value["kind"], "UnsafeUnknownOutcome");
    assert_eq!(value["detail"]["call_block_id"], json!(id));
    assert_eq!(
        serde_json::from_value::<TurnInterruption>(value).unwrap(),
        cause
    );
}

#[test]
fn conversation_state_round_trip_rebuilds_validated_facts() {
    let mut state = ConversationState::new(ConversationId("conv-rt".into()));
    commit_sealed(&mut state, "t1", SealedResult::Completed);
    state.begin_turn(TurnId::new("t2")).unwrap();
    let active_id = new_block_id();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(active_id, TextPayload::new("active"), "user")
        .unwrap();

    let value = serde_json::to_value(&state).unwrap();
    let restored: ConversationState = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&restored).unwrap(), value);
    assert_eq!(restored.history_len(), 1);
    assert_eq!(restored.active_turn().unwrap().blocks()[0].id, active_id);
    assert!(matches!(
        &restored.active_turn().unwrap().blocks()[0].content,
        BlockContent::Parts(_)
    ));
}

#[test]
fn conversation_state_deserialization_rejects_duplicate_block_identity() {
    let mut state = ConversationState::new(ConversationId("conv-duplicate".into()));
    state.begin_turn(TurnId::new("t1")).unwrap();
    let id = new_block_id();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(id, TextPayload::new("first"), "user")
        .unwrap();
    let second = new_block_id();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(second, TextPayload::new("second"), "user")
        .unwrap();

    let mut value = serde_json::to_value(&state).unwrap();
    value["active_turn"]["blocks"][1]["id"] = json!(id);
    assert!(serde_json::from_value::<ConversationState>(value).is_err());
}
