//! Serialization round-trip tests for outcome types.
//!
//! Verifies that the kernel's terminal outcome values (TurnResult,
//! TurnOutcome, ConversationOutcome, and ConversationState) survive a
//! JSON round-trip. The in-memory TurnContext is serialized through its
//! snapshot projection - the wire shape never carries the live mutable
//! state machine.
//!
//! These tests pin the serde tag discipline: adding a new
//! TurnInterruption variant or a new ConversationState field must keep
//! the wire honest (ConversationState reloads through the validating
//! manual Deserialize impl, not a bare derive).

mod common;

use causa_kernel::{
    ConversationId, ModelInvokeErrorKind, ModelStopReason, TextPayload, TurnContext, TurnId,
};
use causa_runtime::{
    Continuation, ConversationOutcome, ConversationState, PausePoint, PreparedApproval,
    SealedResult, TurnInterruption, TurnOutcome, TurnResult, TurnTrace,
};
use common::{commit_sealed, endturn_output, turn_id};
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
    // The wire preserves sealedness: terminal outcomes carry a sealed
    // context, and the round-trip must restore it sealed.
    context.seal();
    let outcome = TurnOutcome {
        context,
        result: TurnResult::Interrupted {
            cause: TurnInterruption::CompactionFailed {
                reason: "test reason".into(),
            },
        },
        trace: TurnTrace::new(),
    };
    let json = serde_json::to_string(&outcome).expect("serialize");
    let restored: TurnOutcome = serde_json::from_str(&json).expect("deserialize");
    let restored_json = serde_json::to_string(&restored).expect("re-serialize");
    assert_eq!(json, restored_json);
    assert!(restored.context.is_sealed());
    assert_eq!(restored.context.turn_id(), turn_id("t-out"));
    assert_eq!(restored.context.blocks().len(), 1);
}

/// A paused outcome carries the open context plus the single continuation —
/// the round-trip restores both. The Paused variant carries NO snapshot of
/// its own: the outcome's `context` is the only fact source.
#[test]
fn turn_outcome_paused_round_trip_preserves_open_context_and_continuation() {
    let mut context = TurnContext::new(turn_id("t-paused"));
    context
        .append_input(TextPayload::new("user said hi"), "user")
        .expect("append input");
    assert!(!context.is_sealed(), "paused turns stay open");
    let continuation = Continuation {
        pause_point: PausePoint::AwaitingApproval {
            prepared: PreparedApproval {
                awaiting: vec![causa_kernel::ToolCallPayload {
                    call_id: causa_kernel::ToolCallId("call-1".into()),
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                }],
                rejected: vec![],
                unknown_decisions: vec![],
            },
            deadline: Some(std::time::Duration::from_secs(30)),
        },
        round: 0,
        accounted_tool_calls: 1,
        queued_inputs: vec![],
    };
    let outcome = TurnOutcome {
        context,
        result: TurnResult::Paused { continuation },
        trace: TurnTrace::new(),
    };
    let json = serde_json::to_string(&outcome).expect("serialize");
    let restored: TurnOutcome = serde_json::from_str(&json).expect("deserialize");
    let restored_json = serde_json::to_string(&restored).expect("re-serialize");
    assert_eq!(json, restored_json);
    assert!(!restored.context.is_sealed());
    match restored.result {
        TurnResult::Paused { continuation } => {
            match continuation.pause_point {
                PausePoint::AwaitingApproval { prepared, deadline } => {
                    assert_eq!(prepared.awaiting.len(), 1);
                    assert_eq!(prepared.awaiting[0].call_id.0, "call-1");
                    assert_eq!(deadline, Some(std::time::Duration::from_secs(30)));
                }
                other => panic!("expected AwaitingApproval, got {other:?}"),
            }
            assert_eq!(continuation.round, 0);
            assert_eq!(continuation.accounted_tool_calls, 1);
        }
        other => panic!("expected Paused, got {other:?}"),
    }
    // The wire carries no snapshot field on the Paused variant.
    assert!(!json.contains(r#""snapshot""#), "{json}");
}

#[test]
fn pause_point_tags_are_pinned() {
    let steering = Continuation {
        pause_point: PausePoint::PausedForSteering,
        round: 2,
        accounted_tool_calls: 0,
        queued_inputs: vec![TextPayload::new("wait")],
    };
    let value = serde_json::to_value(&steering.pause_point).expect("serialize");
    assert_eq!(
        value.get("kind").and_then(|v| v.as_str()),
        Some("paused_for_steering")
    );
    // Queued inputs default-and-skip: an empty queue is absent on the wire.
    let approval = Continuation {
        pause_point: PausePoint::AwaitingApproval {
            prepared: PreparedApproval {
                awaiting: vec![],
                rejected: vec![],
                unknown_decisions: Vec::new(),
            },
            deadline: None,
        },
        round: 0,
        accounted_tool_calls: 0,
        queued_inputs: vec![],
    };
    let value = serde_json::to_value(&approval).expect("serialize");
    assert!(
        !value.to_string().contains("queued_inputs"),
        "empty queued inputs must be skipped: {value}"
    );
    assert_eq!(
        value["pause_point"]["kind"].as_str(),
        Some("awaiting_approval")
    );
}

/// Pre-continuation pause payloads (snapshot + reason riding on the
/// variant) do NOT transparently convert — that shape discarded the hook's
/// prepared work, so no equivalent continuation can be constructed.
/// Deserialization is the explicit rejection point.
#[test]
fn pre_continuation_paused_payloads_are_explicitly_rejected() {
    let old = json!({
        "Paused": {
            "snapshot": {
                "turn_id": "t-old",
                "turn_sequence": 0,
                "blocks": [],
                "source_version": 1,
                "sealed": false
            },
            "reason": {
                "kind": "awaiting_approval",
                "detail": {"pending_calls": [], "deadline": null}
            }
        }
    });
    let err = serde_json::from_value::<TurnResult>(old)
        .expect_err("old pause material must not silently deserialize");
    assert!(err.to_string().contains("continuation"), "{err}");
}

#[test]
fn conversation_outcome_round_trip_preserves_history() {
    let mut state = ConversationState::new(ConversationId("conv-rt".into()));
    commit_sealed(&mut state, "t1", SealedResult::Completed);

    let outcome = ConversationOutcome {
        state,
        result: TurnResult::Completed {
            final_output: endturn_output("done"),
        },
        trace: TurnTrace::new(),
    };
    let json = serde_json::to_string(&outcome).expect("serialize");
    let restored: ConversationOutcome = serde_json::from_str(&json).expect("deserialize");
    let restored_json = serde_json::to_string(&restored).expect("re-serialize");
    assert_eq!(json, restored_json);
    assert_eq!(restored.state.history_len(), 1);
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
fn conversation_state_load_rejects_tampered_block_sequence() {
    let mut state = ConversationState::new(ConversationId("conv-f7".into()));
    commit_sealed(&mut state, "t1", SealedResult::Completed);

    let mut value = serde_json::to_value(&state).expect("serialize");
    serde_json::from_value::<ConversationState>(value.clone())
        .expect("untampered payload still loads");
    value["history"][0]["snapshot"]["blocks"][0]["sequence"] = json!(42);
    let err = serde_json::from_value::<ConversationState>(value)
        .expect_err("tampered block sequence must not load");
    assert!(
        err.to_string().contains("invalid conversation state"),
        "{err}"
    );
}

#[test]
fn conversation_state_load_rejects_non_monotonic_entry_sequence() {
    let mut state = ConversationState::new(ConversationId("conv-f7-seq".into()));
    commit_sealed(&mut state, "t1", SealedResult::Completed);
    commit_sealed(&mut state, "t2", SealedResult::Completed);

    let mut value = serde_json::to_value(&state).expect("serialize");
    value["history"][1]["sequence"] = json!(0);
    let err = serde_json::from_value::<ConversationState>(value)
        .expect_err("non-monotonic turn sequence must not load");
    assert!(err.to_string().contains("strictly increasing"), "{err}");
}

#[test]
fn conversation_state_load_rejects_active_turn_duplicating_committed_id() {
    let mut state = ConversationState::new(ConversationId("conv-f7-dup".into()));
    commit_sealed(&mut state, "t1", SealedResult::Completed);
    state.begin_turn(TurnId::new("a")).expect("begin");

    let mut value = serde_json::to_value(&state).expect("serialize");
    value["active_turn"]["turn_id"] = json!("t1");
    let err = serde_json::from_value::<ConversationState>(value)
        .expect_err("active turn duplicating a committed id must not load");
    assert!(err.to_string().contains("duplicates a committed"), "{err}");
}

#[test]
fn sealed_result_round_trip_variants() {
    for variant in [
        SealedResult::Completed,
        SealedResult::Interrupted,
        SealedResult::Paused,
    ] {
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

// ---- the {result, policy} checkpoint wire --------------------------------------

use causa_kernel::{ToolCallId, ToolOutput, ToolResultPayload, ToolResultStatus};
use causa_runtime::{UnknownDecision, UnknownOutcomePolicy};

fn unknown_result(call: &str) -> ToolResultPayload {
    ToolResultPayload {
        call_id: ToolCallId(call.into()),
        status: ToolResultStatus::UnknownOutcome,
        output: ToolOutput::new(json!({"unk": call})),
        media: Vec::new(),
    }
}

fn decided_result(call: &str) -> ToolResultPayload {
    ToolResultPayload {
        call_id: ToolCallId(call.into()),
        status: ToolResultStatus::Rejected,
        output: ToolOutput::new(json!({"denied": call})),
        media: Vec::new(),
    }
}

/// The old `{result, policy}` envelope: an UnknownOutcome entry writes its
/// saved decision as the policy, every other status writes the canonical
/// Stop — and loading restores exactly the saved actions, never a
/// guessed default.
#[test]
fn checkpoint_wire_round_trips_saved_unknown_actions() {
    let prepared = PreparedApproval {
        awaiting: vec![],
        rejected: vec![unknown_result("call-unk"), decided_result("call-rej")],
        unknown_decisions: vec![UnknownDecision {
            call_id: ToolCallId("call-unk".into()),
            policy: UnknownOutcomePolicy::Continue,
        }],
    };
    let value = serde_json::to_value(&prepared).expect("serialize");
    assert_eq!(value["rejected"][0]["policy"], "Continue");
    assert_eq!(value["rejected"][1]["policy"], "Stop");
    let restored: PreparedApproval = serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(
        restored.unknown_decisions,
        vec![UnknownDecision {
            call_id: ToolCallId("call-unk".into()),
            policy: UnknownOutcomePolicy::Continue,
        }]
    );
    assert_eq!(
        serde_json::to_value(&restored.rejected).unwrap(),
        serde_json::to_value(&prepared.rejected).unwrap()
    );
    assert_eq!(serde_json::to_value(&restored).unwrap(), value);
}

/// Broken decision sets are hard deserialize errors — a missing policy on
/// an UnknownOutcome entry is never guessed into a default Stop, a
/// duplicated call is refused, and a non-unknown entry's stray Continue
/// policy is dropped (never consumed) and re-canonicalized to Stop.
#[test]
fn checkpoint_wire_rejects_broken_or_ambiguous_decision_sets() {
    let base = || PreparedApproval {
        awaiting: vec![],
        rejected: vec![unknown_result("call-unk"), decided_result("call-rej")],
        unknown_decisions: vec![UnknownDecision {
            call_id: ToolCallId("call-unk".into()),
            policy: UnknownOutcomePolicy::Continue,
        }],
    };

    // A saved UnknownOutcome without its policy field: hard error, never
    // a silent Stop.
    let mut missing = serde_json::to_value(base()).unwrap();
    missing["rejected"][0]
        .as_object_mut()
        .unwrap()
        .remove("policy");
    let err =
        serde_json::from_value::<PreparedApproval>(missing).expect_err("missing policy must fail");
    assert!(err.to_string().contains("policy"), "{err}");

    // A stray Continue on a decided (non-unknown) entry: loaded, dropped,
    // and re-canonicalized to Stop on write.
    let mut stray = serde_json::to_value(base()).unwrap();
    stray["rejected"][1]["policy"] = json!("Continue");
    let restored: PreparedApproval = serde_json::from_value(stray).expect("stray policy dropped");
    assert!(
        restored
            .unknown_decisions
            .iter()
            .all(|d| d.call_id.0 == "call-unk"),
        "no decision may come from a non-unknown entry"
    );
    assert_eq!(
        serde_json::to_value(&restored).unwrap()["rejected"][1]["policy"],
        "Stop"
    );

    // The same call saved twice among the results: refused.
    let mut dup = serde_json::to_value(base()).unwrap();
    let rej = dup["rejected"][1].clone();
    dup["rejected"].as_array_mut().unwrap().push(rej);
    let err =
        serde_json::from_value::<PreparedApproval>(dup).expect_err("duplicate call must fail");
    assert!(err.to_string().contains("twice"), "{err}");

    // Two UnknownOutcome entries sharing one call id: conflicting
    // decisions, refused.
    let conflicting = PreparedApproval {
        awaiting: vec![],
        rejected: vec![unknown_result("call-unk"), unknown_result("call-unk")],
        unknown_decisions: vec![
            UnknownDecision {
                call_id: ToolCallId("call-unk".into()),
                policy: UnknownOutcomePolicy::Stop,
            },
            UnknownDecision {
                call_id: ToolCallId("call-unk".into()),
                policy: UnknownOutcomePolicy::Continue,
            },
        ],
    };
    let err =
        serde_json::from_value::<PreparedApproval>(serde_json::to_value(&conflicting).unwrap())
            .expect_err("conflicting decisions must fail");
    assert!(err.to_string().contains("conflicting"), "{err}");
}
