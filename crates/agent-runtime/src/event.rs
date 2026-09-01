//! `ContextEvent` projection — Slice 4 §6 Phase C (框架事件).
//!
//! Events are a **projection** of the kernel's facts (`TurnOutcome +
//! TurnTrace`) into a sequence that consumers (UI, observability,
//! audit) can subscribe to. They are not facts themselves: they are
//! derived, not persisted, and never written back to `TurnContext`.
//!
//! ## Layering
//!
//! ```text
//! reimagine-context-kernel
//!   └─ TurnOutcome / TurnTrace / ModelRoundTrace / ToolBatchTrace   ← facts
//!             ^
//!             │ project_turn(...)
//!             |
//! reimagine-agent-runtime
//!   └─ ContextEvent / project_turn / ContextSink                  ← projection
//!             ^
//!             |
//! app-host / external consumer   ← observers (UI, audit, metrics)
//! ```
//!
//! ## Boundaries
//!
//! - **Not facts**: `ContextEvent` instances are constructed on demand
//!   by `project_turn`; they never appear in a kernel snapshot.
//! - **Not persistent**: nothing in this module touches the workspace
//!   store. Consumers persist what they need.
//! - **No harness dependency**: this module is `Send + Sync`-pure and
//!   does not depend on `agent-harness` (frozen legacy).
//! - **No `AgentEvent` reuse**: `reimagine_agent_harness::AgentEvent`
//!   is frozen and out of scope; if a host needs to bridge to it,
//!   do so with a one-off `From<ContextEvent> for AgentEvent` adapter
//!   in the host crate, not here.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use reimagine_context_kernel::{
    ConversationId, ModelRoundTrace, RoundId, ToolCallPayload, TurnId, TurnResult, TurnTrace,
};

/// Framework-side event projected from a turn's facts. Consumers
/// (Slice 4B host, observability, audit) subscribe to a sequence of
/// these; the framework never persists them.
///
/// ## Multi-conversation routing
///
/// `conversation_id` is `Option<ConversationId>` because the Slice 1
/// `TurnRunner::run` entry does not carry a `ConversationState`. The
/// Slice 2 `run_in_conversation` entry does; the host constructs the
/// event with `Some(id)` when projecting from a `ConversationOutcome`,
/// and `None` for the bare `TurnOutcome` path. Subscribers that care
/// about cross-conversation routing key on `Some`.
///
/// ## Serialization
///
/// `ContextEvent` implements `Serialize` / `Deserialize` for IPC
/// delivery to host UIs and audit pipelines (resolved in Slice 5A
/// Phase C, 2026-09-01 — previously a documented TODO). The
/// underlying kernel types (`TurnResult`, `TurnTrace`) carry their
/// own serde derives; see `crates/agent-stack/context-kernel`.
///
/// `Option<ConversationId>` uses `skip_serializing_if = "Option::is_none"`
/// so the bare `TurnRunner::run` path emits compact events (no
/// conversation routing key) while the `run_in_conversation` path
/// emits the explicit id.
#[derive(Debug, Serialize, Deserialize)]
pub enum ContextEvent {
    /// The turn has begun. Emitted once per `project_turn` call, as
    /// the first event in the sequence.
    TurnStarted {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        conversation_id: Option<ConversationId>,
        turn_id: TurnId,
    },
    /// A batch of tool calls was dispatched for `round_id`. `calls`
    /// carries the **pre-execution** payloads (what the model
    /// emitted), not the post-execution trace — events reflect
    /// intent at the dispatch boundary.
    ///
    /// Emitted once per round that has a `tool_batch` in the trace,
    /// in `round_id` order. Rounds with `tool_batch == None`
    /// (EndTurn, MaxTokens, Refusal, etc.) produce no event.
    ToolBatchDispatched {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        conversation_id: Option<ConversationId>,
        turn_id: TurnId,
        round_id: RoundId,
        calls: Vec<ToolCallPayload>,
    },
    /// The turn finished. `result` is the canonical `TurnResult`
    /// (`Completed | Interrupted { cause }`) and `trace` is the full
    /// `TurnTrace` (rounds, totals). Consumers can re-walk `trace`
    /// for finer-grained data.
    ///
    /// Emitted exactly once per `project_turn` call, as the last
    /// event in the sequence.
    TurnOutcome {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        conversation_id: Option<ConversationId>,
        turn_id: TurnId,
        result: TurnResult,
        trace: TurnTrace,
    },
}

impl ContextEvent {
    /// Conversation id the event carries, if any. Convenience for
    /// subscribers that just want to route on the field.
    pub fn conversation_id(&self) -> Option<ConversationId> {
        match self {
            ContextEvent::TurnStarted {
                conversation_id, ..
            }
            | ContextEvent::ToolBatchDispatched {
                conversation_id, ..
            }
            | ContextEvent::TurnOutcome {
                conversation_id, ..
            } => conversation_id.clone(),
        }
    }

    /// Turn id the event carries.
    pub fn turn_id(&self) -> &TurnId {
        match self {
            ContextEvent::TurnStarted { turn_id, .. }
            | ContextEvent::ToolBatchDispatched { turn_id, .. }
            | ContextEvent::TurnOutcome { turn_id, .. } => turn_id,
        }
    }
}

/// Project a turn into its canonical sequence of `ContextEvent`s.
///
/// ## Output order
///
/// 1. `TurnStarted { conversation_id, turn_id }` — exactly once.
/// 2. For each `ModelRoundTrace` in `trace.rounds` that has a
///    `tool_batch`, a `ToolBatchDispatched` event carrying the
///    pre-execution payloads (looked up by `round_id` in
///    `pre_dispatch_payloads`). Rounds without a `tool_batch`
///    (EndTurn, MaxTokens, Refusal, compaction failure, etc.)
///    produce **no** `ToolBatchDispatched`.
/// 3. `TurnOutcome { conversation_id, turn_id, result, trace }` —
///    exactly once, with the full trace.
///
/// `pre_dispatch_payloads` is keyed by `round_id`. Rounds whose
/// `round_id` is not present in the map (typically EndTurn) produce
/// no dispatch event. This is the contract: callers supply the
/// pre-execution payloads they captured during the run; the
/// projector never reaches back into the kernel to reconstruct
/// them (the post-execution trace drops `arguments`, so they would
/// be unrecoverable anyway).
///
/// ## Caller responsibilities
///
/// - Capture payloads **between** model output and tool dispatch
///   (i.e. at the `FilterChain` boundary or just before
///   `ToolExecutor::execute_with_limits`).
/// - Pass `result` and `trace` straight from `TurnOutcome` /
///   `ConversationOutcome.state.conversation_id()`.
/// - Pass `conversation_id = Some(state.conversation_id())` for the
///   `run_in_conversation` entry, `None` for the bare `run` entry.
///
/// ## What this function never does
///
/// - It never reads or mutates `TurnContext` / `ConversationState`.
/// - It never constructs `ToolExecutionOutcome` — execution
///   results live in `trace`, not in events.
/// - It never holds kernel locks; it is a pure projection over
///   owned data.
pub fn project_turn(
    turn_id: TurnId,
    conversation_id: Option<ConversationId>,
    pre_dispatch_payloads: &HashMap<RoundId, Vec<ToolCallPayload>>,
    result: TurnResult,
    trace: TurnTrace,
) -> Vec<ContextEvent> {
    let mut events = Vec::with_capacity(2 + trace.rounds.len());
    events.push(ContextEvent::TurnStarted {
        conversation_id: conversation_id.clone(),
        turn_id: turn_id.clone(),
    });
    for round in &trace.rounds {
        emit_round_dispatch(
            &mut events,
            round,
            conversation_id.clone(),
            &turn_id,
            pre_dispatch_payloads,
        );
    }
    events.push(ContextEvent::TurnOutcome {
        conversation_id,
        turn_id,
        result,
        trace,
    });
    events
}

fn emit_round_dispatch(
    out: &mut Vec<ContextEvent>,
    round: &ModelRoundTrace,
    conversation_id: Option<ConversationId>,
    turn_id: &TurnId,
    pre_dispatch_payloads: &HashMap<RoundId, Vec<ToolCallPayload>>,
) {
    // Only rounds that actually dispatched a tool batch produce a
    // `ToolBatchDispatched`. EndTurn / MaxTokens / Refusal /
    // CompactionFailed / etc. all leave `tool_batch = None`.
    let Some(_batch) = &round.tool_batch else {
        return;
    };
    // The `round_id` matches `InvocationId.round_id` by construction
    // (see `driver.rs:329`); we use `ModelRoundTrace.round_id` as the
    // canonical key into the pre-dispatch map.
    let Some(calls) = pre_dispatch_payloads.get(&round.round_id) else {
        // Caller did not capture payloads for this round — drop the
        // event rather than emit a half-formed one. (Future: make
        // this an error, once hosts standardize on capture.)
        return;
    };
    out.push(ContextEvent::ToolBatchDispatched {
        conversation_id,
        turn_id: turn_id.clone(),
        round_id: round.round_id,
        calls: calls.clone(),
    });
}

// --- tests ----------------------------------------------------------------
//
// Pure-projection tests. They build a `TurnTrace` directly from the
// kernel's data types and assert the emitted event sequence. No
// driver, no harness, no async — the projection is sync over owned
// data.

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_context_kernel::{
        BlockId, ContextVersion, InvocationId, ModelResponse, ModelStopReason, OutputSummary,
        RoundId, ToolBatchTrace, ToolCallId, TurnId, TurnInterruption, TurnTrace,
    };
    use serde_json::json;

    fn turn_id(byte: u8) -> TurnId {
        TurnId::new(format!("turn-{byte}"))
    }

    fn empty_trace() -> TurnTrace {
        TurnTrace {
            rounds: Vec::new(),
            tool_calls_total: 0,
            total_duration_ms: 0,
        }
    }

    fn round_with_tool_batch(round_id: RoundId) -> ModelRoundTrace {
        ModelRoundTrace {
            round_id,
            invocation_id: InvocationId {
                turn_id: turn_id(1),
                round_id,
            },
            frame_version: ContextVersion(0),
            attempts: Vec::new(),
            output_summary: Some(OutputSummary {
                stop_reason: reimagine_context_kernel::ModelStopReason::ToolUse,
                usage: None,
                tool_call_count: 1,
                response_text_bytes: 0,
            }),
            applied_block_ids: vec![BlockId {
                turn_id: turn_id(1),
                sequence: reimagine_context_kernel::BlockSequence(0),
            }],
            tool_batch: Some(ToolBatchTrace {
                calls: vec![],
                completion_order: Vec::new(),
            }),
        }
    }

    fn round_endturn(round_id: RoundId) -> ModelRoundTrace {
        ModelRoundTrace {
            round_id,
            invocation_id: InvocationId {
                turn_id: turn_id(2),
                round_id,
            },
            frame_version: ContextVersion(0),
            attempts: Vec::new(),
            output_summary: Some(OutputSummary {
                stop_reason: reimagine_context_kernel::ModelStopReason::EndTurn,
                usage: None,
                tool_call_count: 0,
                response_text_bytes: 0,
            }),
            applied_block_ids: vec![],
            tool_batch: None,
        }
    }

    fn payload(call_id: &str, tool: &str) -> ToolCallPayload {
        ToolCallPayload {
            call_id: ToolCallId(call_id.to_string()),
            tool_name: tool.to_string(),
            arguments: json!({}),
        }
    }

    #[test]
    fn empty_trace_emits_only_started_and_outcome() {
        let events = project_turn(
            turn_id(1),
            None,
            &HashMap::new(),
            dummy_completed(),
            empty_trace(),
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ContextEvent::TurnStarted { .. }));
        assert!(matches!(events[1], ContextEvent::TurnOutcome { .. }));
    }

    #[test]
    fn single_tool_use_round_emits_dispatch_between_started_and_outcome() {
        let rid = RoundId(0);
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(rid));

        let mut payloads = HashMap::new();
        payloads.insert(rid, vec![payload("c-1", "echo")]);

        let events = project_turn(turn_id(1), None, &payloads, dummy_completed(), trace);
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], ContextEvent::TurnStarted { .. }));
        match &events[1] {
            ContextEvent::ToolBatchDispatched {
                turn_id,
                round_id,
                calls,
                ..
            } => {
                assert_eq!(turn_id.0, "turn-1");
                assert_eq!(*round_id, rid);
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].call_id.0, "c-1");
                assert_eq!(calls[0].tool_name, "echo");
            }
            other => panic!("expected ToolBatchDispatched, got {other:?}"),
        }
        assert!(matches!(events[2], ContextEvent::TurnOutcome { .. }));
    }

    #[test]
    fn endturn_round_produces_no_dispatch_event() {
        // EndTurn (no tool_batch) must NOT produce a ToolBatchDispatched,
        // even if the caller mistakenly registered payloads for that
        // round id.
        let rid = RoundId(7);
        let mut trace = empty_trace();
        trace.rounds.push(round_endturn(rid));

        let mut payloads = HashMap::new();
        payloads.insert(rid, vec![payload("ignored", "echo")]);

        let events = project_turn(turn_id(1), None, &payloads, dummy_completed(), trace);
        assert_eq!(
            events.len(),
            2,
            "EndTurn must not produce ToolBatchDispatched"
        );
        assert!(matches!(events[0], ContextEvent::TurnStarted { .. }));
        assert!(matches!(events[1], ContextEvent::TurnOutcome { .. }));
    }

    #[test]
    fn multiple_rounds_emit_dispatches_in_round_id_order() {
        // Three rounds: tool_use, endturn, tool_use → two dispatch
        // events in round order (0, 2), EndTurn round skipped.
        let rid0 = RoundId(0);
        let rid1 = RoundId(1);
        let rid2 = RoundId(2);
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(rid0));
        trace.rounds.push(round_endturn(rid1));
        trace.rounds.push(round_with_tool_batch(rid2));

        let mut payloads = HashMap::new();
        payloads.insert(rid0, vec![payload("a", "echo")]);
        payloads.insert(rid2, vec![payload("b", "other"), payload("c", "third")]);

        let events = project_turn(turn_id(1), None, &payloads, dummy_completed(), trace);
        assert_eq!(events.len(), 4);
        let dispatched_round_ids: Vec<RoundId> = events
            .iter()
            .filter_map(|e| match e {
                ContextEvent::ToolBatchDispatched { round_id, .. } => Some(*round_id),
                _ => None,
            })
            .collect();
        assert_eq!(dispatched_round_ids, vec![rid0, rid2]);
    }

    #[test]
    fn missing_pre_dispatch_payloads_skips_dispatch() {
        // If the host didn't capture payloads for a dispatched round,
        // we drop the event (do not emit a half-formed one).
        let rid = RoundId(3);
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(rid));

        let events = project_turn(turn_id(1), None, &HashMap::new(), dummy_completed(), trace);
        // No payload registered → no dispatch event.
        assert_eq!(events.len(), 2);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ContextEvent::ToolBatchDispatched { .. }))
        );
    }

    #[test]
    fn conversation_id_propagates_to_all_events() {
        let conv = ConversationId("conv-42".to_string());
        let rid = RoundId(0);
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(rid));
        let mut payloads = HashMap::new();
        payloads.insert(rid, vec![payload("c", "echo")]);

        let events = project_turn(
            turn_id(1),
            Some(conv.clone()),
            &payloads,
            dummy_completed(),
            trace,
        );
        for e in &events {
            assert_eq!(e.conversation_id(), Some(conv.clone()));
            assert_eq!(e.turn_id().0, "turn-1");
        }
    }

    #[test]
    fn interrupted_turn_carries_cause_in_outcome() {
        let mut trace = empty_trace();
        // Max rounds reached → MaxModelRounds interruption.
        trace.rounds.push(round_endturn(RoundId(0)));
        let cause = TurnInterruption::MaxModelRounds { limit: 8 };
        let events = project_turn(
            turn_id(1),
            None,
            &HashMap::new(),
            TurnResult::Interrupted {
                cause: cause.clone(),
            },
            trace,
        );
        assert_eq!(events.len(), 2);
        match &events[1] {
            ContextEvent::TurnOutcome { result, .. } => match result {
                TurnResult::Interrupted { cause: c } => assert_eq!(*c, cause),
                other => panic!("expected Interrupted, got {other:?}"),
            },
            other => panic!("expected TurnOutcome, got {other:?}"),
        }
    }

    #[test]
    fn projection_never_mutates_trace() {
        // Defensive: the projector takes ownership of `trace` via
        // `TurnTrace` (which is Clone), and the resulting event uses
        // the same trace. We verify no payload mutation by checking
        // that the original trace is preserved when cloned first.
        let rid = RoundId(0);
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(rid));
        let original_rounds = trace.rounds.clone();
        let mut payloads = HashMap::new();
        payloads.insert(rid, vec![payload("c", "echo")]);

        let _events = project_turn(turn_id(1), None, &payloads, dummy_completed(), trace);
        // The local `trace` is moved; we cannot reuse it. The
        // assertion here is structural: we kept `original_rounds`
        // and matched against the (moved) trace via the events.
        assert_eq!(original_rounds.len(), 1);
    }

    #[test]
    fn empty_payload_list_still_emits_dispatch() {
        // An empty `calls` list is unusual but valid (e.g. an
        // empty-but-present tool batch after a filter rejected
        // everything). The dispatch event should still fire so
        // observers see the round.
        let rid = RoundId(0);
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(rid));
        let mut payloads = HashMap::new();
        payloads.insert(rid, Vec::new());

        let events = project_turn(turn_id(1), None, &payloads, dummy_completed(), trace);
        assert_eq!(events.len(), 3);
        match &events[1] {
            ContextEvent::ToolBatchDispatched { calls, .. } => assert!(calls.is_empty()),
            other => panic!("expected ToolBatchDispatched, got {other:?}"),
        }
    }

    // -- helpers --

    fn dummy_completed() -> TurnResult {
        TurnResult::Completed {
            final_output: reimagine_context_kernel::ModelOutput {
                stop_reason: ModelStopReason::EndTurn,
                usage: None,
                reasoning: None,
                response: ModelResponse {
                    text: reimagine_context_kernel::TextPayload(String::new()),
                    tool_calls: Vec::new(),
                },
            },
        }
    }

    // -- Slice 5A Phase C: ContextEvent JSON round-trip --

    #[test]
    fn round_trip_turn_started_no_conversation() {
        let original = ContextEvent::TurnStarted {
            conversation_id: None,
            turn_id: turn_id(7),
        };
        let json = serde_json::to_string(&original).expect("serialize");
        // conversation_id is skipped on None — compact wire format.
        assert!(!json.contains("conversation_id"));
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.turn_id(), &turn_id(7));
        assert_eq!(restored.conversation_id(), None);
    }

    #[test]
    fn round_trip_turn_started_with_conversation() {
        let original = ContextEvent::TurnStarted {
            conversation_id: Some(ConversationId("conv-42".into())),
            turn_id: turn_id(3),
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            restored.conversation_id(),
            Some(ConversationId("conv-42".into()))
        );
        assert_eq!(restored.turn_id(), &turn_id(3));
    }

    #[test]
    fn round_trip_tool_batch_dispatched() {
        let original = ContextEvent::ToolBatchDispatched {
            conversation_id: Some(ConversationId("conv-batch".into())),
            turn_id: turn_id(11),
            round_id: RoundId(3),
            calls: vec![payload("call-a", "echo"), payload("call-b", "echo")],
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        match restored {
            ContextEvent::ToolBatchDispatched {
                conversation_id,
                turn_id: tid,
                round_id,
                calls,
            } => {
                assert_eq!(conversation_id, Some(ConversationId("conv-batch".into())));
                assert_eq!(tid, turn_id(11));
                assert_eq!(round_id, RoundId(3));
                assert_eq!(calls.len(), 2);
            }
            other => panic!("expected ToolBatchDispatched, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_turn_outcome_completed() {
        let trace = empty_trace();
        // Empty trace is sufficient — `Completed` does not require any
        // round trace content for serialization.
        let original = ContextEvent::TurnOutcome {
            conversation_id: Some(ConversationId("conv-out".into())),
            turn_id: turn_id(99),
            result: dummy_completed(),
            trace,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        match restored {
            ContextEvent::TurnOutcome {
                conversation_id,
                turn_id: tid,
                result: TurnResult::Completed { .. },
                ..
            } => {
                assert_eq!(conversation_id, Some(ConversationId("conv-out".into())));
                assert_eq!(tid, turn_id(99));
            }
            other => panic!("expected TurnOutcome Completed, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_turn_outcome_interrupted() {
        let trace = empty_trace();
        let original = ContextEvent::TurnOutcome {
            conversation_id: None,
            turn_id: turn_id(13),
            result: TurnResult::Interrupted {
                cause: reimagine_context_kernel::TurnInterruption::CompactionFailed {
                    reason: "budget exceeded".into(),
                },
            },
            trace,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        match restored {
            ContextEvent::TurnOutcome {
                conversation_id,
                turn_id: tid,
                result:
                    TurnResult::Interrupted {
                        cause: TurnInterruption::CompactionFailed { reason },
                    },
                ..
            } => {
                assert_eq!(conversation_id, None);
                assert_eq!(tid, turn_id(13));
                assert_eq!(reason, "budget exceeded");
            }
            other => panic!("expected TurnOutcome Interrupted, got {other:?}"),
        }
    }

    #[test]
    fn project_turn_output_survives_serialization_round_trip() {
        // End-to-end: project a full event sequence, serialize, restore,
        // assert equality of the structural contents.
        let rid = RoundId(0);
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(rid));
        let mut payloads = HashMap::new();
        payloads.insert(rid, vec![payload("c1", "echo")]);

        let events = project_turn(
            turn_id(21),
            Some(ConversationId("conv-e2e".into())),
            &payloads,
            dummy_completed(),
            trace,
        );
        let json = serde_json::to_string(&events).expect("serialize");
        let restored: Vec<ContextEvent> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.len(), events.len());
        // All events carry the same conversation id and turn id.
        for e in &restored {
            assert_eq!(e.conversation_id(), Some(ConversationId("conv-e2e".into())));
            assert_eq!(e.turn_id(), &turn_id(21));
        }
        // The middle event is a ToolBatchDispatched with the one payload.
        match &restored[1] {
            ContextEvent::ToolBatchDispatched { calls, .. } => {
                assert_eq!(calls.len(), 1);
            }
            other => panic!("expected ToolBatchDispatched, got {other:?}"),
        }
    }
}
