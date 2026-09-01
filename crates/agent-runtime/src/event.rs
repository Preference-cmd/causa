//! `ContextEvent` projection — Slice 4 §6 Phase C (框架事件), reworked by
//! the 2026-09-02 thermo-nuclear review.
//!
//! Events are a **projection** of the kernel's facts (`TurnContext` +
//! `TurnResult` + `TurnTrace`) into a sequence that consumers (UI,
//! observability, audit) can subscribe to. They are not facts themselves:
//! they are derived, not persisted, and never written back into a
//! `TurnContext`.
//!
//! ## Layering
//!
//! ```text
//! reimagine-context-kernel
//!   └─ TurnContext / TurnResult / TurnTrace / ModelRoundTrace   ← facts
//!             ^
//!             │ project_turn(...)
//!             |
//! reimagine-agent-runtime
//!   └─ ContextEvent / project_turn                              ← projection
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
//!
//! ## Serialization
//!
//! `ContextEvent` is serde-derived for IPC delivery to host UIs and
//! audit pipelines (Slice 5A Phase C). The embedded kernel types
//! (`TurnResult`, `TurnTrace`) carry their own serde derives; their
//! serde **shapes** are a load-bearing wire contract for this module —
//! see the note on `reimagine_context_kernel::TurnOutcome` — even
//! though the Rust item paths live in the kernel's staged perimeter.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use reimagine_context_kernel::{
    BlockContent, ConversationId, ModelRoundTrace, RoundId, ToolCallPayload, TurnContext, TurnId,
    TurnResult, TurnTrace,
};

/// Framework-side event projected from a turn's facts.
///
/// Every variant shares the routing envelope (`conversation_id`,
/// `turn_id`); the variant-specific payload lives in `kind`.
///
/// `conversation_id` is `Option<ConversationId>` because the Slice 1
/// `TurnRunner::run` entry does not carry a `ConversationState`. The
/// Slice 2 `run_in_conversation` entry does; the host constructs the
/// event with `Some(id)` when projecting from a `ConversationOutcome`,
/// and `None` for the bare `TurnOutcome` path. Subscribers that care
/// about cross-conversation routing key on `Some`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEvent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<ConversationId>,
    pub turn_id: TurnId,
    pub kind: ContextEventKind,
}

/// The event payload proper. `type` is the snake_case discriminator on
/// the wire (`"turn_started"` / `"tool_batch_dispatched"` /
/// `"turn_outcome"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextEventKind {
    /// The turn has begun. Emitted exactly once, first in the sequence.
    TurnStarted,
    /// A batch of tool calls was dispatched for `round_id`. `calls`
    /// carries the **pre-execution** payloads — the model-emitted tool
    /// calls as committed facts, in draft order. Emitted once per round
    /// that has a `tool_batch` in the trace; rounds without one
    /// (EndTurn, MaxTokens, Refusal, compaction failure, …) produce no
    /// dispatch event.
    ToolBatchDispatched {
        round_id: RoundId,
        calls: Vec<ToolCallPayload>,
    },
    /// The turn finished. `result` is the canonical `TurnResult`
    /// (`Completed | Interrupted { cause }`) and `trace` is the full
    /// `TurnTrace` (rounds, totals). Consumers can re-walk `trace`
    /// for finer-grained data. Emitted exactly once, last in the
    /// sequence.
    TurnOutcome {
        result: TurnResult,
        trace: TurnTrace,
    },
}

/// Project a turn into its canonical sequence of `ContextEvent`s.
///
/// ## Inputs
///
/// - `context` — the turn's fact state. `ToolBatchDispatched.calls` is
///   derived from the committed `tool.call` blocks addressed by each
///   round's `applied_block_ids`, so the projector needs no side-channel
///   capture: the committed facts are the pre-execution batch (the
///   kernel commits every model-emitted call before any dispatch).
/// - `result` / `trace` — straight from `TurnOutcome` (or the
///   `ConversationOutcome` fields).
/// - `conversation_id` — `Some(id)` for the `run_in_conversation` entry,
///   `None` for the bare `run` entry.
///
/// For the conversation path, project **before** `commit`/`abort_turn`
/// (the turn is then still `state.active_turn()`); the projection reads
/// the blocks and copies what it needs.
///
/// ## Output order
///
/// 1. `TurnStarted` — exactly once.
/// 2. One `ToolBatchDispatched` per trace round that has a `tool_batch`,
///    in `round_id` order, carrying that round's committed tool calls.
/// 3. `TurnOutcome` — exactly once, with the full trace.
///
/// ## What this function never does
///
/// - It never reads or mutates kernel state beyond the arguments.
/// - It never constructs `ToolExecutionOutcome` — execution results
///   live in `trace`, not in events.
/// - It never holds kernel locks; it is a pure projection over
///   borrowed data.
pub fn project_turn(
    context: &TurnContext,
    result: &TurnResult,
    trace: &TurnTrace,
    conversation_id: Option<ConversationId>,
) -> Vec<ContextEvent> {
    let turn_id = context.turn_id();
    let mut events = Vec::with_capacity(2 + trace.rounds.len());
    events.push(ContextEvent {
        conversation_id: conversation_id.clone(),
        turn_id: turn_id.clone(),
        kind: ContextEventKind::TurnStarted,
    });
    // tool.call blocks indexed by block id — one pass over the facts,
    // then O(1) resolution per round.
    let call_index: HashMap<reimagine_context_kernel::BlockId, ToolCallPayload> = context
        .blocks()
        .iter()
        .filter_map(|b| match &b.content {
            BlockContent::ToolCall(call) => Some((b.id.clone(), call.clone())),
            _ => None,
        })
        .collect();
    for round in &trace.rounds {
        if round.tool_batch.is_none() {
            continue;
        }
        let calls = committed_calls(&call_index, round);
        events.push(ContextEvent {
            conversation_id: conversation_id.clone(),
            turn_id: turn_id.clone(),
            kind: ContextEventKind::ToolBatchDispatched {
                round_id: round.round_id,
                calls,
            },
        });
    }
    events.push(ContextEvent {
        conversation_id,
        turn_id,
        kind: ContextEventKind::TurnOutcome {
            result: result.clone(),
            trace: trace.clone(),
        },
    });
    events
}

/// The model-emitted tool calls of one round, in draft (commit) order.
/// `applied_block_ids` addresses every block the model door committed for
/// the round (optional text first, then one tool call per draft); only the
/// tool-call blocks resolve here.
fn committed_calls(
    call_index: &HashMap<reimagine_context_kernel::BlockId, ToolCallPayload>,
    round: &ModelRoundTrace,
) -> Vec<ToolCallPayload> {
    round
        .applied_block_ids
        .iter()
        .filter_map(|id| call_index.get(id).cloned())
        .collect()
}

// --- tests ----------------------------------------------------------------
//
// Pure-projection tests. They build the kernel data types directly and
// assert the emitted event sequence and serde shape. No driver, no
// harness, no async — the projection is sync over borrowed data.

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_context_kernel::{
        BlockId, BlockSequence, ContextVersion, InvocationId, ModelResponse, ModelStopReason,
        OutputSummary, TextPayload, ToolBatchTrace, ToolCallId, TurnContext, TurnInterruption,
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

    /// A turn whose facts contain one text block and one committed tool
    /// call (sequence 1) — the shape the model door leaves behind.
    fn turn_with_tool_call(tool: &str) -> TurnContext {
        let mut ctx = TurnContext::new(turn_id(1));
        ctx.append_input(TextPayload::new("hi"), "user")
            .expect("append_input");
        let response = ModelResponse {
            text: TextPayload::new(String::new()),
            tool_calls: vec![reimagine_context_kernel::ToolCallDraft {
                tool_name: tool.to_string(),
                arguments: json!({"k": "v"}),
                provider_call_id: None,
            }],
        };
        ctx.append_model_output(
            InvocationId {
                turn_id: turn_id(1),
                round_id: RoundId(0),
            },
            &response,
            ModelStopReason::ToolUse,
        )
        .expect("append_model_output");
        ctx
    }

    /// Round trace with a tool batch whose `applied_block_ids` point at
    /// the committed tool-call block (block sequence 1 in
    /// `turn_with_tool_call`).
    fn round_with_tool_batch(round_id: RoundId, call_block_seq: u64) -> ModelRoundTrace {
        ModelRoundTrace {
            round_id,
            invocation_id: InvocationId {
                turn_id: turn_id(1),
                round_id,
            },
            frame_version: ContextVersion(0),
            attempts: Vec::new(),
            output_summary: Some(OutputSummary {
                stop_reason: ModelStopReason::ToolUse,
                usage: None,
                tool_call_count: 1,
                response_text_bytes: 0,
            }),
            applied_block_ids: vec![BlockId {
                turn_id: turn_id(1),
                sequence: BlockSequence(call_block_seq),
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
                stop_reason: ModelStopReason::EndTurn,
                usage: None,
                tool_call_count: 0,
                response_text_bytes: 0,
            }),
            applied_block_ids: vec![],
            tool_batch: None,
        }
    }

    fn completed() -> TurnResult {
        TurnResult::Completed {
            final_output: reimagine_context_kernel::ModelOutput {
                stop_reason: ModelStopReason::EndTurn,
                usage: None,
                reasoning: None,
                response: ModelResponse {
                    text: TextPayload::new(String::new()),
                    tool_calls: Vec::new(),
                },
            },
        }
    }

    #[test]
    fn empty_trace_emits_only_started_and_outcome() {
        let ctx = TurnContext::new(turn_id(1));
        let events = project_turn(&ctx, &completed(), &empty_trace(), None);
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0].kind, ContextEventKind::TurnStarted));
        assert!(matches!(
            events[1].kind,
            ContextEventKind::TurnOutcome { .. }
        ));
    }

    #[test]
    fn dispatched_calls_derive_from_committed_tool_call_blocks() {
        let ctx = turn_with_tool_call("echo");
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(RoundId(0), 1));

        let events = project_turn(&ctx, &completed(), &trace, None);
        assert_eq!(events.len(), 3);
        match &events[1].kind {
            ContextEventKind::ToolBatchDispatched { round_id, calls } => {
                assert_eq!(*round_id, RoundId(0));
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].tool_name, "echo");
                assert_eq!(calls[0].arguments, json!({"k": "v"}));
            }
            other => panic!("expected ToolBatchDispatched, got {other:?}"),
        }
    }

    #[test]
    fn endturn_round_produces_no_dispatch_event() {
        let ctx = TurnContext::new(turn_id(1));
        let mut trace = empty_trace();
        trace.rounds.push(round_endturn(RoundId(7)));

        let events = project_turn(&ctx, &completed(), &trace, None);
        assert_eq!(
            events.len(),
            2,
            "EndTurn must not produce ToolBatchDispatched"
        );
        assert!(matches!(events[0].kind, ContextEventKind::TurnStarted));
        assert!(matches!(
            events[1].kind,
            ContextEventKind::TurnOutcome { .. }
        ));
    }

    #[test]
    fn multiple_rounds_emit_dispatches_in_round_id_order() {
        // Three rounds: tool_use, endturn, tool_use → two dispatch
        // events in round order (0, 2), EndTurn round skipped.
        let ctx = turn_with_tool_call("echo");
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(RoundId(0), 1));
        trace.rounds.push(round_endturn(RoundId(1)));
        trace.rounds.push(round_with_tool_batch(RoundId(2), 1));

        let events = project_turn(&ctx, &completed(), &trace, None);
        assert_eq!(events.len(), 4);
        let dispatched_round_ids: Vec<RoundId> = events
            .iter()
            .filter_map(|e| match e.kind {
                ContextEventKind::ToolBatchDispatched { round_id, .. } => Some(round_id),
                _ => None,
            })
            .collect();
        assert_eq!(dispatched_round_ids, vec![RoundId(0), RoundId(2)]);
    }

    #[test]
    fn envelope_routes_conversation_and_turn_ids() {
        let conv = ConversationId("conv-42".to_string());
        let ctx = turn_with_tool_call("echo");
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(RoundId(0), 1));

        let events = project_turn(&ctx, &completed(), &trace, Some(conv.clone()));
        for e in &events {
            assert_eq!(e.conversation_id, Some(conv.clone()));
            assert_eq!(e.turn_id, turn_id(1));
        }
        // The bare-run path omits the key on the wire entirely.
        let bare = project_turn(&ctx, &completed(), &trace, None);
        let json = serde_json::to_string(&bare[0]).expect("serialize");
        assert!(!json.contains("conversation_id"));
    }

    #[test]
    fn interrupted_turn_carries_cause_in_outcome() {
        let ctx = TurnContext::new(turn_id(1));
        let mut trace = empty_trace();
        trace.rounds.push(round_endturn(RoundId(0)));
        let cause = TurnInterruption::MaxModelRounds { limit: 8 };
        let events = project_turn(
            &ctx,
            &TurnResult::Interrupted {
                cause: cause.clone(),
            },
            &trace,
            None,
        );
        assert_eq!(events.len(), 2);
        match &events[1].kind {
            ContextEventKind::TurnOutcome { result, .. } => match result {
                TurnResult::Interrupted { cause: c } => assert_eq!(*c, cause),
                other => panic!("expected Interrupted, got {other:?}"),
            },
            other => panic!("expected TurnOutcome, got {other:?}"),
        }
    }

    // -- Slice 5A Phase C: ContextEvent JSON round-trip ---------------------

    #[test]
    fn round_trip_turn_started_no_conversation() {
        let original = ContextEvent {
            conversation_id: None,
            turn_id: turn_id(7),
            kind: ContextEventKind::TurnStarted,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        // conversation_id is skipped on None — compact wire format; the
        // discriminator is snake_case under `type`.
        assert!(!json.contains("conversation_id"));
        assert!(json.contains("\"type\":\"turn_started\""));
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.turn_id, turn_id(7));
        assert_eq!(restored.conversation_id, None);
    }

    #[test]
    fn round_trip_tool_batch_dispatched() {
        let original = ContextEvent {
            conversation_id: Some(ConversationId("conv-batch".into())),
            turn_id: turn_id(11),
            kind: ContextEventKind::ToolBatchDispatched {
                round_id: RoundId(3),
                calls: vec![ToolCallPayload {
                    call_id: ToolCallId("call-a".into()),
                    tool_name: "echo".into(),
                    arguments: json!({"k": "v"}),
                }],
            },
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        match restored.kind {
            ContextEventKind::ToolBatchDispatched { round_id, calls } => {
                assert_eq!(round_id, RoundId(3));
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].call_id.0, "call-a");
            }
            other => panic!("expected ToolBatchDispatched, got {other:?}"),
        }
        assert_eq!(
            restored.conversation_id,
            Some(ConversationId("conv-batch".into()))
        );
        assert_eq!(restored.turn_id, turn_id(11));
    }

    #[test]
    fn round_trip_turn_outcome_completed_and_interrupted() {
        let completed_event = ContextEvent {
            conversation_id: Some(ConversationId("conv-out".into())),
            turn_id: turn_id(99),
            kind: ContextEventKind::TurnOutcome {
                result: completed(),
                trace: empty_trace(),
            },
        };
        let json = serde_json::to_string(&completed_event).expect("serialize");
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(
            restored.kind,
            ContextEventKind::TurnOutcome {
                result: TurnResult::Completed { .. },
                ..
            }
        ));

        let interrupted_event = ContextEvent {
            conversation_id: None,
            turn_id: turn_id(13),
            kind: ContextEventKind::TurnOutcome {
                result: TurnResult::Interrupted {
                    cause: TurnInterruption::CompactionFailed {
                        reason: "budget exceeded".into(),
                    },
                },
                trace: empty_trace(),
            },
        };
        let json = serde_json::to_string(&interrupted_event).expect("serialize");
        let restored: ContextEvent = serde_json::from_str(&json).expect("deserialize");
        match restored.kind {
            ContextEventKind::TurnOutcome {
                result: TurnResult::Interrupted { cause },
                ..
            } => assert_eq!(
                cause,
                TurnInterruption::CompactionFailed {
                    reason: "budget exceeded".into()
                }
            ),
            other => panic!("expected TurnOutcome Interrupted, got {other:?}"),
        }
    }

    #[test]
    fn project_turn_output_survives_serialization_round_trip() {
        let ctx = turn_with_tool_call("echo");
        let mut trace = empty_trace();
        trace.rounds.push(round_with_tool_batch(RoundId(0), 1));

        let events = project_turn(
            &ctx,
            &completed(),
            &trace,
            Some(ConversationId("conv-e2e".into())),
        );
        let json = serde_json::to_string(&events).expect("serialize");
        let restored: Vec<ContextEvent> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.len(), events.len());
        // All events carry the same conversation id and turn id.
        for e in &restored {
            assert_eq!(e.conversation_id, Some(ConversationId("conv-e2e".into())));
            assert_eq!(e.turn_id, turn_id(1));
        }
        // The middle event is a ToolBatchDispatched with the one call.
        match &restored[1].kind {
            ContextEventKind::ToolBatchDispatched { calls, .. } => {
                assert_eq!(calls.len(), 1);
            }
            other => panic!("expected ToolBatchDispatched, got {other:?}"),
        }
    }
}
