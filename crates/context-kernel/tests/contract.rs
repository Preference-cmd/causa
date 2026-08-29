//! Contract tests — canonical facts, validated transitions, deterministic
//! projections, and fact fidelity. No reference driver involved; every
//! import comes from the public root facade.

mod common;

use common::{assistant_endturn, ctx, turn_id};
use reimagine_context_kernel::{
    AssistantPayload, BlockId, BlockMeta, BlockPayload, BlockSequence, Compaction, CompactionError,
    CompactionInput, CompactionOutput, ContextBlock, ContextError, ContextVersion, FramePolicy,
    InputPayload, InvocationId, ModelOutput, ModelStopReason, ModelUsage, ReasoningPayload,
    RoundId, TextPayload, ToolCallDraft, ToolCallId, ToolCallPayload, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnContext, WindowBudget,
};
use serde_json::json;

#[tokio::test]
async fn empty_frame_deterministic() {
    let c = ctx("t1");
    let f0 = c.frame_sync(RoundId(0)).unwrap();
    let f1 = c.frame_sync(RoundId(0)).unwrap();
    assert_eq!(f0.frame_id, f1.frame_id);
    assert!(f0.model_context.blocks.is_empty());
}

#[tokio::test]
async fn append_input_and_frame_order() {
    let mut c = ctx("t1");
    c.apply_inputs(vec![
        InputPayload::RequestUser(TextPayload::new("hello")),
        InputPayload::InstructionSystem(TextPayload::new("sys")),
    ])
    .unwrap();
    let f = c.frame_sync(RoundId(0)).unwrap();
    assert_eq!(f.model_context.blocks.len(), 2);
    // Order preserved
    assert!(matches!(
        f.model_context.blocks[0].payload,
        BlockPayload::RequestUser(_)
    ));
    assert!(matches!(
        f.model_context.blocks[1].payload,
        BlockPayload::InstructionSystem(_)
    ));
    // After frame, seal should make further append fail
    let mut c2 =
        TurnContext::from_validated_blocks(turn_id("t1"), c.snapshot_blocks(), c.version())
            .unwrap();
    // not sealed, should allow
    c2.append_input(InputPayload::RequestUser(TextPayload::new("more")))
        .unwrap();
}

#[tokio::test]
async fn sealed_turn_append_closed() {
    let mut c = ctx("t1");
    c.append_input(InputPayload::RequestUser(TextPayload::new("hi")))
        .unwrap();
    // seal is a public lifecycle transition owned by the driver (Phase D)
    c.seal();
    assert!(c.is_sealed());
    let mut sealed = c;
    assert!(matches!(
        sealed.append_input(InputPayload::RequestUser(TextPayload::new("x"))),
        Err(ContextError::SealedTurn)
    ));
    assert!(matches!(
        sealed.apply_model_output(
            InvocationId {
                turn_id: turn_id("t1"),
                round_id: RoundId(1)
            },
            assistant_endturn("y")
        ),
        Err(ContextError::SealedTurn)
    ));
}

#[tokio::test]
async fn apply_model_output_rejects_invalid_outputs() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    // EndTurn must not carry tool_calls
    let bad_endturn = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![ToolCallDraft {
                tool_name: "echo".into(),
                arguments: json!({}),
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    };
    assert!(matches!(
        c.apply_model_output(inv.clone(), bad_endturn),
        Err(ContextError::InvalidSequence(_))
    ));
    // ToolUse requires non-empty tool name
    let empty_name = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![ToolCallDraft {
                tool_name: "  ".into(),
                arguments: json!({}),
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    assert!(matches!(
        c.apply_model_output(inv.clone(), empty_name),
        Err(ContextError::InvalidSequence(_))
    ));
    // ToolUse requires object arguments
    let non_object = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![ToolCallDraft {
                tool_name: "echo".into(),
                arguments: json!(42),
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    assert!(matches!(
        c.apply_model_output(inv.clone(), non_object),
        Err(ContextError::InvalidSequence(_))
    ));
    // identical (tool, args) twice in one batch is fine — positions differ
    let dup_batch = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                    provider_call_id: None,
                },
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                    provider_call_id: None,
                },
            ],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    assert!(c.apply_model_output(inv.clone(), dup_batch).is_ok());
    // sealed turn rejects everything — seal is the public lifecycle transition
    c.seal();
    let mut sealed = c;
    assert!(sealed.is_sealed());
    let sealed_inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(1),
    };
    assert!(matches!(
        sealed.apply_model_output(sealed_inv, assistant_endturn("y")),
        Err(ContextError::SealedTurn)
    ));
}

// [P1 Phase 1] from_validated_blocks rejects corrupt state

#[tokio::test]
async fn from_validated_blocks_rejects_corrupt_state() {
    fn block(seq: u64, payload: BlockPayload) -> ContextBlock {
        let tid = turn_id("t1");
        ContextBlock {
            id: BlockId {
                turn_id: tid,
                sequence: BlockSequence(seq),
            },
            sequence: BlockSequence(seq),
            meta: BlockMeta::default(),
            payload,
        }
    }
    // wrong turn_id
    let mut b = block(0, BlockPayload::RequestUser(TextPayload::new("hi")));
    b.id.turn_id = turn_id("other");
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), vec![b], ContextVersion(1)),
        Err(ContextError::InvalidSequence(_))
    ));
    // non-contiguous sequence
    let blocks = vec![
        block(0, BlockPayload::RequestUser(TextPayload::new("a"))),
        block(2, BlockPayload::RequestUser(TextPayload::new("b"))),
    ];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(2)),
        Err(ContextError::InvalidSequence(_))
    ));
    // duplicate tool.call ids
    let blocks = vec![
        block(
            0,
            BlockPayload::ToolCall(ToolCallPayload {
                call_id: ToolCallId::new("dup"),
                tool_name: "echo".into(),
                arguments: json!({}),
                provider_call_id: None,
            }),
        ),
        block(
            1,
            BlockPayload::ToolCall(ToolCallPayload {
                call_id: ToolCallId::new("dup"),
                tool_name: "echo".into(),
                arguments: json!({}),
                provider_call_id: None,
            }),
        ),
    ];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(2)),
        Err(ContextError::DuplicateToolCallId(_))
    ));
    // unpaired tool.result
    let blocks = vec![block(
        0,
        BlockPayload::ToolResult(ToolResultPayload {
            call_id: ToolCallId::new("ghost"),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({})),
        }),
    )];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(1)),
        Err(ContextError::UnpairedToolResult(_))
    ));
}

// [P3] MaxToolCalls counts raw calls (incl. rejected duplicates) and records trace

// ---- Phase B: fact-fidelity fields are recorded verbatim and serde-stable ----

#[test]
fn provider_call_id_passes_through_draft_to_persisted_block() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    let out = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("with provider id"),
            tool_calls: vec![
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"a": 1}),
                    provider_call_id: Some("call_provider_1".into()),
                },
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"b": 2}),
                    provider_call_id: None,
                },
            ],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    let applied = c.apply_model_output(inv, out).expect("record facts");
    let call_blocks: Vec<&ContextBlock> = c
        .blocks()
        .iter()
        .filter(|b| matches!(b.payload, BlockPayload::ToolCall(_)))
        .collect();
    assert_eq!(applied.block_ids.len(), 3); // assistant text + 2 calls
    assert_eq!(call_blocks.len(), 2);
    match (&call_blocks[0].payload, &call_blocks[1].payload) {
        (BlockPayload::ToolCall(a), BlockPayload::ToolCall(b)) => {
            assert_eq!(a.provider_call_id.as_deref(), Some("call_provider_1"));
            assert_eq!(b.provider_call_id, None);
        }
        _ => panic!("expected tool call payloads"),
    }
    // serde round-trip of the fact state preserves the passthrough verbatim
    let blocks_json = serde_json::to_string(&c.snapshot_blocks()).unwrap();
    let blocks: Vec<ContextBlock> = serde_json::from_str(&blocks_json).unwrap();
    let restored: Vec<Option<String>> = blocks
        .iter()
        .filter_map(|b| match &b.payload {
            BlockPayload::ToolCall(p) => Some(p.provider_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(restored, vec![Some("call_provider_1".into()), None]);
}

#[test]
fn apply_model_output_records_max_tokens_and_refusal_as_facts() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    // Structural validation only: any stop reason may be recorded as facts.
    // Interpreting terminal reasons stays driver policy above the kernel.
    let max_tokens = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("partial"),
            tool_calls: vec![],
        },
        usage: None,
        stop_reason: ModelStopReason::MaxTokens,
        reasoning: None,
    };
    let applied = c
        .apply_model_output(inv.clone(), max_tokens)
        .expect("MaxTokens is recordable");
    assert_eq!(applied.block_ids.len(), 1); // ResponseAssistant only
    let refusal_with_calls = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new(""),
            tool_calls: vec![ToolCallDraft {
                tool_name: "echo".into(),
                arguments: json!({}),
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::Refusal,
        reasoning: None,
    };
    assert!(
        c.apply_model_output(inv.clone(), refusal_with_calls)
            .is_ok()
    );
    // Structural rules for EndTurn/ToolUse are untouched by the loosening.
    let bad_endturn = ModelOutput {
        assistant: AssistantPayload {
            text: TextPayload::new("x"),
            tool_calls: vec![ToolCallDraft {
                tool_name: "echo".into(),
                arguments: json!({}),
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    };
    assert!(matches!(
        c.apply_model_output(inv, bad_endturn),
        Err(ContextError::InvalidSequence(_))
    ));
}

#[test]
fn fidelity_fields_are_serde_additive() {
    let usage = ModelUsage {
        input_tokens: 120,
        output_tokens: 30,
        cache_read_tokens: Some(64),
        cache_write_tokens: Some(12),
        reasoning_tokens: Some(8),
    };
    let back: ModelUsage = serde_json::from_str(&serde_json::to_string(&usage).unwrap()).unwrap();
    assert_eq!(back.input_tokens, 120);
    assert_eq!(back.cache_read_tokens, Some(64));
    assert_eq!(back.cache_write_tokens, Some(12));
    assert_eq!(back.reasoning_tokens, Some(8));
    // pre-fidelity payloads (without the new fields) still deserialize
    let legacy: ModelUsage =
        serde_json::from_str(r#"{"input_tokens":1,"output_tokens":2}"#).unwrap();
    assert_eq!(
        (
            legacy.cache_read_tokens,
            legacy.cache_write_tokens,
            legacy.reasoning_tokens
        ),
        (None, None, None)
    );
    let reasoning = ReasoningPayload {
        text: "thinking".into(),
        signature: Some("sig-abc".into()),
    };
    let back: ReasoningPayload =
        serde_json::from_str(&serde_json::to_string(&reasoning).unwrap()).unwrap();
    assert_eq!(back.text, "thinking");
    assert_eq!(back.signature.as_deref(), Some("sig-abc"));
    let legacy_call: ToolCallPayload =
        serde_json::from_str(r#"{"call_id":"echo:abcd1234:0","tool_name":"echo","arguments":{}}"#)
            .unwrap();
    assert_eq!(legacy_call.provider_call_id, None);
}

// ---- Phase D boundary: compaction projection identity ----

struct DropAllCompaction;
#[async_trait::async_trait]
impl Compaction for DropAllCompaction {
    async fn compact(&self, _input: CompactionInput) -> Result<CompactionOutput, CompactionError> {
        Ok(CompactionOutput {
            blocks: vec![],
            summary: None,
            truncated: true,
        })
    }
}

#[tokio::test]
async fn compaction_projection_identity() {
    let mut c = ctx("t1");
    c.append_input(InputPayload::RequestUser(TextPayload::new("hello")))
        .unwrap();
    let lossless = FramePolicy::default();
    let compacting = FramePolicy {
        window_budget: WindowBudget {
            model_window_limit: 100,
            compaction_trigger: 1,
        },
        compaction: Some(std::sync::Arc::new(DropAllCompaction)),
        token_counter: None,
    };
    let sync_frame = c.frame_sync(RoundId(0)).unwrap();
    let lossless_frame = c.frame(RoundId(0), &lossless).await.unwrap();
    // lossless projection == facts; equal inputs materialize equal frames
    assert_eq!(sync_frame.frame_id, lossless_frame.frame_id);
    assert_eq!(
        serde_json::to_string(&sync_frame.model_context.blocks).unwrap(),
        serde_json::to_string(&lossless_frame.model_context.blocks).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&lossless_frame.model_context.blocks).unwrap(),
        serde_json::to_string(&c.snapshot_blocks()).unwrap()
    );
    // policy-shaped projection: compacted content under the SAME frame identity
    let projected = c.frame(RoundId(0), &compacting).await.unwrap();
    assert_eq!(projected.frame_id, sync_frame.frame_id);
    assert!(projected.model_context.blocks.is_empty());
    // facts untouched — compaction is frame-local
    assert_eq!(c.snapshot_blocks().len(), 1);
}
