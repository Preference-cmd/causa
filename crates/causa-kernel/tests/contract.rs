//! Contract tests -- canonical facts, validated transitions, deterministic
//! projections, and fact fidelity. No reference driver involved; every
//! import comes from the public root facade.

mod common;

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, BlockSequence, ContentPart, ContextBlock, ContextError,
    ContextVersion, FrameId, FrameScope, InvocationId, ModelOutput, ModelResponse, ModelStopReason,
    ModelUsage, ReasoningPayload, RoundId, TextPayload, ToolCallDraft, ToolCallId, ToolCallPayload,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnContext,
};
use common::{ctx, endturn_output, turn_id};
use serde_json::json;

#[tokio::test]
async fn empty_frame_deterministic() {
    let c = ctx("t1");
    let f0 = c.frame(RoundId(0));
    let f1 = c.frame(RoundId(0));
    assert_eq!(f0.frame_id, f1.frame_id);
    assert!(f0.model_context.blocks.is_empty());
}

#[tokio::test]
async fn append_input_and_frame_order() {
    let mut c = ctx("t1");
    c.append_input(TextPayload::new("hello"), "user").unwrap();
    c.append_input(TextPayload::new("sys"), "user").unwrap();
    let f = c.frame(RoundId(0));
    assert_eq!(f.model_context.blocks.len(), 2);
    // Order preserved.
    assert!(matches!(
        f.model_context.blocks[0].content,
        BlockContent::Parts(_)
    ));
    assert!(matches!(
        f.model_context.blocks[1].content,
        BlockContent::Parts(_)
    ));
    // Sealed turn rejects further append.
    c.seal();
    let mut sealed = c;
    assert!(matches!(
        sealed.append_input(TextPayload::new("x"), "user"),
        Err(ContextError::SealedTurn)
    ));
}

#[tokio::test]
async fn sealed_turn_append_closed() {
    let mut c = ctx("t1");
    c.append_input(TextPayload::new("hi"), "user").unwrap();
    c.seal();
    assert!(c.is_sealed());
    let mut sealed = c;
    assert!(matches!(
        sealed.append_input(TextPayload::new("x"), "user"),
        Err(ContextError::SealedTurn)
    ));
    assert!(matches!(
        sealed.append_model_output(
            InvocationId {
                turn_id: turn_id("t1"),
                round_id: RoundId(1)
            },
            &endturn_output("y").response,
            ModelStopReason::EndTurn
        ),
        Err(ContextError::SealedTurn)
    ));
}

#[tokio::test]
async fn append_model_output_rejects_invalid_outputs() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    // EndTurn must not carry tool_calls
    let bad_endturn = ModelOutput {
        response: ModelResponse {
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
        c.append_model_output(inv.clone(), &bad_endturn.response, bad_endturn.stop_reason),
        Err(ContextError::InvalidModelOutput(_))
    ));
    // ToolUse requires non-empty tool name
    let empty_name = ModelOutput {
        response: ModelResponse {
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
        c.append_model_output(inv.clone(), &empty_name.response, empty_name.stop_reason),
        Err(ContextError::InvalidModelOutput(_))
    ));
    // ToolUse requires object arguments
    let non_object = ModelOutput {
        response: ModelResponse {
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
        c.append_model_output(inv.clone(), &non_object.response, non_object.stop_reason),
        Err(ContextError::InvalidModelOutput(_))
    ));
    // Identical (tool, args) twice in one batch is fine -- positions differ
    let dup_batch = ModelOutput {
        response: ModelResponse {
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
    assert!(
        c.append_model_output(inv.clone(), &dup_batch.response, dup_batch.stop_reason)
            .is_ok()
    );
    // Sealed turn rejects everything.
    c.seal();
    let mut sealed = c;
    assert!(sealed.is_sealed());
    let sealed_inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(1),
    };
    assert!(matches!(
        sealed.append_model_output(
            sealed_inv,
            &endturn_output("y").response,
            ModelStopReason::EndTurn
        ),
        Err(ContextError::SealedTurn)
    ));
}

#[tokio::test]
async fn from_validated_blocks_rejects_corrupt_state() {
    fn block(seq: u64, content: BlockContent) -> ContextBlock {
        let tid = turn_id("t1");
        ContextBlock {
            id: BlockId {
                turn_id: tid,
                sequence: BlockSequence(seq),
            },
            sequence: BlockSequence(seq),
            content,
            meta: BlockMeta::default(),
        }
    }
    // Wrong turn_id
    let mut b = block(
        0,
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("hi"))]),
    );
    b.id.turn_id = turn_id("other");
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), vec![b], ContextVersion(1)),
        Err(ContextError::InvalidSequence(_))
    ));
    // Non-contiguous sequence
    let blocks = vec![
        block(
            0,
            BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("a"))]),
        ),
        block(
            2,
            BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("b"))]),
        ),
    ];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(2)),
        Err(ContextError::InvalidSequence(_))
    ));
    // Duplicate tool.call ids
    let blocks = vec![
        block(
            0,
            BlockContent::ToolCall(ToolCallPayload {
                call_id: ToolCallId::new("dup"),
                tool_name: "echo".into(),
                arguments: json!({}),
            }),
        ),
        block(
            1,
            BlockContent::ToolCall(ToolCallPayload {
                call_id: ToolCallId::new("dup"),
                tool_name: "echo".into(),
                arguments: json!({}),
            }),
        ),
    ];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(2)),
        Err(ContextError::DuplicateToolCallId(_))
    ));
    // Unpaired tool.result
    let blocks = vec![block(
        0,
        BlockContent::ToolResult(ToolResultPayload {
            call_id: ToolCallId::new("ghost"),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({})),
            media: Vec::new(),
        }),
    )];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks, ContextVersion(1)),
        Err(ContextError::UnpairedToolResult(_))
    ));
}

#[test]
fn provider_call_id_passes_through_draft_to_persisted_block() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    let out = ModelOutput {
        response: ModelResponse {
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
    let applied = c
        .append_model_output(inv, &out.response, out.stop_reason)
        .expect("record facts");
    let call_blocks: Vec<&ContextBlock> = c
        .blocks()
        .iter()
        .filter(|b| matches!(b.content, BlockContent::ToolCall(_)))
        .collect();
    assert_eq!(applied.block_ids.len(), 3);
    assert_eq!(call_blocks.len(), 2);
    // provider_call_id rides on envelope BlockMeta.
    assert_eq!(
        call_blocks[0].meta.provider_call_id.as_deref(),
        Some("call_provider_1")
    );
    assert_eq!(call_blocks[1].meta.provider_call_id, None);
    let blocks_json = serde_json::to_string(&c.snapshot_blocks()).unwrap();
    let blocks: Vec<ContextBlock> = serde_json::from_str(&blocks_json).unwrap();
    let restored: Vec<Option<String>> = blocks
        .iter()
        .filter(|b| matches!(b.content, BlockContent::ToolCall(_)))
        .map(|b| b.meta.provider_call_id.clone())
        .collect();
    assert_eq!(restored, vec![Some("call_provider_1".into()), None]);
}

#[test]
fn append_model_output_records_max_tokens_and_refusal_as_facts() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    let max_tokens = ModelOutput {
        response: ModelResponse {
            text: TextPayload::new("partial"),
            tool_calls: vec![],
        },
        usage: None,
        stop_reason: ModelStopReason::MaxTokens,
        reasoning: None,
    };
    let applied = c
        .append_model_output(inv.clone(), &max_tokens.response, max_tokens.stop_reason)
        .expect("MaxTokens is recordable");
    assert_eq!(applied.block_ids.len(), 1);
    let refusal_with_calls = ModelOutput {
        response: ModelResponse {
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
        c.append_model_output(
            inv.clone(),
            &refusal_with_calls.response,
            refusal_with_calls.stop_reason
        )
        .is_ok()
    );
    let bad_endturn = ModelOutput {
        response: ModelResponse {
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
        c.append_model_output(inv, &bad_endturn.response, bad_endturn.stop_reason),
        Err(ContextError::InvalidModelOutput(_))
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
    // provider_call_id is on BlockMeta (envelope-level); verify BlockMeta
    // serde-additivity.
    let legacy_meta: BlockMeta = serde_json::from_str("{}").unwrap();
    assert_eq!(legacy_meta.provider_call_id, None);
    assert_eq!(legacy_meta.source, None);
    let full_meta = BlockMeta {
        provider_call_id: Some("abc".into()),
        source: Some("kernel".into()),
    };
    let back: BlockMeta =
        serde_json::from_str(&serde_json::to_string(&full_meta).unwrap()).unwrap();
    assert_eq!(back.provider_call_id.as_deref(), Some("abc"));
    assert_eq!(back.source.as_deref(), Some("kernel"));
    // ToolCallPayload stays slim (no provider_call_id field).
    let legacy_call: ToolCallPayload =
        serde_json::from_str(r#"{"call_id":"echo:abcd1234:0","tool_name":"echo","arguments":{}}"#)
            .unwrap();
    assert_eq!(legacy_call.call_id.0, "echo:abcd1234:0");
}

// ---- compaction projection identity ----
// The budget/compaction policy lives in `causa-runtime`; the
// projection-identity test is in `causa-runtime/tests/budget.rs`. What
// stays here is the fact-machine side: the lossless frame is a pure
// function of the committed facts.

#[tokio::test]
async fn lossless_frame_is_a_pure_function_of_facts() {
    let mut c = ctx("t1");
    c.append_input(TextPayload::new("hello"), "user").unwrap();
    let f0 = c.frame(RoundId(0));
    let f1 = c.frame(RoundId(0));
    assert_eq!(f0.frame_id, f1.frame_id);
    assert_eq!(
        serde_json::to_string(&f0.model_context.blocks).unwrap(),
        serde_json::to_string(&f1.model_context.blocks).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&f0.model_context.blocks).unwrap(),
        serde_json::to_string(&c.snapshot_blocks()).unwrap()
    );
    assert_eq!(c.snapshot_blocks().len(), 1);
}

// ---- New: content is the only axis, no kind field ----

#[test]
fn content_is_first_class_with_three_shapes() {
    // Three content shapes: Text, ToolCall, ToolResult. No kind field.
    let t = turn_id("t1");
    let make = |content: BlockContent, seq: u64| ContextBlock {
        id: BlockId {
            turn_id: t.clone(),
            sequence: BlockSequence(seq),
        },
        sequence: BlockSequence(seq),
        content,
        meta: BlockMeta::default(),
    };

    let text = make(
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("any role"))]),
        0,
    );
    assert!(matches!(text.content, BlockContent::Parts(_)));

    let call_id = ToolCallId::new("echo:abcd1234:0");
    let call = make(
        BlockContent::ToolCall(ToolCallPayload {
            call_id: call_id.clone(),
            tool_name: "echo".into(),
            arguments: json!({}),
        }),
        1,
    );
    assert!(matches!(call.content, BlockContent::ToolCall(_)));

    let result = make(
        BlockContent::ToolResult(ToolResultPayload {
            call_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({})),
            media: Vec::new(),
        }),
        2,
    );
    assert!(matches!(result.content, BlockContent::ToolResult(_)));
}

#[test]
fn context_block_serde_format_is_flat_with_content() {
    // No kind field. The shape tag is "shape"; the value is the inner data.
    let mut c = ctx("t1");
    c.append_input(TextPayload::new("sys"), "user").unwrap();
    c.append_input(TextPayload::new("hi"), "user").unwrap();
    let blocks_json = serde_json::to_string(&c.snapshot_blocks()).unwrap();
    // No legacy kind field.
    assert!(!blocks_json.contains("\"kind\""));
    // Content with shape + value; the value is the ordered parts list
    // (no legacy text tag remains).
    assert!(!blocks_json.contains("\"shape\":\"text\""));
    assert!(blocks_json.contains(
        "\"content\":{\"shape\":\"parts\",\"value\":[{\"part\":\"text\",\"value\":\"sys\"}]}"
    ));
    assert!(blocks_json.contains(
        "\"content\":{\"shape\":\"parts\",\"value\":[{\"part\":\"text\",\"value\":\"hi\"}]}"
    ));
    // Round-trip works through the root facade.
    let restored: Vec<ContextBlock> = serde_json::from_str(&blocks_json).unwrap();
    assert_eq!(restored.len(), 2);
    assert!(matches!(restored[0].content, BlockContent::Parts(_)));
    assert!(matches!(restored[1].content, BlockContent::Parts(_)));
}

// ---- door contracts ---------------------------------------------------------

#[test]
fn foreign_invocation_is_rejected() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t2"),
        round_id: RoundId(0),
    };
    let out = endturn_output("hi");
    assert!(matches!(
        c.append_model_output(inv, &out.response, out.stop_reason),
        Err(ContextError::ForeignInvocation { .. })
    ));
    assert!(c.blocks().is_empty());
    assert_eq!(c.version(), ContextVersion(0));
}

#[test]
fn empty_output_commits_nothing_and_does_not_bump_version() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    let empty = ModelOutput {
        response: ModelResponse {
            text: TextPayload::new("   "),
            tool_calls: vec![],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    };
    let applied = c
        .append_model_output(inv, &empty.response, empty.stop_reason)
        .expect("empty output is recordable");
    assert!(applied.block_ids.is_empty());
    assert!(applied.tool_calls.is_empty());
    assert!(c.blocks().is_empty());
    // ContextVersion counts canonical fact commits, not attempts.
    assert_eq!(c.version(), ContextVersion(0));
}

#[test]
fn input_source_is_recorded_verbatim_in_envelope() {
    let mut c = ctx("t1");
    c.append_input(TextPayload::new("sys"), "system").unwrap();
    c.append_input(TextPayload::new("hi"), "user").unwrap();
    let blocks = c.blocks();
    assert_eq!(blocks[0].meta.source.as_deref(), Some("system"));
    assert_eq!(blocks[1].meta.source.as_deref(), Some("user"));
    // Source survives snapshot round-trip — the raw material for role
    // reconstruction at replay time.
    let json = serde_json::to_string(&c.snapshot_blocks()).unwrap();
    let back: Vec<ContextBlock> = serde_json::from_str(&json).unwrap();
    assert_eq!(back[0].meta.source.as_deref(), Some("system"));
    assert_eq!(back[1].meta.source.as_deref(), Some("user"));
}

#[test]
fn tool_results_commit_in_call_order_regardless_of_submission_order() {
    let mut c = ctx("t1");
    let inv = InvocationId {
        turn_id: turn_id("t1"),
        round_id: RoundId(0),
    };
    let out = ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(""),
            tool_calls: vec![
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"n": 1}),
                    provider_call_id: None,
                },
                ToolCallDraft {
                    tool_name: "echo".into(),
                    arguments: json!({"n": 2}),
                    provider_call_id: None,
                },
            ],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    };
    let applied = c
        .append_model_output(inv, &out.response, out.stop_reason)
        .unwrap();
    assert_eq!(applied.tool_calls.len(), 2);
    // The receipt preserves draft order and matches the committed call blocks.
    let call_ids: Vec<ToolCallId> = applied
        .tool_calls
        .iter()
        .map(|p| p.call_id.clone())
        .collect();
    let committed: Vec<ToolCallId> = c
        .blocks()
        .iter()
        .filter_map(|b| match &b.content {
            BlockContent::ToolCall(p) => Some(p.call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(call_ids, committed);
    assert_eq!(c.version(), ContextVersion(1));

    // Submit results in reverse completion order; the kernel commits in
    // the paired calls' block order.
    let results = vec![
        ToolResultPayload {
            call_id: call_ids[1].clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("second")),
            media: Vec::new(),
        },
        ToolResultPayload {
            call_id: call_ids[0].clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("first")),
            media: Vec::new(),
        },
    ];
    c.append_tool_results(results).unwrap();
    let result_order: Vec<ToolCallId> = c
        .blocks()
        .iter()
        .filter_map(|b| match &b.content {
            BlockContent::ToolResult(r) => Some(r.call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(result_order, call_ids);
    // The whole batch was one canonical commit: exactly one version bump.
    assert_eq!(c.version(), ContextVersion(2));
}

// ---- scope-driven frame identity ----------------------------------------------

#[test]
fn frame_id_from_scope_matches_deterministic_for_turn_scope() {
    // The Turn branch of from_scope must replicate the legacy preimage
    // byte-for-byte, so existing frame ids stay stable.
    let scope = FrameScope::Turn {
        turn_id: turn_id("t1"),
        source_version: ContextVersion(7),
    };
    assert_eq!(
        FrameId::from_scope(&scope, RoundId(3)),
        FrameId::deterministic(&turn_id("t1"), ContextVersion(7), RoundId(3))
    );
    // 16-hex truncation unchanged by the generalization.
    assert_eq!(
        FrameId::deterministic(&turn_id("t1"), ContextVersion(7), RoundId(3))
            .0
            .len(),
        16
    );
}

// ---- Parts vocabulary, media references, append_parts --------------------------

use causa_kernel::MediaRef;

#[test]
fn append_parts_commits_one_block_with_a_single_bump() {
    let mut c = ctx("t1");
    let id = c
        .append_parts(
            vec![
                ContentPart::Text(TextPayload::new("look at this")),
                ContentPart::Media(MediaRef::new("image/png", "asset-1")),
            ],
            "user",
        )
        .unwrap();
    // one logical message = one fact block, one version bump
    assert_eq!(c.blocks().len(), 1);
    assert_eq!(c.version(), ContextVersion(1));
    assert_eq!(c.blocks()[0].id, id);
    assert_eq!(c.blocks()[0].sequence, BlockSequence(0));
    // source stamped verbatim on the envelope
    assert_eq!(c.blocks()[0].meta.source.as_deref(), Some("user"));
    // part order preserved
    if let BlockContent::Parts(parts) = &c.blocks()[0].content {
        assert_eq!(parts.len(), 2);
        assert_eq!(
            parts[0],
            ContentPart::Text(TextPayload::new("look at this"))
        );
        assert_eq!(
            parts[1],
            ContentPart::Media(MediaRef::new("image/png", "asset-1"))
        );
    } else {
        panic!("expected a Parts block");
    }
}

#[test]
fn append_parts_rejects_empty_parts_without_committing() {
    let mut c = ctx("t1");
    let e = c.append_parts(vec![], "user").unwrap_err();
    assert!(matches!(e, ContextError::InvalidSequence(_)));
    assert!(c.blocks().is_empty());
    assert_eq!(c.version(), ContextVersion(0));
}

#[test]
fn append_parts_is_rejected_on_a_sealed_turn() {
    let mut c = ctx("t1");
    c.seal();
    let e = c
        .append_parts(vec![ContentPart::Text(TextPayload::new("late"))], "user")
        .unwrap_err();
    assert!(matches!(e, ContextError::SealedTurn));
}

#[test]
fn append_input_is_the_single_text_part_sugar() {
    let mut direct = ctx("t1");
    direct
        .append_parts(vec![ContentPart::Text(TextPayload::new("hi"))], "user")
        .unwrap();
    let mut sugar = ctx("t1");
    sugar.append_input(TextPayload::new("hi"), "user").unwrap();
    assert_eq!(
        serde_json::to_string(&direct.snapshot_blocks()).unwrap(),
        serde_json::to_string(&sugar.snapshot_blocks()).unwrap()
    );
}

#[test]
fn media_reference_round_trips_without_bytes_in_facts() {
    let mut c = ctx("t1");
    c.append_parts(
        vec![
            ContentPart::Text(TextPayload::new("caption")),
            ContentPart::Media(MediaRef::new("image/png", "blake3-asset-id")),
        ],
        "user",
    )
    .unwrap();
    let json = serde_json::to_string(&c.snapshot()).unwrap();
    // the reference is the only media content on the wire shape
    assert!(json.contains(
        r#""part":"media","value":{"media_type":"image/png","reference":"blake3-asset-id"}"#
    ));
    // snapshot size stays proportional to the reference, not to any payload
    assert!(json.len() < 800);
    let restored: causa_kernel::TurnSnapshot = serde_json::from_str(&json).unwrap();
    let causa_kernel::BlockContent::Parts(parts) = &restored.blocks.as_slice()[0].content else {
        panic!("expected Parts after round-trip");
    };
    assert_eq!(
        parts[1],
        ContentPart::Media(MediaRef::new("image/png", "blake3-asset-id"))
    );
}

#[test]
fn tool_result_media_is_serde_additive_both_ways() {
    let payload = ToolResultPayload {
        call_id: ToolCallId::new("c1"),
        status: ToolResultStatus::Succeeded,
        output: ToolOutput::new(json!("ok")),
        media: vec![MediaRef::new("image/png", "a1")],
    };
    let json = serde_json::to_string(&payload).unwrap();
    assert!(json.contains(r#""media":[{"media_type":"image/png","reference":"a1"}]"#));
    let restored: ToolResultPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.media, vec![MediaRef::new("image/png", "a1")]);

    // empty media is skipped on the wire...
    let empty = ToolResultPayload {
        call_id: ToolCallId::new("c2"),
        status: ToolResultStatus::Failed,
        output: ToolOutput::new(json!("no")),
        media: Vec::new(),
    };
    assert!(!serde_json::to_string(&empty).unwrap().contains("media"));
    // ...and a snapshot without the field defaults to empty.
    let old = json!({
        "call_id": "c1",
        "status": "Succeeded",
        "output": {"content": "ok", "truncation": "none", "meta": null, "artifact": null},
    });
    let from_old: ToolResultPayload = serde_json::from_value(old).unwrap();
    assert!(from_old.media.is_empty());
}

#[test]
fn empty_parts_block_is_rejected_on_recovery() {
    let mut c = ctx("t1");
    c.append_input(TextPayload::new("hi"), "user").unwrap();
    let mut json = serde_json::to_value(c.snapshot()).unwrap();
    // Corrupt the history: a Parts block the doors could never produce.
    json["blocks"][0]["content"]["value"] = json!([]);
    let snap: causa_kernel::TurnSnapshot = serde_json::from_value(json).unwrap();
    let err = causa_kernel::TurnContext::from_validated_blocks(
        snap.turn_id.clone(),
        snap.blocks.into_inner(),
        snap.source_version,
    )
    .unwrap_err();
    assert!(matches!(err, ContextError::InvalidSequence(_)));
}
