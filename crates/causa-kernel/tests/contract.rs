//! Contract tests -- canonical facts, validated transitions, deterministic
//! projections, and fact fidelity. No runtime involved; every
//! import comes from the public root facade.

mod common;

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, ContentPart, ContextBlock, ContextError, EditError,
    InvocationId, ModelOutput, ModelResponse, ModelStopReason, ModelUsage, ReasoningPayload,
    RoundId, TextPayload, ToolCallDraft, ToolCallPayload, ToolOutput, ToolResultPayload,
    ToolResultStatus, TurnContext,
};
use common::{ctx, endturn_output, turn_id};
use serde_json::json;

#[tokio::test]
async fn empty_frame_deterministic() {
    let c = ctx("t1");
    let f0 = c.frame(RoundId(0));
    let f1 = c.frame(RoundId(0));
    assert_eq!(f0.scope, f1.scope);
    assert!(f0.model_context.blocks.is_empty());
}

#[tokio::test]
async fn append_input_and_frame_order() {
    let mut c = ctx("t1");
    c.append_input(common::block_id(), TextPayload::new("hello"), "user")
        .unwrap();
    c.append_input(common::block_id(), TextPayload::new("sys"), "user")
        .unwrap();
    let f = c.frame(RoundId(0));
    assert_eq!(f.model_context.blocks.len(), 2);
    // Order preserved.
    assert!(matches!(
        f.model_context.blocks[0].content(),
        BlockContent::Parts(_)
    ));
    assert!(matches!(
        f.model_context.blocks[1].content(),
        BlockContent::Parts(_)
    ));
    // Sealed turn rejects further append.
    c.seal();
    let mut sealed = c;
    assert!(matches!(
        sealed.append_input(common::block_id(), TextPayload::new("x"), "user"),
        Err(ContextError::Edit(EditError::SealedTurn))
    ));
}

#[tokio::test]
async fn sealed_turn_append_closed() {
    let mut c = ctx("t1");
    c.append_input(common::block_id(), TextPayload::new("hi"), "user")
        .unwrap();
    c.seal();
    assert!(c.is_sealed());
    let mut sealed = c;
    assert!(matches!(
        sealed.append_input(common::block_id(), TextPayload::new("x"), "user"),
        Err(ContextError::Edit(EditError::SealedTurn))
    ));
    assert!(matches!(
        sealed.append_model_output(
            InvocationId {
                turn_id: turn_id("t1"),
                round_id: RoundId(1)
            },
            &endturn_output("y").response,
            ModelStopReason::EndTurn,
            common::block_ids_for(&endturn_output("y").response)
        ),
        Err(ContextError::Edit(EditError::SealedTurn))
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
        c.append_model_output(
            inv.clone(),
            &bad_endturn.response,
            bad_endturn.stop_reason,
            common::block_ids_for(&bad_endturn.response)
        ),
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
        c.append_model_output(
            inv.clone(),
            &empty_name.response,
            empty_name.stop_reason,
            common::block_ids_for(&empty_name.response)
        ),
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
        c.append_model_output(
            inv.clone(),
            &non_object.response,
            non_object.stop_reason,
            common::block_ids_for(&non_object.response)
        ),
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
        c.append_model_output(
            inv.clone(),
            &dup_batch.response,
            dup_batch.stop_reason,
            common::block_ids_for(&dup_batch.response)
        )
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
            ModelStopReason::EndTurn,
            common::block_ids_for(&endturn_output("y").response)
        ),
        Err(ContextError::Edit(EditError::SealedTurn))
    ));
}

#[test]
fn from_validated_blocks_checks_only_context_local_identity() {
    fn block(id: BlockId, content: BlockContent) -> ContextBlock {
        ContextBlock::new(id, content, BlockMeta::default())
    }
    let duplicate = common::block_id();
    let blocks = vec![
        block(
            duplicate,
            BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("a"))]),
        ),
        block(
            duplicate,
            BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("b"))]),
        ),
    ];
    assert!(matches!(
        TurnContext::from_validated_blocks(turn_id("t1"), blocks),
        Err(ContextError::Edit(EditError::DuplicateBlockId(_)))
    ));
    let call_id = common::block_id();
    let partial_material = vec![
        block(
            common::block_id(),
            BlockContent::ToolResult(ToolResultPayload {
                call_block_id: call_id,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!({})),
                media: Vec::new(),
                notes: Vec::new(),
            }),
        ),
        block(common::block_id(), BlockContent::Parts(Vec::new())),
        block(
            common::block_id(),
            BlockContent::ToolResult(ToolResultPayload {
                call_block_id: call_id,
                status: ToolResultStatus::Failed,
                output: ToolOutput::new(json!("later")),
                media: Vec::new(),
                notes: Vec::new(),
            }),
        ),
    ];
    let imported = TurnContext::from_validated_blocks(turn_id("t1"), partial_material).unwrap();
    assert_eq!(imported.blocks().len(), 3);
    let wire = serde_json::to_string(&imported).unwrap();
    let restored: TurnContext = serde_json::from_str(&wire).unwrap();
    assert_eq!(restored.blocks(), imported.blocks());
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
        .append_model_output(
            inv,
            &out.response,
            out.stop_reason,
            common::block_ids_for(&out.response),
        )
        .expect("record facts");
    let call_blocks: Vec<&ContextBlock> = c
        .blocks()
        .iter()
        .filter(|b| matches!(b.content(), BlockContent::ToolCall(_)))
        .collect();
    assert_eq!(applied.block_ids.len(), 3);
    assert_eq!(call_blocks.len(), 2);
    // provider_call_id rides on envelope BlockMeta.
    assert_eq!(
        call_blocks[0].meta().provider_call_id.as_deref(),
        Some("call_provider_1")
    );
    assert_eq!(call_blocks[1].meta().provider_call_id, None);
    let blocks_json = serde_json::to_string(&c.blocks()).unwrap();
    let blocks: Vec<ContextBlock> = serde_json::from_str(&blocks_json).unwrap();
    let restored: Vec<Option<String>> = blocks
        .iter()
        .filter(|b| matches!(b.content(), BlockContent::ToolCall(_)))
        .map(|b| b.meta().provider_call_id.clone())
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
        .append_model_output(
            inv.clone(),
            &max_tokens.response,
            max_tokens.stop_reason,
            common::block_ids_for(&max_tokens.response),
        )
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
            refusal_with_calls.stop_reason,
            common::block_ids_for(&refusal_with_calls.response)
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
        c.append_model_output(
            inv,
            &bad_endturn.response,
            bad_endturn.stop_reason,
            common::block_ids_for(&bad_endturn.response)
        ),
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
    let call = ToolCallPayload {
        tool_name: "echo".into(),
        arguments: json!({"a": 1}),
    };
    let round_trip: ToolCallPayload =
        serde_json::from_str(&serde_json::to_string(&call).unwrap()).unwrap();
    assert_eq!(round_trip, call);
}

// ---- compaction projection identity ----
// The budget/compaction policy lives in `causa-runtime`; the
// projection-identity test is in `causa-runtime/tests/budget.rs`. What
// stays here is the fact-machine side: the lossless frame is a pure
// function of the committed facts.

#[tokio::test]
async fn lossless_frame_is_a_pure_function_of_facts() {
    let mut c = ctx("t1");
    c.append_input(common::block_id(), TextPayload::new("hello"), "user")
        .unwrap();
    let f0 = c.frame(RoundId(0));
    let f1 = c.frame(RoundId(0));
    assert_eq!(f0.scope, f1.scope);
    assert_eq!(
        serde_json::to_string(&f0.model_context.blocks).unwrap(),
        serde_json::to_string(&f1.model_context.blocks).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&f0.model_context.blocks).unwrap(),
        serde_json::to_string(&c.blocks()).unwrap()
    );
    assert_eq!(c.blocks().len(), 1);
}

// ---- New: content is the only axis, no kind field ----

#[test]
fn content_is_first_class_with_three_shapes() {
    // Three content shapes: Text, ToolCall, ToolResult. No kind field.
    let make = |content: BlockContent| {
        ContextBlock::new(common::block_id(), content, BlockMeta::default())
    };

    let text = make(BlockContent::Parts(vec![ContentPart::Text(
        TextPayload::new("any role"),
    )]));
    assert!(matches!(text.content(), BlockContent::Parts(_)));

    let call = make(BlockContent::ToolCall(ToolCallPayload {
        tool_name: "echo".into(),
        arguments: json!({}),
    }));
    assert!(matches!(call.content(), BlockContent::ToolCall(_)));

    let result = make(BlockContent::ToolResult(ToolResultPayload {
        call_block_id: call.id(),
        status: ToolResultStatus::Succeeded,
        output: ToolOutput::new(json!({})),
        media: Vec::new(),
        notes: Vec::new(),
    }));
    assert!(matches!(result.content(), BlockContent::ToolResult(_)));
}

#[test]
fn context_block_serde_format_is_flat_with_content() {
    // No kind field. The shape tag is "shape"; the value is the inner data.
    let mut c = ctx("t1");
    c.append_input(common::block_id(), TextPayload::new("sys"), "user")
        .unwrap();
    c.append_input(common::block_id(), TextPayload::new("hi"), "user")
        .unwrap();
    let blocks_json = serde_json::to_string(&c.blocks()).unwrap();
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
    assert!(matches!(restored[0].content(), BlockContent::Parts(_)));
    assert!(matches!(restored[1].content(), BlockContent::Parts(_)));
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
        c.append_model_output(
            inv,
            &out.response,
            out.stop_reason,
            common::block_ids_for(&out.response)
        ),
        Err(ContextError::ForeignInvocation { .. })
    ));
    assert!(c.blocks().is_empty());
}

#[test]
fn empty_output_commits_nothing() {
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
        .append_model_output(
            inv,
            &empty.response,
            empty.stop_reason,
            common::block_ids_for(&empty.response),
        )
        .expect("empty output is recordable");
    assert!(applied.block_ids.is_empty());
    assert!(applied.tool_calls.is_empty());
    assert!(c.blocks().is_empty());
}

#[test]
fn input_source_is_recorded_verbatim_in_envelope() {
    let mut c = ctx("t1");
    c.append_input(common::block_id(), TextPayload::new("sys"), "system")
        .unwrap();
    c.append_input(common::block_id(), TextPayload::new("hi"), "user")
        .unwrap();
    let blocks = c.blocks();
    assert_eq!(blocks[0].meta().source.as_deref(), Some("system"));
    assert_eq!(blocks[1].meta().source.as_deref(), Some("user"));
    // Source survives a direct TurnContext round-trip — the raw material for role
    // reconstruction at replay time.
    let json = serde_json::to_string(&c).unwrap();
    let back: TurnContext = serde_json::from_str(&json).unwrap();
    assert_eq!(back.blocks()[0].meta().source.as_deref(), Some("system"));
    assert_eq!(back.blocks()[1].meta().source.as_deref(), Some("user"));
}

#[test]
fn tool_results_commit_in_supplied_order_with_identity_bound_to_payload() {
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
        .append_model_output(
            inv,
            &out.response,
            out.stop_reason,
            common::block_ids_for(&out.response),
        )
        .unwrap();
    assert_eq!(applied.tool_calls.len(), 2);
    // The receipt preserves draft order and matches the committed call blocks.
    let call_ids: Vec<BlockId> = applied.tool_calls.iter().map(|(id, _)| *id).collect();
    let committed: Vec<BlockId> = c
        .blocks()
        .iter()
        .filter_map(|b| match b.content() {
            BlockContent::ToolCall(_) => Some(b.id()),
            _ => None,
        })
        .collect();
    assert_eq!(call_ids, committed);

    // Submit in reverse declaration order; the kernel preserves this order.
    let result_ids = [common::block_id(), common::block_id()];
    let results = vec![
        (
            result_ids[0],
            ToolResultPayload {
                call_block_id: call_ids[1],
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("second")),
                media: Vec::new(),
                notes: Vec::new(),
            },
        ),
        (
            result_ids[1],
            ToolResultPayload {
                call_block_id: call_ids[0],
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("first")),
                media: Vec::new(),
                notes: Vec::new(),
            },
        ),
    ];
    let committed_result_ids = c.append_tool_results(results).unwrap();
    assert_eq!(committed_result_ids, result_ids);
    let result_blocks: Vec<&ContextBlock> = c
        .blocks()
        .iter()
        .filter(|b| matches!(b.content(), BlockContent::ToolResult(_)))
        .collect();
    assert_eq!(result_blocks[0].id(), result_ids[0]);
    assert!(
        matches!(result_blocks[0].content(), BlockContent::ToolResult(r) if r.call_block_id == call_ids[1])
    );
    assert_eq!(result_blocks[1].id(), result_ids[1]);
    assert!(
        matches!(result_blocks[1].content(), BlockContent::ToolResult(r) if r.call_block_id == call_ids[0])
    );
    // The batch preserves its supplied commit order.
}

// ---- Parts vocabulary, media references, append_parts --------------------------

use causa_kernel::MediaRef;

#[test]
fn append_parts_commits_one_block() {
    let mut c = ctx("t1");
    let id = c
        .append_parts(
            common::block_id(),
            vec![
                ContentPart::Text(TextPayload::new("look at this")),
                ContentPart::Media(MediaRef::new("image/png", "asset-1")),
            ],
            "user",
        )
        .unwrap();
    // one logical message = one fact block
    assert_eq!(c.blocks().len(), 1);
    assert_eq!(c.blocks()[0].id(), id);
    // source stamped verbatim on the envelope
    assert_eq!(c.blocks()[0].meta().source.as_deref(), Some("user"));
    // part order preserved
    if let BlockContent::Parts(parts) = c.blocks()[0].content() {
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
    let e = c
        .append_parts(common::block_id(), vec![], "user")
        .unwrap_err();
    assert!(matches!(e, ContextError::InvalidContext(_)));
    assert!(c.blocks().is_empty());
}

#[test]
fn append_parts_is_rejected_on_a_sealed_turn() {
    let mut c = ctx("t1");
    c.seal();
    let e = c
        .append_parts(
            common::block_id(),
            vec![ContentPart::Text(TextPayload::new("late"))],
            "user",
        )
        .unwrap_err();
    assert!(matches!(e, ContextError::Edit(EditError::SealedTurn)));
}

#[test]
fn append_input_is_the_single_text_part_sugar() {
    let mut direct = ctx("t1");
    let id = common::block_id();
    direct
        .append_parts(id, vec![ContentPart::Text(TextPayload::new("hi"))], "user")
        .unwrap();
    let mut sugar = ctx("t1");
    sugar
        .append_input(id, TextPayload::new("hi"), "user")
        .unwrap();
    assert_eq!(
        serde_json::to_string(&direct.blocks()).unwrap(),
        serde_json::to_string(&sugar.blocks()).unwrap()
    );
}

#[test]
fn media_reference_round_trips_without_bytes_in_facts() {
    let mut c = ctx("t1");
    c.append_parts(
        common::block_id(),
        vec![
            ContentPart::Text(TextPayload::new("caption")),
            ContentPart::Media(MediaRef::new("image/png", "blake3-asset-id")),
        ],
        "user",
    )
    .unwrap();
    let json = serde_json::to_string(&c).unwrap();
    // the reference is the only media content on the wire shape
    assert!(json.contains(
        r#""part":"media","value":{"media_type":"image/png","reference":"blake3-asset-id"}"#
    ));
    // The saved turn stays proportional to the reference, not to any payload.
    assert!(json.len() < 800);
    let restored: TurnContext = serde_json::from_str(&json).unwrap();
    let BlockContent::Parts(parts) = restored.blocks()[0].content() else {
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
        call_block_id: common::block_id(),
        status: ToolResultStatus::Succeeded,
        output: ToolOutput::new(json!("ok")),
        media: vec![MediaRef::new("image/png", "a1")],
        notes: vec![TextPayload::new("note")],
    };
    let json = serde_json::to_string(&payload).unwrap();
    assert!(json.contains(r#""media":[{"media_type":"image/png","reference":"a1"}]"#));
    let restored: ToolResultPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.media, vec![MediaRef::new("image/png", "a1")]);

    // empty media is skipped on the wire...
    let empty = ToolResultPayload {
        call_block_id: common::block_id(),
        status: ToolResultStatus::Failed,
        output: ToolOutput::new(json!("no")),
        media: Vec::new(),
        notes: Vec::new(),
    };
    assert!(!serde_json::to_string(&empty).unwrap().contains("media"));
    // ...and a stored record without the field defaults to empty.
    let old = json!({
        "call_block_id": common::block_id(),
        "status": "Succeeded",
        "output": {"content": "ok", "truncation": "none", "meta": null, "artifact": null},
    });
    let from_old: ToolResultPayload = serde_json::from_value(old).unwrap();
    assert!(from_old.media.is_empty());
    assert!(from_old.notes.is_empty());
}

#[test]
fn turn_context_serde_requires_all_fields_and_preserves_partial_material() {
    let mut c = ctx("t1");
    c.append_input(common::block_id(), TextPayload::new("hi"), "user")
        .unwrap();
    c.seal();
    let value = serde_json::to_value(&c).unwrap();
    let restored: TurnContext = serde_json::from_value(value.clone()).unwrap();
    assert!(restored.is_sealed());
    assert_eq!(restored.blocks(), c.blocks());

    let mut open = TurnContext::new(turn_id("empty"));
    let empty_open = serde_json::to_value(&open).unwrap();
    assert_eq!(empty_open["lifecycle"], "open");
    assert!(serde_json::from_value::<TurnContext>(empty_open.clone()).is_ok());
    open.seal();
    let empty_sealed = serde_json::to_value(&open).unwrap();
    assert_eq!(empty_sealed["lifecycle"], "sealed");
    assert!(
        serde_json::from_value::<TurnContext>(empty_sealed)
            .unwrap()
            .is_sealed()
    );

    for field in ["turn_id", "blocks", "lifecycle"] {
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<TurnContext>(missing).is_err());
    }
    let mut unknown_lifecycle = value.clone();
    unknown_lifecycle["lifecycle"] = json!("paused");
    assert!(serde_json::from_value::<TurnContext>(unknown_lifecycle).is_err());

    let mut duplicate = value;
    duplicate["blocks"] = json!([c.blocks()[0], c.blocks()[0]]);
    assert!(serde_json::from_value::<TurnContext>(duplicate).is_err());

    let partial: TurnContext = serde_json::from_value(json!({
        "turn_id": "partial-material",
        "blocks": [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "content": {"shape": "tool_result", "value": {
                    "call_block_id": "00000000-0000-0000-0000-000000000003",
                    "status": "Succeeded",
                    "output": {"content": "first", "truncation": "none", "meta": null, "artifact": null}
                }},
                "meta": {}
            },
            {
                "id": "00000000-0000-0000-0000-000000000002",
                "content": {"shape": "parts", "value": []},
                "meta": {}
            },
            {
                "id": "00000000-0000-0000-0000-000000000004",
                "content": {"shape": "tool_result", "value": {
                    "call_block_id": "00000000-0000-0000-0000-000000000003",
                    "status": "Failed",
                    "output": {"content": "second", "truncation": "none", "meta": null, "artifact": null}
                }},
                "meta": {}
            }
        ],
        "lifecycle": "open"
    }))
    .unwrap();
    let partial_round_trip: TurnContext =
        serde_json::from_value(serde_json::to_value(&partial).unwrap()).unwrap();
    assert_eq!(partial_round_trip.blocks(), partial.blocks());
    assert_eq!(
        partial_round_trip
            .blocks()
            .iter()
            .map(|block| block.id())
            .collect::<Vec<_>>(),
        vec![
            BlockId::new(uuid::Uuid::from_u128(1)),
            BlockId::new(uuid::Uuid::from_u128(2)),
            BlockId::new(uuid::Uuid::from_u128(4)),
        ]
    );
}
