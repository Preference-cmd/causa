//! Direct consumer tests for Slice 8 Context editing (Q4/Q5).
//!
//! Copy into `crates/causa-kernel/tests/` once the edit API is implemented.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, ContentPart, Context, ContextBlock, Replacement, TextPayload,
    ToolCallPayload, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use serde_json::json;
use std::ops::Range;

fn id(n: u128) -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(n))
}

fn note(id_value: u128, text: &str, source: &str) -> ContextBlock {
    ContextBlock::new(
        id(id_value),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
        BlockMeta {
            source: Some(source.to_owned()),
            provider_call_id: None,
        },
    )
}

fn call(id_value: u128, name: &str) -> ContextBlock {
    ContextBlock::new(
        id(id_value),
        BlockContent::ToolCall(ToolCallPayload {
            tool_name: name.to_owned(),
            arguments: json!({"round": id_value}),
        }),
        BlockMeta {
            source: Some("model".to_owned()),
            provider_call_id: Some(format!("provider-call-{id_value}")),
        },
    )
}

fn result(id_value: u128, call_block_id: BlockId, value: &str) -> ContextBlock {
    ContextBlock::new(
        id(id_value),
        BlockContent::ToolResult(ToolResultPayload {
            call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({"value": value})),
            media: Vec::new(),
            notes: Vec::new(),
        }),
        BlockMeta {
            source: Some("tool-result".to_owned()),
            provider_call_id: None,
        },
    )
}

fn replacement(range: Range<usize>, with: Vec<ContextBlock>) -> Replacement {
    Replacement { range, with }
}

fn context(blocks: Vec<ContextBlock>) -> Context {
    Context::from_blocks(blocks).unwrap()
}

fn assert_frame_is_current(ctx: &Context) {
    let frame = ctx.frame();
    assert_eq!(frame.blocks, ctx.blocks());
}

#[test]
fn saved_material_and_full_turn_survive_local_summary_reuse_and_later_edit() {
    // Two model/tool rounds are facts in one current turn. The caller chooses
    // what to retain; the edit API does not decide archival or lineage policy.
    let original_blocks = vec![
        note(1, "research question", "user"),
        call(2, "search"),
        result(3, id(2), "first source"),
        note(4, "intermediate finding", "model"),
        call(5, "fetch"),
        result(6, id(5), "second source"),
        note(7, "current instruction", "user"),
    ];
    let mut ctx = context(original_blocks.clone());
    let selected = ctx.blocks()[1..3].to_vec();
    let full_turn = ctx.clone();

    // Replace only the first round's call/result pair with a local summary.
    ctx.apply(
        vec![replacement(
            1..3,
            vec![note(20, "first-round summary", "host")],
        )],
        Vec::new(),
    )
    .unwrap();
    assert_eq!(ctx.blocks()[0].id(), id(1));
    assert_eq!(ctx.blocks()[1].id(), id(20));
    assert_eq!(ctx.blocks()[2..], original_blocks[3..]);
    assert_eq!(selected, original_blocks[1..3]);
    assert_eq!(full_turn.blocks(), original_blocks);
    assert_frame_is_current(&ctx);

    // Explicitly reinsert caller-owned saved materials, preserving their
    // complete records and IDs, then perform another ordinary local edit.
    ctx.apply(vec![replacement(1..2, selected.clone())], Vec::new())
        .unwrap();
    assert_eq!(ctx.blocks()[1..3], selected);
    assert_eq!(selected, original_blocks[1..3]);

    ctx.apply(
        vec![replacement(
            1..3,
            vec![note(21, "revised first-round summary", "host")],
        )],
        Vec::new(),
    )
    .unwrap();
    assert!(ctx.blocks().iter().all(|block| block.id() != id(2)));
    assert!(ctx.blocks().iter().all(|block| block.id() != id(3)));
    assert_eq!(selected, original_blocks[1..3]);
    assert_eq!(full_turn.blocks(), original_blocks);
    assert_frame_is_current(&ctx);
}

#[test]
fn generic_edit_accepts_local_tool_material_without_pairing_or_execution_policy() {
    let non_call_target = note(1, "ordinary fact", "user");
    let mut ctx = context(vec![non_call_target]);

    // The first result is isolated (its declaration is absent). The next two
    // are multiple results for the same absent declaration. Another result
    // refers to a Parts block, and the last result precedes its ToolCall
    // declaration. Generic material editing preserves all these supplied
    // facts in order; it does not execute calls or synthesize missing facts.
    let supplied = vec![
        result(2, id(900), "isolated"),
        result(3, id(901), "first result"),
        result(4, id(901), "second result"),
        result(5, id(1), "reference points to Parts"),
        result(6, id(7), "result appears before declaration"),
        call(7, "later-declared-tool"),
    ];
    ctx.apply(vec![replacement(1..1, supplied.clone())], Vec::new())
        .unwrap();

    let expected_ids = [1, 2, 3, 4, 5, 6, 7].map(id);
    assert_eq!(
        ctx.blocks()
            .iter()
            .map(ContextBlock::id)
            .collect::<Vec<_>>(),
        expected_ids
    );
    assert_eq!(ctx.blocks()[1..], supplied);
    assert_frame_is_current(&ctx);
}

#[test]
fn context_serializes_only_blocks_and_clones_preserve_material() {
    let context = context(vec![note(40, "material", "host")]);
    let cloned = context.clone();
    assert_eq!(cloned.blocks(), context.blocks());
    let encoded = serde_json::to_value(&context).unwrap();
    let object = encoded.as_object().unwrap();
    assert_eq!(object.len(), 1, "unexpected serialized fields: {object:?}");
    assert!(object.contains_key("blocks"));
    let restored: Context = serde_json::from_value(encoded).unwrap();
    assert_eq!(restored.into_blocks(), context.into_blocks());
}
