//! Material identity is independent of invocation identity and content keys.

mod common;

use causa_kernel::{
    BlockContent, BlockId, Context, EditError, ModelResponse, ModelStopReason, TextPayload,
    ToolCallDraft,
};
use serde_json::json;

#[test]
fn block_id_is_a_uuid_value_with_no_legacy_position_shape() {
    let id = common::block_id();
    let encoded = serde_json::to_string(&id).unwrap();
    assert_eq!(serde_json::from_str::<BlockId>(&encoded).unwrap(), id);
    assert!(serde_json::from_str::<BlockId>(r#""not-a-uuid""#).is_err());
    assert!(serde_json::from_str::<BlockId>(r#"{"turn_id":"t","sequence":0}"#).is_err());
}

#[test]
fn identical_calls_have_distinct_material_ids() {
    let mut context = Context::new();
    let response = ModelResponse {
        text: TextPayload::new(" "),
        tool_calls: vec![ToolCallDraft {
            tool_name: "lookup".into(),
            arguments: json!({"query":"same"}),
            provider_call_id: None,
        }],
    };
    assert_eq!(response.block_count(), 1);
    for _ in 0..2 {
        let id = common::block_id();
        context
            .apply(
                Vec::new(),
                response.to_blocks(ModelStopReason::ToolUse, &[id]).unwrap(),
            )
            .unwrap();
    }
    assert_ne!(context.blocks()[0].id(), context.blocks()[1].id());
    assert_eq!(context.blocks()[0].content(), context.blocks()[1].content());
    assert!(matches!(
        context.blocks()[0].content(),
        BlockContent::ToolCall(_)
    ));
}

#[test]
fn duplicate_appends_reject_atomically_and_return_all_material() {
    let id = common::block_id();
    let block = common::text_block(id, TextPayload::new("kept"), "user");
    let mut context = Context::from_blocks(vec![block.clone()]).unwrap();
    let before = serde_json::to_value(&context).unwrap();
    let changed = common::text_block(id, TextPayload::new("rejected"), "user");
    let error = context
        .apply(Vec::new(), vec![changed.clone()])
        .unwrap_err();
    assert_eq!(error.reason, EditError::BlockIdentityMismatch(id));
    assert_eq!(error.appended, vec![changed]);
    let error = context.apply(Vec::new(), vec![block.clone()]).unwrap_err();
    assert_eq!(error.reason, EditError::DuplicateBlockId(id));
    assert_eq!(error.appended, vec![block]);
    assert_eq!(serde_json::to_value(&context).unwrap(), before);
}

#[test]
fn model_count_ignores_blank_text_and_counts_each_declaration() {
    let response = ModelResponse {
        text: TextPayload::new("\n  \t"),
        tool_calls: vec![common::draft("a", json!({})), common::draft("b", json!({}))],
    };
    assert_eq!(response.block_count(), 2);
}

#[test]
fn import_failure_returns_duplicate_blocks_and_independent_import_keeps_ids() {
    let block = common::text_block(common::block_id(), TextPayload::new("same fact"), "user");
    let inputs = vec![block.clone(), block.clone()];
    let error = Context::from_blocks(inputs.clone()).unwrap_err();
    assert_eq!(error.reason, EditError::DuplicateBlockId(block.id()));
    assert_eq!(error.appended, inputs);
    assert!(error.replacements.is_empty());
    let original = Context::from_blocks(vec![block.clone()]).unwrap();
    let restored = Context::from_blocks(vec![block.clone()]).unwrap();
    assert_eq!(original.blocks(), restored.blocks());
    assert_eq!(restored.into_blocks(), vec![block]);
}

#[test]
fn pure_conversion_and_atomic_submission_have_distinct_validation_responsibilities() {
    let existing = common::block_id();
    let mut context = Context::from_blocks(vec![common::text_block(
        existing,
        TextPayload::new("initial"),
        "user",
    )])
    .unwrap();
    let response = ModelResponse {
        text: TextPayload::new("searching"),
        tool_calls: vec![common::draft("search", json!({}))],
    };
    let a = common::block_id();
    let b = common::block_id();
    let before = serde_json::to_value(&context).unwrap();
    assert!(response.to_blocks(ModelStopReason::ToolUse, &[a]).is_err());
    for ids in [vec![a, a], vec![a, existing]] {
        let converted = response.to_blocks(ModelStopReason::ToolUse, &ids).unwrap();
        let error = context.apply(Vec::new(), converted.clone()).unwrap_err();
        assert_eq!(error.appended, converted);
        assert_eq!(serde_json::to_value(&context).unwrap(), before);
    }
    context
        .apply(
            Vec::new(),
            response
                .to_blocks(ModelStopReason::ToolUse, &[a, b])
                .unwrap(),
        )
        .unwrap();
    assert!(matches!(
        context.blocks()[1].content(),
        BlockContent::Parts(_)
    ));
    assert_eq!(context.blocks()[1].id(), a);
    assert_eq!(context.blocks()[2].id(), b);
}
