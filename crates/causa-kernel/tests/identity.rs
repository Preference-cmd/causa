mod common;

use causa_kernel::{
    BlockContent, BlockId, ContextError, InvocationId, ModelResponse, ModelStopReason, RoundId,
    TextPayload, ToolCallDraft, TurnContext, TurnId, model_output_block_count,
};
use serde_json::json;

#[test]
fn block_id_is_a_uuid_value_with_no_legacy_position_shape() {
    let id = common::block_id();
    let encoded = serde_json::to_string(&id).unwrap();
    let decoded: BlockId = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, id);
    assert!(serde_json::from_str::<BlockId>(r#""not-a-uuid""#).is_err());
    assert!(serde_json::from_str::<BlockId>(r#"{"turn_id":"t","sequence":0}"#).is_err());
}

#[test]
fn same_invocation_can_record_identical_calls_with_distinct_block_ids() {
    let mut context = TurnContext::new(TurnId::new("identity-turn"));
    let invocation = InvocationId {
        turn_id: context.turn_id(),
        round_id: RoundId(4),
    };
    let response = ModelResponse {
        text: TextPayload::new(" "),
        tool_calls: vec![ToolCallDraft {
            tool_name: "lookup".into(),
            arguments: json!({"query": "same"}),
            provider_call_id: None,
        }],
    };
    assert_eq!(model_output_block_count(&response), 1);
    let first_id = common::block_id();
    let second_id = common::block_id();
    assert_ne!(first_id, second_id);
    let first = context
        .append_model_output(
            invocation.clone(),
            &response,
            ModelStopReason::ToolUse,
            vec![first_id],
        )
        .unwrap();
    let second = context
        .append_model_output(
            invocation,
            &response,
            ModelStopReason::ToolUse,
            vec![second_id],
        )
        .unwrap();
    assert_eq!(first.tool_calls[0].1, second.tool_calls[0].1);
    assert_eq!(first.tool_calls[0].0, first_id);
    assert_eq!(second.tool_calls[0].0, second_id);
    assert!(matches!(
        context.blocks()[0].content(),
        BlockContent::ToolCall(_)
    ));
    assert!(matches!(
        context.blocks()[1].content(),
        BlockContent::ToolCall(_)
    ));
}

#[test]
fn duplicate_imported_id_is_rejected_without_any_context_mutation() {
    let mut context = TurnContext::new(TurnId::new("atomic-turn"));
    let id = common::block_id();
    context
        .append_input(id, TextPayload::new("kept"), "user")
        .unwrap();
    let before = serde_json::to_string(&context).unwrap();
    let err = context
        .append_input(id, TextPayload::new("rejected"), "user")
        .unwrap_err();
    assert!(matches!(err, ContextError::DuplicateBlockId(found) if found == id));
    assert_eq!(before, serde_json::to_string(&context).unwrap());
}

#[test]
fn model_output_count_ignores_blank_text_and_counts_each_declaration() {
    let response = ModelResponse {
        text: TextPayload::new("\n  \t"),
        tool_calls: vec![
            ToolCallDraft {
                tool_name: "a".into(),
                arguments: json!({}),
                provider_call_id: None,
            },
            ToolCallDraft {
                tool_name: "b".into(),
                arguments: json!({}),
                provider_call_id: None,
            },
        ],
    };
    assert_eq!(model_output_block_count(&response), 2);
}

#[test]
fn importing_duplicate_blocks_rejects_the_material() {
    let mut context = TurnContext::new(TurnId::new("source"));
    context
        .append_input(common::block_id(), TextPayload::new("same fact"), "user")
        .unwrap();
    let block = context.blocks()[0].clone();
    let result = TurnContext::from_validated_blocks(
        TurnId::new("destination"),
        vec![block.clone(), block.clone()],
    );
    assert!(matches!(result, Err(ContextError::DuplicateBlockId(id)) if id == block.id()));
    // The same material can independently appear in another context.
    let restored =
        TurnContext::from_validated_blocks(TurnId::new("destination"), vec![block.clone()])
            .unwrap();
    assert_eq!(restored.blocks()[0].id(), block.id());
}

#[test]
fn model_ids_bind_text_then_calls_and_all_validation_is_atomic() {
    let mut context = TurnContext::new(TurnId::new("binding"));
    let existing = common::block_id();
    context
        .append_input(existing, TextPayload::new("initial"), "user")
        .unwrap();
    let response = ModelResponse {
        text: TextPayload::new("searching"),
        tool_calls: vec![ToolCallDraft {
            tool_name: "search".into(),
            arguments: json!({}),
            provider_call_id: None,
        }],
    };
    let invocation = InvocationId {
        turn_id: context.turn_id(),
        round_id: RoundId(0),
    };
    let a = common::block_id();
    let b = common::block_id();
    let before = serde_json::to_value(&context).unwrap();
    for ids in [vec![a], vec![a, a], vec![a, existing]] {
        assert!(
            context
                .append_model_output(invocation.clone(), &response, ModelStopReason::ToolUse, ids,)
                .is_err()
        );
        assert_eq!(serde_json::to_value(&context).unwrap(), before);
    }
    let applied = context
        .append_model_output(invocation, &response, ModelStopReason::ToolUse, vec![a, b])
        .unwrap();
    assert_eq!(applied.block_ids, vec![a, b]);
    assert_eq!(applied.tool_calls[0].0, b);
    assert!(matches!(
        context.blocks()[1].content(),
        BlockContent::Parts(_)
    ));
    assert_eq!(context.blocks()[1].id(), a);
    assert_eq!(context.blocks()[2].id(), b);
}
