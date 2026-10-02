//! C1: kernel-only material driver using the public facade.

use causa::kernel::{
    BlockContent, BlockId, BlockMeta, ContentPart, Context, ContextBlock, EditError,
    ModelBlockError, ModelResponse, ModelStopReason, TextPayload, ToolBatch, ToolCallContext,
    ToolCallDraft, ToolOutput, ToolResultError, ToolResultPayload, ToolResultStatus,
    validate_tool_result_append,
};
use serde_json::json;

fn id(value: u128) -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(value))
}
fn text(value: u128, content: &str) -> ContextBlock {
    ContextBlock::new(
        id(value),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(content))]),
        BlockMeta::default(),
    )
}
fn response(tool: &str) -> ModelResponse {
    ModelResponse {
        text: TextPayload::new("selected output"),
        tool_calls: vec![ToolCallDraft {
            tool_name: tool.into(),
            arguments: json!({"query":"same"}),
            provider_call_id: Some("provider-new".into()),
        }],
    }
}

#[test]
fn material_driver_builds_work_only_from_new_blocks_and_preserves_failed_inputs() {
    let old = response("old")
        .to_blocks(ModelStopReason::ToolUse, &[id(1), id(2)])
        .unwrap();
    let mut context = Context::from_blocks(old).unwrap();
    context
        .edit()
        .append([text(3, "new input")])
        .commit()
        .unwrap();
    let saved = context.frame();
    let output = response("new");
    let ids = [id(4), id(5)];
    let blocks = output.to_blocks(ModelStopReason::ToolUse, &ids).unwrap();
    let calls = blocks
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolCall(call) => {
                Some(ToolCallContext::from_declaration(block.id(), call))
            }
            _ => None,
        })
        .collect();
    let mut batch = ToolBatch::new(calls).unwrap();
    assert_eq!(batch.calls().len(), 1);
    assert_eq!(batch.calls()[0].call().input.tool_name, "new");
    context.apply(Vec::new(), blocks).unwrap();
    batch.calls_mut()[0].push_note(TextPayload::new("adjusted input"));
    batch
        .resolve_at(
            0,
            id(6),
            ToolResultPayload {
                call_block_id: id(5),
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!({"answer":42})),
                media: vec![],
                notes: vec![],
            },
        )
        .unwrap();
    let results = batch
        .results()
        .iter()
        .map(|entry| {
            let (id, payload) = entry.result().unwrap();
            (*id, payload.clone())
        })
        .collect::<Vec<_>>();
    validate_tool_result_append(context.blocks(), &results).unwrap();
    let result_blocks = results
        .iter()
        .map(|(id, payload)| {
            ContextBlock::new(
                *id,
                BlockContent::ToolResult(payload.clone()),
                BlockMeta::default(),
            )
        })
        .collect();
    context.apply(Vec::new(), result_blocks).unwrap();
    let wire = serde_json::to_value(&context).unwrap();
    assert_eq!(wire.as_object().unwrap().len(), 1);
    let mut restored: Context = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(restored.blocks(), context.blocks());
    let conflict = text(1, "different material");
    let error = restored
        .apply(Vec::new(), vec![conflict.clone()])
        .unwrap_err();
    assert_eq!(error.reason, EditError::BlockIdentityMismatch(id(1)));
    assert_eq!(error.appended, vec![conflict]);
    assert_eq!(serde_json::to_value(&restored).unwrap(), wire);
    assert_eq!(saved.blocks.len(), 3);
    assert_eq!(context.into_blocks().len(), 6);
    assert_eq!(
        batch.results()[0].result().unwrap().1.notes,
        vec![TextPayload::new("adjusted input")]
    );
}

#[test]
fn pure_errors_leave_response_ids_and_batch_available_to_the_caller() {
    let response = response("new");
    let ids = [id(1)];
    assert_eq!(
        response
            .to_blocks(ModelStopReason::ToolUse, &ids)
            .unwrap_err(),
        ModelBlockError::BlockIdCountMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(
        response.tool_calls[0].provider_call_id.as_deref(),
        Some("provider-new")
    );
    assert_eq!(ids, [id(1)]);
    let blocks = response
        .to_blocks(ModelStopReason::ToolUse, &[id(1), id(2)])
        .unwrap();
    let calls = blocks
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolCall(call) => {
                Some(ToolCallContext::from_declaration(block.id(), call))
            }
            _ => None,
        })
        .collect();
    let mut batch = ToolBatch::new(calls).unwrap();
    batch
        .resolve_at(
            0,
            id(3),
            ToolResultPayload {
                call_block_id: id(2),
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("ok")),
                media: vec![],
                notes: vec![],
            },
        )
        .unwrap();
    let (result_id, result) = batch.results()[0].result().unwrap();
    assert_eq!(
        validate_tool_result_append(&[], &[(*result_id, result.clone())]).unwrap_err(),
        ToolResultError::MissingDeclaration {
            call_block_id: id(2)
        }
    );
    assert_eq!(batch.completed_len(), 1);
    let failure = Context::from_blocks(vec![blocks[0].clone(), blocks[0].clone()]).unwrap_err();
    assert_eq!(failure.appended, vec![blocks[0].clone(), blocks[0].clone()]);
    assert!(failure.replacements.is_empty());
}
