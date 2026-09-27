//! Projection-independence evidence: hosts transform projection copies while
//! committed facts and sibling contexts remain unchanged.

mod common;

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, ContentPart, ContextBlock, ContextVersion, ConversationId,
    InvocationId, MediaRef, ModelResponse, ModelStopReason, RoundId, TextPayload, ToolCallDraft,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnContext, TurnId, merged_frame,
};
use serde_json::json;

fn parts_block(parts: Vec<ContentPart>, source: &str) -> ContextBlock {
    ContextBlock {
        id: common::block_id(),
        content: BlockContent::Parts(parts),
        meta: BlockMeta {
            provider_call_id: None,
            source: Some(source.into()),
        },
    }
}

fn pending_calls(blocks: &[ContextBlock]) -> Vec<BlockId> {
    let mut calls = Vec::new();
    let mut answered = Vec::new();
    for block in blocks {
        match &block.content {
            BlockContent::ToolCall(_) => calls.push(block.id),
            BlockContent::ToolResult(result) => answered.push(result.call_block_id),
            _ => {}
        }
    }
    calls.retain(|id| !answered.contains(id));
    calls
}

fn call_response(tool_name: &str) -> ModelResponse {
    ModelResponse {
        text: TextPayload::new("calling"),
        tool_calls: vec![ToolCallDraft {
            tool_name: tool_name.into(),
            arguments: json!({ "path": "notes.txt" }),
            provider_call_id: None,
        }],
    }
}

fn endturn_response(text: &str) -> ModelResponse {
    ModelResponse {
        text: TextPayload::new(text),
        tool_calls: vec![],
    }
}

fn declare_call(ctx: &mut TurnContext, round: RoundId, tool_name: &str) -> BlockId {
    let invocation = InvocationId {
        turn_id: ctx.turn_id(),
        round_id: round,
    };
    let response = call_response(tool_name);
    let applied = ctx
        .append_model_output(
            invocation,
            &response,
            ModelStopReason::ToolUse,
            common::block_ids_for(&response),
        )
        .unwrap();
    applied.tool_calls[0].0
}

#[tokio::test]
async fn projection_transforms_never_write_back_into_facts() {
    let mut ctx = TurnContext::new(TurnId::new("proj-1"));
    ctx.append_parts(
        common::block_id(),
        vec![
            ContentPart::Text(TextPayload::new("read this")),
            ContentPart::Media(MediaRef::new("image/png", "asset-1")),
        ],
        "user",
    )
    .unwrap();
    let declaration_id = declare_call(&mut ctx, RoundId(0), "read_file");
    ctx.append_tool_results(vec![(
        common::block_id(),
        ToolResultPayload {
            call_block_id: declaration_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({ "contents": "original body" })),
            media: vec![],
            notes: vec![],
        },
    )])
    .unwrap();

    let facts_before = serde_json::to_string(ctx.blocks()).unwrap();
    let version_before = ctx.version();
    let pending_before = pending_calls(ctx.blocks());
    let snapshot_before = serde_json::to_string(&ctx.snapshot()).unwrap();

    let mut projection = ctx.frame(RoundId(0));
    let blocks = &mut projection.model_context.blocks;
    blocks.retain(|block| !matches!(block.content, BlockContent::ToolResult(_)));
    blocks.push(parts_block(
        vec![ContentPart::Text(TextPayload::new("host context"))],
        "host.injected",
    ));
    for block in blocks.iter_mut() {
        if let BlockContent::Parts(parts) = &mut block.content {
            for part in parts.iter_mut() {
                if let ContentPart::Media(media) = part {
                    *media = MediaRef::new("image/png", "asset-host-copy");
                }
            }
            parts.push(ContentPart::Text(TextPayload::new("rewritten")));
        }
    }

    assert_eq!(facts_before, serde_json::to_string(ctx.blocks()).unwrap());
    assert_eq!(version_before, ctx.version());
    assert_eq!(pending_before, pending_calls(ctx.blocks()));
    assert_eq!(
        snapshot_before,
        serde_json::to_string(&ctx.snapshot()).unwrap()
    );
    assert_eq!(projection.model_context.blocks.len(), 4);
    assert!(matches!(
        projection.model_context.blocks[3].content,
        BlockContent::Parts(_)
    ));
    assert!(
        serde_json::to_string(&projection.model_context.blocks)
            .unwrap()
            .contains("asset-host-copy")
    );
}

#[tokio::test]
async fn shared_history_projection_feeds_two_records_without_pollution() {
    let mut shared = TurnContext::new(TurnId::new("shared-1"));
    shared
        .append_input(
            common::block_id(),
            TextPayload::new("the shared question"),
            "user",
        )
        .unwrap();
    let shared_invocation = InvocationId {
        turn_id: TurnId::new("shared-1"),
        round_id: RoundId(0),
    };
    let response = endturn_response("the shared answer");
    shared
        .append_model_output(
            shared_invocation,
            &response,
            ModelStopReason::EndTurn,
            common::block_ids_for(&response),
        )
        .unwrap();
    shared.seal();
    let history = vec![shared.snapshot()];
    let history_wire = serde_json::to_string(&history).unwrap();
    let history_block_count: usize = history.iter().map(|s| s.blocks.as_slice().len()).sum();
    let conversation = ConversationId("shared-proj".into());

    let mut record_a = TurnContext::new(TurnId::new("child-a"));
    record_a
        .append_input(
            common::block_id(),
            TextPayload::new("a's follow-up"),
            "user",
        )
        .unwrap();
    let mut record_b = TurnContext::new(TurnId::new("child-b"));
    record_b
        .append_input(
            common::block_id(),
            TextPayload::new("b's follow-up"),
            "user",
        )
        .unwrap();

    let frame_a = merged_frame(&conversation, &history, &record_a, RoundId(1));
    let frame_b = merged_frame(&conversation, &history, &record_b, RoundId(1));
    assert_eq!(
        serde_json::to_string(&frame_a.model_context.blocks[..history_block_count]).unwrap(),
        serde_json::to_string(&frame_b.model_context.blocks[..history_block_count]).unwrap()
    );
    assert_eq!(history_wire, serde_json::to_string(&history).unwrap());
    let a_wire = serde_json::to_string(&frame_a.model_context.blocks).unwrap();
    let b_wire = serde_json::to_string(&frame_b.model_context.blocks).unwrap();
    assert!(a_wire.contains("a's follow-up") && !a_wire.contains("b's follow-up"));
    assert!(b_wire.contains("b's follow-up") && !b_wire.contains("a's follow-up"));

    let call_a = declare_call(&mut record_a, RoundId(1), "echo");
    let call_b = declare_call(&mut record_b, RoundId(1), "echo");
    record_a
        .append_tool_results(vec![(
            common::block_id(),
            ToolResultPayload {
                call_block_id: call_a,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!({ "who": "a" })),
                media: vec![],
                notes: vec![],
            },
        )])
        .unwrap();
    record_b
        .append_tool_results(vec![(
            common::block_id(),
            ToolResultPayload {
                call_block_id: call_b,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!({ "who": "b" })),
                media: vec![],
                notes: vec![],
            },
        )])
        .unwrap();

    let a_facts = serde_json::to_string(record_a.blocks()).unwrap();
    let b_facts = serde_json::to_string(record_b.blocks()).unwrap();
    assert!(a_facts.contains("\"who\":\"a\"") && !a_facts.contains("\"who\":\"b\""));
    assert!(b_facts.contains("\"who\":\"b\"") && !b_facts.contains("\"who\":\"a\""));
    let a_ids: Vec<_> = record_a.blocks().iter().map(|block| block.id).collect();
    let b_ids: Vec<_> = record_b.blocks().iter().map(|block| block.id).collect();
    assert!(a_ids.iter().all(|id| !b_ids.contains(id)));
    assert_eq!(history_wire, serde_json::to_string(&history).unwrap());
}

#[tokio::test]
async fn rebuilt_record_from_validated_blocks_matches_original_projection() {
    let mut original = TurnContext::new(TurnId::new("rebuild-1"));
    original
        .append_input(common::block_id(), TextPayload::new("hi"), "user")
        .unwrap();
    let invocation = InvocationId {
        turn_id: TurnId::new("rebuild-1"),
        round_id: RoundId(0),
    };
    let response = endturn_response("hello");
    original
        .append_model_output(
            invocation,
            &response,
            ModelStopReason::EndTurn,
            common::block_ids_for(&response),
        )
        .unwrap();
    let snapshot = original.snapshot();
    let rebuilt = TurnContext::from_validated_blocks(
        TurnId::new("rebuild-1"),
        snapshot.blocks.as_slice().to_vec(),
        ContextVersion(snapshot.source_version.0),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_string(&original.frame(RoundId(0)).model_context.blocks).unwrap(),
        serde_json::to_string(&rebuilt.frame(RoundId(0)).model_context.blocks).unwrap()
    );
    assert_eq!(
        original.frame(RoundId(0)).scope,
        rebuilt.frame(RoundId(0)).scope
    );
}
