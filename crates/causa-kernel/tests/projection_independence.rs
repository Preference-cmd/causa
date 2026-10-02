//! Independent frames and caller-owned saved material remain isolated.

mod common;

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, ContentPart, Context, ContextBlock, ContextFrame, MediaRef,
    ModelResponse, ModelStopReason, TextPayload, ToolOutput, ToolResultPayload, ToolResultStatus,
    validate_tool_result_append,
};
use serde_json::json;

fn declaration(context: &mut Context, name: &str) -> BlockId {
    let response = ModelResponse {
        text: TextPayload::new("calling"),
        tool_calls: vec![common::draft(name, json!({"path":"notes.txt"}))],
    };
    let ids = common::block_ids_for(&response);
    context
        .apply(
            Vec::new(),
            response.to_blocks(ModelStopReason::ToolUse, &ids).unwrap(),
        )
        .unwrap();
    ids[1]
}

fn complete(context: &mut Context, call_block_id: BlockId, output: serde_json::Value) {
    let id = common::block_id();
    let result = ToolResultPayload {
        call_block_id,
        status: ToolResultStatus::Succeeded,
        output: ToolOutput::new(output),
        media: vec![],
        notes: vec![],
    };
    validate_tool_result_append(context.blocks(), &[(id, result.clone())]).unwrap();
    context
        .apply(
            Vec::new(),
            vec![ContextBlock::new(
                id,
                BlockContent::ToolResult(result),
                BlockMeta::default(),
            )],
        )
        .unwrap();
}

#[test]
fn projection_transforms_never_write_back_into_material() {
    let input = common::parts_block(
        common::block_id(),
        vec![
            ContentPart::Text(TextPayload::new("read this")),
            ContentPart::Media(MediaRef::new("image/png", "asset-1")),
        ],
        "user",
    );
    let mut context = Context::from_blocks(vec![input]).unwrap();
    let call = declaration(&mut context, "read_file");
    complete(&mut context, call, json!({"contents":"original body"}));
    let before = serde_json::to_value(&context).unwrap();
    let mut projection = context.frame();
    projection
        .blocks
        .retain(|block| !matches!(block.content(), BlockContent::ToolResult(_)));
    projection.blocks.push(common::text_block(
        common::block_id(),
        TextPayload::new("host context"),
        "host.injected",
    ));
    for block in &mut projection.blocks {
        if let BlockContent::Parts(parts) = block.content() {
            let mut parts = parts.clone();
            for part in &mut parts {
                if let ContentPart::Media(media) = part {
                    *media = MediaRef::new("image/png", "asset-host-copy");
                }
            }
            parts.push(ContentPart::Text(TextPayload::new("rewritten")));
            *block = ContextBlock::new(
                common::block_id(),
                BlockContent::Parts(parts),
                block.meta().clone(),
            );
        }
    }
    assert_eq!(serde_json::to_value(&context).unwrap(), before);
    assert_eq!(projection.blocks.len(), 4);
    assert!(
        serde_json::to_string(&projection.blocks)
            .unwrap()
            .contains("asset-host-copy")
    );
    context.edit().replace(0..1, vec![]).commit().unwrap();
    assert_eq!(projection.blocks.len(), 4);
    assert_eq!(context.blocks().len(), 3);
}

#[test]
fn caller_selected_shared_material_feeds_two_contexts_without_pollution() {
    let shared = vec![
        common::text_block(
            common::block_id(),
            TextPayload::new("shared question"),
            "user",
        ),
        common::text_block(
            common::block_id(),
            TextPayload::new("shared answer"),
            "model",
        ),
    ];
    let saved = serde_json::to_value(&shared).unwrap();
    let mut a = Context::from_blocks(vec![common::text_block(
        common::block_id(),
        TextPayload::new("a's follow-up"),
        "user",
    )])
    .unwrap();
    let mut b = Context::from_blocks(vec![common::text_block(
        common::block_id(),
        TextPayload::new("b's follow-up"),
        "user",
    )])
    .unwrap();
    let frame = |context: &Context| ContextFrame {
        blocks: shared.iter().chain(context.blocks()).cloned().collect(),
    };
    let frame_a = frame(&a);
    let frame_b = frame(&b);
    assert_eq!(frame_a.blocks[..2], frame_b.blocks[..2]);
    let wire_a = serde_json::to_string(&frame_a.blocks).unwrap();
    let wire_b = serde_json::to_string(&frame_b.blocks).unwrap();
    assert!(wire_a.contains("a's follow-up") && !wire_a.contains("b's follow-up"));
    assert!(wire_b.contains("b's follow-up") && !wire_b.contains("a's follow-up"));
    let call_a = declaration(&mut a, "echo");
    let call_b = declaration(&mut b, "echo");
    complete(&mut a, call_a, json!({"who":"a"}));
    complete(&mut b, call_b, json!({"who":"b"}));
    assert!(
        a.blocks()
            .iter()
            .all(|block| !b.blocks().iter().any(|other| other.id() == block.id()))
    );
    assert_eq!(serde_json::to_value(&shared).unwrap(), saved);
    assert_eq!(frame_a.blocks.len(), 3);
    assert_eq!(frame_b.blocks.len(), 3);
}

#[test]
fn imported_material_matches_original_projection_and_remains_editable() {
    let mut original = Context::from_blocks(vec![common::text_block(
        common::block_id(),
        TextPayload::new("hi"),
        "user",
    )])
    .unwrap();
    let response = common::endturn_output("hello");
    original
        .apply(
            Vec::new(),
            response
                .response
                .to_blocks(
                    response.stop_reason,
                    &common::block_ids_for(&response.response),
                )
                .unwrap(),
        )
        .unwrap();
    let mut rebuilt = Context::from_blocks(original.blocks().to_vec()).unwrap();
    assert_eq!(original.frame().blocks, rebuilt.frame().blocks);
    let retained = rebuilt.frame();
    rebuilt
        .apply(
            Vec::new(),
            vec![common::text_block(
                common::block_id(),
                TextPayload::new("new work"),
                "user",
            )],
        )
        .unwrap();
    assert_eq!(retained.blocks, original.blocks());
    assert_eq!(rebuilt.blocks().len(), original.blocks().len() + 1);
}
