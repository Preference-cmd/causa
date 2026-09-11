//! Projection-independence evidence: hosts select, rewrite, and inject on
//! *copies* of the kernel's lossless projection, and two independent records
//! can consume one shared read-only history without polluting each other or
//! the shared input. Kernel-only — no reference driver, no runtime types.

use causa_kernel::{
    AppliedModelOutput, BlockContent, BlockId, BlockMeta, BlockSequence, ContentPart, ContextBlock,
    ContextVersion, ConversationId, InvocationId, MediaRef, ModelResponse, ModelStopReason,
    RoundId, TextPayload, ToolCallDraft, ToolCallId, ToolOutput, ToolResultPayload,
    ToolResultStatus, TurnContext, TurnId, merged_frame,
};
use serde_json::json;

fn parts_block(turn: &TurnId, seq: u64, parts: Vec<ContentPart>, source: &str) -> ContextBlock {
    ContextBlock {
        id: BlockId {
            turn_id: turn.clone(),
            sequence: BlockSequence(seq),
        },
        sequence: BlockSequence(seq),
        content: BlockContent::Parts(parts),
        meta: BlockMeta {
            provider_call_id: None,
            source: Some(source.into()),
        },
    }
}

fn pending_call_ids(blocks: &[ContextBlock]) -> Vec<ToolCallId> {
    let mut calls = Vec::new();
    let mut answered = Vec::new();
    for b in blocks {
        match &b.content {
            BlockContent::ToolCall(c) => calls.push(c.call_id.clone()),
            BlockContent::ToolResult(r) => answered.push(r.call_id.clone()),
            _ => {}
        }
    }
    calls.retain(|id| !answered.contains(id));
    calls
}

/// A tool-use model response. The kernel generates the `ToolCallId` itself
/// at commit time — take it from the [`AppliedModelOutput`] receipt.
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

fn tool_use(ctx: &mut TurnContext, round: RoundId, tool_name: &str) -> ToolCallId {
    let invocation = InvocationId {
        turn_id: ctx.turn_id(),
        round_id: round,
    };
    let applied: AppliedModelOutput = ctx
        .append_model_output(
            invocation,
            &call_response(tool_name),
            ModelStopReason::ToolUse,
        )
        .unwrap();
    applied.tool_calls[0].call_id.clone()
}

/// A host reworks the projection — selects, rewrites a result body,
/// replaces a media reference, injects a sourced block — and the committed
/// facts are byte-identical afterwards: same blocks, same version, same
/// pending set, same snapshot wire shape.
#[tokio::test]
async fn projection_transforms_never_write_back_into_facts() {
    let turn = TurnId::new("proj-1");
    let mut ctx = TurnContext::new(turn.clone());
    ctx.append_parts(
        vec![
            ContentPart::Text(TextPayload::new("read this")),
            ContentPart::Media(MediaRef::new("image/png", "asset-1")),
        ],
        "user",
    )
    .unwrap();
    let call_id = tool_use(&mut ctx, RoundId(0), "read_file");
    ctx.append_tool_results(vec![ToolResultPayload {
        call_id,
        status: ToolResultStatus::Succeeded,
        output: ToolOutput::new(json!({ "contents": "original body" })),
        media: vec![],
    }])
    .unwrap();

    let facts_before = serde_json::to_string(ctx.blocks()).unwrap();
    let version_before = ctx.version();
    let pending_before = pending_call_ids(ctx.blocks());
    let snapshot_before = serde_json::to_string(&ctx.snapshot()).unwrap();

    // The host's working copy: the lossless projection, reworked in place.
    let mut projection = ctx.frame(RoundId(0));
    let blocks = &mut projection.model_context.blocks;
    // select: drop the tool result from the model's view
    blocks.retain(|b| !matches!(b.content, BlockContent::ToolResult(_)));
    // inject: a host-sourced block that was never committed
    blocks.push(parts_block(
        &turn,
        blocks.len() as u64,
        vec![ContentPart::Text(TextPayload::new("host context"))],
        "host.injected",
    ));
    // rewrite + media swap live on the remaining parts block
    for b in blocks.iter_mut() {
        if let BlockContent::Parts(parts) = &mut b.content {
            for part in parts.iter_mut() {
                if let ContentPart::Media(media) = part {
                    *media = MediaRef::new("image/png", "asset-host-copy");
                }
            }
            parts.push(ContentPart::Text(TextPayload::new("rewritten")));
        }
    }

    // Facts are untouched: the committed record, version, pending set, and
    // the snapshot wire shape are all identical to before the rework.
    assert_eq!(facts_before, serde_json::to_string(ctx.blocks()).unwrap());
    assert_eq!(version_before, ctx.version());
    assert_eq!(pending_before, pending_call_ids(ctx.blocks()));
    assert_eq!(
        snapshot_before,
        serde_json::to_string(&ctx.snapshot()).unwrap()
    );
    // The reworked copy really did change: it has the injected block and the
    // swapped media reference. Facts were [parts, response text, tool call,
    // tool result]; the projection drops the result and adds the injection.
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

/// Two independent records consume one shared read-only history projection
/// (the same committed snapshots feed both merged frames); each appends its
/// own results under its own invocation identity, and neither the shared
/// input nor the sibling record is rewritten.
#[tokio::test]
async fn shared_history_projection_feeds_two_records_without_pollution() {
    // One committed turn of shared history.
    let mut shared = TurnContext::new(TurnId::new("shared-1"));
    shared
        .append_input(TextPayload::new("the shared question"), "user")
        .unwrap();
    let shared_invocation = InvocationId {
        turn_id: TurnId::new("shared-1"),
        round_id: RoundId(0),
    };
    shared
        .append_model_output(
            shared_invocation,
            &endturn_response("the shared answer"),
            ModelStopReason::EndTurn,
        )
        .unwrap();
    shared.seal();
    let history = vec![shared.snapshot()];
    let history_wire = serde_json::to_string(&history).unwrap();
    let history_block_count: usize = history.iter().map(|s| s.blocks.as_slice().len()).sum();

    let conversation = ConversationId("shared-proj".into());
    // Two fresh active records, each pairing the same read-only history
    // with its own in-flight turn.
    let mut record_a = TurnContext::new(TurnId::new("child-a"));
    record_a
        .append_input(TextPayload::new("a's follow-up"), "user")
        .unwrap();
    let mut record_b = TurnContext::new(TurnId::new("child-b"));
    record_b
        .append_input(TextPayload::new("b's follow-up"), "user")
        .unwrap();

    let frame_a = merged_frame(&conversation, &history, &record_a, RoundId(1));
    let frame_b = merged_frame(&conversation, &history, &record_b, RoundId(1));
    // The shared segment is identical in both projections and the history
    // snapshots themselves are unchanged.
    assert_eq!(
        serde_json::to_string(&frame_a.model_context.blocks[..history_block_count]).unwrap(),
        serde_json::to_string(&frame_b.model_context.blocks[..history_block_count]).unwrap()
    );
    assert_eq!(history_wire, serde_json::to_string(&history).unwrap());
    // Each projection carries its own active blocks, not the sibling's.
    let a_wire = serde_json::to_string(&frame_a.model_context.blocks).unwrap();
    let b_wire = serde_json::to_string(&frame_b.model_context.blocks).unwrap();
    assert!(a_wire.contains("a's follow-up") && !a_wire.contains("b's follow-up"));
    assert!(b_wire.contains("b's follow-up") && !b_wire.contains("a's follow-up"));

    // Each record commits its own result under its own invocation identity.
    let call_a = tool_use(&mut record_a, RoundId(1), "echo");
    let call_b = tool_use(&mut record_b, RoundId(1), "echo");
    record_a
        .append_tool_results(vec![ToolResultPayload {
            call_id: call_a,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({ "who": "a" })),
            media: vec![],
        }])
        .unwrap();
    record_b
        .append_tool_results(vec![ToolResultPayload {
            call_id: call_b,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({ "who": "b" })),
            media: vec![],
        }])
        .unwrap();

    // The sibling's result content never crosses records, and every block
    // is owned by the record whose turn_id it carries.
    let a_facts = serde_json::to_string(record_a.blocks()).unwrap();
    let b_facts = serde_json::to_string(record_b.blocks()).unwrap();
    assert!(a_facts.contains("\"who\":\"a\"") && !a_facts.contains("\"who\":\"b\""));
    assert!(b_facts.contains("\"who\":\"b\"") && !b_facts.contains("\"who\":\"a\""));
    for b in record_a.blocks() {
        assert_eq!(b.id.turn_id, TurnId::new("child-a"));
    }
    for b in record_b.blocks() {
        assert_eq!(b.id.turn_id, TurnId::new("child-b"));
    }
    // The shared history projection is still pristine.
    assert_eq!(history_wire, serde_json::to_string(&history).unwrap());
}

/// Rebuild-from-validated-blocks parity: an independent record rebuilt from
/// committed facts behaves like the original record — projection identity
/// survives the reload path.
#[tokio::test]
async fn rebuilt_record_from_validated_blocks_matches_original_projection() {
    let mut original = TurnContext::new(TurnId::new("rebuild-1"));
    original
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    let invocation = InvocationId {
        turn_id: TurnId::new("rebuild-1"),
        round_id: RoundId(0),
    };
    original
        .append_model_output(
            invocation,
            &endturn_response("hello"),
            ModelStopReason::EndTurn,
        )
        .unwrap();
    let snapshot = original.snapshot();
    let version = ContextVersion(snapshot.source_version.0);

    let rebuilt = TurnContext::from_validated_blocks(
        TurnId::new("rebuild-1"),
        snapshot.blocks.as_slice().to_vec(),
        version,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_string(&original.frame(RoundId(0)).model_context.blocks).unwrap(),
        serde_json::to_string(&rebuilt.frame(RoundId(0)).model_context.blocks).unwrap()
    );
    assert_eq!(
        original.frame(RoundId(0)).frame_id,
        rebuilt.frame(RoundId(0)).frame_id
    );
}
