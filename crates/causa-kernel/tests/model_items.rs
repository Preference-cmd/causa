//! One model item becomes one block: ordering, wire pass-through, text
//! retention and the stored shape of the new fields.

mod common;

use causa_kernel::{
    BlockContent, BlockMeta, ContentPart, Context, ContextBlock, ModelBlockError, ModelItem,
    ModelItemContent, ModelResponse, ModelStopReason, Role, SealedPart, TextPayload, ToolCallDraft,
    WireMeta,
};
use serde_json::json;

fn wire(format: &str, fields: serde_json::Value) -> Option<WireMeta> {
    Some(WireMeta::new(format, fields))
}

fn parts_item(parts: Vec<ContentPart>, wire: Option<WireMeta>) -> ModelItem {
    {
        let mut item = ModelItem::new(ModelItemContent::Parts(parts));
        item.wire = wire;
        item
    }
}

fn text_part(text: &str) -> ContentPart {
    ContentPart::Text(TextPayload::new(text))
}

fn call(name: &str, id: &str) -> ToolCallDraft {
    ToolCallDraft::new(name, json!({"n": name})).with_provider_call_id(id)
}

#[test]
fn interleaved_items_become_blocks_in_item_order() {
    let response = ModelResponse {
        items: vec![
            ModelItem::text("first"),
            ModelItem::tool_call(call("a", "id-a")),
            ModelItem::text("second"),
            ModelItem::tool_call(call("b", "id-b")),
            ModelItem::text("third"),
        ],
    };
    assert_eq!(response.block_count(), 5);
    let ids = common::block_ids_for(&response);
    let blocks = response.to_blocks(ModelStopReason::ToolUse, &ids).unwrap();
    assert_eq!(blocks.iter().map(ContextBlock::id).collect::<Vec<_>>(), ids);
    let shapes: Vec<String> = blocks
        .iter()
        .map(|b| match b.content() {
            BlockContent::Message { parts, .. } => match &parts[0] {
                ContentPart::Text(t) => t.0.clone(),
                other => panic!("unexpected part {other:?}"),
            },
            BlockContent::ToolCall(call) => format!("call:{}", call.tool_name),
            BlockContent::ToolResult(_) | BlockContent::ToolChange(_) => unreachable!(),
        })
        .collect();
    assert_eq!(shapes, ["first", "call:a", "second", "call:b", "third"]);
    assert_eq!(blocks[1].meta().provider_call_id.as_deref(), Some("id-a"));
    assert_eq!(blocks[3].meta().provider_call_id.as_deref(), Some("id-b"));
}

#[test]
fn readers_follow_item_order() {
    let response = ModelResponse {
        items: vec![
            ModelItem::text("a"),
            ModelItem::tool_call(call("x", "1")),
            parts_item(
                vec![
                    text_part("b"),
                    ContentPart::Sealed(SealedPart::new("f", json!({"hidden": true}))),
                    text_part(" c"),
                ],
                None,
            ),
            ModelItem::tool_call(call("y", "2")),
        ],
    };
    // concatenated in order, no separator; sealed parts contribute nothing
    assert_eq!(response.text(), "ab c");
    let names: Vec<&str> = response
        .tool_calls()
        .map(|c| c.tool_name.as_str())
        .collect();
    assert_eq!(names, ["x", "y"]);
}

#[test]
fn whitespace_only_text_is_kept_and_zero_length_is_dropped() {
    let response = ModelResponse {
        items: vec![
            ModelItem::text(" \n\t"),
            ModelItem::text(""),
            parts_item(vec![text_part(""), text_part("kept"), text_part("")], None),
            parts_item(vec![], None),
        ],
    };
    assert_eq!(response.block_count(), 2);
    let ids = common::block_ids_for(&response);
    let blocks = response.to_blocks(ModelStopReason::EndTurn, &ids).unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(
        blocks[0].content(),
        &BlockContent::Message {
            role: Role::Assistant,
            parts: vec![text_part(" \n\t")],
        }
    );
    assert_eq!(
        blocks[1].content(),
        &BlockContent::Message {
            role: Role::Assistant,
            parts: vec![text_part("kept")],
        }
    );
}

#[test]
fn an_item_left_without_parts_produces_no_block_and_loses_its_wire() {
    let response = ModelResponse {
        items: vec![parts_item(
            vec![text_part("")],
            wire("openai_responses", json!({"id": "msg_1"})),
        )],
    };
    assert_eq!(response.block_count(), 0);
    assert!(
        response
            .to_blocks(ModelStopReason::EndTurn, &[])
            .unwrap()
            .is_empty()
    );
}

#[test]
fn item_wire_is_copied_to_block_meta_untouched() {
    let parts_wire = wire(
        "openai_responses",
        json!({"id": "msg_1", "phase": "commentary"}),
    );
    let call_wire = wire(
        "openai_responses",
        json!({"id": "fc_1", "arguments": "{ }"}),
    );
    let response = ModelResponse {
        items: vec![
            parts_item(vec![text_part("hi")], parts_wire.clone()),
            {
                let mut item = ModelItem::new(ModelItemContent::ToolCall(call("a", "id-a")));
                item.wire = call_wire.clone();
                item
            },
            ModelItem::text("plain"),
        ],
    };
    let ids = common::block_ids_for(&response);
    let blocks = response.to_blocks(ModelStopReason::ToolUse, &ids).unwrap();
    assert_eq!(blocks[0].meta().wire, parts_wire);
    assert_eq!(blocks[0].meta().provider_call_id, None);
    assert_eq!(blocks[1].meta().wire, call_wire);
    assert_eq!(blocks[1].meta().provider_call_id.as_deref(), Some("id-a"));
    assert_eq!(blocks[2].meta().wire, None);
}

#[test]
fn stop_reason_pairing_and_tool_checks_use_the_tool_call_items() {
    let response = ModelResponse {
        items: vec![
            ModelItem::text("x"),
            ModelItem::tool_call(call("ok", "1")),
            ModelItem::text("y"),
            ModelItem::tool_call(ToolCallDraft::new("bad", json!([1]))),
        ],
    };
    let ids = common::block_ids_for(&response);
    assert!(matches!(
        response
            .to_blocks(ModelStopReason::EndTurn, &ids)
            .unwrap_err(),
        ModelBlockError::EndTurnWithToolCalls { count: 2, .. }
    ));
    // `tool_index` counts tool-call items, not all items
    assert!(matches!(
        response
            .to_blocks(ModelStopReason::ToolUse, &ids)
            .unwrap_err(),
        ModelBlockError::ArgumentsNotObject { tool_index: 1, .. }
    ));
    assert_eq!(
        ModelResponse {
            items: vec![ModelItem::text("only text")]
        }
        .to_blocks(ModelStopReason::ToolUse, &[common::block_id()])
        .unwrap_err(),
        ModelBlockError::ToolUseWithoutToolCalls
    );
}

#[test]
fn blocks_written_before_the_wire_field_still_load() {
    let old = r#"{
        "id": "00000000-0000-0000-0000-000000000001",
        "content": {"shape": "message", "value": {"role": "user", "parts": [{"part": "text", "value": "hello"}]}},
        "meta": {"source": "user"}
    }"#;
    let block: ContextBlock = serde_json::from_str(old).unwrap();
    assert_eq!(block.meta().source.as_deref(), Some("user"));
    assert_eq!(block.meta().wire, None);
    // and a block without wire is written without the key
    let written = serde_json::to_value(&block).unwrap();
    assert!(written["meta"].get("wire").is_none());

    let meta: BlockMeta = serde_json::from_str("{}").unwrap();
    assert_eq!(meta, BlockMeta::default());
}

#[test]
fn wire_and_sealed_parts_round_trip_through_a_stored_context() {
    let sealed = ContentPart::Sealed(SealedPart::new(
        "openai_chat_completions",
        json!({"reasoning_content": "think", "content": null, "n": [1, 2.5]}),
    ));
    let block = ContextBlock::new(
        common::block_id(),
        BlockContent::Message {
            role: Role::Assistant,
            parts: vec![sealed, text_part("hi")],
        },
        BlockMeta::default().with_wire(WireMeta::new(
            "anthropic_messages",
            json!({"type": "text", "x": {"a": 1}}),
        )),
    );
    let call_block = ContextBlock::new(
        common::block_id(),
        BlockContent::ToolCall(causa_kernel::ToolCallPayload::new("t", json!({}))),
        BlockMeta::default()
            .with_provider_call_id("c1")
            .with_wire(WireMeta::new(
                "openai_responses",
                json!({"id": "fc", "arguments": "{ }"}),
            )),
    );
    let context = Context::from_blocks(vec![block, call_block]).unwrap();
    let stored = serde_json::to_string(&context).unwrap();
    assert!(stored.contains(r#""part":"sealed""#));
    let restored: Context = serde_json::from_str(&stored).unwrap();
    assert_eq!(restored.blocks(), context.blocks());
}

#[test]
fn sealed_parts_are_edited_and_removed_through_the_existing_context_api() {
    let original = ContextBlock::new(
        common::block_id(),
        BlockContent::Message {
            role: Role::Assistant,
            parts: vec![ContentPart::Sealed(SealedPart::new(
                "anthropic_messages",
                json!({"type": "thinking", "thinking": "hmm", "signature": "sig"}),
            ))],
        },
        BlockMeta::from_source("assistant"),
    );
    let mut context = Context::from_blocks(vec![original]).unwrap();

    // Replace it with a new identity carrying a different sealed item: the
    // kernel validates ranges and block identity, never the provider item, so
    // no new API and no sealed-specific rule is involved.
    let edited = ContextBlock::new(
        common::block_id(),
        BlockContent::Message {
            role: Role::Assistant,
            parts: vec![ContentPart::Sealed(SealedPart::new(
                "anthropic_messages",
                json!({"type": "thinking", "thinking": "edited", "signature": "sig"}),
            ))],
        },
        BlockMeta::from_source("assistant"),
    );
    context
        .edit()
        .replace(0..1, [edited.clone()])
        .commit()
        .unwrap();
    assert_eq!(context.blocks().len(), 1);
    assert_eq!(context.blocks()[0].id(), edited.id());
    assert!(matches!(
        context.blocks()[0].content(),
        BlockContent::Message { parts, .. } if matches!(parts.as_slice(), [ContentPart::Sealed(_)])
    ));

    // Remove the sealed material entirely; nothing else changes.
    context.edit().replace(0..1, []).commit().unwrap();
    assert!(context.blocks().is_empty());
}

#[test]
fn model_response_round_trips_through_serde() {
    let response = ModelResponse {
        items: vec![
            parts_item(
                vec![text_part("hi")],
                wire("anthropic_messages", json!({"type": "text"})),
            ),
            ModelItem::tool_call(call("a", "id-a")),
        ],
    };
    let again: ModelResponse =
        serde_json::from_value(serde_json::to_value(&response).unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(&again).unwrap(),
        serde_json::to_value(&response).unwrap()
    );
}
