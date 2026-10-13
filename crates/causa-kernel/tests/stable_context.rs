//! The serialized form of a `Context` is the stable set: it is what a host
//! stores. This test holds one context that uses every content shape and
//! every optional field against a committed file, so a change to any covered
//! form fails here and has to be a recorded break.

use causa_kernel::{
    ArtifactKind, ArtifactRef, BlockContent, BlockId, BlockMeta, ContentPart, Context,
    ContextBlock, MediaRef, SealedPart, TextPayload, ToolCallPayload, ToolOutput, ToolOutputMeta,
    ToolResultPayload, ToolResultStatus, Truncation, WireMeta,
};
use serde_json::{Value, json};

const FIXTURE: &str = include_str!("fixtures/stable_context.json");

fn id(n: u128) -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(n))
}

/// One block per content shape, with every optional field set, followed by
/// the same shapes with every optional field left out.
fn context() -> Context {
    let full_meta = BlockMeta::from_source("user")
        .with_provider_call_id("call_1")
        .with_wire(WireMeta::new("anthropic", json!({"index": 0})));
    let parts = BlockContent::Parts(vec![
        ContentPart::Text(TextPayload::new("hello")),
        ContentPart::Media(MediaRef::new("image/png", "asset://1")),
        ContentPart::Sealed(SealedPart::new("anthropic", json!({"type": "thinking"}))),
    ]);
    let call = BlockContent::ToolCall(ToolCallPayload::new("read_file", json!({"path": "a"})));

    let mut meta = ToolOutputMeta::default();
    meta.duration_ms = Some(12);
    meta.original_tokens = Some(3400);
    meta.extra = Some(json!({"reason": "trimmed"}));
    let mut output = ToolOutput::new(json!({"text": "body"}));
    output.truncation = Truncation::Middle;
    output.meta = Some(meta);
    output.artifact = Some(ArtifactRef::new(
        "asset-1",
        9000,
        ArtifactKind::FullOutput,
        true,
    ));
    let mut full_result = ToolResultPayload::new(id(2), ToolResultStatus::Succeeded, output);
    full_result.media = vec![MediaRef::new("image/png", "asset://2")];
    full_result.notes = vec![TextPayload::new("note")];

    let bare_result = ToolResultPayload::new(
        id(5),
        ToolResultStatus::UnknownOutcome,
        ToolOutput::new(json!("plain")),
    );

    Context::from_blocks(vec![
        ContextBlock::new(id(1), parts, full_meta.clone()),
        ContextBlock::new(id(2), call.clone(), full_meta.clone()),
        ContextBlock::new(id(3), BlockContent::ToolResult(full_result), full_meta),
        ContextBlock::new(
            id(4),
            BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("bare"))]),
            BlockMeta::default(),
        ),
        ContextBlock::new(id(5), call, BlockMeta::default()),
        ContextBlock::new(
            id(6),
            BlockContent::ToolResult(bare_result),
            BlockMeta::default(),
        ),
    ])
    .unwrap()
}

#[test]
fn a_context_serializes_to_the_committed_form() {
    let written = serde_json::to_value(context()).unwrap();
    let committed: Value = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(
        written,
        committed,
        "the stable set changed; written form:\n{}",
        serde_json::to_string_pretty(&written).unwrap()
    );
}

#[test]
fn the_committed_form_reads_back_as_the_same_context() {
    let read: Context = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(
        serde_json::to_value(&read).unwrap(),
        serde_json::to_value(context()).unwrap()
    );
    assert_eq!(read.blocks(), context().blocks());
}

#[test]
fn a_field_this_version_does_not_know_is_ignored() {
    // A later version may add an optional field anywhere in the stable set;
    // a reader built against this one has to accept the record.
    fn add_everywhere(value: &mut Value) {
        match value {
            Value::Object(map) => {
                // Tagged enums are objects too, so this also covers a new
                // key next to `shape`/`value` and `part`/`value`.
                map.values_mut().for_each(add_everywhere);
                map.insert("added_later".into(), json!(true));
            }
            Value::Array(items) => items.iter_mut().for_each(add_everywhere),
            _ => {}
        }
    }
    let mut record: Value = serde_json::from_str(FIXTURE).unwrap();
    let committed = record.clone();
    add_everywhere(&mut record);
    let read: Context = serde_json::from_value(record).unwrap();
    let reread: Context = serde_json::from_value(committed).unwrap();
    assert_eq!(read.blocks().len(), reread.blocks().len());
    for (a, b) in read.blocks().iter().zip(reread.blocks()) {
        assert_eq!(a.id(), b.id());
        assert_eq!(a.meta().source, b.meta().source);
    }
}

#[test]
fn an_unset_optional_field_is_left_out_and_null_still_reads() {
    let mut meta = ToolOutputMeta::default();
    meta.duration_ms = Some(12);
    assert_eq!(
        serde_json::to_value(&meta).unwrap(),
        json!({"duration_ms": 12})
    );
    // Records written before this rule spelled an unset field as null.
    let old = json!({"content": "plain", "truncation": "none", "meta": null, "artifact": null});
    assert_eq!(
        serde_json::from_value::<ToolOutput>(old).unwrap(),
        ToolOutput::new(json!("plain"))
    );
}
