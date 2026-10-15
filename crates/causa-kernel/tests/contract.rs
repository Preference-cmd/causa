//! Material, pure conversion and tool-result append contracts.

mod common;

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, ContentPart, Context, ContextBlock, EditError, MediaRef,
    ModelBlockError, ModelItem, ModelItemContent, ModelResponse, ModelStopReason, ModelUsage,
    ReasoningPayload, Role, TextPayload, ToolCallDraft, ToolCallPayload, ToolOutput,
    ToolResultError, ToolResultPayload, ToolResultStatus, validate_tool_result_append,
};
use serde_json::json;

fn response(text: &str, calls: Vec<ToolCallDraft>) -> ModelResponse {
    let text = (!text.is_empty()).then(|| ModelItem::text(text));
    ModelResponse {
        items: text
            .into_iter()
            .chain(calls.into_iter().map(ModelItem::tool_call))
            .collect(),
    }
}

fn result(call_block_id: BlockId) -> ToolResultPayload {
    ToolResultPayload::new(
        call_block_id,
        ToolResultStatus::Succeeded,
        ToolOutput::new(json!({"ok": true})),
    )
    .with_media(vec![MediaRef::new("image/png", "result-asset")])
    .with_notes(vec![TextPayload::new("parameters adjusted")])
}

#[test]
fn empty_and_lossless_frames_are_deterministic_and_identity_free() {
    let mut context = Context::new();
    assert!(context.frame().blocks.is_empty());
    let blocks = vec![
        common::text_block(common::block_id(), TextPayload::new("hello"), "user"),
        common::text_block(common::block_id(), TextPayload::new("sys"), "system"),
    ];
    context.apply(Vec::new(), blocks.clone()).unwrap();
    assert_eq!(context.frame().blocks, blocks);
    assert_eq!(context.frame().blocks, context.frame().blocks);
    assert_eq!(context.blocks(), blocks);
    assert_eq!(context.into_blocks(), blocks);
}

#[test]
fn pure_conversion_validates_in_documented_order_and_preserves_inputs() {
    let mut value = response("text", vec![common::draft(" ", json!(42))]);
    let ids = common::block_ids_for(&value);
    let original = serde_json::to_value(&value).unwrap();
    assert!(matches!(
        value.to_blocks(ModelStopReason::EndTurn, &[]).unwrap_err(),
        ModelBlockError::EndTurnWithToolCalls { count: 1, .. }
    ));
    assert!(matches!(
        value.to_blocks(ModelStopReason::ToolUse, &[]).unwrap_err(),
        ModelBlockError::BlockIdCountMismatch {
            expected: 2,
            actual: 0,
            ..
        }
    ));
    assert!(matches!(
        value.to_blocks(ModelStopReason::ToolUse, &ids).unwrap_err(),
        ModelBlockError::EmptyToolName { tool_index: 0, .. }
    ));
    assert_eq!(serde_json::to_value(&value).unwrap(), original);
    let ModelItemContent::ToolCall(call) = &mut value.items[1].content else {
        panic!("second item is the tool call");
    };
    call.tool_name = "valid".into();
    assert!(matches!(
        value.to_blocks(ModelStopReason::ToolUse, &ids).unwrap_err(),
        ModelBlockError::ArgumentsNotObject { tool_index: 0, .. }
    ));
    let no_calls = response("text", vec![]);
    assert_eq!(
        no_calls
            .to_blocks(ModelStopReason::ToolUse, &[])
            .unwrap_err(),
        ModelBlockError::ToolUseWithoutToolCalls
    );
    let late_invalid = response(
        "text",
        vec![
            common::draft("ok", json!({})),
            common::draft("bad", json!(null)),
        ],
    );
    assert!(matches!(
        late_invalid
            .to_blocks(
                ModelStopReason::ToolUse,
                &common::block_ids_for(&late_invalid)
            )
            .unwrap_err(),
        ModelBlockError::ArgumentsNotObject { tool_index: 1, .. }
    ));
}

#[test]
fn conversion_binds_text_then_calls_preserving_provider_ids_names_and_arguments() {
    let value = response(
        "  unchanged text\n",
        vec![
            ToolCallDraft::new(" echo ", json!({"a":1})).with_provider_call_id("provider-1"),
            common::draft("echo", json!({"a":1})),
        ],
    );
    let ids = common::block_ids_for(&value);
    let before = serde_json::to_value(&value).unwrap();
    let blocks = value.to_blocks(ModelStopReason::ToolUse, &ids).unwrap();
    assert_eq!(blocks.iter().map(ContextBlock::id).collect::<Vec<_>>(), ids);
    assert_eq!(
        blocks[0].content(),
        &BlockContent::text(Role::Assistant, value.text())
    );
    assert!(
        matches!(blocks[1].content(), BlockContent::ToolCall(call) if call.tool_name == " echo " && call.arguments == json!({"a":1}))
    );
    assert_eq!(
        blocks[1].meta().provider_call_id.as_deref(),
        Some("provider-1")
    );
    assert_eq!(blocks[2].meta().provider_call_id, None);
    assert_eq!(serde_json::to_value(&value).unwrap(), before);
    let imported = Context::from_blocks(blocks).unwrap();
    let restored: Context =
        serde_json::from_value(serde_json::to_value(&imported).unwrap()).unwrap();
    assert_eq!(restored.blocks(), imported.blocks());
}

#[test]
fn empty_output_requires_no_ids_and_terminal_recording_policy_remains_external() {
    let blank = ModelResponse {
        items: vec![ModelItem::text("")],
    };
    assert_eq!(blank.block_count(), 0);
    assert!(
        blank
            .to_blocks(ModelStopReason::EndTurn, &[])
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        blank
            .to_blocks(ModelStopReason::EndTurn, &[common::block_id()])
            .unwrap_err(),
        ModelBlockError::BlockIdCountMismatch {
            expected: 0,
            actual: 1,
            ..
        }
    ));
    for reason in [ModelStopReason::MaxTokens, ModelStopReason::Refusal] {
        let value = response("partial", vec![common::draft("echo", json!({}))]);
        assert_eq!(
            value
                .to_blocks(reason, &common::block_ids_for(&value))
                .unwrap()
                .len(),
            2
        );
    }
}

#[test]
fn scope_validation_distinguishes_missing_completed_and_duplicate_results() {
    let declaration = common::block_id();
    let value = response("", vec![common::draft("echo", json!({}))]);
    let mut blocks = value
        .to_blocks(ModelStopReason::ToolUse, &[declaration])
        .unwrap();
    let valid = (common::block_id(), result(declaration));
    let before = serde_json::to_value(&valid.1).unwrap();
    assert!(validate_tool_result_append(&blocks, std::slice::from_ref(&valid)).is_ok());
    assert_eq!(serde_json::to_value(&valid.1).unwrap(), before);
    let missing = common::block_id();
    assert!(matches!(
            validate_tool_result_append(&blocks, &[(common::block_id(), result(missing))]).unwrap_err(),
    ToolResultError::MissingDeclaration { call_block_id, .. } if call_block_id == missing
    ));
    let part = common::text_block(missing, TextPayload::new("not a declaration"), "user");
    blocks.push(part);
    assert!(matches!(
            validate_tool_result_append(&blocks, &[(common::block_id(), result(missing))]).unwrap_err(),
    ToolResultError::MissingDeclaration { call_block_id, .. } if call_block_id == missing
    ));
    assert!(matches!(
            validate_tool_result_append(
                &blocks,
                &[valid.clone(), (common::block_id(), result(declaration))]
            )
            .unwrap_err(),
    ToolResultError::DuplicateResult { call_block_id, .. } if call_block_id == declaration
    ));
    blocks.push(ContextBlock::new(
        valid.0,
        BlockContent::ToolResult(valid.1.clone()),
        BlockMeta::default(),
    ));
    assert!(matches!(
            validate_tool_result_append(&blocks, &[valid]).unwrap_err(),
    ToolResultError::AlreadyCompleted { call_block_id, .. } if call_block_id == declaration
    ));
    assert!(validate_tool_result_append(&blocks, &[]).is_ok());
}

#[test]
fn scope_validation_reports_first_input_error_and_ignores_unrelated_local_material() {
    let declaration = common::block_id();
    let missing = common::block_id();
    let value = response("", vec![common::draft("echo", json!({}))]);
    let mut blocks = value
        .to_blocks(ModelStopReason::ToolUse, &[declaration])
        .unwrap();
    blocks.push(ContextBlock::new(
        common::block_id(),
        BlockContent::ToolResult(result(missing)),
        BlockMeta::default(),
    ));
    let valid = (common::block_id(), result(declaration));
    assert!(validate_tool_result_append(&blocks, std::slice::from_ref(&valid)).is_ok());
    assert!(matches!(
            validate_tool_result_append(
                &blocks,
                &[(common::block_id(), result(missing)), valid.clone(), valid]
            )
            .unwrap_err(),
    ToolResultError::MissingDeclaration { call_block_id, .. } if call_block_id == missing
    ));
}

#[test]
fn results_keep_supplied_order_identity_notes_media_and_status() {
    let value = response(
        "",
        vec![
            common::draft("echo", json!({"n":1})),
            common::draft("echo", json!({"n":2})),
        ],
    );
    let declarations = common::block_ids_for(&value);
    let mut context = Context::from_blocks(
        value
            .to_blocks(ModelStopReason::ToolUse, &declarations)
            .unwrap(),
    )
    .unwrap();
    let mut second = result(declarations[1]);
    second.status = ToolResultStatus::Failed;
    let results = vec![
        (common::block_id(), second),
        (common::block_id(), result(declarations[0])),
    ];
    validate_tool_result_append(context.blocks(), &results).unwrap();
    let blocks = results
        .iter()
        .map(|(id, payload)| {
            ContextBlock::new(
                *id,
                BlockContent::ToolResult(payload.clone()),
                BlockMeta::default(),
            )
        })
        .collect();
    context.apply(Vec::new(), blocks).unwrap();
    for (block, (id, payload)) in context.blocks()[2..].iter().zip(&results) {
        assert_eq!(block.id(), *id);
        assert_eq!(block.content(), &BlockContent::ToolResult(payload.clone()));
    }
}

#[test]
fn directly_constructed_parts_preserve_empty_values_sources_and_media() {
    let media = MediaRef::new("image/png", "blake3-asset-id");
    let block = common::parts_block(
        common::block_id(),
        vec![
            ContentPart::Text(TextPayload::new("caption")),
            ContentPart::Media(media.clone()),
        ],
        "user",
    );
    let empty = common::parts_block(common::block_id(), vec![], "host");
    let context = Context::from_blocks(vec![block.clone(), empty.clone()]).unwrap();
    let wire = serde_json::to_string(&context).unwrap();
    assert!(wire.contains(
        r#""part":"media","value":{"media_type":"image/png","reference":"blake3-asset-id"}"#
    ));
    assert!(wire.len() < 800);
    assert!(!wire.contains("\"kind\""));
    let restored: Context = serde_json::from_str(&wire).unwrap();
    assert_eq!(restored.blocks(), &[block, empty]);
    assert_eq!(restored.blocks()[0].meta().source.as_deref(), Some("user"));
}

#[test]
fn direct_serde_requires_blocks_checks_identity_and_preserves_partial_material() {
    assert!(serde_json::from_value::<Context>(json!({})).is_err());
    let empty: Context = serde_json::from_value(json!({"blocks":[]})).unwrap();
    assert!(empty.blocks().is_empty());
    assert_eq!(serde_json::to_value(empty).unwrap(), json!({"blocks":[]}));
    let id = common::block_id();
    let partial = Context::from_blocks(vec![
        ContextBlock::new(
            common::block_id(),
            BlockContent::ToolResult(result(id)),
            BlockMeta::default(),
        ),
        common::parts_block(common::block_id(), vec![], "host"),
        ContextBlock::new(
            common::block_id(),
            BlockContent::ToolResult(result(id)),
            BlockMeta::default(),
        ),
    ])
    .unwrap();
    let encoded = serde_json::to_value(&partial).unwrap();
    assert_eq!(encoded.as_object().unwrap().len(), 1);
    let restored: Context = serde_json::from_value(encoded).unwrap();
    assert_eq!(restored.blocks(), partial.blocks());
    let duplicate = vec![partial.blocks()[0].clone(), partial.blocks()[0].clone()];
    let error = Context::from_blocks(duplicate.clone()).unwrap_err();
    assert_eq!(error.reason, EditError::DuplicateBlockId(duplicate[0].id()));
    assert!(error.replacements.is_empty());
    assert_eq!(error.appended, duplicate);
    assert!(serde_json::from_value::<Context>(json!({"blocks":duplicate})).is_err());
}

#[test]
fn fidelity_fields_are_serde_additive() {
    let usage = {
        let mut usage = ModelUsage::new(120, 30);
        usage.cache_read_tokens = Some(64);
        usage.cache_write_tokens = Some(12);
        usage.reasoning_tokens = Some(8);
        usage
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
    let reasoning = ReasoningPayload::new("thinking");
    let back: ReasoningPayload =
        serde_json::from_str(&serde_json::to_string(&reasoning).unwrap()).unwrap();
    assert_eq!(back.text, "thinking");
    // The removed `signature` field is a plain unknown-field skip: a payload
    // persisted before the removal still deserializes, and the signature now
    // lives only inside the sealed provider item.
    let legacy: ReasoningPayload =
        serde_json::from_str(r#"{"text":"thinking","signature":"sig-abc"}"#).unwrap();
    assert_eq!(legacy.text, "thinking");
    // provider_call_id is on BlockMeta (envelope-level); verify BlockMeta
    // serde-additivity.
    let legacy_meta: BlockMeta = serde_json::from_str("{}").unwrap();
    assert_eq!(legacy_meta.provider_call_id, None);
    assert_eq!(legacy_meta.source, None);
    let full_meta = BlockMeta::default()
        .with_provider_call_id("abc")
        .with_source("kernel");
    let back: BlockMeta =
        serde_json::from_str(&serde_json::to_string(&full_meta).unwrap()).unwrap();
    assert_eq!(back.provider_call_id.as_deref(), Some("abc"));
    assert_eq!(back.source.as_deref(), Some("kernel"));
    let call = ToolCallPayload::new("echo", json!({"a": 1}));
    let round_trip: ToolCallPayload =
        serde_json::from_str(&serde_json::to_string(&call).unwrap()).unwrap();
    assert_eq!(round_trip, call);
}

#[test]
fn content_is_first_class_with_three_shapes() {
    // Three content shapes: Text, ToolCall, ToolResult. No kind field.
    let make = |content: BlockContent| {
        ContextBlock::new(common::block_id(), content, BlockMeta::default())
    };

    let text = make(BlockContent::text(Role::User, "any role"));
    assert!(matches!(text.content(), BlockContent::Message { .. }));

    let call = make(BlockContent::ToolCall(ToolCallPayload::new(
        "echo",
        json!({}),
    )));
    assert!(matches!(call.content(), BlockContent::ToolCall(_)));

    let result = make(BlockContent::ToolResult(ToolResultPayload::new(
        call.id(),
        ToolResultStatus::Succeeded,
        ToolOutput::new(json!({})),
    )));
    assert!(matches!(result.content(), BlockContent::ToolResult(_)));
}

#[test]
fn tool_result_media_is_serde_additive_both_ways() {
    let payload = ToolResultPayload::new(
        common::block_id(),
        ToolResultStatus::Succeeded,
        ToolOutput::new(json!("ok")),
    )
    .with_media(vec![MediaRef::new("image/png", "a1")])
    .with_notes(vec![TextPayload::new("note")]);
    let json = serde_json::to_string(&payload).unwrap();
    assert!(json.contains(r#""media":[{"media_type":"image/png","reference":"a1"}]"#));
    let restored: ToolResultPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.media, vec![MediaRef::new("image/png", "a1")]);

    // empty media is skipped on the wire...
    let empty = ToolResultPayload::new(
        common::block_id(),
        ToolResultStatus::Failed,
        ToolOutput::new(json!("no")),
    );
    assert!(!serde_json::to_string(&empty).unwrap().contains("media"));
    // ...and a stored record without the field defaults to empty.
    let old = json!({
        "call_block_id": common::block_id(),
        "status": "succeeded",
        "output": {"content": "ok", "truncation": "none", "meta": null, "artifact": null},
    });
    let from_old: ToolResultPayload = serde_json::from_value(old).unwrap();
    assert!(from_old.media.is_empty());
    assert!(from_old.notes.is_empty());
}

#[test]
fn block_content_and_context_block_text_helpers_equal_long_form() {
    let id = common::block_id();
    for role in [Role::System, Role::User, Role::Assistant] {
        let content_helper = BlockContent::text(role, "hello world");
        let content_long = BlockContent::Message {
            role,
            parts: vec![ContentPart::Text(TextPayload::new("hello world"))],
        };
        assert_eq!(content_helper, content_long);

        let block_helper = ContextBlock::text(id, role, "hello world");
        let block_long = ContextBlock::new(id, content_long, BlockMeta::default());
        assert_eq!(block_helper, block_long);
    }
}
