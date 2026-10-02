//! Request constraints are separate from the kernel's permissive material edits.
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, ContentPart, Context, ContextBlock,
    ContextFrame, GenerationOptions, MediaRef, ModelInvokeErrorKind, ModelRef, TextPayload,
    ToolCallPayload, ToolOutput, ToolResultPayload, ToolResultStatus, ToolSurface,
};
use causa_protocol::translation::{anthropic, media::MediaSet, openai_chat, openai_responses};
use serde_json::{Value, json};

fn id(value: u128) -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(value))
}
fn call(value: u128, wire: &str) -> ContextBlock {
    ContextBlock::new(
        id(value),
        BlockContent::ToolCall(ToolCallPayload {
            tool_name: "lookup".into(),
            arguments: json!({}),
        }),
        BlockMeta {
            provider_call_id: Some(wire.into()),
            ..Default::default()
        },
    )
}
fn result(value: u128, declaration: u128, media: bool) -> ContextBlock {
    ContextBlock::new(
        id(value),
        BlockContent::ToolResult(ToolResultPayload {
            call_block_id: id(declaration),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("found")),
            notes: vec![TextPayload::new("note")],
            media: if media {
                vec![MediaRef::new("image/png", format!("asset-{value}"))]
            } else {
                vec![]
            },
        }),
        BlockMeta::default(),
    )
}
fn text(value: u128, source: &str) -> ContextBlock {
    ContextBlock::new(
        id(value),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("text"))]),
        BlockMeta {
            source: Some(source.into()),
            ..Default::default()
        },
    )
}
fn render(protocol: usize, frame: &ContextFrame) -> Result<Value, causa_kernel::ModelInvokeError> {
    let renderer = match protocol {
        0 => anthropic::render_anthropic_messages,
        1 => openai_chat::render_openai_chat_messages,
        _ => openai_responses::render_openai_responses_input,
    };
    renderer(
        frame,
        &MediaSet::new(),
        &ToolSurface::empty(),
        &GenerationOptions::default(),
        &ModelRef::new("test"),
        CacheDirective::None,
    )
}

#[test]
fn partial_material_is_legal_but_self_contained_requests_require_complete_unambiguous_exchanges() {
    let cases = [
        vec![result(2, 1, false)],
        vec![call(1, "a")],
        vec![result(2, 1, false), call(1, "a")],
        vec![call(1, "a"), result(2, 1, false), result(3, 1, false)],
        vec![
            call(1, "a"),
            call(2, "a"),
            result(3, 1, false),
            result(4, 2, false),
        ],
        vec![call(1, ""), result(2, 1, false)],
    ];
    for blocks in cases {
        let context = Context::from_blocks(blocks).expect("material remains valid");
        for protocol in 0..3 {
            let error = render(protocol, &context.frame()).unwrap_err();
            assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
        }
    }
}

#[test]
fn duplicate_declaration_occurrences_are_rejected_even_in_a_directly_assembled_frame() {
    let frame = ContextFrame {
        blocks: vec![call(1, "a"), call(1, "a"), result(2, 1, false)],
    };
    for protocol in 0..3 {
        assert!(render(protocol, &frame).is_err());
    }
}

#[test]
fn results_may_follow_final_processor_order_without_losing_identity() {
    let frame = ContextFrame {
        blocks: vec![
            call(1, "a"),
            call(2, "b"),
            result(3, 2, false),
            result(4, 1, false),
        ],
    };
    for protocol in 0..3 {
        render(protocol, &frame).expect("reordered complete results");
    }
    let chat = render(1, &frame).unwrap();
    assert_eq!(chat["messages"][1]["tool_call_id"], "b");
    assert_eq!(chat["messages"][2]["tool_call_id"], "a");
}

#[test]
fn adjacency_is_checked_on_each_protocols_wire_shape() {
    let frame = ContextFrame {
        blocks: vec![call(1, "a"), text(2, "user"), result(3, 1, false)],
    };
    assert!(render(0, &frame).is_err());
    assert!(render(1, &frame).is_err());
    render(2, &frame)
        .expect("Responses has flat item references rather than adjacent message roles");
    let with_system = ContextFrame {
        blocks: vec![call(1, "a"), text(2, "system"), result(3, 1, false)],
    };
    render(0, &with_system).expect("system is a top-level parameter");
    assert!(render(1, &with_system).is_err());
    render(2, &with_system).unwrap();
}

#[test]
fn anthropic_user_text_can_follow_all_results_but_not_split_them() {
    let valid = ContextFrame {
        blocks: vec![call(1, "a"), result(2, 1, false), text(3, "user")],
    };
    render(0, &valid).unwrap();
    let split = ContextFrame {
        blocks: vec![
            call(1, "a"),
            call(2, "b"),
            result(3, 1, false),
            text(4, "user"),
            result(5, 2, false),
        ],
    };
    assert!(render(0, &split).is_err());
    assert!(render(1, &split).is_err());
}

#[test]
fn chat_media_hoists_after_the_complete_tool_reply_group_in_result_order() {
    let frame = ContextFrame {
        blocks: vec![
            call(1, "a"),
            call(2, "b"),
            result(3, 2, true),
            result(4, 1, true),
        ],
    };
    let body = render(1, &frame).unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(
        messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["assistant", "tool", "tool", "user", "user"]
    );
    assert_eq!(messages[1]["tool_call_id"], "b");
    assert_eq!(messages[2]["tool_call_id"], "a");
    assert_eq!(
        messages[3]["content"][0]["text"],
        "[tool result media for call b]"
    );
    assert_eq!(
        messages[4]["content"][0]["text"],
        "[tool result media for call a]"
    );
    assert_eq!(
        messages[3]["content"][1]["text"],
        "[media: image/png asset-3]"
    );
    assert_eq!(
        messages[4]["content"][1]["text"],
        "[media: image/png asset-4]"
    );
    assert!(messages[1]["content"].as_str().unwrap().contains("note"));
    assert_eq!(frame.blocks.len(), 4);
}
