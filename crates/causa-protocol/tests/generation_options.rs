//! Pure generation mappings preserve options and reject unsupported combinations.
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, ContentPart, ContextBlock, ContextFrame,
    GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ModelRef, ReasoningEffort,
    TextPayload, ToolSurface,
};
use causa_protocol::translation::{
    anthropic::render_anthropic_messages, media::MediaSet,
    openai_chat::render_openai_chat_messages, openai_responses::render_openai_responses_input,
};
use serde_json::{Value, json};

fn render(protocol: usize, generation: &GenerationOptions) -> Result<Value, ModelInvokeError> {
    let frame = ContextFrame {
        blocks: vec![ContextBlock::new(
            BlockId::new(uuid::Uuid::from_u128(1)),
            BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("hi"))]),
            BlockMeta {
                source: Some("user".into()),
                ..Default::default()
            },
        )],
    };
    let renderer = match protocol {
        0 => render_openai_chat_messages,
        1 => render_openai_responses_input,
        _ => render_anthropic_messages,
    };
    renderer(
        &frame,
        &MediaSet::new(),
        &ToolSurface::empty(),
        generation,
        &ModelRef::new("configured-model"),
        CacheDirective::None,
    )
}

#[test]
fn openai_maps_every_effort_and_completion_budget_without_enabling_replay() {
    for (effort, wire) in [
        (ReasoningEffort::None, "none"),
        (ReasoningEffort::Minimal, "minimal"),
        (ReasoningEffort::Low, "low"),
        (ReasoningEffort::Medium, "medium"),
        (ReasoningEffort::High, "high"),
        (ReasoningEffort::XHigh, "xhigh"),
        (ReasoningEffort::Max, "max"),
    ] {
        let generation = GenerationOptions {
            reasoning_effort: Some(effort),
            max_tokens: Some(8192),
            ..Default::default()
        };
        let chat = render(0, &generation).unwrap();
        assert_eq!(chat["reasoning_effort"], wire);
        assert_eq!(chat["max_completion_tokens"], 8192);
        assert!(chat.get("max_tokens").is_none());
        assert!(chat.get("reasoning").is_none());
        let responses = render(1, &generation).unwrap();
        assert_eq!(responses["reasoning"], json!({"effort": wire}));
        assert_eq!(responses["max_output_tokens"], 8192);
        assert!(responses.get("reasoning_effort").is_none());
        assert_eq!(responses, render(1, &generation).unwrap());
    }
    for protocol in [0, 1] {
        let body = render(protocol, &GenerationOptions::default()).unwrap();
        for key in [
            "reasoning",
            "reasoning_effort",
            "max_completion_tokens",
            "max_output_tokens",
            "temperature",
        ] {
            assert!(body.get(key).is_none());
        }
        let body = render(
            protocol,
            &GenerationOptions {
                reasoning_effort: Some(ReasoningEffort::None),
                temperature: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(body["temperature"], 0.0);
    }
}

#[test]
fn invalid_options_fail_locally_instead_of_being_dropped_or_clipped() {
    for protocol in [0, 1, 2] {
        for generation in [
            GenerationOptions {
                max_tokens: Some(0),
                ..Default::default()
            },
            GenerationOptions {
                temperature: Some(f32::NAN),
                ..Default::default()
            },
            GenerationOptions {
                temperature: Some(f32::INFINITY),
                ..Default::default()
            },
            GenerationOptions {
                temperature: Some(-0.1),
                ..Default::default()
            },
            GenerationOptions {
                temperature: Some(2.1),
                ..Default::default()
            },
            GenerationOptions {
                reasoning_effort: Some(ReasoningEffort::High),
                temperature: Some(1.0),
                ..Default::default()
            },
        ] {
            assert_eq!(
                render(protocol, &generation).unwrap_err().kind,
                ModelInvokeErrorKind::InvalidRequest
            );
        }
    }
    for effort in [
        ReasoningEffort::None,
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
        ReasoningEffort::Max,
    ] {
        let error = render(
            2,
            &GenerationOptions {
                reasoning_effort: Some(effort),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
        assert!(error.message.contains("thinking-block replay"));
    }
    assert!(
        render(
            2,
            &GenerationOptions {
                output_schema: Some(json!({"type": "object"})),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        render(
            2,
            &GenerationOptions {
                temperature: Some(1.1),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert_eq!(
        render(2, &GenerationOptions::default()).unwrap()["max_tokens"],
        4096
    );
}
