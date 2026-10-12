//! Pure generation mappings preserve options and reject unsupported combinations.
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, ContentPart, ContextBlock, ContextFrame,
    GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ModelRef, ReasoningEffort,
    TextPayload, ToolSurface,
};
use causa_protocol::translation::render_options::{
    AnthropicRenderOptions, ChatRenderOptions, ResponsesRenderOptions,
};
use causa_protocol::translation::{
    anthropic::render_anthropic_messages, media::MediaSet,
    openai_chat::render_openai_chat_messages, openai_responses::render_openai_responses_input,
};
use serde_json::{Value, json};

fn frame() -> ContextFrame {
    ContextFrame {
        blocks: vec![ContextBlock::new(
            BlockId::new(uuid::Uuid::from_u128(1)),
            BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("hi"))]),
            BlockMeta::from_source("user"),
        )],
    }
}

fn render_chat(
    generation: &GenerationOptions,
    options: &ChatRenderOptions,
) -> Result<Value, ModelInvokeError> {
    render_openai_chat_messages(
        &frame(),
        &MediaSet::new(),
        &ToolSurface::empty(),
        generation,
        &ModelRef::new("configured-model"),
        CacheDirective::None,
        options,
    )
}

fn render_responses(
    generation: &GenerationOptions,
    options: &ResponsesRenderOptions,
) -> Result<Value, ModelInvokeError> {
    render_openai_responses_input(
        &frame(),
        &MediaSet::new(),
        &ToolSurface::empty(),
        generation,
        &ModelRef::new("configured-model"),
        CacheDirective::None,
        options,
    )
}

fn render_anthropic(
    generation: &GenerationOptions,
    options: &AnthropicRenderOptions,
) -> Result<Value, ModelInvokeError> {
    render_anthropic_messages(
        &frame(),
        &MediaSet::new(),
        &ToolSurface::empty(),
        generation,
        &ModelRef::new("configured-model"),
        CacheDirective::None,
        options,
    )
}

fn render(protocol: usize, generation: &GenerationOptions) -> Result<Value, ModelInvokeError> {
    match protocol {
        0 => render_chat(generation, &ChatRenderOptions::default()),
        1 => render_responses(generation, &ResponsesRenderOptions::default()),
        _ => render_anthropic(generation, &AnthropicRenderOptions::default()),
    }
}

/// `max_tokens` every case supplies unless the case is about the limit itself:
/// the Anthropic wire requires one.
const LIMIT: u32 = 8192;

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
        let generation = {
            let mut generation = GenerationOptions::default();
            generation.reasoning_effort = Some(effort);
            generation.max_tokens = Some(LIMIT);
            generation
        };
        let chat = render(0, &generation).unwrap();
        assert_eq!(chat["reasoning_effort"], wire);
        assert_eq!(chat["max_completion_tokens"], LIMIT);
        assert!(chat.get("max_tokens").is_none());
        assert!(chat.get("reasoning").is_none());
        let responses = render(1, &generation).unwrap();
        assert_eq!(responses["reasoning"], json!({"effort": wire}));
        assert_eq!(responses["max_output_tokens"], LIMIT);
        assert!(responses.get("reasoning_effort").is_none());
        assert_eq!(responses, render(1, &generation).unwrap());
    }
    for protocol in [0, 1] {
        // Unset generation options render nothing on the OpenAI wires
        // (`include`/`store` are separate, covered below).
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
        let body = render(protocol, &{
            let mut generation = GenerationOptions::default();
            generation.reasoning_effort = Some(ReasoningEffort::None);
            generation.temperature = Some(0.0);
            generation
        })
        .unwrap();
        assert_eq!(body["temperature"], 0.0);
    }
}

#[test]
fn anthropic_renders_the_effort_table_and_requires_max_tokens() {
    // Unset effort: no thinking field and no output_config.
    let body = render_anthropic(
        &{
            let mut generation = GenerationOptions::default();
            generation.max_tokens = Some(LIMIT);
            generation
        },
        &AnthropicRenderOptions::default(),
    )
    .unwrap();
    assert_eq!(body["max_tokens"], LIMIT);
    assert!(body.get("thinking").is_none());
    assert!(body.get("output_config").is_none());

    // `None`: thinking explicitly disabled, no effort level.
    let body = render_anthropic(
        &{
            let mut generation = GenerationOptions::default();
            generation.reasoning_effort = Some(ReasoningEffort::None);
            generation.max_tokens = Some(LIMIT);
            generation
        },
        &AnthropicRenderOptions::default(),
    )
    .unwrap();
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("output_config").is_none());

    // Low..Max: adaptive thinking plus the effort level.
    for (effort, wire) in [
        (ReasoningEffort::Low, "low"),
        (ReasoningEffort::Medium, "medium"),
        (ReasoningEffort::High, "high"),
        (ReasoningEffort::XHigh, "xhigh"),
        (ReasoningEffort::Max, "max"),
    ] {
        let body = render_anthropic(
            &{
                let mut generation = GenerationOptions::default();
                generation.reasoning_effort = Some(effort);
                generation.max_tokens = Some(LIMIT);
                generation
            },
            &AnthropicRenderOptions::default(),
        )
        .unwrap();
        assert_eq!(body["thinking"], json!({"type": "adaptive"}), "{wire}");
        assert_eq!(body["output_config"], json!({"effort": wire}), "{wire}");
        assert!(body.get("reasoning_effort").is_none());
    }

    // `Minimal` has no wire level.
    let error = render_anthropic(
        &{
            let mut generation = GenerationOptions::default();
            generation.reasoning_effort = Some(ReasoningEffort::Minimal);
            generation.max_tokens = Some(LIMIT);
            generation
        },
        &AnthropicRenderOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
    assert!(error.message.contains("minimal"), "{error}");

    // No default ceiling: the limit is the host's to set.
    let error = render_anthropic(
        &GenerationOptions::default(),
        &AnthropicRenderOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
    assert!(error.message.contains("max_tokens"), "{error}");

    // The prefix-mismatch option renders only alongside a thinking object.
    let options = {
        let mut options = AnthropicRenderOptions::default();
        options.prefix_mismatch_behavior =
            Some(causa_protocol::translation::render_options::PrefixMismatchBehavior::DropBlock);
        options
    };
    let body = render_anthropic(
        &{
            let mut generation = GenerationOptions::default();
            generation.max_tokens = Some(LIMIT);
            generation
        },
        &options,
    )
    .unwrap();
    assert!(body.get("thinking").is_none());
    let body = render_anthropic(
        &{
            let mut generation = GenerationOptions::default();
            generation.reasoning_effort = Some(ReasoningEffort::High);
            generation.max_tokens = Some(LIMIT);
            generation
        },
        &options,
    )
    .unwrap();
    assert_eq!(
        body["thinking"],
        json!({
            "type": "adaptive",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        })
    );
}

#[test]
fn responses_sends_encrypted_reasoning_and_store_false_by_default() {
    let generation = {
        let mut generation = GenerationOptions::default();
        generation.reasoning_effort = Some(ReasoningEffort::High);
        generation.max_tokens = Some(LIMIT);
        generation
    };
    let body = render_responses(&generation, &ResponsesRenderOptions::default()).unwrap();
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["store"], json!(false));
    assert_eq!(body["reasoning"], json!({"effort": "high"}));

    // Each option changes exactly its own field.
    let body = render_responses(&generation, &{
        let mut options = ResponsesRenderOptions::default();
        options.include_encrypted_reasoning = false;
        options
    })
    .unwrap();
    assert!(body.get("include").is_none());
    assert_eq!(body["store"], json!(false));
    let body = render_responses(&generation, &{
        let mut options = ResponsesRenderOptions::default();
        options.store = None;
        options
    })
    .unwrap();
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert!(body.get("store").is_none());
    let body = render_responses(&generation, &{
        let mut options = ResponsesRenderOptions::default();
        options.store = Some(true);
        options
    })
    .unwrap();
    assert_eq!(body["store"], json!(true));
}

#[test]
fn invalid_options_fail_locally_instead_of_being_dropped_or_clipped() {
    for protocol in [0, 1, 2] {
        for generation in [
            {
                let mut generation = GenerationOptions::default();
                generation.max_tokens = Some(0);
                generation
            },
            {
                let mut generation = GenerationOptions::default();
                generation.max_tokens = Some(LIMIT);
                generation.temperature = Some(f32::NAN);
                generation
            },
            {
                let mut generation = GenerationOptions::default();
                generation.max_tokens = Some(LIMIT);
                generation.temperature = Some(f32::INFINITY);
                generation
            },
            {
                let mut generation = GenerationOptions::default();
                generation.max_tokens = Some(LIMIT);
                generation.temperature = Some(-0.1);
                generation
            },
            {
                let mut generation = GenerationOptions::default();
                generation.max_tokens = Some(LIMIT);
                generation.temperature = Some(2.1);
                generation
            },
            {
                let mut generation = GenerationOptions::default();
                generation.max_tokens = Some(LIMIT);
                generation.reasoning_effort = Some(ReasoningEffort::High);
                generation.temperature = Some(1.0);
                generation
            },
        ] {
            assert_eq!(
                render(protocol, &generation).unwrap_err().kind,
                ModelInvokeErrorKind::InvalidRequest
            );
        }
    }
    assert!(
        render(2, &{
            let mut generation = GenerationOptions::default();
            generation.max_tokens = Some(LIMIT);
            generation.output_schema = Some(json!({"type": "object"}));
            generation
        })
        .is_err()
    );
    assert!(
        render(2, &{
            let mut generation = GenerationOptions::default();
            generation.max_tokens = Some(LIMIT);
            generation.temperature = Some(1.1);
            generation
        })
        .is_err()
    );
}
