//! Pure generation mappings: the adapter's own options are checked, and the
//! reasoning effort and extra request fields are written as the host gave them.
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, ContentPart, ContextBlock, ContextFrame,
    GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ModelRef, TextPayload, ToolSurface,
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

/// Efforts a host might send: today's common levels, one a server asked for
/// in place of `none`, one nobody has defined yet, and the empty string.
const EFFORTS: [&str; 6] = ["none", "minimal", "high", "off", "level-of-tomorrow", ""];

fn with_effort(effort: &str) -> GenerationOptions {
    let mut generation = GenerationOptions::default();
    generation.reasoning_effort = Some(effort.into());
    generation.max_tokens = Some(LIMIT);
    generation
}

#[test]
fn every_protocol_writes_any_effort_string_to_its_own_key() {
    for effort in EFFORTS {
        let generation = with_effort(effort);
        let chat = render(0, &generation).unwrap();
        assert_eq!(chat["reasoning_effort"], effort);
        assert_eq!(chat["max_completion_tokens"], LIMIT);
        assert!(chat.get("max_tokens").is_none());
        assert!(chat.get("reasoning").is_none());
        let responses = render(1, &generation).unwrap();
        assert_eq!(responses["reasoning"], json!({"effort": effort}));
        assert_eq!(responses["max_output_tokens"], LIMIT);
        assert!(responses.get("reasoning_effort").is_none());
        assert_eq!(responses, render(1, &generation).unwrap());
        let anthropic = render(2, &generation).unwrap();
        assert_eq!(anthropic["output_config"], json!({"effort": effort}));
        assert_eq!(anthropic["max_tokens"], LIMIT);
        // No `thinking` object is derived from the effort.
        assert!(anthropic.get("thinking").is_none());
        assert!(anthropic.get("reasoning_effort").is_none());
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
            "thinking",
        ] {
            assert!(body.get(key).is_none());
        }
    }
    let body = render(2, &{
        let mut generation = GenerationOptions::default();
        generation.max_tokens = Some(LIMIT);
        generation
    })
    .unwrap();
    assert!(body.get("thinking").is_none());
    assert!(body.get("output_config").is_none());
}

#[test]
fn extra_request_fields_reach_every_body_unchanged() {
    let thinking = json!({
        "type": "adaptive",
        "budget_tokens": 1234,
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        "nested": [1, "two", {"three": null}],
    });
    let mut generation = with_effort("high");
    generation
        .extra_body
        .insert("thinking".into(), thinking.clone());
    generation
        .extra_body
        .insert("enable_thinking".into(), json!(true));
    generation.extra_body.insert("top_k".into(), json!(40));
    for protocol in [0, 1, 2] {
        let body = render(protocol, &generation).unwrap();
        assert_eq!(body["thinking"], thinking, "{protocol}");
        assert_eq!(body["enable_thinking"], json!(true), "{protocol}");
        assert_eq!(body["top_k"], json!(40), "{protocol}");
        // The serialized value is the same bytes the host supplied.
        assert_eq!(
            serde_json::to_string(&body["thinking"]).unwrap(),
            serde_json::to_string(&thinking).unwrap()
        );
    }
}

#[test]
fn an_extra_field_never_replaces_a_key_the_adapter_wrote() {
    let mut generation = with_effort("high");
    generation.temperature = Some(0.5);
    for (key, value) in [
        ("model", json!("replaced")),
        ("messages", json!([])),
        ("input", json!([])),
        ("temperature", json!(2)),
        ("max_tokens", json!(1)),
        ("max_completion_tokens", json!(1)),
        ("max_output_tokens", json!(1)),
        ("reasoning_effort", json!("low")),
        ("reasoning", json!({"effort": "low", "summary": "auto"})),
        ("output_config", json!({"effort": "low"})),
    ] {
        generation.extra_body.insert(key.into(), value);
    }
    let chat = render(0, &generation).unwrap();
    assert_eq!(chat["model"], "configured-model");
    assert_eq!(chat["messages"].as_array().unwrap().len(), 1);
    assert_eq!(chat["temperature"], 0.5);
    assert_eq!(chat["max_completion_tokens"], LIMIT);
    assert_eq!(chat["reasoning_effort"], "high");
    let responses = render(1, &generation).unwrap();
    assert_eq!(responses["model"], "configured-model");
    assert_eq!(responses["input"].as_array().unwrap().len(), 1);
    assert_eq!(responses["max_output_tokens"], LIMIT);
    assert_eq!(responses["reasoning"], json!({"effort": "high"}));
    let anthropic = render(2, &generation).unwrap();
    assert_eq!(anthropic["model"], "configured-model");
    assert_eq!(anthropic["messages"].as_array().unwrap().len(), 1);
    assert_eq!(anthropic["max_tokens"], LIMIT);
    assert_eq!(anthropic["output_config"], json!({"effort": "high"}));

    // With no effort set, the host's own reasoning object is what is sent.
    let mut generation = GenerationOptions::default();
    generation.max_tokens = Some(LIMIT);
    let reasoning = json!({"effort": "high", "summary": "auto"});
    generation
        .extra_body
        .insert("reasoning".into(), reasoning.clone());
    assert_eq!(render(1, &generation).unwrap()["reasoning"], reasoning);
}

#[test]
fn no_renderer_judges_the_effort_or_its_combination_with_a_temperature() {
    for protocol in [0, 1, 2] {
        for effort in EFFORTS {
            let mut generation = with_effort(effort);
            generation.temperature = Some(1.0);
            generation
                .extra_body
                .insert("thinking".into(), json!({"type": "disabled"}));
            let body = render(protocol, &generation).unwrap();
            assert_eq!(body["temperature"], 1.0, "{protocol} {effort}");
        }
    }
    // What the adapter itself needs is still checked: no default ceiling on
    // Anthropic; the limit is the host's to set.
    let error = render_anthropic(
        &GenerationOptions::default(),
        &AnthropicRenderOptions::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest);
    assert!(error.message.contains("max_tokens"), "{error}");
}

#[test]
fn responses_sends_encrypted_reasoning_and_store_false_by_default() {
    let generation = {
        let mut generation = GenerationOptions::default();
        generation.reasoning_effort = Some("high".into());
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
