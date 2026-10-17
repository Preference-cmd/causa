//! Pure generation mappings: the adapter's own options are checked, and the
//! reasoning effort and extra request fields are written as the host gave them.
use causa_kernel::{
    BlockId, CacheDirective, ContextBlock, ContextFrame, GenerationOptions, ModelInvokeError,
    ModelInvokeErrorKind, ModelRef, Role,
};
use causa_protocol::translation::render_options::{
    AnthropicRenderOptions, ChatMaxTokensField, ChatRenderOptions, ResponsesRenderOptions,
};
use causa_protocol::translation::{
    anthropic::render_anthropic_messages, media::MediaSet,
    openai_chat::render_openai_chat_messages, openai_responses::render_openai_responses_input,
};
use serde_json::{Value, json};

fn frame() -> ContextFrame {
    ContextFrame {
        blocks: vec![ContextBlock::text(
            BlockId::new(uuid::Uuid::from_u128(1)),
            Role::User,
            "hi",
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
    GenerationOptions::default()
        .with_reasoning_effort(effort)
        .with_max_tokens(LIMIT)
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
    let body = render(2, &GenerationOptions::default().with_max_tokens(LIMIT)).unwrap();
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
    }
}

#[test]
fn the_merge_continues_below_the_first_level_and_stops_at_the_schema() {
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    });
    // What a host might send hoping to adjust the schema in place.
    let fragment = json!({
        "type": "array",
        "properties": {"answer": {"description": "added"}, "other": {"type": "number"}},
        "additionalProperties": false,
    });
    let mut generation = GenerationOptions::default();
    generation.output_schema = Some(schema.clone());
    generation.extra_body.insert(
        "response_format".into(),
        json!({"json_schema": {"description": "an answer", "strict": true, "schema": fragment}}),
    );
    generation.extra_body.insert(
        "text".into(),
        json!({"format": {"description": "an answer", "strict": true, "schema": fragment}}),
    );

    // Chat: `description` sits two objects below the body.
    let chat = &render(0, &generation).unwrap()["response_format"];
    assert_eq!(chat["type"], "json_schema");
    assert_eq!(chat["json_schema"]["description"], "an answer");
    assert_eq!(chat["json_schema"]["strict"], false);
    assert_eq!(chat["json_schema"]["schema"], schema);

    let responses = &render(1, &generation).unwrap()["text"]["format"];
    assert_eq!(responses["description"], "an answer");
    assert_eq!(responses["strict"], false);
    assert_eq!(responses["schema"], schema);

    // Without `output_schema` the host's object is the whole value.
    generation.output_schema = None;
    assert_eq!(
        render(0, &generation).unwrap()["response_format"]["json_schema"]["schema"],
        fragment
    );
    assert_eq!(
        render(1, &generation).unwrap()["text"]["format"]["schema"],
        fragment
    );
}

#[test]
fn a_null_extra_field_is_sent_and_a_null_extra_body_reads_as_empty() {
    let mut generation = with_effort("high");
    generation.extra_body.insert("user".into(), Value::Null);
    // A null does not unset a key the adapter wrote.
    generation
        .extra_body
        .insert("reasoning".into(), Value::Null);
    for protocol in [0, 1, 2] {
        let body = render(protocol, &generation).unwrap();
        assert_eq!(body.get("user"), Some(&Value::Null), "{protocol}");
    }
    assert_eq!(
        render(1, &generation).unwrap()["reasoning"],
        json!({"effort": "high"})
    );

    let read: GenerationOptions = serde_json::from_value(json!({"extra_body": null})).unwrap();
    assert!(read.extra_body.is_empty());
    let read: GenerationOptions =
        serde_json::from_value(json!({"extra_body": {"top_k": 40}})).unwrap();
    assert_eq!(read.extra_body["top_k"], 40);
    assert!(serde_json::from_value::<GenerationOptions>(json!({"extra_body": []})).is_err());
}

#[test]
fn chat_writes_the_limit_under_the_chosen_field_only() {
    let mut options = ChatRenderOptions::default();
    let limited = GenerationOptions::default().with_max_tokens(LIMIT);

    let body = render_chat(&limited, &options).unwrap();
    assert_eq!(body["max_completion_tokens"], LIMIT);
    assert!(body.get("max_tokens").is_none());

    options.max_tokens_field = ChatMaxTokensField::MaxTokens;
    let body = render_chat(&limited, &options).unwrap();
    assert_eq!(body["max_tokens"], LIMIT);
    assert!(body.get("max_completion_tokens").is_none());

    // No limit writes neither field, whichever is chosen.
    let body = render_chat(&GenerationOptions::default(), &options).unwrap();
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("max_completion_tokens").is_none());

    // The chosen field is the adapter's key: an extra value under it loses,
    // and the other name is merged like any extra key.
    let mut generation = limited.clone();
    generation.extra_body.insert("max_tokens".into(), json!(1));
    generation
        .extra_body
        .insert("max_completion_tokens".into(), json!(2));
    let body = render_chat(&generation, &options).unwrap();
    assert_eq!(body["max_tokens"], LIMIT);
    assert_eq!(body["max_completion_tokens"], 2);
}

#[test]
fn kept_goes_by_the_key_the_adapter_wrote() {
    // Chat writes the limit as `max_completion_tokens`; the older name is a
    // different key and goes out beside it.
    let mut generation = with_effort("high");
    generation.extra_body.insert("max_tokens".into(), json!(1));
    let chat = render(0, &generation).unwrap();
    assert_eq!(chat["max_completion_tokens"], LIMIT);
    assert_eq!(chat["max_tokens"], 1);

    // Responses writes `include` and `store` by default, so the host's lose;
    // with the render options turned off, the host's are sent.
    generation
        .extra_body
        .insert("include".into(), json!(["message.output_text.logprobs"]));
    generation.extra_body.insert("store".into(), json!(true));
    let body = render(1, &generation).unwrap();
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["store"], false);
    let mut options = ResponsesRenderOptions::default();
    options.include_encrypted_reasoning = false;
    options.store = None;
    let body = render_responses(&generation, &options).unwrap();
    assert_eq!(body["include"], json!(["message.output_text.logprobs"]));
    assert_eq!(body["store"], true);
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
    assert_eq!(
        responses["reasoning"],
        json!({"effort": "high", "summary": "auto"})
    );
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
fn an_extra_object_joins_the_object_the_adapter_wrote_under_the_same_key() {
    // Responses: a summary beside the effort.
    let mut generation = with_effort("high");
    generation
        .extra_body
        .insert("reasoning".into(), json!({"summary": "auto"}));
    assert_eq!(
        render(1, &generation).unwrap()["reasoning"],
        json!({"effort": "high", "summary": "auto"})
    );

    // Responses: a verbosity beside the output format, which stays whole.
    let mut generation = GenerationOptions::default();
    generation.output_schema = Some(json!({"type": "object"}));
    generation.extra_body.insert(
        "text".into(),
        json!({"verbosity": "low", "format": {"type": "text", "strict": true}}),
    );
    let text = &render(1, &generation).unwrap()["text"];
    assert_eq!(text["verbosity"], "low");
    assert_eq!(text["format"]["type"], "json_schema");
    assert_eq!(text["format"]["schema"], json!({"type": "object"}));

    // Anthropic: an output format beside the effort, and a thinking object.
    let mut generation = with_effort("high");
    generation
        .extra_body
        .insert("thinking".into(), json!({"type": "adaptive"}));
    generation.extra_body.insert(
        "output_config".into(),
        json!({"effort": "low", "format": {"type": "json_schema"}}),
    );
    let anthropic = render(2, &generation).unwrap();
    assert_eq!(anthropic["thinking"], json!({"type": "adaptive"}));
    assert_eq!(
        anthropic["output_config"],
        json!({"effort": "high", "format": {"type": "json_schema"}})
    );

    // Chat: the effort is a string, so nothing merges into it.
    let mut generation = with_effort("off");
    generation
        .extra_body
        .insert("thinking".into(), json!({"type": "disabled"}));
    generation
        .extra_body
        .insert("reasoning_effort".into(), json!({"level": "low"}));
    let chat = render(0, &generation).unwrap();
    assert_eq!(chat["reasoning_effort"], "off");
    assert_eq!(chat["thinking"], json!({"type": "disabled"}));

    // An array the adapter wrote is never merged into or replaced.
    let mut generation = with_effort("high");
    generation
        .extra_body
        .insert("messages".into(), json!({"0": "replaced"}));
    assert!(render(0, &generation).unwrap()["messages"].is_array());
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
    let generation = GenerationOptions::default()
        .with_reasoning_effort("high")
        .with_max_tokens(LIMIT);
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
            GenerationOptions::default().with_max_tokens(0),
            GenerationOptions::default()
                .with_max_tokens(LIMIT)
                .with_temperature(f32::NAN),
            GenerationOptions::default()
                .with_max_tokens(LIMIT)
                .with_temperature(f32::INFINITY),
            GenerationOptions::default()
                .with_max_tokens(LIMIT)
                .with_temperature(-0.1),
            GenerationOptions::default()
                .with_max_tokens(LIMIT)
                .with_temperature(2.1),
        ] {
            assert_eq!(
                render(protocol, &generation).unwrap_err().kind,
                ModelInvokeErrorKind::InvalidRequest
            );
        }
    }
    assert!(
        render(
            2,
            &GenerationOptions::default()
                .with_max_tokens(LIMIT)
                .with_output_schema(json!({"type": "object"}))
        )
        .is_err()
    );
    assert!(
        render(
            2,
            &GenerationOptions::default()
                .with_max_tokens(LIMIT)
                .with_temperature(1.1)
        )
        .is_err()
    );
}
