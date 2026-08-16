//! Normalized sampling-parameter consumption from `AgentRequest::options()`.
//!
//! The options blob is provider-specific and opaque at the agent boundary;
//! this module gives the request builders typed reads of the well-known
//! sampling keys plus a `remaining` passthrough for everything else. It also
//! owns the model-level applicability rule set: reasoning models reject
//! parameters like `temperature` and `top_p` (see Vercel AI SDK issue #10932
//! for the OpenAI behavior; Anthropic extended thinking similarly rejects
//! `temperature`), so those are stripped when reasoning is enabled.

use serde_json::{Map, Value};

pub const MAX_TOKENS: &str = "max_tokens";
pub const TEMPERATURE: &str = "temperature";
pub const TOP_P: &str = "top_p";
pub const TOP_K: &str = "top_k";
pub const STOP: &str = "stop";
pub const SEED: &str = "seed";
pub const USER: &str = "user";
pub const REASONING_EFFORT: &str = "reasoning_effort";
/// Explicit reasoning switch. Presence of a non-null value (bool or
/// string) marks the request as reasoning-enabled; `reasoning_effort`
/// and this key are interchangeable triggers.
pub const REASONING: &str = "reasoning";
/// Anthropic extended-thinking token budget (`thinking.budget_tokens`).
pub const REASONING_BUDGET_TOKENS: &str = "reasoning_budget_tokens";
/// JSON Schema the model output must conform to (structured output).
/// The same schema is shared across all three protocols.
pub const OUTPUT_SCHEMA: &str = "output_schema";
/// Optional name for the structured-output schema. Defaults to
/// `"structured_output"` (required by OpenAI's json_schema format).
pub const OUTPUT_SCHEMA_NAME: &str = "output_schema_name";
/// Prompt-caching switch. Defaults to **on**; set `false` to disable
/// Anthropic `cache_control` breakpoints.
pub const CACHE_CONTROL: &str = "cache_control";

/// Wire name for Anthropic's stop list (the typed `stop` field maps onto it).
pub const ANTHROPIC_STOP_SEQUENCES: &str = "stop_sequences";

const KNOWN_KEYS: &[&str] = &[
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    STOP,
    SEED,
    USER,
    REASONING_EFFORT,
    REASONING,
    REASONING_BUDGET_TOKENS,
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_NAME,
    CACHE_CONTROL,
];

/// Sampling parameters that reasoning models reject. V1 is a simple table:
/// OpenAI reasoning models reject `temperature` and `top_p` (Vercel AI SDK
/// issue #10932), and Anthropic extended thinking rejects `temperature` too.
pub const REASONING_MODELS_DISABLE: &[&str] = &[TEMPERATURE, TOP_P];

/// Typed read of `AgentRequest::options()`. Present-but-invalid values read
/// as `None` (the field is omitted from the wire request); `remaining`
/// carries every key not recognized by this module.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub user: Option<String>,
    pub reasoning_effort: Option<String>,
    /// JSON Schema for structured output (shared across protocols).
    pub output_schema: Option<Value>,
    /// Name for the structured-output schema; defaults to
    /// `"structured_output"` on the wire.
    pub output_schema_name: Option<String>,
    pub remaining: Value,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            seed: None,
            user: None,
            reasoning_effort: None,
            output_schema: None,
            output_schema_name: None,
            remaining: Value::Object(Map::new()),
        }
    }
}

impl SamplingParams {
    pub fn from_options(options: &Value) -> Self {
        let map = options.as_object();
        Self {
            max_tokens: map.and_then(|m| positive_u32(m.get(MAX_TOKENS))),
            temperature: map.and_then(|m| m.get(TEMPERATURE)).and_then(Value::as_f64),
            top_p: map.and_then(|m| m.get(TOP_P)).and_then(Value::as_f64),
            top_k: map.and_then(|m| positive_u32(m.get(TOP_K))),
            stop: map.and_then(|m| m.get(STOP)).and_then(stop_list),
            seed: map.and_then(|m| m.get(SEED)).and_then(Value::as_u64),
            user: map
                .and_then(|m| m.get(USER))
                .and_then(Value::as_str)
                .map(String::from),
            reasoning_effort: map
                .and_then(|m| m.get(REASONING_EFFORT))
                .and_then(Value::as_str)
                .map(String::from),
            output_schema: map
                .and_then(|m| m.get(OUTPUT_SCHEMA))
                .filter(|v| !v.is_null())
                .cloned(),
            output_schema_name: map
                .and_then(|m| m.get(OUTPUT_SCHEMA_NAME))
                .and_then(Value::as_str)
                .map(String::from),
            remaining: remaining(map),
        }
    }

    /// Insert the wire fields this protocol consumes into `body`, including
    /// the `remaining` passthrough.
    pub fn apply_openai(&self, body: &mut Map<String, Value>) {
        self.apply_common(body);
        if let Some(v) = &self.stop {
            body.insert(
                STOP.to_string(),
                Value::Array(v.iter().cloned().map(Value::String).collect()),
            );
        }
        if let Some(v) = self.seed {
            body.insert(SEED.to_string(), Value::from(v));
        }
        if let Some(v) = &self.user {
            body.insert(USER.to_string(), Value::String(v.clone()));
        }
        if let Some(v) = &self.reasoning_effort {
            body.insert(REASONING_EFFORT.to_string(), Value::String(v.clone()));
        }
        if let Some(schema) = &self.output_schema {
            body.insert(
                "response_format".to_string(),
                json_schema_format(self.output_schema_name.as_deref(), schema),
            );
        }
        apply_remaining(body, &self.remaining);
    }

    /// Insert the wire fields this protocol consumes into `body`, including
    /// the `remaining` passthrough.
    pub fn apply_anthropic(&self, body: &mut Map<String, Value>) {
        self.apply_common(body);
        if let Some(v) = self.top_k {
            body.insert(TOP_K.to_string(), Value::from(v));
        }
        if let Some(v) = &self.stop {
            body.insert(
                ANTHROPIC_STOP_SEQUENCES.to_string(),
                Value::Array(v.iter().cloned().map(Value::String).collect()),
            );
        }
        // Anthropic structured output (GA `output_config.format`; no beta
        // header needed). Anthropic's json_schema format does not take a
        // name — only the schema itself.
        if let Some(schema) = &self.output_schema {
            body.insert(
                "output_config".to_string(),
                Value::Object(Map::from_iter([(
                    "format".to_string(),
                    Value::Object(Map::from_iter([
                        ("type".to_string(), Value::String("json_schema".to_string())),
                        ("schema".to_string(), schema.clone()),
                    ])),
                )])),
            );
        }
        apply_remaining(body, &self.remaining);
    }

    /// Insert the wire fields the OpenAI Responses API consumes.
    /// The Responses API names the output budget `max_output_tokens`
    /// (unlike chat completions' `max_tokens`); the remaining fields
    /// match the chat-completions wire names. Reasoning effort maps to
    /// the nested `reasoning.effort` field.
    pub fn apply_responses(&self, body: &mut Map<String, Value>) {
        if let Some(v) = self.max_tokens {
            body.insert("max_output_tokens".to_string(), Value::from(v));
        }
        if let Some(v) = self.temperature {
            body.insert(TEMPERATURE.to_string(), Value::from(v));
        }
        if let Some(v) = self.top_p {
            body.insert(TOP_P.to_string(), Value::from(v));
        }
        if let Some(v) = &self.stop {
            body.insert(
                STOP.to_string(),
                Value::Array(v.iter().cloned().map(Value::String).collect()),
            );
        }
        if let Some(v) = self.seed {
            body.insert(SEED.to_string(), Value::from(v));
        }
        if let Some(v) = &self.user {
            body.insert(USER.to_string(), Value::String(v.clone()));
        }
        if let Some(v) = &self.reasoning_effort {
            body.insert(
                "reasoning".to_string(),
                Value::Object(Map::from_iter([(
                    "effort".to_string(),
                    Value::String(v.clone()),
                )])),
            );
        }
        // Responses structured output lives under `text.format`.
        if let Some(schema) = &self.output_schema {
            body.insert(
                "text".to_string(),
                Value::Object(Map::from_iter([(
                    "format".to_string(),
                    json_schema_format(self.output_schema_name.as_deref(), schema),
                )])),
            );
        }
        apply_remaining(body, &self.remaining);
    }

    fn apply_common(&self, body: &mut Map<String, Value>) {
        if let Some(v) = self.max_tokens {
            body.insert(MAX_TOKENS.to_string(), Value::from(v));
        }
        if let Some(v) = self.temperature {
            body.insert(TEMPERATURE.to_string(), Value::from(v));
        }
        if let Some(v) = self.top_p {
            body.insert(TOP_P.to_string(), Value::from(v));
        }
    }
}

/// Whether the request enables reasoning. A non-null `reasoning` value or
/// the presence of `reasoning_effort` in `options` marks the request as
/// reasoning-enabled; the same trigger stands in for Anthropic extended
/// thinking in V1.
pub fn reasoning_enabled(options: &Value) -> bool {
    options.get(REASONING).is_some_and(|v| !v.is_null())
        || options.get(REASONING_EFFORT).is_some_and(|v| !v.is_null())
}

/// OpenAI `json_schema` response-format payload shared by chat
/// completions (`response_format`) and the Responses API (`text.format`).
/// Strict mode is on: the caller's schema must be fully constrained
/// (`additionalProperties: false` plus all fields `required`).
fn json_schema_format(name: Option<&str>, schema: &Value) -> Value {
    Value::Object(Map::from_iter([
        ("type".to_string(), Value::String("json_schema".to_string())),
        (
            "json_schema".to_string(),
            Value::Object(Map::from_iter([
                (
                    "name".to_string(),
                    Value::String(name.unwrap_or("structured_output").to_string()),
                ),
                ("schema".to_string(), schema.clone()),
                ("strict".to_string(), Value::Bool(true)),
            ])),
        ),
    ]))
}

/// Read the Anthropic extended-thinking budget from `options`, defaulting
/// to 4096 when reasoning is enabled. Returns `None` when reasoning is off.
pub fn reasoning_budget(options: &Value) -> Option<u32> {
    if !reasoning_enabled(options) {
        return None;
    }
    let explicit = options.get(REASONING_BUDGET_TOKENS).and_then(Value::as_u64);
    Some(
        explicit
            .filter(|n| *n > 0)
            .map(|n| n as u32)
            .unwrap_or(4096),
    )
}

/// Whether prompt caching is enabled. Defaults to **on** (grill Q10:
/// `cache: on`); an explicit `false` under the `cache_control` key opts
/// out. Any other value (or a non-map `options`) keeps the default.
pub fn cache_control_enabled(options: &Value) -> bool {
    match options.get(CACHE_CONTROL) {
        Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) | Some(_) | None => true,
    }
}

/// Whether a sampling key is inapplicable when reasoning is enabled.
pub fn inapplicable_for_reasoning(key: &str) -> bool {
    REASONING_MODELS_DISABLE.contains(&key)
}

/// Remove inapplicable sampling keys when reasoning is enabled. Returns the
/// cleaned options map plus the keys that were stripped so the caller can
/// warn. With reasoning disabled (or a non-map `options`) this is a no-op.
pub fn strip_inapplicable(options: &Value, reasoning_enabled: bool) -> (Value, Vec<&'static str>) {
    let Some(map) = options.as_object() else {
        return (options.clone(), Vec::new());
    };
    if !reasoning_enabled {
        return (Value::Object(map.clone()), Vec::new());
    }
    let mut cleaned = map.clone();
    let mut stripped = Vec::new();
    for key in REASONING_MODELS_DISABLE {
        if cleaned.get(*key).is_some_and(|v| !v.is_null()) {
            cleaned.remove(*key);
            stripped.push(*key);
        }
    }
    (Value::Object(cleaned), stripped)
}

/// Read `options`, applying the reasoning-model applicability rule set.
/// Returns the typed params plus the keys stripped so the caller can warn.
pub fn from_options_with_applicability(options: &Value) -> (SamplingParams, Vec<&'static str>) {
    let (cleaned, stripped) = strip_inapplicable(options, reasoning_enabled(options));
    (SamplingParams::from_options(&cleaned), stripped)
}

fn positive_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_u64)
        .filter(|n| *n > 0)
        .map(|n| n as u32)
}

fn stop_list(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::String(s) => Some(vec![s.clone()]),
        Value::Array(items) => items
            .iter()
            .map(Value::as_str)
            .map(|s| s.map(String::from))
            .collect(),
        _ => None,
    }
}

fn remaining(map: Option<&Map<String, Value>>) -> Value {
    let mut out = Map::new();
    if let Some(map) = map {
        for (k, v) in map {
            if !KNOWN_KEYS.contains(&k.as_str()) {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    Value::Object(out)
}

fn apply_remaining(body: &mut Map<String, Value>, remaining: &Value) {
    if let Some(extra) = remaining.as_object() {
        for (k, v) in extra {
            body.insert(k.clone(), v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn options(entries: &[(&str, Value)]) -> Value {
        Value::Object(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    fn invalid_values() -> Vec<(&'static str, Value)> {
        vec![
            (MAX_TOKENS, json!("many")),
            (TEMPERATURE, json!("hot")),
            (TOP_P, json!("high")),
            (TOP_K, json!("lots")),
            (STOP, json!(42)),
            (SEED, json!("seed")),
            (USER, json!(123)),
            (REASONING_EFFORT, json!(true)),
        ]
    }

    fn params_for(key: &str, value: Value) -> SamplingParams {
        SamplingParams::from_options(&options(&[(key, value)]))
    }

    fn assert_key_none(key: &str, invalid: Value) {
        let absent = SamplingParams::from_options(&options(&[]));
        let null = SamplingParams::from_options(&options(&[(key, Value::Null)]));
        let invalid = params_for(key, invalid);
        assert_eq!(
            absent, null,
            "absent and explicit null must read the same for {key}"
        );
        match key {
            MAX_TOKENS => {
                assert_eq!(absent.max_tokens, None);
                assert_eq!(invalid.max_tokens, None);
            }
            TEMPERATURE => {
                assert_eq!(absent.temperature, None);
                assert_eq!(invalid.temperature, None);
            }
            TOP_P => {
                assert_eq!(absent.top_p, None);
                assert_eq!(invalid.top_p, None);
            }
            TOP_K => {
                assert_eq!(absent.top_k, None);
                assert_eq!(invalid.top_k, None);
            }
            STOP => {
                assert_eq!(absent.stop, None);
                assert_eq!(invalid.stop, None);
            }
            SEED => {
                assert_eq!(absent.seed, None);
                assert_eq!(invalid.seed, None);
            }
            USER => {
                assert_eq!(absent.user, None);
                assert_eq!(invalid.user, None);
            }
            REASONING_EFFORT => {
                assert_eq!(absent.reasoning_effort, None);
                assert_eq!(invalid.reasoning_effort, None);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn every_key_reads_none_when_absent_or_null_or_invalid_type() {
        for (key, invalid) in invalid_values() {
            assert_key_none(key, invalid);
        }
    }

    #[test]
    fn max_tokens_reads_positive_u32_and_rejects_zero_and_negative() {
        let p = params_for(MAX_TOKENS, json!(2048));
        assert_eq!(p.max_tokens, Some(2048));
        assert_eq!(params_for(MAX_TOKENS, json!(0)).max_tokens, None);
        assert_eq!(params_for(MAX_TOKENS, json!(-5)).max_tokens, None);
        assert_eq!(params_for(MAX_TOKENS, json!(2.5)).max_tokens, None);
    }

    #[test]
    fn temperature_and_top_p_read_f64() {
        let p = params_for(TEMPERATURE, json!(1.25));
        assert_eq!(p.temperature, Some(1.25));
        assert_eq!(params_for(TEMPERATURE, json!(0)).temperature, Some(0.0));
        let p = params_for(TOP_P, json!(0.9));
        assert_eq!(p.top_p, Some(0.9));
    }

    #[test]
    fn top_k_reads_positive_u32() {
        let p = params_for(TOP_K, json!(40));
        assert_eq!(p.top_k, Some(40));
        assert_eq!(params_for(TOP_K, json!(0)).top_k, None);
    }

    #[test]
    fn stop_reads_single_string_and_string_array() {
        assert_eq!(
            params_for(STOP, json!("end")).stop,
            Some(vec!["end".to_string()])
        );
        let p = params_for(STOP, json!(["a", "b"]));
        assert_eq!(p.stop, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(params_for(STOP, json!(["a", 1])).stop, None);
        assert_eq!(params_for(STOP, json!(42)).stop, None);
    }

    #[test]
    fn seed_reads_u64() {
        let p = params_for(SEED, json!(1234567890));
        assert_eq!(p.seed, Some(1234567890));
        assert_eq!(params_for(SEED, json!(-1)).seed, None);
        assert_eq!(params_for(SEED, json!(0)).seed, Some(0));
    }

    #[test]
    fn user_and_reasoning_effort_read_strings() {
        assert_eq!(
            params_for(USER, json!("user-abc")).user,
            Some("user-abc".to_string())
        );
        let p = params_for(REASONING_EFFORT, json!("high"));
        assert_eq!(p.reasoning_effort, Some("high".to_string()));
    }

    #[test]
    fn non_map_options_read_as_empty() {
        let p = SamplingParams::from_options(&Value::Null);
        assert_eq!(p, SamplingParams::default());
        let p = SamplingParams::from_options(&json!("nope"));
        assert_eq!(p, SamplingParams::default());
    }

    #[test]
    fn remaining_holds_only_unknown_keys() {
        let p = SamplingParams::from_options(&options(&[
            (MAX_TOKENS, json!(100)),
            (TEMPERATURE, json!(0.5)),
            (TOP_P, json!(0.9)),
            (TOP_K, json!(20)),
            (STOP, json!("end")),
            (SEED, json!(7)),
            (USER, json!("u")),
            (REASONING_EFFORT, json!("low")),
            ("frequency_penalty", json!(0.2)),
            ("presence_penalty", json!(-1.0)),
        ]));
        assert_eq!(
            p.remaining,
            json!({"frequency_penalty": 0.2, "presence_penalty": -1.0})
        );
    }

    #[test]
    fn reasoning_enabled_is_presence_based() {
        assert!(!reasoning_enabled(&Value::Null));
        assert!(!reasoning_enabled(&options(&[(TEMPERATURE, json!(0.5))])));
        assert!(reasoning_enabled(&options(&[(
            REASONING_EFFORT,
            json!("high")
        )])));
        assert!(reasoning_enabled(&options(&[(
            REASONING_EFFORT,
            json!("low")
        )])));
    }

    #[test]
    fn inapplicable_for_reasoning_covers_temperature_and_top_p_only() {
        assert!(inapplicable_for_reasoning(TEMPERATURE));
        assert!(inapplicable_for_reasoning(TOP_P));
        assert!(!inapplicable_for_reasoning(MAX_TOKENS));
        assert!(!inapplicable_for_reasoning(TOP_K));
        assert!(!inapplicable_for_reasoning(STOP));
        assert!(!inapplicable_for_reasoning(SEED));
        assert!(!inapplicable_for_reasoning(USER));
        assert!(!inapplicable_for_reasoning(REASONING_EFFORT));
        assert!(!inapplicable_for_reasoning("unknown"));
    }

    #[test]
    fn strip_inapplicable_is_a_no_op_without_reasoning() {
        let opts = options(&[(TEMPERATURE, json!(0.5)), (TOP_P, json!(0.9))]);
        let (cleaned, stripped) = strip_inapplicable(&opts, false);
        assert!(stripped.is_empty());
        assert_eq!(cleaned, opts);
    }

    #[test]
    fn strip_inapplicable_removes_only_reasoning_incompatible_keys() {
        let opts = options(&[
            (TEMPERATURE, json!(0.5)),
            (TOP_P, json!(0.9)),
            (STOP, json!("end")),
            (SEED, json!(7)),
            ("frequency_penalty", json!(0.2)),
        ]);
        let (cleaned, stripped) = strip_inapplicable(&opts, true);
        assert_eq!(stripped, vec![TEMPERATURE, TOP_P]);
        let cleaned = cleaned.as_object().unwrap();
        assert!(!cleaned.contains_key(TEMPERATURE));
        assert!(!cleaned.contains_key(TOP_P));
        assert_eq!(cleaned.get(STOP), Some(&json!("end")));
        assert_eq!(cleaned.get(SEED), Some(&json!(7)));
        assert_eq!(cleaned.get("frequency_penalty"), Some(&json!(0.2)));
    }

    #[test]
    fn strip_inapplicable_ignores_null_values_and_non_map_options() {
        let opts = options(&[(TEMPERATURE, Value::Null)]);
        let (cleaned, stripped) = strip_inapplicable(&opts, true);
        assert!(stripped.is_empty());
        assert_eq!(cleaned, opts);
        let (cleaned, stripped) = strip_inapplicable(&Value::Null, true);
        assert!(stripped.is_empty());
        assert_eq!(cleaned, Value::Null);
    }

    #[test]
    fn from_options_with_applicability_strips_and_types() {
        let opts = options(&[
            (REASONING_EFFORT, json!("high")),
            (TEMPERATURE, json!(0.5)),
            (TOP_P, json!(0.9)),
            (MAX_TOKENS, json!(1024)),
        ]);
        let (params, stripped) = from_options_with_applicability(&opts);
        assert_eq!(stripped, vec![TEMPERATURE, TOP_P]);
        assert_eq!(params.max_tokens, Some(1024));
        assert_eq!(params.temperature, None);
        assert_eq!(params.top_p, None);
        assert_eq!(params.reasoning_effort, Some("high".to_string()));
    }

    #[test]
    fn apply_openai_wire_shape() {
        let params = SamplingParams {
            max_tokens: Some(100),
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: Some(20),
            stop: Some(vec!["a".to_string(), "b".to_string()]),
            seed: Some(7),
            user: Some("u-1".to_string()),
            reasoning_effort: Some("low".to_string()),
            output_schema: None,
            output_schema_name: None,
            remaining: json!({"frequency_penalty": 0.2}),
        };
        let mut body = Map::new();
        body.insert("model".to_string(), json!("m"));
        params.apply_openai(&mut body);
        let body = Value::Object(body);
        assert_eq!(
            body,
            json!({
                "model": "m",
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9,
                "stop": ["a", "b"],
                "seed": 7,
                "user": "u-1",
                "reasoning_effort": "low",
                "frequency_penalty": 0.2,
            })
        );
        assert!(body.get(TOP_K).is_none());
    }

    #[test]
    fn apply_openai_omits_absent_fields() {
        let params = SamplingParams::default();
        let mut body = Map::new();
        body.insert("model".to_string(), json!("m"));
        params.apply_openai(&mut body);
        assert_eq!(body, Map::from_iter([("model".to_string(), json!("m"))]));
    }

    #[test]
    fn apply_anthropic_wire_shape() {
        let params = SamplingParams {
            max_tokens: Some(512),
            temperature: Some(0.3),
            top_p: Some(0.8),
            top_k: Some(20),
            stop: Some(vec!["a".to_string()]),
            seed: Some(7),
            user: Some("u-1".to_string()),
            reasoning_effort: Some("low".to_string()),
            output_schema: None,
            output_schema_name: None,
            remaining: json!({"metadata": {"user_id": "x"}}),
        };
        let mut body = Map::new();
        body.insert("model".to_string(), json!("m"));
        params.apply_anthropic(&mut body);
        let body = Value::Object(body);
        assert_eq!(
            body,
            json!({
                "model": "m",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 20,
                "stop_sequences": ["a"],
                "metadata": {"user_id": "x"},
            })
        );
        assert!(body.get(SEED).is_none());
        assert!(body.get(USER).is_none());
        assert!(body.get(REASONING_EFFORT).is_none());
        assert!(body.get(STOP).is_none());
    }

    #[test]
    fn apply_anthropic_omits_absent_fields() {
        let params = SamplingParams::default();
        let mut body = Map::new();
        body.insert("model".to_string(), json!("m"));
        params.apply_anthropic(&mut body);
        assert_eq!(body, Map::from_iter([("model".to_string(), json!("m"))]));
    }

    #[test]
    fn output_schema_reads_json_value_and_name() {
        let schema = json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let p = SamplingParams::from_options(&options(&[
            (OUTPUT_SCHEMA, schema.clone()),
            (OUTPUT_SCHEMA_NAME, json!("person")),
        ]));
        assert_eq!(p.output_schema, Some(schema));
        assert_eq!(p.output_schema_name.as_deref(), Some("person"));
        assert_eq!(
            SamplingParams::from_options(&options(&[])).output_schema,
            None
        );
        let p = SamplingParams::from_options(&options(&[(OUTPUT_SCHEMA, Value::Null)]));
        assert_eq!(p.output_schema, None);
    }

    #[test]
    fn output_schema_never_leaks_into_remaining() {
        let schema = json!({"type": "object"});
        let p = SamplingParams::from_options(&options(&[
            (OUTPUT_SCHEMA, schema),
            (OUTPUT_SCHEMA_NAME, json!("n")),
        ]));
        assert_eq!(p.remaining, json!({}));
    }

    #[test]
    fn apply_openai_structured_output_wire_shape() {
        let schema = json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let params = SamplingParams {
            output_schema: Some(schema.clone()),
            output_schema_name: Some("person".to_string()),
            ..SamplingParams::default()
        };
        let mut body = Map::new();
        body.insert("model".to_string(), json!("m"));
        params.apply_openai(&mut body);
        assert_eq!(
            body.get("response_format"),
            Some(&json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": schema,
                    "strict": true,
                }
            }))
        );
    }

    #[test]
    fn apply_openai_structured_output_default_name() {
        let schema = json!({"type": "object"});
        let params = SamplingParams {
            output_schema: Some(schema.clone()),
            output_schema_name: None,
            ..SamplingParams::default()
        };
        let mut body = Map::new();
        params.apply_openai(&mut body);
        assert_eq!(
            body.get("response_format")
                .and_then(|v| v.get("json_schema"))
                .and_then(|v| v.get("name")),
            Some(&json!("structured_output"))
        );
    }

    #[test]
    fn apply_anthropic_structured_output_wire_shape() {
        let schema = json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let params = SamplingParams {
            output_schema: Some(schema.clone()),
            ..SamplingParams::default()
        };
        let mut body = Map::new();
        params.apply_anthropic(&mut body);
        assert_eq!(
            body.get("output_config"),
            Some(&json!({
                "format": { "type": "json_schema", "schema": schema }
            }))
        );
    }

    #[test]
    fn apply_responses_structured_output_wire_shape() {
        let schema = json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let params = SamplingParams {
            output_schema: Some(schema.clone()),
            output_schema_name: Some("person".to_string()),
            ..SamplingParams::default()
        };
        let mut body = Map::new();
        params.apply_responses(&mut body);
        assert_eq!(
            body.get("text"),
            Some(&json!({
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person",
                        "schema": schema,
                        "strict": true,
                    }
                }
            }))
        );
    }
}
