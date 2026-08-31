//! Shared usage-field parsing helpers.
//!
//! Cache and reasoning token counts appear in different wire namespaces:
//! Chat Completions reports prompt-cache hits under
//! `prompt_tokens_details.cached_tokens`, while the Responses API uses
//! `input_tokens_details.cached_tokens`. Both are mapped onto the shared
//! `cache_read` slot of the budget convention (see
//! `reimagine_agent_harness::Usage`).

use reimagine_context_kernel::ModelUsage;
use serde_json::Value;

/// Read OpenAI's prompt-cache hit count from a `usage` object.
///
/// Prefers the Chat Completions name (`prompt_tokens_details.cached_tokens`)
/// and falls back to the Responses name (`input_tokens_details.cached_tokens`)
/// for proxies that normalize shapes. Returns `None` when absent.
pub fn openai_cached_tokens(usage: &Value) -> Option<u64> {
    usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            usage
                .get("input_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(Value::as_u64)
        })
}

/// Read OpenAI's reasoning token count from a `usage` object.
///
/// Prefers `completion_tokens_details.reasoning_tokens` (OpenAI), falling
/// back to a top-level `reasoning_tokens` (DeepSeek-style servers).
pub fn openai_reasoning_tokens(usage: &Value) -> Option<u64> {
    usage
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| usage.get("reasoning_tokens").and_then(Value::as_u64))
}

/// Anthropic Messages `usage` → `ModelUsage`.
///
/// Wire keys: `input_tokens` / `output_tokens` / `cache_read_input_tokens`
/// / `cache_creation_input_tokens`. Reasoning tokens are not reported on
/// this wire.
pub fn model_usage_from_anthropic(usage: &Value) -> ModelUsage {
    ModelUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        output_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        cache_read_tokens: usage
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as usize),
        cache_write_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64)
            .map(|v| v as usize),
        reasoning_tokens: None,
    }
}

/// OpenAI Chat Completions `usage` → `ModelUsage`.
///
/// Wire keys: `prompt_tokens` / `completion_tokens` plus
/// `prompt_tokens_details.cached_tokens` and
/// `completion_tokens_details.reasoning_tokens`.
pub fn model_usage_from_openai_chat(usage: &Value) -> ModelUsage {
    ModelUsage {
        input_tokens: usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        output_tokens: usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        cache_read_tokens: openai_cached_tokens(usage).map(|v| v as usize),
        cache_write_tokens: None,
        reasoning_tokens: openai_reasoning_tokens(usage).map(|v| v as usize),
    }
}

/// OpenAI Responses `usage` → `ModelUsage`.
///
/// Wire keys: `input_tokens` / `output_tokens` plus
/// `input_tokens_details.cached_tokens` and
/// `output_tokens_details.reasoning_tokens`.
pub fn model_usage_from_openai_responses(usage: &Value) -> ModelUsage {
    ModelUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        output_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        cache_read_tokens: usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as usize),
        cache_write_tokens: None,
        reasoning_tokens: usage
            .get("output_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as usize),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cached_tokens_reads_chat_completions_name_first() {
        let usage = json!({
            "prompt_tokens_details": { "cached_tokens": 7 },
            "input_tokens_details": { "cached_tokens": 9 },
        });
        assert_eq!(openai_cached_tokens(&usage), Some(7));
    }

    #[test]
    fn cached_tokens_falls_back_to_responses_name() {
        let usage = json!({ "input_tokens_details": { "cached_tokens": 9 } });
        assert_eq!(openai_cached_tokens(&usage), Some(9));
        assert_eq!(openai_cached_tokens(&json!({})), None);
    }

    #[test]
    fn reasoning_tokens_reads_details_then_top_level() {
        let usage = json!({
            "completion_tokens_details": { "reasoning_tokens": 3 },
            "reasoning_tokens": 5,
        });
        assert_eq!(openai_reasoning_tokens(&usage), Some(3));
        assert_eq!(
            openai_reasoning_tokens(&json!({ "reasoning_tokens": 5 })),
            Some(5)
        );
        assert_eq!(openai_reasoning_tokens(&json!({})), None);
    }

    #[test]
    fn anthropic_usage_maps_expected_fields() {
        let usage = json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        });
        let u = model_usage_from_anthropic(&usage);
        assert_eq!((u.input_tokens, u.output_tokens), (10, 5));
        assert_eq!(u.cache_read_tokens, Some(3));
        assert_eq!(u.cache_write_tokens, Some(2));
        assert_eq!(u.reasoning_tokens, None);
    }

    #[test]
    fn openai_chat_usage_maps_expected_fields() {
        let usage = json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": { "cached_tokens": 7 },
            "completion_tokens_details": { "reasoning_tokens": 3 },
        });
        let u = model_usage_from_openai_chat(&usage);
        assert_eq!((u.input_tokens, u.output_tokens), (10, 5));
        assert_eq!(u.cache_read_tokens, Some(7));
        assert_eq!(u.reasoning_tokens, Some(3));
        assert_eq!(u.cache_write_tokens, None);
    }

    #[test]
    fn openai_responses_usage_maps_expected_fields() {
        let usage = json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "input_tokens_details": { "cached_tokens": 11 },
            "output_tokens_details": { "reasoning_tokens": 5 },
        });
        let u = model_usage_from_openai_responses(&usage);
        assert_eq!((u.input_tokens, u.output_tokens), (10, 5));
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.reasoning_tokens, Some(5));
        assert_eq!(u.cache_write_tokens, None);
    }

    #[test]
    fn usage_defaults_when_fields_missing() {
        assert_eq!(model_usage_from_anthropic(&json!({})).input_tokens, 0);
        assert_eq!(model_usage_from_openai_chat(&json!({})).output_tokens, 0);
        assert_eq!(
            model_usage_from_openai_responses(&json!({})).output_tokens,
            0
        );
    }
}
