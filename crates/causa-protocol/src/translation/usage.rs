//! Shared usage-field parsing helpers for the three protocol renderers.
//!
//! Maps provider-specific wire usage objects onto the shared kernel
//! [`ModelUsage`] representation reporting non-overlapping raw counts:
//! uncached input tokens, total output tokens, prompt cache read tokens,
//! prompt cache write tokens, and reasoning tokens.

use causa_kernel::ModelUsage;
use serde_json::Value;

/// Read OpenAI's prompt-cache hit count from a `usage` object.
///
/// Prefers the Chat Completions name (`prompt_tokens_details.cached_tokens`),
/// falling back to the Responses name (`input_tokens_details.cached_tokens`)
/// for proxies that normalize shapes, and top-level `prompt_cache_hit_tokens`.
/// Returns `None` when absent.
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
        .or_else(|| usage.get("prompt_cache_hit_tokens").and_then(Value::as_u64))
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
/// Wire mapping:
/// - `input_tokens` → `input_tokens` (uncached prompt tokens)
/// - `output_tokens` → `output_tokens`
/// - `cache_read_input_tokens` → `cache_read_tokens`
/// - `cache_creation_input_tokens` → `cache_write_tokens`
/// - `output_tokens_details.thinking_tokens` → `reasoning_tokens`
pub fn model_usage_from_anthropic(usage: &Value) -> ModelUsage {
    let mut counts = ModelUsage::new(
        usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
    );
    counts.cache_read_tokens = usage
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    counts.cache_write_tokens = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    counts.reasoning_tokens = usage
        .get("output_tokens_details")
        .and_then(|d| d.get("thinking_tokens"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    counts
}

/// OpenAI Chat Completions `usage` → `ModelUsage`.
///
/// Wire mapping:
/// - `prompt_tokens - read - write` (saturating) → `input_tokens` (uncached prompt tokens)
/// - `completion_tokens` → `output_tokens`
/// - `prompt_tokens_details.cached_tokens` (or fallback `prompt_cache_hit_tokens`) → `cache_read_tokens`
/// - `prompt_tokens_details.cache_write_tokens` → `cache_write_tokens`
/// - `completion_tokens_details.reasoning_tokens` (or fallback top-level `reasoning_tokens`) → `reasoning_tokens`
pub fn model_usage_from_openai_chat(usage: &Value) -> ModelUsage {
    let raw_prompt = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let output = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let cache_read = openai_cached_tokens(usage).map(|v| v as usize);
    let cache_write = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    let uncached_input = raw_prompt
        .saturating_sub(cache_read.unwrap_or(0))
        .saturating_sub(cache_write.unwrap_or(0));
    let mut counts = ModelUsage::new(uncached_input, output);
    counts.cache_read_tokens = cache_read;
    counts.cache_write_tokens = cache_write;
    counts.reasoning_tokens = openai_reasoning_tokens(usage).map(|v| v as usize);
    counts
}

/// OpenAI Responses `usage` → `ModelUsage`.
///
/// Wire mapping:
/// - `input_tokens - read - write` (saturating) → `input_tokens` (uncached prompt tokens)
/// - `output_tokens` → `output_tokens`
/// - `input_tokens_details.cached_tokens` → `cache_read_tokens`
/// - `input_tokens_details.cache_write_tokens` → `cache_write_tokens`
/// - `output_tokens_details.reasoning_tokens` → `reasoning_tokens`
pub fn model_usage_from_openai_responses(usage: &Value) -> ModelUsage {
    let raw_input = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let output = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let cache_read = usage
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    let cache_write = usage
        .get("input_tokens_details")
        .and_then(|d| d.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    let uncached_input = raw_input
        .saturating_sub(cache_read.unwrap_or(0))
        .saturating_sub(cache_write.unwrap_or(0));
    let mut counts = ModelUsage::new(uncached_input, output);
    counts.cache_read_tokens = cache_read;
    counts.cache_write_tokens = cache_write;
    counts.reasoning_tokens = usage
        .get("output_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    counts
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
    fn cached_tokens_falls_back_to_top_level_prompt_cache_hit_tokens() {
        let usage = json!({ "prompt_cache_hit_tokens": 11 });
        assert_eq!(openai_cached_tokens(&usage), Some(11));
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
        assert_eq!((u.input_tokens, u.output_tokens), (3, 5));
        assert_eq!(u.cache_read_tokens, Some(7));
        assert_eq!(u.reasoning_tokens, Some(3));
        assert_eq!(u.cache_write_tokens, None);
    }

    #[test]
    fn openai_responses_usage_maps_expected_fields() {
        let usage = json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "input_tokens_details": { "cached_tokens": 4 },
            "output_tokens_details": { "reasoning_tokens": 5 },
        });
        let u = model_usage_from_openai_responses(&usage);
        assert_eq!((u.input_tokens, u.output_tokens), (6, 5));
        assert_eq!(u.cache_read_tokens, Some(4));
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

    #[test]
    fn pt1_anthropic_non_stream() {
        let usage = json!({
            "input_tokens": 100,
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": 3,
            "output_tokens": 20,
            "output_tokens_details": { "thinking_tokens": 8 },
        });
        let u = model_usage_from_anthropic(&usage);
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.cache_write_tokens, Some(3));
        assert_eq!(u.output_tokens, 20);
        assert_eq!(u.reasoning_tokens, Some(8));

        let usage_without_details = json!({
            "input_tokens": 100,
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": 3,
            "output_tokens": 20,
        });
        let u2 = model_usage_from_anthropic(&usage_without_details);
        assert_eq!(u2.reasoning_tokens, None);
    }

    #[test]
    fn pt3_chat_uncached_input_tokens() {
        let usage = json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": { "cached_tokens": 11 },
        });
        let u = model_usage_from_openai_chat(&usage);
        assert_eq!(u.input_tokens, 89);
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.cache_write_tokens, None);
        assert_eq!(u.output_tokens, 20);
        assert_ne!(u.input_tokens, 100);
    }

    #[test]
    fn pt4_chat_compat_cache_write_and_top_level_cache_hit() {
        let usage_with_write = json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {
                "cached_tokens": 11,
                "cache_write_tokens": 5,
            },
        });
        let u = model_usage_from_openai_chat(&usage_with_write);
        assert_eq!(u.input_tokens, 84);
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.cache_write_tokens, Some(5));

        let usage_top_level_hit = json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_cache_hit_tokens": 11,
        });
        let u_top = model_usage_from_openai_chat(&usage_top_level_hit);
        assert_eq!(u_top.input_tokens, 89);
        assert_eq!(u_top.cache_read_tokens, Some(11));
        assert_eq!(u_top.cache_write_tokens, None);
    }

    #[test]
    fn pt5_responses_usage_cached_and_write() {
        let usage = json!({
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": { "cached_tokens": 11 },
        });
        let u = model_usage_from_openai_responses(&usage);
        assert_eq!(u.input_tokens, 89);
        assert_eq!(u.cache_read_tokens, Some(11));
        assert_eq!(u.cache_write_tokens, None);

        let usage_with_write = json!({
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {
                "cached_tokens": 11,
                "cache_write_tokens": 5,
            },
        });
        let u2 = model_usage_from_openai_responses(&usage_with_write);
        assert_eq!(u2.input_tokens, 84);
        assert_eq!(u2.cache_read_tokens, Some(11));
        assert_eq!(u2.cache_write_tokens, Some(5));
    }

    #[test]
    fn pt7_usage_edges_no_cache_and_saturating_overflow() {
        // No cache fields -> both None, input equals raw
        let anthropic_raw = model_usage_from_anthropic(&json!({
            "input_tokens": 50,
            "output_tokens": 20,
        }));
        assert_eq!(anthropic_raw.input_tokens, 50);
        assert_eq!(anthropic_raw.cache_read_tokens, None);
        assert_eq!(anthropic_raw.cache_write_tokens, None);

        let chat_raw = model_usage_from_openai_chat(&json!({
            "prompt_tokens": 50,
            "completion_tokens": 20,
        }));
        assert_eq!(chat_raw.input_tokens, 50);
        assert_eq!(chat_raw.cache_read_tokens, None);
        assert_eq!(chat_raw.cache_write_tokens, None);

        let responses_raw = model_usage_from_openai_responses(&json!({
            "input_tokens": 50,
            "output_tokens": 20,
        }));
        assert_eq!(responses_raw.input_tokens, 50);
        assert_eq!(responses_raw.cache_read_tokens, None);
        assert_eq!(responses_raw.cache_write_tokens, None);

        // read+write > total -> input 0, read/write kept, no panic
        let chat_overflow = json!({
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 8,
                "cache_write_tokens": 5,
            },
        });
        let u_chat = model_usage_from_openai_chat(&chat_overflow);
        assert_eq!(u_chat.input_tokens, 0);
        assert_eq!(u_chat.cache_read_tokens, Some(8));
        assert_eq!(u_chat.cache_write_tokens, Some(5));

        let responses_overflow = json!({
            "input_tokens": 10,
            "output_tokens": 5,
            "input_tokens_details": {
                "cached_tokens": 8,
                "cache_write_tokens": 5,
            },
        });
        let u_resp = model_usage_from_openai_responses(&responses_overflow);
        assert_eq!(u_resp.input_tokens, 0);
        assert_eq!(u_resp.cache_read_tokens, Some(8));
        assert_eq!(u_resp.cache_write_tokens, Some(5));
    }
}
