//! Shared usage-field parsing helpers.
//!
//! Cache and reasoning token counts appear in different wire namespaces:
//! Chat Completions reports prompt-cache hits under
//! `prompt_tokens_details.cached_tokens`, while the Responses API uses
//! `input_tokens_details.cached_tokens`. Both are mapped onto the shared
//! `cache_read` slot of the budget convention (see
//! `reimagine_agent_harness::Usage`).

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
        assert_eq!(openai_reasoning_tokens(&json!({ "reasoning_tokens": 5 })), Some(5));
        assert_eq!(openai_reasoning_tokens(&json!({})), None);
    }
}
