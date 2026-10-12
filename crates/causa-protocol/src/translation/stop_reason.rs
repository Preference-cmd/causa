//! One stop-value mapping per protocol, shared by the batch parsers and the
//! streaming decoders so the two entry points cannot classify a response
//! differently.
//!
//! A stop value the kernel has no meaning for is reported as what it is:
//! [`ModelStopReason::Unrecognized`]. The output is not committed and the raw
//! value reaches the host through [`ModelOutput::raw_stop_reason`]. The
//! framework neither guesses a meaning nor prescribes a recovery.
//!
//! | wire value | result |
//! |---|---|
//! | Anthropic `end_turn`, `stop_sequence` | [`ModelStopReason::EndTurn`] |
//! | Anthropic `tool_use` | [`ModelStopReason::ToolUse`] |
//! | Anthropic `max_tokens` | [`ModelStopReason::MaxTokens`] |
//! | Anthropic `refusal` | [`ModelStopReason::Refusal`] |
//! | Anthropic `model_context_window_exceeded` | [`ModelStopReason::ContextWindowExceeded`] |
//! | Anthropic `pause_turn`, any other string | [`ModelStopReason::Unrecognized`] |
//! | Chat `stop`, `end` | [`ModelStopReason::EndTurn`] |
//! | Chat `tool_calls`, `function_call` | [`ModelStopReason::ToolUse`] |
//! | Chat `length` | [`ModelStopReason::MaxTokens`] |
//! | Chat `content_filter` | [`ModelStopReason::Refusal`] |
//! | Chat any other string | [`ModelStopReason::Unrecognized`] |
//! | Responses `incomplete` with `max_output_tokens`, `length` | [`ModelStopReason::MaxTokens`] |
//! | Responses `incomplete` with `content_filter` | [`ModelStopReason::Refusal`] |
//! | Responses `incomplete` with any other reason | [`ModelStopReason::Unrecognized`] |
//!
//! `length` is not a Responses value. It is the Chat word for the same stop,
//! and servers that translate a Chat backend into Responses pass it through.
//!
//! The Responses raw value is `status`, or `status.reason` for an
//! `incomplete` status that carries a reason (for example
//! `incomplete.max_output_tokens`).
//!
//! A malformed field is still an error. A missing or non-string stop field is
//! a `Permanent` parse failure; a Responses `status` outside
//! `completed` / `failed` / `incomplete` is `UnknownOutcome`, because the
//! response reports no outcome this adapter can act on. Stop reason and tool calls that
//! disagree are reclassified only on Responses, whose protocol has no
//! tool-call stop value.

use causa_kernel::ModelStopReason;
use serde_json::Value;

/// Map an Anthropic `stop_reason` value.
pub(crate) fn anthropic(value: &str) -> ModelStopReason {
    match value {
        "end_turn" | "stop_sequence" => ModelStopReason::EndTurn,
        "tool_use" => ModelStopReason::ToolUse,
        "max_tokens" => ModelStopReason::MaxTokens,
        "refusal" => ModelStopReason::Refusal,
        "model_context_window_exceeded" => ModelStopReason::ContextWindowExceeded,
        // `pause_turn` and every unknown value: the output is handed over
        // whole and the runner does not continue.
        _ => ModelStopReason::Unrecognized,
    }
}

/// Map an OpenAI Chat Completions `finish_reason` value.
pub(crate) fn chat(value: &str) -> ModelStopReason {
    match value {
        "stop" | "end" => ModelStopReason::EndTurn,
        "tool_calls" | "function_call" => ModelStopReason::ToolUse,
        "length" => ModelStopReason::MaxTokens,
        "content_filter" => ModelStopReason::Refusal,
        _ => ModelStopReason::Unrecognized,
    }
}

/// Map an OpenAI Responses terminal `status` plus its
/// `incomplete_details.reason`, when present.
pub(crate) fn responses(status: &str, incomplete_reason: Option<&str>) -> ModelStopReason {
    match (status, incomplete_reason) {
        ("completed", _) => ModelStopReason::EndTurn,
        ("incomplete", Some("max_output_tokens" | "length")) => ModelStopReason::MaxTokens,
        ("incomplete", Some("content_filter")) => ModelStopReason::Refusal,
        ("incomplete", _) => ModelStopReason::Unrecognized,
        // Callers reject every other status before mapping; this keeps the
        // function total.
        _ => ModelStopReason::Unrecognized,
    }
}

/// The `stop_reason` a batch Anthropic body carries, if it is a string.
/// An absent, null or non-string field yields `None`, and the caller reports
/// it as `Permanent`.
pub(crate) fn anthropic_raw(value: &Value) -> Option<&str> {
    value.get("stop_reason").and_then(Value::as_str)
}

/// The `finish_reason` a batch Chat body carries, if it is a string.
pub(crate) fn chat_raw(choice: &Value) -> Option<&str> {
    choice.get("finish_reason").and_then(Value::as_str)
}

/// The Responses stop value as `pi-ai` builds its `rawStopReason`: the
/// `status`, or `status.reason` when the status is `incomplete` and a reason
/// is present (for example `incomplete.max_output_tokens`).
pub(crate) fn responses_raw(
    value: &Value,
    status: &str,
    incomplete_reason: Option<&str>,
) -> String {
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or(status);
    match (status, incomplete_reason) {
        ("incomplete", Some(reason)) => format!("{status}.{reason}"),
        _ => status.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::protocol::Protocol;

    fn mapped(protocol: Protocol, value: &str) -> ModelStopReason {
        match protocol {
            Protocol::AnthropicMessages => anthropic(value),
            Protocol::OpenAiChatCompletions => chat(value),
            Protocol::OpenAiResponses => responses(value, None),
        }
    }

    #[test]
    fn anthropic_values_map_to_the_table() {
        assert_eq!(anthropic("end_turn"), ModelStopReason::EndTurn);
        assert_eq!(anthropic("stop_sequence"), ModelStopReason::EndTurn);
        assert_eq!(anthropic("tool_use"), ModelStopReason::ToolUse);
        assert_eq!(anthropic("max_tokens"), ModelStopReason::MaxTokens);
        assert_eq!(anthropic("refusal"), ModelStopReason::Refusal);
        assert_eq!(
            anthropic("model_context_window_exceeded"),
            ModelStopReason::ContextWindowExceeded
        );
        assert_eq!(anthropic("pause_turn"), ModelStopReason::Unrecognized);
        assert_eq!(anthropic("something_new"), ModelStopReason::Unrecognized);
    }

    #[test]
    fn chat_values_map_to_the_table() {
        assert_eq!(chat("stop"), ModelStopReason::EndTurn);
        assert_eq!(chat("end"), ModelStopReason::EndTurn);
        assert_eq!(chat("tool_calls"), ModelStopReason::ToolUse);
        assert_eq!(chat("function_call"), ModelStopReason::ToolUse);
        assert_eq!(chat("length"), ModelStopReason::MaxTokens);
        assert_eq!(chat("content_filter"), ModelStopReason::Refusal);
        assert_eq!(chat("something_new"), ModelStopReason::Unrecognized);
    }

    #[test]
    fn responses_values_map_to_the_table() {
        assert_eq!(responses("completed", None), ModelStopReason::EndTurn);
        assert_eq!(
            responses("incomplete", Some("max_output_tokens")),
            ModelStopReason::MaxTokens
        );
        assert_eq!(
            responses("incomplete", Some("length")),
            ModelStopReason::MaxTokens
        );
        assert_eq!(
            responses("incomplete", Some("content_filter")),
            ModelStopReason::Refusal
        );
        assert_eq!(
            responses("incomplete", Some("something_new")),
            ModelStopReason::Unrecognized
        );
        assert_eq!(responses("incomplete", None), ModelStopReason::Unrecognized);
    }

    #[test]
    fn responses_raw_is_the_status_or_the_status_reason_form() {
        // Completed: the status alone.
        assert_eq!(
            responses_raw(&json!({"status": "completed"}), "completed", None),
            "completed"
        );
        // Incomplete with a reason: the dotted form, as `pi-ai` builds it.
        assert_eq!(
            responses_raw(
                &json!({"status": "incomplete"}),
                "incomplete",
                Some("max_output_tokens")
            ),
            "incomplete.max_output_tokens"
        );
        // Incomplete with no reason on the wire: the status alone. Nothing on
        // the wire said "unknown", so the raw value must not invent it.
        assert_eq!(
            responses_raw(&json!({"status": "incomplete"}), "incomplete", None),
            "incomplete"
        );
        // An unknown reason is still reported verbatim.
        assert_eq!(
            responses_raw(
                &json!({"status": "incomplete"}),
                "incomplete",
                Some("brand_new_reason")
            ),
            "incomplete.brand_new_reason"
        );
    }

    #[test]
    fn every_known_stop_value_is_recognized_and_exactly_one_probe_is_not() {
        // The three tables are exhaustive over the values the kernel names.
        let known = [
            (Protocol::AnthropicMessages, "end_turn"),
            (Protocol::AnthropicMessages, "tool_use"),
            (Protocol::AnthropicMessages, "max_tokens"),
            (Protocol::AnthropicMessages, "refusal"),
            (Protocol::AnthropicMessages, "model_context_window_exceeded"),
            (Protocol::OpenAiChatCompletions, "stop"),
            (Protocol::OpenAiChatCompletions, "end"),
            (Protocol::OpenAiChatCompletions, "tool_calls"),
            (Protocol::OpenAiChatCompletions, "function_call"),
            (Protocol::OpenAiChatCompletions, "length"),
            (Protocol::OpenAiChatCompletions, "content_filter"),
        ];
        for (protocol, value) in known {
            assert_ne!(
                mapped(protocol, value),
                ModelStopReason::Unrecognized,
                "{protocol} {value} must be a known value"
            );
        }
        for protocol in [
            Protocol::AnthropicMessages,
            Protocol::OpenAiChatCompletions,
            Protocol::OpenAiResponses,
        ] {
            assert_eq!(
                mapped(protocol, "definitely_not_a_stop_value"),
                ModelStopReason::Unrecognized,
                "{protocol} must not guess a meaning"
            );
        }
    }
}
