//! The driver's internal fallback opinions. Private module: the noop port
//! defaults were removed (a zero counter is behaviorally distinct from no
//! counter — see `ExecutionOptions` docs), leaving only the token-estimate
//! fallback the executor needs.
//!
//! # Driver policy surface
//!
//! Every default the driver bakes in is either a config/policy object
//! (`RetryPolicy`, `TurnPolicy`, `HookCtx`, `TokenCounter`,
//! `UnknownOutcomePolicy`, `ToolOutputLimits`) or documented at the impl
//! site as the driver's opinion. The fallback below is the single home of
//! the chars/4 heuristic — no other copy exists.

/// The driver's token-estimate fallback opinion: serialized JSON length
/// divided by 4.
///
/// NOT the kernel's policy. `ToolExecutor` falls back to this when no
/// `TokenCounter` is wired; hosts that don't yet have a real tokenizer can
/// wire their own `TokenCounter` with the same logic. This function is the
/// heuristic's single home in library code — do not inline a copy
/// elsewhere.
pub(crate) fn placeholder_token_estimate_value(value: &serde_json::Value) -> usize {
    serde_json::to_string(value)
        .map(|s| s.len() / 4)
        .unwrap_or(0)
}
