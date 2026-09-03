//! Tool-use filter policies — the framework's concrete behavior for the
//! tool-use batch between model output and tool dispatch.
//!
//! ## One trait, one seam
//!
//! This crate defines `ToolUseHook` (the seam `TurnRunner::with_hook`
//! consumes, since Slice 12 alongside the driver itself) and the concrete
//! policies below. Filters implement that trait directly — there is no
//! second extension trait and no alias layer. The default is
//! `PassthroughHook` (no opinion); dedup / kill-switch / approval are
//! framework or host concerns, never kernel facts.

use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;

use crate::hook::{HookCtx, HookOutcome, ToolUseHook};
use reimagine_context_kernel::{
    ToolCallPayload, ToolExecutionOutcome, ToolOutput, ToolResultPayload, ToolResultStatus,
};

/// Default deduplication policy — same-batch `(tool_name, arguments)` dedup.
/// Subsequent occurrences of an identical pair are pushed to `rejected` with
/// the original `call_id` and a `{"error": "duplicate tool call"}` payload.
///
/// This is a *business policy*, not a fact-layer invariant. The kernel
/// pair-validation rules (ToolCallId pairing, `append_tool_results`
/// `call_seq` checks) are independent and remain.
#[derive(Debug, Default, Clone)]
pub struct DedupFilter;

#[async_trait]
impl ToolUseHook for DedupFilter {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        let mut seen: HashSet<(String, serde_json::Value)> = HashSet::new();
        let mut to_execute = Vec::new();
        let mut rejected = Vec::new();
        for payload in calls {
            let key = (payload.tool_name.clone(), payload.arguments.clone());
            if seen.insert(key) {
                to_execute.push(payload);
            } else {
                rejected.push(ToolExecutionOutcome::new(ToolResultPayload {
                    call_id: payload.call_id.clone(),
                    status: ToolResultStatus::Rejected,
                    output: ToolOutput::new(serde_json::json!({"error": "duplicate tool call"})),
                }));
            }
        }
        HookOutcome {
            to_execute,
            rejected,
        }
    }
}

/// A filter that lets every call through unchanged.
///
/// Useful for tests, opt-out scenarios, and as a baseline for chains
/// that only add policy filters.
#[derive(Debug, Default, Clone)]
pub struct AllowAllFilter;

#[async_trait]
impl ToolUseHook for AllowAllFilter {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        HookOutcome::passthrough(calls)
    }
}

/// A filter that rejects every call with a fixed reason.
///
/// Useful for tests, kill-switch scenarios, and as a baseline for
/// "deny by default" chains. The reason is recorded in the rejected
/// outcome's output payload.
#[derive(Debug, Clone)]
pub struct DenyAllFilter {
    /// The fixed reason recorded in every rejected outcome's output
    /// payload.
    pub reason: String,
}

impl DenyAllFilter {
    /// A filter that rejects every call with `reason`.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl Default for DenyAllFilter {
    fn default() -> Self {
        Self {
            reason: "denied by filter".into(),
        }
    }
}

#[async_trait]
impl ToolUseHook for DenyAllFilter {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        let rejected = calls
            .into_iter()
            .map(|payload| {
                ToolExecutionOutcome::new(ToolResultPayload {
                    call_id: payload.call_id.clone(),
                    status: ToolResultStatus::Rejected,
                    output: ToolOutput::new(serde_json::json!({"error": self.reason.clone()})),
                })
            })
            .collect();
        HookOutcome {
            to_execute: Vec::new(),
            rejected,
        }
    }
}

/// Composes multiple filters sequentially.
///
/// `FilterChain::apply` walks `filters` in order, feeding each
/// filter's `to_execute` into the next. `rejected` accumulates across
/// the chain. `FilterChain::default()` is empty (no opinion — callers opt
/// in via `DedupFilter` or `dedup_only()`).
///
/// A chain plugs into the driver through the same trait it is made of:
/// `TurnRunner::with_hook(gateway, executor, Arc::new(chain))`.
#[derive(Clone)]
pub struct FilterChain {
    filters: Vec<Arc<dyn ToolUseHook>>,
}

impl FilterChain {
    /// Build a chain from an explicit filter list. Order matters: the
    /// first filter sees the raw batch, the next sees the previous
    /// filter's `to_execute`, etc.
    pub fn new(filters: Vec<Arc<dyn ToolUseHook>>) -> Self {
        Self { filters }
    }

    /// True if the chain has no filters — in that case `apply` is a
    /// pure passthrough.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Returns the number of filters in the chain.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Convenience: a chain containing only `DedupFilter`.
    pub fn dedup_only() -> Self {
        Self {
            filters: vec![Arc::new(DedupFilter)],
        }
    }

    /// Convenience: an empty chain (pure passthrough). Equivalent to
    /// `FilterChain::new(vec![])` but documents intent.
    pub fn passthrough() -> Self {
        Self {
            filters: Vec::new(),
        }
    }
}

impl Default for FilterChain {
    /// Empty by default — the framework carries no opinion.
    fn default() -> Self {
        Self {
            filters: Vec::new(),
        }
    }
}

impl std::fmt::Debug for FilterChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterChain")
            .field("len", &self.filters.len())
            .finish()
    }
}

#[async_trait]
impl ToolUseHook for FilterChain {
    async fn apply(&self, mut calls: Vec<ToolCallPayload>, ctx: &HookCtx<'_>) -> HookOutcome {
        let mut all_rejected = Vec::new();
        for filter in &self.filters {
            let outcome = filter.apply(std::mem::take(&mut calls), ctx).await;
            all_rejected.extend(outcome.rejected);
            calls = outcome.to_execute;
        }
        HookOutcome {
            to_execute: calls,
            rejected: all_rejected,
        }
    }
}

// --- tests ----------------------------------------------------------------
//
// Unit tests for the filter policies. They live in the same file to keep
// the test surface local; they do not require a full driver stack.

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_context_kernel::{RoundId, ToolCallId, TurnId};
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    fn call(call_id: &str, tool_name: &str, args: serde_json::Value) -> ToolCallPayload {
        ToolCallPayload {
            call_id: ToolCallId(call_id.to_string()),
            tool_name: tool_name.to_string(),
            arguments: args,
        }
    }

    fn dummy_turn_id() -> TurnId {
        TurnId::new("test-turn")
    }

    /// Construct a `HookCtx` bound to a freshly built `CallControl`
    /// via `Box::leak`. The control lives for the test's process —
    /// acceptable for unit tests, never used in production.
    fn make_ctx<'a>(turn_id: &'a TurnId, round_id: RoundId) -> HookCtx<'a> {
        let cancellation = CancellationToken::new();
        let control = Box::leak(Box::new(reimagine_context_kernel::CallControl::new(
            cancellation,
            None,
        )));
        HookCtx {
            turn_id,
            conversation_id: None,
            round_id,
            control,
        }
    }

    #[tokio::test]
    async fn dedup_filter_collapses_same_tool_name_and_arguments() {
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let calls = vec![
            call("a", "echo", json!({"x": 1})),
            call("b", "echo", json!({"x": 1})), // dup
            call("c", "echo", json!({"x": 2})),
            call("d", "other", json!({"x": 1})), // different tool name
        ];
        let outcome = DedupFilter.apply(calls, &ctx).await;
        assert_eq!(outcome.to_execute.len(), 3);
        assert_eq!(outcome.rejected.len(), 1);
        let ids: Vec<&str> = outcome
            .to_execute
            .iter()
            .map(|p| p.call_id.0.as_str())
            .collect();
        assert_eq!(ids, vec!["a", "c", "d"]);
        // Rejected carries the original call_id (b), not a fresh one.
        assert_eq!(outcome.rejected[0].result.call_id.0, "b");
        assert_eq!(
            outcome.rejected[0].result.status,
            ToolResultStatus::Rejected
        );
    }

    #[tokio::test]
    async fn dedup_filter_preserves_call_id_pairing_invariant() {
        // Each rejected outcome must reuse the source payload's call_id;
        // otherwise append_tool_results would reject it with
        // UnpairedToolResult.
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let calls = vec![
            call("id-1", "echo", json!({"k": "v"})),
            call("id-2", "echo", json!({"k": "v"})),
            call("id-3", "echo", json!({"k": "v"})),
        ];
        let outcome = DedupFilter.apply(calls, &ctx).await;
        let rejected_ids: Vec<String> = outcome
            .rejected
            .iter()
            .map(|o| o.result.call_id.0.clone())
            .collect();
        assert_eq!(rejected_ids, vec!["id-2", "id-3"]);
    }

    #[tokio::test]
    async fn allow_all_filter_passes_through() {
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let calls = vec![
            call("a", "echo", json!({"x": 1})),
            call("b", "echo", json!({"x": 1})), // would be deduped, but AllowAll lets it through
        ];
        let outcome = AllowAllFilter.apply(calls, &ctx).await;
        assert_eq!(outcome.to_execute.len(), 2);
        assert!(outcome.rejected.is_empty());
    }

    #[tokio::test]
    async fn deny_all_filter_rejects_every_call() {
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let calls = vec![call("a", "echo", json!({})), call("b", "other", json!({}))];
        let outcome = DenyAllFilter::new("test deny").apply(calls, &ctx).await;
        assert!(outcome.to_execute.is_empty());
        assert_eq!(outcome.rejected.len(), 2);
        assert_eq!(outcome.rejected[0].result.call_id.0, "a");
        assert_eq!(outcome.rejected[1].result.call_id.0, "b");
        assert_eq!(
            outcome.rejected[0].result.status,
            ToolResultStatus::Rejected
        );
        // Reason is recorded in the output payload's content field.
        let json = &outcome.rejected[0].result.output.content;
        assert_eq!(json["error"], "test deny");
    }

    #[tokio::test]
    async fn filter_chain_default_is_empty() {
        let chain = FilterChain::default();
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());
    }

    #[tokio::test]
    async fn filter_chain_passthrough_lets_calls_through() {
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let chain = FilterChain::passthrough();
        assert!(chain.is_empty());
        let calls = vec![
            call("a", "echo", json!({"x": 1})),
            call("b", "echo", json!({"x": 1})),
        ];
        let outcome = chain.apply(calls, &ctx).await;
        assert_eq!(outcome.to_execute.len(), 2);
        assert!(outcome.rejected.is_empty());
    }

    #[tokio::test]
    async fn filter_chain_dedup_then_deny_rejects_everything() {
        // DedupFilter passes both calls (different arguments).
        // DenyAllFilter then rejects them all.
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let chain = FilterChain::new(vec![
            Arc::new(DedupFilter),
            Arc::new(DenyAllFilter::new("kill switch")),
        ]);
        let calls = vec![
            call("a", "echo", json!({"x": 1})),
            call("b", "echo", json!({"x": 2})),
        ];
        let outcome = chain.apply(calls, &ctx).await;
        assert!(outcome.to_execute.is_empty());
        assert_eq!(outcome.rejected.len(), 2);
        let ids: Vec<String> = outcome
            .rejected
            .iter()
            .map(|o| o.result.call_id.0.clone())
            .collect();
        // Both ids must be preserved (call_id pairing invariant).
        assert!(ids.contains(&"a".to_string()));
        assert!(ids.contains(&"b".to_string()));
    }

    #[tokio::test]
    async fn filter_chain_deny_then_dedup_is_pre_filter_pattern() {
        // DenyAllFilter runs first; everything is rejected before DedupFilter
        // even sees it. This is the canonical pattern for "deny by default,
        // permit allow-listed later".
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let chain = FilterChain::new(vec![
            Arc::new(DenyAllFilter::new("block all")),
            Arc::new(DedupFilter),
        ]);
        let calls = vec![
            call("a", "echo", json!({"x": 1})),
            call("b", "echo", json!({"x": 1})),
        ];
        let outcome = chain.apply(calls, &ctx).await;
        assert!(outcome.to_execute.is_empty());
        // Both rejected at the first filter; DedupFilter never sees them.
        assert_eq!(outcome.rejected.len(), 2);
    }

    #[tokio::test]
    async fn filter_chain_is_clone() {
        let chain = FilterChain::new(vec![Arc::new(DedupFilter)]);
        let cloned = chain.clone();
        assert_eq!(cloned.len(), 1);
    }
}
