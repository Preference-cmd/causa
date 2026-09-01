//! Tool-use filter primitives — Slice 4 Phase A (选项 B).
//!
//! `ToolUseFilter` is the framework's user-facing extension point for the
//! tool-use batch between model output and tool dispatch. The kernel
//! exposes only the minimum adapter (`ToolUseHook` in
//! `reimagine_context_kernel`); all filter types live here.
//!
//! ## Layering
//!
//! ```text
//! reimagine-context-kernel   ← ports stay zero new user-facing types
//!   └─ internal::hook::ToolUseHook   ← kernel-side adapter (the seam)
//!            ▲
//!            │ impl ToolUseHook for FilterChain
//!            │
//! reimagine-agent-runtime
//!   ├─ ToolUseFilter trait   ← user-facing extension point
//!   ├─ FilterContext / FilterResult   ← type aliases to kernel types
//!   ├─ DedupFilter, AllowAllFilter, DenyAllFilter
//!   └─ FilterChain
//! ```
//!
//! ## Why aliases
//!
//! `FilterContext` and `FilterResult` are exposed as `pub use` aliases of
//! the kernel's `HookCtx` and `HookOutcome` so consumers see one type name
//! (`FilterContext` / `FilterResult`) while the underlying struct lives
//! in the kernel adapter crate. Field sets are identical.
//!
//! ## Defaults
//!
//! `FilterChain::default() == [DedupFilter]`, matching the proposal
//! §3 "`FilterChain::default() == [DedupFilter]`". A `FilterChain::new()`
//! with explicit filters is allowed (dedup is non-mandatory, see §3 B2).

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use reimagine_context_kernel::{
    ToolCallPayload, ToolExecutionOutcome, ToolOutput, ToolResultPayload, ToolResultStatus,
    ToolUseHook,
};

// Type aliases for the user-facing extension surface. They re-export
// the kernel-side adapter types under framework names so consumers
// see `FilterContext` / `FilterResult` regardless of underlying crate.
pub use reimagine_context_kernel::HookCtx as FilterContext;
pub use reimagine_context_kernel::HookOutcome as FilterResult;

/// User-facing extension point for the tool-use batch.
///
/// Filters compose sequentially: the next filter's `to_execute` is the
/// previous filter's `to_execute`. `rejected` accumulates across the
/// chain. Open `FilterResult` semantics (Slice 4 §4): filters may
/// rewrite `arguments` on `to_execute`, defer by pushing to `rejected`,
/// or split / merge calls — provided `rejected` always reuses the
/// original `payload.call_id` (forging a new id would make
/// `append_tool_results` reject the entry with `UnpairedToolResult`).
#[async_trait]
pub trait ToolUseFilter: Send + Sync {
    async fn filter(&self, calls: Vec<ToolCallPayload>, ctx: &FilterContext<'_>) -> FilterResult;
}

/// Default deduplication filter — same-batch `(tool_name, arguments)`
/// dedup. Subsequent occurrences of an identical `(tool_name, arguments)`
/// pair are pushed to `rejected` with the original `call_id` and a
/// `{"error": "duplicate tool call"}` payload.
///
/// This is a *business policy*, not a fact-layer invariant. The kernel
/// pair-validation rules (ToolCallId pairing, `append_tool_results`
/// `call_seq` checks) are independent and remain.
#[derive(Debug, Default, Clone)]
pub struct DedupFilter;

#[async_trait]
impl ToolUseFilter for DedupFilter {
    async fn filter(&self, calls: Vec<ToolCallPayload>, _ctx: &FilterContext<'_>) -> FilterResult {
        let mut seen: HashMap<(String, serde_json::Value), ()> = HashMap::new();
        let mut to_execute = Vec::new();
        let mut rejected = Vec::new();
        for payload in calls {
            let key = (payload.tool_name.clone(), payload.arguments.clone());
            if seen.insert(key, ()).is_some() {
                rejected.push(ToolExecutionOutcome::new(ToolResultPayload {
                    call_id: payload.call_id.clone(),
                    status: ToolResultStatus::Rejected,
                    output: ToolOutput::new(serde_json::json!({"error": "duplicate tool call"})),
                }));
            } else {
                to_execute.push(payload);
            }
        }
        FilterResult {
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
impl ToolUseFilter for AllowAllFilter {
    async fn filter(&self, calls: Vec<ToolCallPayload>, _ctx: &FilterContext<'_>) -> FilterResult {
        FilterResult::passthrough(calls)
    }
}

/// A filter that rejects every call with a fixed reason.
///
/// Useful for tests, kill-switch scenarios, and as a baseline for
/// "deny by default" chains. The reason is recorded in the rejected
/// outcome's output payload.
#[derive(Debug, Clone)]
pub struct DenyAllFilter {
    pub reason: String,
}

impl DenyAllFilter {
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
impl ToolUseFilter for DenyAllFilter {
    async fn filter(&self, calls: Vec<ToolCallPayload>, _ctx: &FilterContext<'_>) -> FilterResult {
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
        FilterResult {
            to_execute: Vec::new(),
            rejected,
        }
    }
}

/// Composes multiple filters sequentially.
///
/// `FilterChain::filter` walks `filters` in order, feeding each
/// filter's `to_execute` into the next. `rejected` accumulates across
/// the chain. `FilterChain::default() == [DedupFilter]`.
///
/// ## Bridge to the kernel
///
/// `FilterChain` also implements the kernel-side `ToolUseHook` trait,
/// so it can plug into `TurnRunner::with_hook` directly. This is the
/// bridge that lets the framework own the user-facing types without
/// the kernel depending on `agent-runtime`.
#[derive(Clone)]
pub struct FilterChain {
    filters: Vec<Arc<dyn ToolUseFilter>>,
}

impl FilterChain {
    /// Build a chain from an explicit filter list. Order matters: the
    /// first filter sees the raw batch, the next sees the previous
    /// filter's `to_execute`, etc.
    pub fn new(filters: Vec<Arc<dyn ToolUseFilter>>) -> Self {
        Self { filters }
    }

    /// True if the chain has no filters — in that case `filter` is a
    /// pure passthrough. `AgentRuntime` checks this to avoid plugging
    /// in a no-op `FilterChain` if the consumer opted out of filtering.
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
    fn default() -> Self {
        Self {
            filters: vec![Arc::new(DedupFilter)],
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
impl ToolUseFilter for FilterChain {
    async fn filter(
        &self,
        mut calls: Vec<ToolCallPayload>,
        ctx: &FilterContext<'_>,
    ) -> FilterResult {
        let mut all_rejected = Vec::new();
        for filter in &self.filters {
            let outcome = filter.filter(std::mem::take(&mut calls), ctx).await;
            all_rejected.extend(outcome.rejected);
            calls = outcome.to_execute;
        }
        FilterResult {
            to_execute: calls,
            rejected: all_rejected,
        }
    }
}

/// Bridge: `FilterChain` is a `ToolUseHook`. This lets
/// `AgentRuntime::run_turn` call `TurnRunner::with_hook(filter_chain)`
/// directly without the kernel needing to know about agent-runtime.
#[async_trait]
impl ToolUseHook for FilterChain {
    async fn apply(
        &self,
        calls: Vec<ToolCallPayload>,
        ctx: &reimagine_context_kernel::HookCtx<'_>,
    ) -> reimagine_context_kernel::HookOutcome {
        // `FilterContext = HookCtx` and `FilterResult = HookOutcome`
        // (type aliases), so we can pass through directly.
        <Self as ToolUseFilter>::filter(self, calls, ctx).await
    }
}

// --- tests ----------------------------------------------------------------
//
// Unit tests for the filter primitives. They live in the same file to keep
// the test surface local; they do not require a full driver stack.

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_context_kernel::{
        HookCtx, RoundId, ToolCallId, ToolExecutionOutcome, ToolResultPayload, ToolResultStatus,
        TurnId,
    };
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
        let outcome = DedupFilter.filter(calls, &ctx).await;
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
        let outcome = DedupFilter.filter(calls, &ctx).await;
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
        let outcome = AllowAllFilter.filter(calls, &ctx).await;
        assert_eq!(outcome.to_execute.len(), 2);
        assert!(outcome.rejected.is_empty());
    }

    #[tokio::test]
    async fn deny_all_filter_rejects_every_call() {
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let calls = vec![call("a", "echo", json!({})), call("b", "other", json!({}))];
        let outcome = DenyAllFilter::new("test deny").filter(calls, &ctx).await;
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
    async fn filter_chain_default_is_dedup_only() {
        let chain = FilterChain::default();
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
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
        let outcome = chain.filter(calls, &ctx).await;
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
        let outcome = chain.filter(calls, &ctx).await;
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
        let outcome = chain.filter(calls, &ctx).await;
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

    #[tokio::test]
    async fn filter_chain_implements_tool_use_hook_bridge() {
        // FilterChain implements both ToolUseFilter and ToolUseHook.
        // This is the bridge that lets AgentRuntime plug a chain into
        // TurnRunner::with_hook. Verifying via direct call:
        let chain = FilterChain::default();
        let turn_id = dummy_turn_id();
        let ctx = make_ctx(&turn_id, RoundId(0));
        let calls = vec![
            call("a", "echo", json!({"x": 1})),
            call("b", "echo", json!({"x": 1})),
        ];
        let outcome: reimagine_context_kernel::HookOutcome =
            <FilterChain as ToolUseHook>::apply(&chain, calls, &ctx).await;
        assert_eq!(outcome.to_execute.len(), 1);
        assert_eq!(outcome.rejected.len(), 1);
        // Same as direct ToolUseFilter::filter call would produce.
    }

    // Suppress unused-import warnings if a test is disabled.
    #[allow(dead_code)]
    fn _ensure_tool_payload_types_used(_o: &ToolExecutionOutcome, _p: &ToolResultPayload) {}
}
