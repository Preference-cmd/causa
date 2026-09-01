//! Allow / deny examples for `FilterChain`.
//!
//! These are the same shapes as the gated test filters in
//! `crates/agent-stack/agent-runtime/src/filter.rs`, copied here as
//! runnable examples. The kernel defaults to `PassthroughHook` and
//! empty `FilterChain`; dedup / kill-switch / approval are host
//! concerns and live here, not in the kernel.

use reimagine_agent_runtime::{FilterChain, FilterContext, FilterResult, ToolUseFilter};
use reimagine_context_kernel::{
    ToolCallPayload, ToolExecutionOutcome, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use std::sync::Arc;

struct AllowAll;
#[async_trait::async_trait]
impl ToolUseFilter for AllowAll {
    async fn filter(&self, calls: Vec<ToolCallPayload>, _ctx: &FilterContext<'_>) -> FilterResult {
        FilterResult::passthrough(calls)
    }
}

struct DenyAll {
    reason: String,
}
#[async_trait::async_trait]
impl ToolUseFilter for DenyAll {
    async fn filter(&self, calls: Vec<ToolCallPayload>, _ctx: &FilterContext<'_>) -> FilterResult {
        let rejected = calls
            .into_iter()
            .map(|p| {
                ToolExecutionOutcome::new(ToolResultPayload {
                    call_id: p.call_id,
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

#[tokio::main]
async fn main() {
    let dedup_only = FilterChain::new(vec![Arc::new(reimagine_agent_runtime::DedupFilter)]);
    let allow = AllowAll;
    let deny = DenyAll {
        reason: "external reviewer".into(),
    };
    // Example: `FilterChain::passthrough()` is the explicit empty chain; an
    // empty `FilterChain::default()` also does nothing by design.
    let _passthrough = FilterChain::passthrough();
    // Compose: dedup -> example policy (no-op here, typed to satisfy the chain).
    let _composed = FilterChain::new(vec![Arc::new(reimagine_agent_runtime::DedupFilter)]);
    // Suppress "unused" in release.
    let _ = (allow, deny, dedup_only);
    println!("allow/deny examples compiled -- framework defaults remain passive.");
}
