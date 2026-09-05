//! Allow / deny examples for `ToolUseHook` filters.
//!
//! The same shapes as the filters in
//! `crates/agent-stack/agent-runtime/src/filter.rs`, written here as
//! runnable examples: a custom policy implements the kernel's
//! `ToolUseHook` directly and joins a `FilterChain`. The kernel defaults
//! to `PassthroughHook`; dedup / kill-switch / approval are host
//! concerns and live here, not in the kernel.

use async_trait::async_trait;
use causa_kernel::{
    ToolCallPayload, ToolExecutionOutcome, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use causa_runtime::{DedupFilter, FilterChain, HookCtx, HookOutcome, ToolUseHook};
use std::sync::Arc;

struct DenyAll {
    reason: String,
}

#[async_trait]
impl ToolUseHook for DenyAll {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        let rejected = calls
            .into_iter()
            .map(|p| {
                ToolExecutionOutcome::new(ToolResultPayload {
                    call_id: p.call_id,
                    status: ToolResultStatus::Rejected,
                    output: ToolOutput::new(serde_json::json!({"error": self.reason.clone()})),
                    media: Vec::new(),
                })
            })
            .collect();
        HookOutcome {
            to_execute: Vec::new(),
            rejected,
        }
    }
}

#[tokio::main]
async fn main() {
    let deny = DenyAll {
        reason: "external reviewer".into(),
    };
    // Compose: dedup first, then the kill switch. An empty
    // `FilterChain::default()` is a pure passthrough by design.
    let chain: FilterChain = FilterChain::new(vec![Arc::new(DedupFilter), Arc::new(deny)]);
    let _ = chain.len();
    println!("allow/deny examples compiled -- framework defaults remain passive.");
}
