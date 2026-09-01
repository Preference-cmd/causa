//! Kernel-side tool-use filter adapter — Slice 4 Phase A (选项 B).
//!
//! The user-facing extension trait `ToolUseFilter` and the concrete
//! filters (`DedupFilter`, `AllowAllFilter`, `DenyAllFilter`, `FilterChain`)
//! live in `reimagine-agent-runtime`. This hook is the minimal adapter the
//! driver calls inside the kernel: the kernel ports stay zero new
//! user-facing extension types, but the driver still needs *something* to
//! dispatch a filter on a tool-use batch.
//!
//! ## Why not just import the agent-runtime trait?
//!
//! `context-kernel` is a lower layer than `agent-runtime`; depending on
//! `agent-runtime` from the kernel would invert the layering and create a
//! cycle. The hook is the kernel's own seam; agent-runtime bridges it via
//! `impl ToolUseHook for FilterChain`.
//!
//! ## Field shapes
//!
//! `HookCtx` and `HookOutcome` are structurally identical to
//! agent-runtime's `FilterContext` / `FilterResult`. agent-runtime exposes
//! them as type aliases (`pub use reimagine_context_kernel::HookCtx as
//! FilterContext`) so consumers see one type even though two crates
//! reference it.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::context::block::ToolCallPayload;
use crate::context::ids::{ConversationId, RoundId, TurnId};
use crate::context::tool_data::{ToolOutput, ToolResultPayload, ToolResultStatus};
use crate::ports::control::CallControl;
use crate::ports::tool::ToolExecutionOutcome;

/// Context supplied to every hook invocation.
///
/// `control: &CallControl` exposes the bounded attempt / call cancellation
/// so a filter can `select!` on user-approval responses (Slice 4 §4 B3).
#[derive(Debug)]
pub struct HookCtx<'a> {
    pub turn_id: &'a TurnId,
    pub conversation_id: Option<&'a ConversationId>,
    pub round_id: RoundId,
    pub control: &'a CallControl,
}

/// Result of a hook pass.
///
/// Invariant: `rejected` must reuse the original `payload.call_id` from
/// the input batch; forging a new id would make `append_tool_results`
/// reject the entry with `UnpairedToolResult`. `to_execute` may rewrite
/// `arguments` (open `FilterResult` — approval can rewrite, defer, split).
pub struct HookOutcome {
    pub to_execute: Vec<ToolCallPayload>,
    pub rejected: Vec<ToolExecutionOutcome>,
}

impl HookOutcome {
    pub fn passthrough(calls: Vec<ToolCallPayload>) -> Self {
        Self {
            to_execute: calls,
            rejected: Vec::new(),
        }
    }
}

/// Kernel-side tool-use filter adapter.
///
/// `agent-runtime`'s `FilterChain` implements this trait. The driver
/// calls `hook.apply(calls, ctx).await` between receiving the model's
/// `ToolCallPayload` batch and dispatching it to `ToolExecutor`.
#[async_trait]
pub trait ToolUseHook: Send + Sync {
    async fn apply(&self, calls: Vec<ToolCallPayload>, ctx: &HookCtx<'_>) -> HookOutcome;
}

/// Default kernel-side dedup hook — preserves the historical
/// `TurnRunner::new()` behavior of rejecting same-batch duplicates with
/// identical `(tool_name, arguments)`. agent-runtime's
/// `FilterChain::default() == [DedupFilter]` provides the framework-level
/// equivalent.
///
/// This is the kernel-side adapter for "Slice 1.x — `driver.rs:489`
/// inline `DedupKey` logic". The dedup policy itself is preserved; only
/// its routing changed (private hook struct, not inline logic).
pub struct KernelDedupHook;

#[async_trait]
impl ToolUseHook for KernelDedupHook {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
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
        HookOutcome {
            to_execute,
            rejected,
        }
    }
}
