//! Kernel-side tool-use filter seam — Slice 4 Phase A (选项 B).
//!
//! `ToolUseHook` is THE extension point for the tool-use batch between
//! model output and tool dispatch: the driver calls it here, and
//! concrete filter policies (`DedupFilter`, `DenyAllFilter`,
//! `FilterChain`) in `reimagine-agent-runtime` implement it directly.
//! There is exactly one trait for this concept across both crates.
//!
//! ## Why the seam lives in the kernel
//!
//! `context-kernel` is a lower layer than `agent-runtime`; depending on
//! `agent-runtime` from the kernel would invert the layering and create a
//! cycle. The kernel owns the trait, agent-runtime owns the policies and
//! re-exports the trait so a custom filter can be written against either
//! path.
//!
//! ## Minimum invariant, not a policy
//!
//! The kernel ships **no opinion** about what a hook should do. The only
//! built-in impl is `PassthroughHook` — a zero-sized type that admits every
//! call unchanged. That is the literal absence of behavior, the
//! minimum the trait requires to be callable.
//!
//! Specific filter policies (dedup by `(tool_name, arguments)`,
//! approval-rewrites-amount, kill-switch denial, etc.) are *not*
//! kernel concerns — they belong to `agent-runtime` or to the host. The
//! driver applies whatever hook the caller plugged via
//! `TurnRunner::with_hook(_, _, your_hook)`; `TurnRunner::new()` defaults
//! to `PassthroughHook`.

use async_trait::async_trait;

use crate::context::block::ToolCallPayload;

// This trait is the tool-use extension seam between the kernel and the
// framework layer. It is not a `crate::ports::` item — a port is a
// host-facing contract with stable API obligations; this seam currently
// serves one adjacent crate and can still move without a port-level
// stability promise.
use crate::context::ids::{ConversationId, RoundId, TurnId};
use crate::ports::control::CallControl;

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
    pub rejected: Vec<crate::ports::tool::ToolExecutionOutcome>,
}

impl HookOutcome {
    pub fn passthrough(calls: Vec<ToolCallPayload>) -> Self {
        Self {
            to_execute: calls,
            rejected: Vec::new(),
        }
    }
}

/// Kernel-side tool-use filter seam.
///
/// Filters in `agent-runtime::filter` (`FilterChain` and friends)
/// implement this trait. The driver calls `hook.apply(calls, ctx).await`
/// between receiving the model's `ToolCallPayload` batch and dispatching
/// it to `ToolExecutor`.
#[async_trait]
pub trait ToolUseHook: Send + Sync {
    async fn apply(&self, calls: Vec<ToolCallPayload>, ctx: &HookCtx<'_>) -> HookOutcome;
}

/// Zero-sized hook that admits every call unchanged — the literal
/// absence of behavior, not a policy.
///
/// `TurnRunner::new()` defaults to this hook: callers who want a filter
/// chain opt in via `TurnRunner::with_hook(_, _, filter_chain)`.
/// Concrete filter policies (dedup, approval, kill-switch) are host
/// concerns and live in `agent-runtime::filter` or beyond.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughHook;

#[async_trait]
impl ToolUseHook for PassthroughHook {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        HookOutcome::passthrough(calls)
    }
}
