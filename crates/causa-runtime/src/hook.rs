//! The tool-use filter seam (Slice 4 Phase A, graduated from the kernel's
//! staged perimeter by Slice 12).
//!
//! `ToolUseHook` is THE extension point for the tool-use batch between
//! model output and tool dispatch: `TurnRunner` calls it here, and the
//! concrete filter policies (`DedupFilter`, `DenyAllFilter`,
//! `FilterChain`) in this crate implement it directly. Defining the trait
//! next to both its consumer (the driver) and its policies (the filters)
//! closes the Phase E split where `agent-runtime` re-exported a kernel
//! type it was the sole real consumer of.
//!
//! ## Minimum invariant, not a policy
//!
//! This crate ships **no opinion** in the trait: the only built-in impl is
//! `PassthroughHook` — a zero-sized type that admits every call unchanged.
//! That is the literal absence of behavior, the minimum the trait requires
//! to be callable.
//!
//! Specific filter policies (dedup by `(tool_name, arguments)`,
//! approval-rewrites-amount, kill-switch denial, etc.) are host concerns;
//! the driver applies whatever hook the caller plugged via
//! `TurnRunner::with_hook(_, _, your_hook)`; `TurnRunner::new()` defaults
//! to `PassthroughHook`.

use async_trait::async_trait;

use causa_kernel::{CallControl, ConversationId, RoundId, ToolCallPayload, TurnId};

// This trait is deliberately NOT a `causa_kernel::ports` item:
// a port there is a host-facing contract third parties implement against
// the facts crate alone. The hook's sole consumer is this crate's driver,
// so the contract lives with the driver.

/// Context supplied to every hook invocation.
///
/// `control: &CallControl` exposes the bounded attempt / call cancellation
/// so a filter can `select!` on user-approval responses (Slice 4 §4 B3).
#[derive(Debug)]
pub struct HookCtx<'a> {
    /// The turn whose batch is being filtered.
    pub turn_id: &'a TurnId,
    /// `Some` on the conversation entries, `None` for bare turns.
    pub conversation_id: Option<&'a ConversationId>,
    /// The round that emitted the batch.
    pub round_id: RoundId,
    /// Attempt/call-scoped control for `select!`-ing on approval
    /// responses or deadlines.
    pub control: &'a CallControl,
}

/// Result of a hook pass.
///
/// Invariant: `rejected` must reuse the original `payload.call_id` from
/// the input batch; forging a new id would make `append_tool_results`
/// reject the entry with `UnpairedToolResult`. `to_execute` may rewrite
/// `arguments` (open `FilterResult` — approval can rewrite, defer, split).
pub struct HookOutcome {
    /// Calls that pass the hook and reach the executor (arguments may
    /// have been rewritten).
    pub to_execute: Vec<ToolCallPayload>,
    /// Calls the hook rejected; each outcome must reuse the input's
    /// `call_id`.
    pub rejected: Vec<causa_kernel::ToolExecutionOutcome>,
}

impl HookOutcome {
    /// Admit every call unchanged — no rejections.
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
    /// Filter the model-emitted batch between receipt and dispatch:
    /// return what to execute and what to reject.
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
