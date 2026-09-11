//! The tool-use filter seam.
//!
//! `ToolUseHook` is THE extension point for the tool-use batch between
//! model output and tool dispatch: `TurnRunner` calls it here, and the
//! concrete filter policies (`DedupFilter`, `DenyAllFilter`,
//! `FilterChain`) in this crate implement it directly. Defining the trait
//! next to both its consumer (the driver) and its policies (the filters)
//! keeps the seam, its only consumer, and its implementations together.
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

use crate::config::UnknownOutcomePolicy;
use causa_kernel::ToolResultPayload;
use causa_kernel::{CallControl, ConversationId, RoundId, ToolCallId, ToolCallPayload, TurnId};

// This trait is deliberately NOT a `causa_kernel::ports` item:
// a port there is a host-facing contract third parties implement against
// the facts crate alone. The hook's sole consumer is this crate's driver,
// so the contract lives with the driver.

/// One explicit per-call unknown-outcome decision —
/// host decision / checkpoint data attached to a precomputed result whose
/// status is `UnknownOutcome`, not a result envelope: the recorded result
/// is committed verbatim either way. Where a decision set is a *new* host
/// input (hook, `BatchDecision::Reject`, resume), entries may be omitted —
/// the missing ones resolve through [`UnknownOutcomeConfig`](crate::config::UnknownOutcomeConfig)
/// by the executed tool name. Where it is a checkpoint
/// ([`PreparedApproval`](crate::driver::PreparedApproval)), it must
/// exactly cover the saved `UnknownOutcome` results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownDecision {
    /// The precomputed result's call id — must be one of the same
    /// decision's rejected results, and its status must be
    /// `UnknownOutcome`.
    pub call_id: ToolCallId,
    /// The fixed action for that result.
    pub policy: UnknownOutcomePolicy,
}

/// Context supplied to every hook invocation.
///
/// `control: &CallControl` exposes the bounded attempt / call cancellation
/// so a filter can `select!` on user-approval responses.
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
///
/// `rejected` carries the recorded results only. A host precomputing an
/// `UnknownOutcome` result may pin its continuation action in
/// `unknown_decisions`; entries may be omitted and the driver then resolves
/// through its unknown-outcome configuration by the executed tool name.
/// Rejections with any other status take no action and need no entry.
pub struct HookOutcome {
    /// Calls that pass the hook and reach the executor (arguments may
    /// have been rewritten).
    pub to_execute: Vec<ToolCallPayload>,
    /// Calls the hook rejected; each result must reuse the input's
    /// `call_id`.
    pub rejected: Vec<ToolResultPayload>,
    /// Explicit unknown-outcome actions for precomputed results whose
    /// status is `UnknownOutcome`. Optional per entry; must not point at
    /// `to_execute`, foreign ids, or results with any other status.
    pub unknown_decisions: Vec<UnknownDecision>,
}

impl HookOutcome {
    /// Admit every call unchanged — no rejections.
    pub fn passthrough(calls: Vec<ToolCallPayload>) -> Self {
        Self {
            to_execute: calls,
            rejected: Vec::new(),
            unknown_decisions: Vec::new(),
        }
    }

    /// Builder: pin an explicit unknown-outcome action for one precomputed
    /// result (its `call_id`, which must appear in `rejected` with status
    /// `UnknownOutcome`).
    pub fn with_unknown_decision(
        mut self,
        call_id: ToolCallId,
        policy: UnknownOutcomePolicy,
    ) -> Self {
        self.unknown_decisions
            .push(UnknownDecision { call_id, policy });
        self
    }
}

/// The reference harness's tool-use filter seam.
///
/// Filters in `causa_runtime::filter` (`FilterChain` and friends)
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
/// concerns and live in `causa_runtime::filter` or beyond.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughHook;

#[async_trait]
impl ToolUseHook for PassthroughHook {
    async fn apply(&self, calls: Vec<ToolCallPayload>, _ctx: &HookCtx<'_>) -> HookOutcome {
        HookOutcome::passthrough(calls)
    }
}
