//! `AgentRuntime` — Slice 4 §3 outer wrapper that wires a
//! `TurnRunner` together with a `FilterChain`.
//!
//! Per proposal §3 (选项 B): the wrapper applies `filters` between the
//! model's tool-use batch and the executor's dispatch. The driver
//! calls the kernel-side `ToolUseHook` adapter, and `FilterChain`
//! implements that adapter (see `filter.rs`).
//!
//! This module owns *only* the wrapper. Session lifecycle (create /
//! list / get), persistence, UI event bus, and WorkspaceHost
//! assembly are out of scope — they are the consumer's job (Slice 4B).

use std::sync::Arc;

use reimagine_context_kernel::{
    ConversationError, ConversationOutcome, ConversationState, ModelGateway, RunControl,
    ToolExecutor, ToolUseHook, TurnContext, TurnOutcome, TurnRunner,
};

use crate::filter::FilterChain;

/// The framework's outer wiring for an agent turn loop.
///
/// Holds:
/// - `runner`: a `TurnRunner` from `reimagine-context-kernel`, the
///   canonical state machine.
/// - `filters`: a `FilterChain` whose `impl ToolUseHook` is plugged
///   into the runner via `TurnRunner::with_hook`.
///
/// `AgentRuntime::from_parts(gateway, executor, filters)` is the
/// canonical entry point: it builds a `TurnRunner` with the chain as
/// its hook. `AgentRuntime` then owns the runner; callers go through
/// `run_turn` / `run_in_conversation` instead of touching it directly.
///
/// ## Defaults
///
/// - `FilterChain::default()` is `[DedupFilter]`.
/// - An empty `FilterChain` is allowed and acts as pure passthrough
///   (use `FilterChain::passthrough()` to opt out).
/// - `KernelDedupHook` is the historical default for `TurnRunner::new`;
///   when constructing via `from_parts` with `FilterChain::default()`,
///   the kernel-side default is replaced by `[DedupFilter]` (same
///   effective behavior — the framework's `DedupFilter` is the
///   user-facing twin of the kernel's private `KernelDedupHook`).
pub struct AgentRuntime {
    runner: TurnRunner,
    filters: FilterChain,
}

impl AgentRuntime {
    /// Canonical constructor — caller provides gateway, executor, and
    /// the filter chain. The wrapper builds a `TurnRunner` with the
    /// chain as its hook.
    ///
    /// If `filters` is empty (`FilterChain::passthrough()` or
    /// `FilterChain::new(vec![])`), this still re-binds the runner
    /// — the new hook will pass every call through unchanged.
    ///
    /// This is the recommended entry point.
    pub fn from_parts(
        gateway: Arc<dyn ModelGateway>,
        executor: Arc<ToolExecutor>,
        filters: FilterChain,
    ) -> Self {
        let hook: Arc<dyn ToolUseHook> = Arc::new(filters.clone());
        let runner = TurnRunner::with_hook(gateway, executor, hook);
        Self { runner, filters }
    }

    /// Build from an existing `TurnRunner` plus a filter chain. The
    /// runner is consumed and rebuilt internally so the chain takes
    /// over as the hook. To avoid re-exposing the runner's internal
    /// `gateway` / `executor` fields, prefer `from_parts` when you
    /// already have gateway + executor on hand.
    pub fn from_runner(runner: TurnRunner, filters: FilterChain) -> Self {
        // We replace the hook by re-binding through the kernel's
        // internal hook slot. Since `TurnRunner` does not expose
        // `gateway` / `executor`, we cannot re-bind in place without
        // an accessor. As a transitional pattern, we ask callers to
        // use `from_parts` (the documented canonical entry point).
        //
        // For now, this method is reserved for future use once the
        // kernel exposes accessor methods on `TurnRunner`. See
        // `from_parts` for the supported entry.
        let _ = runner;
        let _ = filters;
        unimplemented!(
            "TurnRunner does not yet expose gateway/executor accessors; \
             use AgentRuntime::from_parts(gateway, executor, filters)"
        )
    }

    /// Borrow the inner `TurnRunner`. Useful for tests / debug
    /// inspection; the runner is not re-exposed for mutation.
    pub fn runner(&self) -> &TurnRunner {
        &self.runner
    }

    /// Borrow the filter chain.
    pub fn filters(&self) -> &FilterChain {
        &self.filters
    }

    /// Run a turn (Slice 1 entry shape). Equivalent to
    /// `TurnRunner::run`, but routes the tool-use batch through
    /// `self.filters` first.
    pub async fn run_turn(
        &self,
        context: TurnContext,
        options: reimagine_context_kernel::TurnRunOptions,
        ctrl: RunControl,
    ) -> TurnOutcome {
        self.runner.run(context, options, ctrl).await
    }

    /// Run a turn within a `ConversationState` (Slice 2 entry).
    pub async fn run_in_conversation(
        &self,
        state: ConversationState,
        options: reimagine_context_kernel::TurnRunOptions,
        ctrl: RunControl,
    ) -> Result<ConversationOutcome, ConversationError> {
        self.runner.run_in_conversation(state, options, ctrl).await
    }
}

impl std::fmt::Debug for AgentRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentRuntime")
            .field("filters", &self.filters)
            .finish_non_exhaustive()
    }
}
