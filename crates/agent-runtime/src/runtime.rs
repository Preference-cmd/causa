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

// This wrapper is a convenience for callers that already hold a
// `TurnRunner` or want a ready-wired `FilterChain`. It is not the
// single entry — callers can compose a `TurnRunner` directly via
// `TurnRunner::with_hook(_, _, hook)` without going through
// `AgentRuntime`.

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
/// - `FilterChain::default()` is empty (no filter — the framework
///   carries no opinion). Callers who want dedup wire
///   `FilterChain::dedup_only()` or `FilterChain::new(vec![Arc::new(DedupFilter)])`.
/// - `TurnRunner::new()` defaults to `PassthroughHook` (no filter in
///   the kernel either). Together that means a default-constructed
///   `AgentRuntime` is a pure passthrough — behavior is opt-in.
/// - `FilterChain::passthrough()` is the explicit alias for the empty chain.
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
