//! causa-runtime — the framework layer over the context kernel.
//!
//! Since Slice 12 this crate owns the canonical driver stack that was once
//! staged inside the kernel's `internal/` perimeter; the kernel itself is
//! facts (`context`) + contracts (`ports`) only.
//!
//! - **Driver stack** (`driver`, `executor`, `hook`, `config`, `control`,
//!   `defaults`): `TurnRunner` orchestrates turns over the kernel's ports —
//!   retry scheduling, tool batch dispatch, artifact spill, traces, run
//!   control, and noop port defaults.
//! - **Tool-use filters** (`DedupFilter`, `AllowAllFilter`, `DenyAllFilter`,
//!   `FilterChain`) implement this crate's `ToolUseHook` directly — the
//!   trait, its consumer, and its policies share one crate (the Phase E
//!   re-export split is closed). The default is `PassthroughHook` (no
//!   opinion); composition and policy are opt-in.
//! - **Events** (`ContextEvent` / `project_turn`) project a finished turn's
//!   facts (`TurnContext` + `TurnResult` + `TurnTrace`) into an IPC-ready
//!   event sequence for UI / observability / audit consumers.
//!
//! Implementation-side note: implementing a `ModelGateway` or a `Tool`
//! requires only `causa-kernel`; this crate is required to
//! *drive* turns, not to fill the kernel's ports.
//!
//! # Driver policy surface (7.6)
//!
//! Every default the driver bakes in is one of two kinds — a swappable
//! policy object, or a documented opinion. No trait seams exist for
//! single-implementation policies; a seam is added only when a second real
//! shape appears.
//!
//! | Policy | Kind | Where |
//! |---|---|---|
//! | Retry schedule (500 ms base, 8 s ceiling, exponential) | config object [`RetryPolicy`] | `config` |
//! | Turn limits (rounds, tool calls, deadline) | config object [`TurnPolicy`] | `config` |
//! | Interaction / approval gate | port object [`NoopInteraction`] default | `config` |
//! | Tool-use filtering | port object [`HookCtx`] / [`ToolUseHook`], default [`PassthroughHook`] | `hook`, `filter` |
//! | Token estimation fallback (chars/4) | documented opinion, single home | `defaults::placeholder_token_estimate_value` |
//! | Output truncation shape (retained head 60% + tail 40% sized to the declared token budget — notice and JSON-string wrapping measured, notice-only floor at tiny budgets; artifact spill; content replaced by a JSON string; `Truncation::Middle` marker) | documented opinion | `executor::ToolExecutor::execute_with_limits` |
//! | Unknown-outcome policy (`Stop` unless the trusted tool declares otherwise) | port vocabulary default | kernel `UnknownOutcomePolicy` |
//! | Batch semantics (dedup-then-parallel, per-call panic isolation → `Failed`, call-deadline backstop → `UnknownOutcome`) | documented opinion | `executor` |
//!
//! Anything not in this table is facts, not policy.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod composition;
pub mod config;
pub mod control;
pub mod defaults;
pub mod driver;
pub mod event;
pub mod executor;
pub mod filter;
pub mod hook;
pub mod resume;

// --- driver stack (graduated from context-kernel internal/, Slice 12) --------
pub use composition::ToolBridge;
pub use config::{
    ExecutionOptions, NoopInteraction, RetryPolicy, TurnInvocation, TurnLimits, TurnPolicy,
    TurnRunOptions,
};
pub use control::RunControl;
pub use defaults::{NoopCompaction, NoopTokenCounter};
pub use driver::{
    AttemptTrace, ConversationOutcome, ModelRoundTrace, OutputSummary, PausedReason,
    ToolBatchTrace, ToolCallTrace, TurnInterruption, TurnOutcome, TurnResult, TurnRunner,
    TurnTrace,
};
pub use executor::{ToolExecutor, ToolRegistryError};
pub use hook::{HookCtx, HookOutcome, PassthroughHook, ToolUseHook};
pub use resume::{ResumeRequest, resume_turn};

// --- framework policies and projections --------------------------------------
pub use event::{
    ContextEvent, ContextEventKind, StreamEventCollector, project_streaming_turn, project_turn,
};
pub use filter::{AllowAllFilter, DedupFilter, DenyAllFilter, FilterChain};
