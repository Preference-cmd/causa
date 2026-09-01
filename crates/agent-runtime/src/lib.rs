//! reimagine-agent-runtime — the framework layer over the context kernel.
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
//! requires only `reimagine-context-kernel`; this crate is required to
//! *drive* turns, not to fill the kernel's ports.

#![deny(unsafe_code)]

pub mod config;
pub mod control;
pub mod defaults;
pub mod driver;
pub mod event;
pub mod executor;
pub mod filter;
pub mod hook;

// --- driver stack (graduated from context-kernel internal/, Slice 12) --------
pub use config::{
    ExecutionOptions, NoopInteraction, RetryPolicy, TurnInvocation, TurnLimits, TurnPolicy,
    TurnRunOptions,
};
pub use control::RunControl;
pub use defaults::{NoopCompaction, NoopTokenCounter};
pub use driver::{
    AttemptTrace, ConversationOutcome, ModelRoundTrace, OutputSummary, ToolBatchTrace,
    ToolCallTrace, TurnInterruption, TurnOutcome, TurnResult, TurnRunner, TurnTrace,
};
pub use executor::ToolExecutor;
pub use hook::{HookCtx, HookOutcome, PassthroughHook, ToolUseHook};

// --- framework policies and projections --------------------------------------
pub use event::{
    ContextEvent, ContextEventKind, StreamEventCollector, project_streaming_turn, project_turn,
};
pub use filter::{AllowAllFilter, DedupFilter, DenyAllFilter, FilterChain};
