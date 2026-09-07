//! causa-runtime — the framework layer over the context kernel.
//!
//! Since Slice 12 this crate owns the canonical driver stack that was once
//! staged inside the kernel's `internal/` perimeter; the kernel itself is
//! facts (`context`) + contracts (`ports`) only.
//!
//! - **Driver stack** (`driver`, `executor`, `hook`, `config`, `control`):
//!   `TurnRunner` orchestrates turns over the kernel's ports —
//!   retry scheduling, tool batch dispatch, artifact spill, traces, and
//!   run control.
//! - **Reference budget & interaction** (`budget`, `interaction`): the
//!   frame-materialization policy (`FramePolicy`, window budget,
//!   compaction, token counting) and the host↔driver `TurnInteraction`
//!   seam — moved here from the kernel in Slice 13 because they are the
//!   reference harness's opinions, not fact-layer invariants. A custom
//!   harness composes the kernel's lossless `TurnContext::frame`
//!   differently.
//! - **Session aggregate** (`conversation`): `ConversationState` (single
//!   active slot, completed-only history, commit-time `TurnSequence` as
//!   `HistoryEntry`), the `SealedResult` stamp, and the `ConversationStore`
//!   archive port — migrated here from the kernel in Slice 6.5. These are
//!   reference-harness decisions, not fact-layer invariants; a custom
//!   harness composes the kernel facts differently.
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
//! requires only `causa-kernel`; this crate is required to *use the
//! reference driver*, not to fill the kernel's ports.
//!
//! # Reference-harness usage paths (Slice 13)
//!
//! Everything here is optional; the four responsibilities a host can adopt
//! independently:
//!
//! 1. **Reference execution** — [`TurnRunner`] plus its config
//!    ([`TurnRunOptions`], policy axes), resume ([`resume_turn`]), hook
//!    seam, and the migrated reference budget ([`FramePolicy`]) and
//!    interaction ([`TurnInteraction`]) seams. The defaults are this
//!    crate's documented opinions, not fact-layer invariants.
//! 2. **Standalone tool execution** — [`ToolExecutor::execute_with_limits`]
//!    runs one call (panic isolation, deadline backstop, limit truncation,
//!    error mapping) with no runner and no session. Its catalog cache and
//!    static-name priority are executor policies, documented at the impl
//!    site.
//! 3. **Optional reference session** — [`ConversationState`] owns the
//!    single-active slot, completed-only history, and commit ordering
//!    (`HistoryEntry`). A host with different session needs composes
//!    kernel facts directly.
//! 4. **Optional observation** — traces and [`ContextEvent`] are
//!    projections of a finished turn for UI / audit consumers. The host
//!    chooses what to persist; there is no separate wire model.
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
//! | Token estimation fallback (chars/4) | documented opinion, single home | `defaults::placeholder_token_estimate_value` (crate-internal) |
//! | Output truncation shape (retained head 60% + tail 40% sized to the declared token budget — notice and JSON-string wrapping measured, notice-only floor at tiny budgets; artifact spill; content replaced by a JSON string; `Truncation::Middle` marker) | documented opinion | `executor::ToolExecutor::execute_with_limits` |
//! | Unknown-outcome policy (`Stop` unless the trusted tool declares otherwise) | port vocabulary default | kernel `UnknownOutcomePolicy` |
//! | Batch semantics (dedup-then-parallel, per-call panic isolation → `Failed`, call-deadline backstop → `UnknownOutcome`) | documented opinion | `executor` |
//!
//! Anything not in this table is facts, not policy.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod budget;
pub mod composition;
pub mod config;
pub mod control;
pub mod conversation;
mod defaults;
pub mod driver;
pub mod event;
pub mod executor;
pub mod filter;
pub mod hook;
pub mod interaction;
pub mod resume;

// --- reference budget & interaction (moved from the kernel, Slice 13) --------
pub use budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, FrameError, FramePolicy,
    TokenCounter, WindowBudget,
};
pub use interaction::{BatchDecision, TurnInteraction};

// --- driver stack (graduated from context-kernel internal/, Slice 12) --------
pub use composition::ToolBridge;
pub use config::{
    ExecutionOptions, NoopInteraction, RetryPolicy, ToolOutputLimits, TurnInvocation, TurnLimits,
    TurnPolicy, TurnRunOptions, UnknownOutcomeConfig, UnknownOutcomePolicy,
};
pub use control::RunControl;
pub use conversation::{
    ConversationError, ConversationState, ConversationStore, ConversationStoreError,
    ConversationVersion, HistoryEntry, SealedResult, TurnSequence,
};
pub use driver::{
    AttemptTrace, Continuation, ConversationOutcome, ModelRoundTrace, OutputSummary, PausePoint,
    PreparedApproval, ToolBatchTrace, ToolCallTrace, TurnInterruption, TurnOutcome, TurnResult,
    TurnRunner, TurnTrace,
};
pub use executor::{ToolExecutor, ToolRegistryError};
pub use hook::{HookCtx, HookOutcome, PassthroughHook, ToolUseHook, UnknownDecision};
pub use resume::{ResumeRejection, ResumeRequest, resume_turn};

// --- framework policies and projections --------------------------------------
pub use event::{
    ContextEvent, ContextEventKind, StreamEventCollector, project_streaming_turn, project_turn,
};
pub use filter::{AllowAllFilter, DedupFilter, DenyAllFilter, FilterChain};
