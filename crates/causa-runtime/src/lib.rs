//! causa-runtime — optional components over the context kernel.
//!
//! The kernel is facts (`context`) + contracts (`ports`) only. This crate
//! is a set of optional, reference components on top of them: adopt the
//! execution stack, take a single piece, or read them as a reference and
//! build your own. Each component documents the policy it bakes in and what
//! adopting it brings along.
//!
//! - **Driver stack** (`driver`, `executor`, `hook`, `config`, `control`):
//!   `TurnRunner` orchestrates turns over the kernel's ports —
//!   retry scheduling, tool batch dispatch, artifact spill, traces, and
//!   run control.
//! - **Budget & interaction components** (`budget`, `interaction`): the
//!   frame-materialization policy (`FramePolicy`, window budget,
//!   compaction, token counting) and the host↔driver `TurnInteraction`
//!   seam — the component's documented opinions, not fact-layer invariants.
//!   A custom host composes the kernel's lossless `TurnContext::frame`
//!   differently.
//! - **Session aggregate** (`conversation`): `ConversationState` (single
//!   active slot, completed-only history, commit-time `TurnSequence` as
//!   `HistoryEntry`), the `SealedResult` stamp, and the `ConversationStore`
//!   archive port. These are component decisions, not fact-layer invariants;
//!   a custom host composes the kernel facts differently.
//! - **Session coordination** (`session`): one [`Session`] owner per
//!   conversation drives accepted work through the assembled [`TurnRunner`]
//!   — cloneable [`SessionHandle`]s submit / observe / wait / resume / cancel,
//!   one active work at a time, per-work `TurnId`s assigned at acceptance and
//!   never reused, local `request_key` dedup, explicit retained-work capacity,
//!   and an observable `Faulted` state. The material-free save loop is
//!   implemented: an idle or paused session exports a versioned
//!   [`SessionCheckpoint`], and `SessionCheckpoint::restore` registers it
//!   against freshly assembled capabilities without calling the model or a
//!   tool. Multi-session coordination and per-work materials are out of
//!   scope.
//! - **Tool-use filters** (`DedupFilter`, `AllowAllFilter`, `DenyAllFilter`,
//!   `FilterChain`) implement this crate's `ToolUseHook` directly — the
//!   trait, its consumer, and its policies share one crate. The default is
//!   `PassthroughHook` (no opinion); composition and policy are opt-in.
//! - **Events** (`ContextEvent` / `project_turn`) project a finished turn's
//!   facts (`TurnContext` + `TurnResult` + `TurnTrace`) into an IPC-ready
//!   event sequence for UI / observability / audit consumers.
//!
//! Implementation-side note: implementing a `ModelGateway` or a `Tool`
//! requires only `causa-kernel`; this crate exists to *use the optional
//! execution components*, not to fill the kernel's ports.
//!
//! # Optional components
//!
//! Everything here is optional, and each piece is offered as a reference:
//! adopt it whole, or read it and build your own. The four responsibilities
//! below are how the crate is organized — not a promise about how finely it
//! can be split. What adopting a piece brings along is noted as cost, not as
//! a contract.
//!
//! 1. **Execution stack** — [`TurnRunner`] plus its config
//!    ([`TurnRunOptions`], policy axes), resume ([`resume_turn`]), hook
//!    seam, the budget ([`FramePolicy`]) and interaction
//!    ([`TurnInteraction`]) components, and the session owner
//!    ([`Session`] / [`SessionHandle`] / [`SessionCheckpoint`]).
//!    Adopting it brings those policy components along; the defaults they
//!    bake in are documented in the policy table below, not fact-layer
//!    invariants.
//! 2. **Tool execution** — [`ToolExecutor::execute_with_limits`] runs one
//!    call (panic isolation, deadline backstop, limit truncation, error
//!    mapping) with no runner and no session; adopting it also brings the
//!    config and budget vocabulary. Its catalog cache and static-name
//!    priority are executor policies, documented at the impl site.
//! 3. **Session aggregate** — [`ConversationState`] owns the single-active
//!    slot, completed-only history, and commit ordering ([`HistoryEntry`]).
//!    A host with different session needs composes kernel facts directly.
//! 4. **Observation** — traces and [`ContextEvent`] are projections of a
//!    finished turn for UI / audit consumers; reading them brings the
//!    execution stack along. The host chooses what to persist; there is no
//!    separate wire model.
//!
//! # Policy surface
//!
//! Every default the driver bakes in is one of two kinds — a swappable
//! policy object, or a documented opinion. No trait seams exist for
//! single-implementation policies; a seam is added only when a second real
//! shape appears.
//!
//! | Policy | Kind | Where |
//! |---|---|---|
//! | Retry schedule (disabled by default; when enabled, 500 ms base, 8 s ceiling, exponential) | config object [`RetryPolicy`] | `config` |
//! | Turn limits (10 model rounds, 64 tool calls by default) | config object [`TurnLimits`] in [`TurnPolicy`] | `config` |
//! | Frame compaction (none by default; host supplies compactor, token counter and thresholds) | config object [`FramePolicy`] | `budget` |
//! | Interaction / approval gate | port object [`NoopInteraction`] default | `config` |
//! | Tool-use filtering | port object [`HookCtx`] / [`ToolUseHook`], default [`PassthroughHook`] | `hook`, `filter` |
//! | Tool-output token estimation fallback (chars/4; frame estimation defaults to zero) | documented opinion, single home | `defaults::placeholder_token_estimate_value` (crate-internal), [`FramePolicy`] |
//! | Output truncation shape (retained head 60% + tail 40% sized to the declared token budget — notice and JSON-string wrapping measured, notice-only floor at tiny budgets; artifact spill; content replaced by a JSON string; `Truncation::Middle` marker) | documented opinion | `executor::ToolExecutor::execute_with_limits` |
//! | Unknown-outcome continuation (default `Stop`, per-executed-name overrides, explicit per-call host decisions) | config object [`UnknownOutcomeConfig`] + checkpoint [`UnknownDecision`] data | `config`, `hook`, `driver` |
//! | Batch semantics (dedup-then-parallel, per-call panic isolation → `Failed`, call-deadline backstop → `UnknownOutcome`) | documented opinion | `executor` |
//! | Session retention (256 works, rejects new work at capacity) and work deadline (none by default) | config object [`SessionConfig`] | `session` |
//! | Session admission (one active work, no queue), completed-only history and checkpoint eligibility | documented component opinions | [`Session`], [`ConversationState`], [`SessionCheckpoint`] |
//!
//! This table summarizes the main execution and session policies. Component
//! and configuration docs describe further defaults and constraints; absence
//! from this table does not make a runtime decision a kernel invariant.

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
pub mod session;

// --- reference budget & interaction -----------------------------------------
pub use budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, FrameError, FramePolicy,
    TokenCounter, WindowBudget,
};
pub use interaction::{BatchDecision, TurnInteraction};

// --- driver stack -----------------------------------------------------------
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
pub use session::{
    CancelOutcome, CancelReceipt, CheckpointPhase, FinishedKind, SESSION_CHECKPOINT_VERSION,
    SavedCancelKey, SavedResumeKey, SavedSubmitKey, SavedWork, Session, SessionBuildRejection,
    SessionCheckpoint, SessionConfig, SessionConfigDescription, SessionError, SessionHandle,
    SessionRestoreRejection, SubmitRequest, WaitEnd, WaitOutcome, WorkObservation, WorkReceipt,
    WorkRef, WorkState,
};

// --- framework policies and projections --------------------------------------
pub use event::{
    ContextEvent, ContextEventKind, StreamEventCollector, project_streaming_turn, project_turn,
};
pub use filter::{AllowAllFilter, DedupFilter, DenyAllFilter, FilterChain};
