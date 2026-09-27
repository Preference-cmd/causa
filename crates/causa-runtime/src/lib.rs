//! Optional execution and session components over the Causa context kernel.
//!
//! The kernel holds facts and contracts. This crate provides reference
//! components that applications can adopt individually or assemble into a
//! turn runner. Tools, storage, approval logic and retention remain host
//! choices.
//!
//! # Components
//!
//! - [`TurnRunner`] drives model rounds, retry scheduling, streaming and
//!   tool batches. [`TurnRunOptions`] configures model invocation, limits,
//!   execution resources, frame construction and streaming interaction.
//! - [`ToolProcessingChain`] runs ordered pre-processors, a fixed executor
//!   stage, then ordered post-processors over one borrowed
//!   [`ToolBatch`](causa_kernel::ToolBatch). Processors implement the kernel's
//!   [`ToolBatchProcessor`](causa_kernel::ToolBatchProcessor) contract.
//!   Approval can await an external decision inside a processor; rejection
//!   resolves that call with a normal result. Arguments may change before
//!   execution, while committed model declarations remain unchanged.
//! - [`ToolExecutor`] dispatches static or dynamic tools, isolates tool
//!   panics, and applies a call-deadline backstop. [`ToolBridge`] also exposes
//!   a dynamic source as a static tool. Neither applies an output budget.
//! - [`FramePolicy`] applies optional token counting and frame-local
//!   compaction. [`TurnInteraction`] receives streaming deltas and supplies
//!   steering input. Neither mechanism mutates previously committed facts.
//! - [`ConversationState`] owns one active turn and completed history
//!   ordered by [`TurnSequence`]. The [`ConversationStore`] port lets hosts
//!   persist current [`HistoryEntry`] values.
//! - [`Session`] coordinates one conversation and runner. Cloneable
//!   [`SessionHandle`]s submit, observe, wait and cancel; request keys
//!   deduplicate retained submissions and cancellations. Admission permits
//!   one active work, with no built-in queue.
//! - [`ContextEvent`] and traces project facts and execution observations
//!   for UI and audit consumers. Hosts decide what to retain.
//!
//! # Batch ownership and interruption
//!
//! The chain awaits each processor before entering the next stage. Only
//! unfinished calls reach the fixed executor, which records results in U as
//! they arrive. Post-processors may edit output, append notes and reorder
//! completed entries. Handoff validation protects declaration membership,
//! pairing and completed identities. Kernel commits preserve the resulting
//! order; no implicit sort or truncation follows the chain.
//!
//! A processing error, cancellation or turn deadline stops further progress.
//! [`TurnOutcome::uncommitted_tool_batch`] and
//! [`ConversationOutcome::uncommitted_tool_batch`] return current U when it
//! has not been committed. Session observations retain it through an `Arc`.
//! Known results remain known; started calls without an observed outcome
//! become `UnknownOutcome`, and calls never started remain pending. The host
//! can inspect, save or discard U. There is no automatic replay or partial
//! commit, and discarding material does not undo external tool effects.
//!
//! Long-running tools can return an ordinary successful observation such as
//! `running` and a tool-owned handle. Later queries are new calls; the runner
//! does not schedule polling or interpret the handle.
//!
//! [`new_block_id`] provides the runtime's UUID v7 convenience generator.
//! The kernel receives explicit IDs. Content-based tool deduplication is a
//! separate opt-in processor policy, not an exactly-once execution guarantee.
//!
//! # Policy surface
//!
//! | Policy | Default and owner |
//! |---|---|
//! | Retry schedule | Disabled; [`RetryPolicy`] configures exponential backoff when enabled |
//! | Turn limits | [`TurnLimits`]: 10 model rounds and 64 tool calls |
//! | Tool processing | Empty pre/post chain; policy processors are opt-in |
//! | Tool dispatch | Parallel unfinished calls; panic becomes `Failed`, no observed deadline outcome becomes `UnknownOutcome` |
//! | Unknown outcomes | [`UnknownOutcomeConfig`] defaults to `Stop`, with effective-tool-name overrides |
//! | Output retention | No default truncation or artifact spill; optional post-processors own budgets and storage |
//! | Frame compaction | None; [`FramePolicy`] accepts a host counter, compactor and thresholds |
//! | Streaming interaction | [`NoopInteraction`] observes no deltas and supplies no inputs |
//! | Session admission | One active work, no queue; request-key replay is local to retained operations |
//! | Session capacity/deadline | [`SessionConfig`]: 256 retained works, no work deadline |
//!
//! A port implementation such as a tool or gateway only needs
//! `causa-kernel`; it need not depend on these reference components.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod budget;
pub mod composition;
pub mod config;
pub mod control;
pub mod conversation;
pub mod driver;
pub mod event;
pub mod executor;
pub mod ids;
pub mod interaction;
pub mod processors;
pub mod session;

// --- reference budget & interaction -----------------------------------------
pub use budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, FrameError, FramePolicy,
    TokenCounter, WindowBudget,
};
pub use interaction::TurnInteraction;

// --- driver stack -----------------------------------------------------------
pub use composition::ToolBridge;
pub use config::{
    ExecutionOptions, NoopInteraction, RetryPolicy, TurnInvocation, TurnLimits, TurnPolicy,
    TurnRunOptions, UnknownOutcomeConfig, UnknownOutcomePolicy,
};
pub use control::RunControl;
pub use conversation::{
    ConversationError, ConversationState, ConversationStore, ConversationStoreError,
    ConversationVersion, HistoryEntry, SealedResult, TurnSequence,
};
pub use driver::{
    AttemptTrace, ConversationOutcome, ModelRoundTrace, OutputSummary, ToolBatchTrace,
    ToolCallTrace, TurnInterruption, TurnOutcome, TurnResult, TurnRunner, TurnTrace,
};
pub use executor::{ToolExecutor, ToolRegistryError};
pub use ids::new_block_id;
pub use processors::{
    DeduplicateProcessor, PassThroughProcessor, RejectAllProcessor, ToolOutputBudgetProcessor,
    ToolProcessingBuilder, ToolProcessingChain, ToolProcessingError,
};
pub use session::{
    CancelOutcome, CancelReceipt, FinishedKind, Session, SessionBuildRejection, SessionConfig,
    SessionError, SessionHandle, SubmitRequest, WaitEnd, WaitOutcome, WorkObservation, WorkReceipt,
    WorkRef, WorkState,
};

// --- framework policies and projections --------------------------------------
pub use event::{
    ContextEvent, ContextEventKind, StreamEventCollector, project_streaming_turn, project_turn,
};
