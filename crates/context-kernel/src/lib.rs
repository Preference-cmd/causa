//! reimagine-context-kernel — ContextBlock conversation kernel.
//! No dependency on reimagine-core / app-host / Tauri / agent-harness.
//!
//! # Layering (Slice 1.5)
//!
//! - **Canonical facts** — block/turn/ids/model values: the recorded turn
//!   state, its validated transitions, and deterministic projections.
//! - **Public ports** — `ModelGateway`, `Tool` + `ArtifactStore`, control
//!   planes (`RunControl`/`AttemptControl`/`CallControl`), budget seams
//!   (`WindowBudget`/`Compaction`/`TokenCounter`/`FramePolicy`). External
//!   implementors supply the behavior; the crate root is the only advertised
//!   surface (physical modules are private by design).
//! - **Staged perimeter** — the reference driver, config axes, executor,
//!   fakes, and noop defaults under `internal` (root-exported deliberately,
//!   but holding no claim on the kernel contract).
//!
//! Everything below re-exports exactly what the public facade promises;
//! nothing else is a cross-crate commitment.

#![deny(unsafe_code)]

mod block;
mod budget;
mod control;
mod gateway;
mod ids;
mod internal;
mod model;
mod tool;
mod tool_data;
mod turn;

// --- canonical facts -------------------------------------------------------
pub use block::{
    BlockMeta, BlockPayload, ContextBlock, ContextInjectPayload, InputPayload, TextPayload,
    ToolCallPayload,
};
pub use ids::{
    AttemptNumber, BlockId, BlockSequence, ContextVersion, ConversationId, ConversationVersion,
    FrameId, InvocationId, RoundId, TurnId, TurnSequence,
};
pub use model::{
    AssistantPayload, GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ModelOutput,
    ModelRef, ModelStopReason, ModelUsage, ReasoningPayload, ToolCallDraft, ToolSurface,
};
pub use tool_data::{
    ArtifactKind, ArtifactRef, ToolCallContext, ToolCallId, ToolDefinition, ToolExecutionOutcome,
    ToolOutput, ToolOutputLimits, ToolOutputMeta, ToolResultPayload, ToolResultStatus, Truncation,
    UnknownOutcomePolicy,
};
pub use turn::{
    AppliedModelOutput, Compaction, CompactionError, CompactionInput, CompactionOutput,
    ContextError, ContextFrame, FrameError, FramePolicy, FrameScope, ModelContext, OrderedBlocks,
    TokenCounter, TurnContext, TurnLifecycle, TurnSnapshot, WindowBudget,
};

// --- public ports ----------------------------------------------------------
pub use control::{AttemptControl, CallControl, ControlError, RunControl};
pub use gateway::{ModelGateway, ModelRequest};
/// The cancellation primitive behind the control planes, re-exported so the
/// port is self-contained for external drivers.
pub use tokio_util::sync::CancellationToken;
pub use tool::{ArtifactHint, ArtifactStore, IsolationLevel, StoreError, Tool};

// --- staged perimeter (deliberately root-exported reference wiring; holds no
//     claim on the kernel contract and may move or decompose without notice) --
pub use internal::config::{
    ExecutionOptions, RetryPolicy, TurnInvocation, TurnLimits, TurnPolicy, TurnRunOptions,
};
pub use internal::defaults::{NoopCompaction, NoopTokenCounter};
pub use internal::driver::{
    AttemptTrace, ModelRoundTrace, OutputSummary, ToolBatchTrace, ToolCallTrace, TurnInterruption,
    TurnOutcome, TurnResult, TurnRunner, TurnTrace,
};
pub use internal::executor::ToolExecutor;
pub use internal::fakes::FakeGateway;
