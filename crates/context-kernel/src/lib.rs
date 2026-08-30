//! reimagine-context-kernel — ContextBlock conversation kernel.
//! No dependency on reimagine-core / app-host / Tauri / agent-harness.
//!
//! # Layering (Slice 1.5)
//!
//! - **`context`** — the external rule interface: exactly what the fact
//!   machine stores and validates — block content shapes, the turn state
//!   machine and its deterministic projections, and ids.
//! - **`ports`** — the behavior seams external implementors fill in, each
//!   self-contained: `ModelGateway` (request params, result envelope,
//!   transport error), `Tool` + `ArtifactStore` (definitions, execution
//!   context, outcome policy, limits), control planes, budget seams.
//! - **`internal`** — staged reference implementation (driver, config axes,
//!   executor, fakes, noop defaults), root-exported deliberately but holding
//!   no claim on the kernel contract.
//!
//! The physical modules are private; every re-export below is the entire
//! public contract. Nothing else is a cross-crate commitment.

#![deny(unsafe_code)]

mod context;
mod internal;
mod ports;

// --- context: the external rule interface ------------------------------------
pub use context::block::{BlockContent, BlockMeta, ContextBlock, TextPayload, ToolCallPayload};
pub use context::conversation::{ConversationError, ConversationState, OrderedTurns, SealedResult};
pub use context::ids::{
    BlockId, BlockSequence, ContextVersion, ConversationId, ConversationVersion, FrameId,
    InvocationId, RoundId, TurnId, TurnSequence,
};
pub use context::model::{ModelResponse, ModelStopReason, ToolCallDraft};
pub use context::tool_data::{
    ArtifactKind, ArtifactRef, ToolCallId, ToolOutput, ToolOutputMeta, ToolResultPayload,
    ToolResultStatus, Truncation,
};
pub use context::turn::{
    AppliedModelOutput, ContextError, ContextFrame, FrameScope, ModelContext, OrderedBlocks,
    TurnContext, TurnLifecycle, TurnSnapshot,
};

// --- ports: behavior seams for external implementors ------------------------
pub use ports::budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, FrameError, FramePolicy,
    TokenCounter, WindowBudget,
};
pub use ports::control::{AttemptControl, CallControl, ControlError};
pub use ports::gateway::{
    AttemptNumber, GenerationOptions, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRef, ModelRequest, ModelUsage, ReasoningPayload, ToolSurface,
};
pub use ports::tool::{
    ArtifactHint, ArtifactStore, IsolationLevel, StoreError, Tool, ToolCallContext, ToolDefinition,
    ToolExecutionOutcome, ToolOutputLimits, UnknownOutcomePolicy,
};
/// The cancellation primitive behind the control planes, re-exported so the
/// port is self-contained for external drivers.
pub use tokio_util::sync::CancellationToken;

// --- staged perimeter (deliberately root-exported reference wiring; holds no
//     claim on the kernel contract and may move or decompose without notice) --
pub use internal::config::{
    ExecutionOptions, RetryPolicy, TurnInvocation, TurnLimits, TurnPolicy, TurnRunOptions,
};
pub use internal::control::RunControl;
pub use internal::defaults::{NoopCompaction, NoopTokenCounter};
pub use internal::driver::{
    AttemptTrace, ModelRoundTrace, OutputSummary, ToolBatchTrace, ToolCallTrace, TurnInterruption,
    TurnOutcome, TurnResult, TurnRunner, TurnTrace,
};
pub use internal::executor::ToolExecutor;
pub use internal::fakes::FakeGateway;
