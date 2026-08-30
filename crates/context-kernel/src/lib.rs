//! reimagine-context-kernel — ContextBlock conversation kernel.
//! No dependency on reimagine-core / app-host / Tauri / agent-harness.
//!
//! # Layering (Slice 1.5)
//!
//! - **`context`** — the external rule interface: block taxonomy, turn fact
//!   machine and deterministic projections, model/tool value shapes, ids.
//! - **`ports`** — the behavior seams external implementors fill in:
//!   `ModelGateway`, `Tool` + `ArtifactStore`, control planes, budget seams.
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
pub use context::ids::{
    AttemptNumber, BlockId, BlockSequence, ContextVersion, ConversationId, ConversationVersion,
    FrameId, InvocationId, RoundId, TurnId, TurnSequence,
};
pub use context::model::{
    GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRef,
    ModelResponse, ModelStopReason, ModelUsage, ReasoningPayload, ToolCallDraft, ToolSurface,
};
pub use context::tool_data::{
    ArtifactKind, ArtifactRef, ToolCallContext, ToolCallId, ToolDefinition, ToolExecutionOutcome,
    ToolOutput, ToolOutputLimits, ToolOutputMeta, ToolResultPayload, ToolResultStatus, Truncation,
    UnknownOutcomePolicy,
};
pub use context::turn::{
    AppliedModelOutput, ContextError, ContextFrame, FrameError, FrameScope, ModelContext,
    OrderedBlocks, TurnContext, TurnLifecycle, TurnSnapshot,
};

// --- ports: behavior seams for external implementors ------------------------
pub use ports::budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, FramePolicy, TokenCounter,
    WindowBudget,
};
pub use ports::control::{AttemptControl, CallControl, ControlError, RunControl};
pub use ports::gateway::{ModelGateway, ModelRequest};
pub use ports::tool::{ArtifactHint, ArtifactStore, IsolationLevel, StoreError, Tool};
/// The cancellation primitive behind the control planes, re-exported so the
/// port is self-contained for external drivers.
pub use tokio_util::sync::CancellationToken;

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
