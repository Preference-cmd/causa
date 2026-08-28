//! reimagine-context-kernel — ContextBlock conversation kernel (Slice 1, Phase 1-4).
//! No dependency on reimagine-core / app-host / Tauri / agent-harness.

#![deny(unsafe_code)]

pub mod block;
mod budget;
pub mod control;
pub mod gateway;
pub mod ids;
pub mod model;
pub mod runtime;
pub mod tool;
pub mod turn;

pub use block::{
    BlockMeta, BlockPayload, ContextBlock, ContextInjectPayload, InputPayload, TextPayload,
    ToolCallPayload,
};
pub use control::{AttemptControl, CallControl, ControlError, RunControl};
pub use gateway::{FakeGateway, ModelGateway, ModelRequest, TurnLimits, TurnRunConfig};
pub use ids::{
    AttemptNumber, BlockId, BlockSequence, ContextVersion, ConversationId, ConversationVersion,
    FrameId, InvocationId, RoundId, TurnId, TurnSequence,
};
pub use model::{
    AssistantPayload, GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ModelOutput,
    ModelRef, ModelStopReason, ModelUsage, RetryPolicy, ToolCallDraft, ToolSurface,
};
pub use runtime::{
    AttemptTrace, ModelRoundTrace, OutputSummary, ToolBatchTrace, ToolCallTrace, TurnInterruption,
    TurnOutcome, TurnResult, TurnRunner, TurnTrace,
};
pub use tool::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, IsolationLevel, StoreError, Tool,
    ToolCallContext, ToolCallId, ToolDefinition, ToolExecutionOutcome, ToolExecutor, ToolOutput,
    ToolOutputLimits, ToolOutputMeta, ToolResultPayload, ToolResultStatus, Truncation,
    UnknownOutcomePolicy,
};
pub use turn::{
    AppliedModelOutput, Compaction, CompactionError, CompactionInput, CompactionOutput,
    ContextError, ContextFrame, FrameError, FrameScope, ModelContext, NoopCompaction,
    NoopTokenCounter, OrderedBlocks, TokenCounter, TurnContext, TurnLifecycle, TurnSnapshot,
    WindowBudget,
};
