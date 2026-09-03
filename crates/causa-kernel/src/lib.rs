//! causa-kernel — ContextBlock conversation kernel.
//! No dependency on reimagine-core / app-host / Tauri / agent-harness.
//!
//! # Layering (Slice 12: the kernel is facts + contracts, nothing else)
//!
//! - **`context`** — the external rule interface: exactly what the fact
//!   machine stores and validates — block content shapes, the turn state
//!   machine and its deterministic projections, and ids.
//! - **`ports`** — the behavior seams external implementors fill in, each
//!   self-contained: `ModelGateway` (request params, result envelope,
//!   transport error), `Tool` + `ArtifactStore` (definitions, execution
//!   context, outcome policy, limits), control planes, budget seams. A
//!   type belongs here iff it is the contract surface third parties
//!   implement or call against; the kernel itself consumes none of it.
//!
//! The reference driver, executor, hook seam, config axes, and run
//! control that once lived in a staged perimeter inside this crate
//! graduated to `causa-runtime` (Slice 12). Anything left here
//! is either a fact or a contract; both are load-bearing and neither is
//! staged.
//!
//! The physical modules are private; every re-export below is the entire
//! public contract. Nothing else is a cross-crate commitment.

#![deny(unsafe_code)]
#![deny(missing_docs)]

mod context;
mod ports;

// --- context: the external rule interface ------------------------------------
pub use context::block::{BlockContent, BlockMeta, ContextBlock, TextPayload, ToolCallPayload};
pub use context::conversation::{
    ConversationError, ConversationState, OrderedTurns, SealedResult, merged_frame,
};
pub use context::ids::{
    BlockId, BlockSequence, ContextVersion, ConversationId, ConversationVersion, FrameId,
    FrameScope, InvocationId, RoundId, TurnId, TurnSequence,
};
pub use context::model::{ModelResponse, ModelStopReason, ToolCallDraft};
pub use context::tool_data::{
    ArtifactKind, ArtifactRef, ToolCallId, ToolOutput, ToolOutputMeta, ToolResultPayload,
    ToolResultStatus, Truncation,
};
pub use context::turn::{
    AppliedModelOutput, ContextError, ContextFrame, ModelContext, OrderedBlocks, TurnContext,
    TurnLifecycle, TurnSnapshot, turn_context_as_snapshot,
};

// --- ports: behavior seams for external implementors ------------------------
pub use ports::budget::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, FrameError, FramePolicy,
    TokenCounter, WindowBudget,
};
pub use ports::control::{AttemptControl, CallControl, ControlError, effective_deadline};
pub use ports::gateway::{
    AttemptNumber, GenerationOptions, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRef, ModelRequest, ModelStream, ModelUsage, ReasoningPayload, StreamDelta,
    ToolSurface, completed_model_stream,
};
pub use ports::interaction::{BatchDecision, TurnInteraction};
pub use ports::source::{DynamicToolSource, SourceError, ToolExecutionError};
pub use ports::store::{ConversationStore, ConversationStoreError};
pub use ports::tool::{
    ArtifactHint, ArtifactStore, IsolationLevel, StoreError, Tool, ToolCallContext, ToolDefinition,
    ToolExecutionOutcome, ToolOutputLimits, UnknownOutcomePolicy,
};
/// The cancellation primitive behind the control planes, re-exported so the
/// port is self-contained for external drivers.
pub use tokio_util::sync::CancellationToken;
