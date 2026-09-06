//! causa-kernel — ContextBlock conversation kernel.
//! No dependency on reimagine-core / app-host / Tauri / agent-harness.
//!
//! # Layering (Slice 12: the kernel is facts + contracts, nothing else)
//!
//! - **`context`** — the external rule interface: exactly what the fact
//!   machine stores and validates — block content shapes, the turn state
//!   machine and its deterministic projections, and ids. Session-level
//!   vocabulary (the `ConversationState` aggregate, its eligibility stamp,
//!   ordering, and the conversation store port) is runtime territory since
//!   Slice 6.5; the kernel keeps the facts (`TurnContext` / `TurnSnapshot`),
//!   the validated recovery entries, and the shared `merged_frame`
//!   projection.
//! - **`ports`** — the behavior seams external implementors fill in, each
//!   self-contained: `ModelGateway` (request params, result envelope,
//!   transport error), `Tool` + `ArtifactStore` (definitions, execution
//!   context, outcome policy, limits), `DynamicToolSource`, control planes. A
//!   type belongs here iff it is the contract surface third parties
//!   implement or call against; the kernel itself consumes none of it.
//!   The reference budget/compaction seam and the host↔driver interaction
//!   seam are the reference harness's opinions, not cross-harness
//!   contracts — they moved to `causa-runtime` in Slice 13 (the
//!   conversation-persistence port moved with the session aggregate in
//!   Slice 6.5).
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
pub use context::block::{
    BlockContent, BlockMeta, ContentPart, ContextBlock, MediaRef, TextPayload, ToolCallPayload,
};
pub use context::ids::{
    BlockId, BlockSequence, ContextVersion, ConversationId, FrameId, FrameScope, InvocationId,
    RoundId, TurnId,
};
pub use context::model::{ModelResponse, ModelStopReason, ToolCallDraft};
pub use context::tool_data::{
    ArtifactKind, ArtifactRef, ToolCallId, ToolOutput, ToolOutputMeta, ToolResultPayload,
    ToolResultStatus, Truncation,
};
pub use context::turn::{
    AppliedModelOutput, ContextError, ContextFrame, ModelContext, OrderedBlocks, TurnContext,
    TurnLifecycle, TurnSnapshot, merged_frame, option_turn_context_as_snapshot,
    turn_context_as_snapshot,
};

// --- ports: behavior seams for external implementors ------------------------
pub use ports::control::{AttemptControl, CallControl, ControlError, effective_deadline};
pub use ports::gateway::{
    AttemptNumber, CacheDirective, GenerationOptions, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelOutput, ModelRef, ModelRequest, ModelStream, ModelUsage,
    ReasoningPayload, StreamDelta, ToolSurface, completed_model_stream,
};
pub use ports::source::{DynamicToolSource, SourceError, ToolExecutionError};
pub use ports::tool::{
    ArtifactHint, ArtifactStore, StoreError, Tool, ToolCallContext, ToolDefinition,
    ToolExecutionOutcome, ToolOutputLimits, UnknownOutcomePolicy,
};
/// The cancellation primitive behind the control planes, re-exported so the
/// port is self-contained for external drivers.
pub use tokio_util::sync::CancellationToken;
