//! causa-kernel — ContextBlock conversation kernel.
//! No dependency on a host application, a UI shell, or another agent framework.
//!
//! The kernel holds ordered material and atomic identity-preserving edits, plus
//! third-party model, preparation, tool and cancellation contracts. It performs
//! no I/O, execution policy or conversation management. Reference execution
//! lives in causa-runtime; concrete transports live in edge crates.

#![deny(unsafe_code)]
#![deny(missing_docs)]

mod context;
mod ports;

// --- context: the external rule interface ------------------------------------
pub use context::block::{
    BlockContent, BlockMeta, ContentPart, ContextBlock, MediaRef, TextPayload, ToolCallPayload,
};
pub use context::ids::{BlockId, InvocationId, RoundId, TurnId};
pub use context::material::{
    Context, ContextEdit, ContextFrame, EditError, EditFailure, Replacement, ToolResultError,
    validate_tool_result_append,
};
pub use context::model::{ModelBlockError, ModelResponse, ModelStopReason, ToolCallDraft};
pub use context::tool_data::{
    ArtifactKind, ArtifactRef, ToolCallId, ToolOutput, ToolOutputMeta, ToolResultPayload,
    ToolResultStatus, Truncation,
};

// --- ports: behavior seams for external implementors ------------------------
pub use ports::batch::{
    BatchError, ProcessorContext, ProcessorError, ToolBatch, ToolBatchProcessor, ToolCallEntry,
};
pub use ports::control::{CallControl, ControlError};
pub use ports::gateway::{
    CacheDirective, GenerationOptions, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRef, ModelRequest, ModelStream, ModelUsage, ReasoningPayload, StreamDelta,
    ToolSurface, completed_model_stream,
};
pub use ports::prepare::{ContextPreparer, PrepareError, RoundInfo};
pub use ports::source::{DynamicToolSource, SourceError, ToolExecutionError};
pub use ports::tool::{
    ArtifactHint, ArtifactStore, StoreError, Tool, ToolCallContext, ToolDefinition,
};
/// The cancellation primitive behind the control planes, re-exported so the
/// port is self-contained for external drivers.
pub use tokio_util::sync::CancellationToken;
