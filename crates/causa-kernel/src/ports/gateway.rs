//! Logical model invocation contracts.

mod contract;

pub use crate::ports::tool::ToolSurface;
pub use contract::{
    CacheDirective, GenerationOptions, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRef, ModelRequest, ModelStream, ModelUsage, ReasoningEffort,
    ReasoningPayload, StreamDelta, completed_model_stream,
};
