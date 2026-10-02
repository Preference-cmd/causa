//! Logical model invocation contracts.

mod contract;

pub use contract::{
    CacheDirective, GenerationOptions, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRef, ModelRequest, ModelStream, ModelUsage, ReasoningPayload, StreamDelta,
    ToolSurface, completed_model_stream,
};
