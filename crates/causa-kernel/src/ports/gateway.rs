//! Logical model invocation contracts.

mod contract;

pub use contract::{
    CacheDirective, GenerationOptions, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRef, ModelRequest, ModelStream, ModelUsage, ReasoningEffort,
    ReasoningPayload, StreamDelta, completed_model_stream,
};
