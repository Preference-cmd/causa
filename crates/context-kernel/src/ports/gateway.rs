//! `ModelGateway` port — the model-invocation seam drivers call into.
//! Transport-free; concrete gateways live outside the kernel. The port is
//! self-contained: its request parameters, result envelope, and transport
//! error all live here. Fact vocabulary (`ModelResponse`, `ModelStopReason`,
//! ids) comes from `crate::context`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::context::ids::InvocationId;
use crate::context::model::{ModelResponse, ModelStopReason};
use crate::context::turn::ContextFrame;
use crate::ports::control::AttemptControl;
use crate::ports::tool::ToolDefinition;

/// Attempt-loop ordinal, carried on requests and attempt traces. Not a fact:
/// the kernel's doors never branch on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttemptNumber(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef(pub String);
impl ModelRef {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolSurface {
    pub definitions: Vec<ToolDefinition>,
}
impl ToolSurface {
    pub fn empty() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }
    pub fn from_definitions(definitions: Vec<ToolDefinition>) -> Self {
        Self { definitions }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    /// Provider-reported prompt-cache read (hit) tokens, if disclosed.
    #[serde(default)]
    pub cache_read_tokens: Option<usize>,
    /// Provider-reported prompt-cache write (population) tokens, if disclosed.
    #[serde(default)]
    pub cache_write_tokens: Option<usize>,
    /// Provider-reported reasoning/thinking tokens, if disclosed.
    #[serde(default)]
    pub reasoning_tokens: Option<usize>,
}

/// Structured reasoning content: the model's thinking text plus the optional
/// provider signature some APIs attach so it can be replayed on later turns.
/// Recorded as-is by callers; the kernel does not interpret or persist it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningPayload {
    pub text: String,
    #[serde(default)]
    pub signature: Option<String>,
}

/// The gateway's result envelope: the model door consumes only the
/// `ModelResponse` and `ModelStopReason` facts from it; usage and reasoning
/// stay caller-retained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    pub response: ModelResponse,
    pub usage: Option<ModelUsage>,
    pub stop_reason: ModelStopReason,
    pub reasoning: Option<ReasoningPayload>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelInvokeErrorKind {
    Transient,
    TimedOut,
    Cancelled,
    Permanent,
    InvalidRequest,
    UnknownOutcome,
}

#[derive(Debug)]
pub struct ModelInvokeError {
    pub kind: ModelInvokeErrorKind,
    pub message: String,
}
impl ModelInvokeError {
    pub fn new(kind: ModelInvokeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub fn kind(&self) -> &ModelInvokeErrorKind {
        &self.kind
    }
}
impl std::fmt::Display for ModelInvokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for ModelInvokeError {}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub invocation_id: InvocationId,
    pub attempt: AttemptNumber,
    pub frame: ContextFrame,
    pub model: ModelRef,
    pub tool_surface: ToolSurface,
    pub generation: GenerationOptions,
}

#[async_trait]
pub trait ModelGateway: Send + Sync {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError>;

    /// Streaming completion. Default implementation: degenerates to a
    /// single [`StreamDelta::Done`] after `invoke` — providers that never
    /// implemented streaming keep working through the same port; streaming
    /// providers override to expose real token increments.
    async fn stream(
        &self,
        request: &ModelRequest,
        control: &AttemptControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        let output = self.invoke(request, control).await?;
        Ok(completed_model_stream(output))
    }
}

/// Bounded stream of provider deltas. Cancellation is observable through
/// the same `AttemptControl` the `invoke` path uses — drivers race
/// `next()` against the cancellation token, never poll.
pub type ModelStream = std::pin::Pin<Box<dyn futures_util::Stream<Item = StreamDelta> + Send>>;

/// Wrap an already-complete output as a single-`Done` stream — the
/// degenerate form the default `stream` implementation and tests use.
pub fn completed_model_stream(output: ModelOutput) -> ModelStream {
    let stop_reason = output.stop_reason;
    Box::pin(futures_util::stream::once(async move {
        StreamDelta::Done {
            stop_reason,
            final_output: output,
        }
    }))
}

/// One incremental observation from a streaming provider.
///
/// Contract: `Done` MUST carry the fully-assembled [`ModelOutput`] (text,
/// tool-call drafts with stable ids, usage) — deltas are advisory
/// observations for hosts, never the source of truth. A stream that ends
/// without `Done` is an `UnknownOutcome`-shaped failure at the driver.
#[derive(Debug, Clone)]
pub enum StreamDelta {
    TextDelta {
        delta: String,
    },
    ReasoningDelta {
        delta: String,
    },
    ToolCallDelta {
        call_index: usize,
        /// Provider-issued id, bound to the stable kernel `ToolCallId` at
        /// first sight by the gateway assembling `Done`.
        provider_call_id: Option<String>,
        name_delta: Option<String>,
        arguments_delta: Option<String>,
    },
    Usage(ModelUsage),
    Done {
        stop_reason: ModelStopReason,
        final_output: ModelOutput,
    },
    Error {
        kind: ModelInvokeErrorKind,
        message: String,
    },
}
