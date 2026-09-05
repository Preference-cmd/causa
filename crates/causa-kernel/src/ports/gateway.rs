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

/// Opaque model identifier, rendered verbatim into the provider request's
/// `model` field. The kernel never parses it; resolution is the gateway's job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef(pub String);
impl ModelRef {
    /// Wraps a raw model-id string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Sampling knobs a caller may pin for the invocation. `None` means
/// "unspecified": renderers omit the knob and the provider default applies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// Sampling temperature; `None` leaves the provider default in force.
    pub temperature: Option<f32>,
    /// Upper bound on generated tokens; `None` omits the limit from the
    /// request.
    pub max_tokens: Option<u32>,
    /// JSON Schema the final response should satisfy, rendered natively
    /// where the provider supports it (Chat `response_format`, Responses
    /// `text.format`; Anthropic has no native mapping — schema validation
    /// and corrective retry stay host-side). `None` omits the knob.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
}

/// The tool definitions offered to the model for one invocation; renderers
/// turn this into the protocol's `tools` array.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolSurface {
    /// Model-facing [`ToolDefinition`]s, in offer order.
    pub definitions: Vec<ToolDefinition>,
}
impl ToolSurface {
    /// A surface offering no tools.
    pub fn empty() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }
    /// Builds a surface from the given definitions.
    pub fn from_definitions(definitions: Vec<ToolDefinition>) -> Self {
        Self { definitions }
    }
}

/// Provider-reported token accounting for one model response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    /// Tokens counted on the prompt side of the exchange.
    pub input_tokens: usize,
    /// Tokens counted on the generated side of the exchange.
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
    /// The model's thinking text.
    pub text: String,
    /// Provider signature that lets the reasoning be replayed on later turns,
    /// when the provider issues one.
    #[serde(default)]
    pub signature: Option<String>,
}

/// The gateway's result envelope: the model door consumes only the
/// `ModelResponse` and `ModelStopReason` facts from it; usage and reasoning
/// stay caller-retained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    /// What the model said and asked: the [`ModelResponse`] fact.
    pub response: ModelResponse,
    /// Token accounting, when the provider disclosed it.
    pub usage: Option<ModelUsage>,
    /// Why generation ended: the [`ModelStopReason`] fact.
    pub stop_reason: ModelStopReason,
    /// Structured reasoning payload, when the provider emitted one.
    pub reasoning: Option<ReasoningPayload>,
}

/// Classification of a model-invocation failure — the fact the retry policy
/// branches on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelInvokeErrorKind {
    /// Recoverable failure a retry policy may allow: transport hiccups,
    /// rate limits, server-side errors.
    Transient,
    /// The request exceeded its deadline.
    TimedOut,
    /// The attempt was cancelled through its [`AttemptControl`], before send
    /// or while in flight.
    Cancelled,
    /// Non-retryable failure: auth, unusable provider responses, parse
    /// failures, and everything else not classified transient.
    Permanent,
    /// The provider rejected the request itself as malformed.
    InvalidRequest,
    /// No observable outcome — e.g. a stream that ended without `Done`.
    UnknownOutcome,
}

/// A model-invocation failure: a [`ModelInvokeErrorKind`] plus human-readable
/// detail (provider text included where available).
#[derive(Debug)]
pub struct ModelInvokeError {
    /// The failure classification.
    pub kind: ModelInvokeErrorKind,
    /// Human-readable failure detail.
    pub message: String,
}
impl ModelInvokeError {
    /// Assembles an error from its kind and message.
    pub fn new(kind: ModelInvokeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    /// The failure classification.
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

/// Prompt-cache instruction carried on a [`ModelRequest`]. The Anthropic
/// translation face renders explicit `cache_control` breakpoints at the
/// stable-prefix anchors; OpenAI-family providers cache server-side and
/// accept the directive as a documented no-op.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheDirective {
    /// Mark no cache breakpoints (the default; OpenAI server-side caching
    /// applies regardless).
    #[default]
    None,
    /// Mark the stable-prefix anchors so provider caches cover the tool
    /// surface, the system prefix, and the latest stable conversation
    /// message.
    StablePrefix,
}

/// Everything one gateway attempt needs: invocation identity, the context to
/// render, and the invocation knobs. Invariant across retries of the same
/// logical invocation except for [`ModelRequest::attempt`].
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// Identifies the turn + round this attempt belongs to; the same id
    /// recurs on every retry.
    pub invocation_id: InvocationId,
    /// This attempt's [`AttemptNumber`] within the invocation's retry loop.
    pub attempt: AttemptNumber,
    /// The [`ContextFrame`] the gateway renders into provider messages.
    pub frame: ContextFrame,
    /// Which model to invoke.
    pub model: ModelRef,
    /// The [`ToolSurface`] offered alongside the frame.
    pub tool_surface: ToolSurface,
    /// The [`GenerationOptions`] governing sampling.
    pub generation: GenerationOptions,
    /// The prompt-cache instruction the translation face renders. Invariant
    /// across retries like the rest of the request.
    pub cache: CacheDirective,
}

/// The model-invocation port: concrete gateways (provider adapters) live
/// outside the kernel and translate [`ModelRequest`]s onto their transport,
/// honoring the attempt's control plane.
#[async_trait]
pub trait ModelGateway: Send + Sync {
    /// Runs one attempt: renders `request` for the provider and returns the
    /// assembled [`ModelOutput`], or a classified [`ModelInvokeError`].
    /// Cancellation and the deadline arrive through `control`.
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
    /// An increment of assistant text.
    TextDelta {
        /// The new text fragment.
        delta: String,
    },
    /// An increment of reasoning/thinking text.
    ReasoningDelta {
        /// The new reasoning fragment.
        delta: String,
    },
    /// An increment of one streamed tool call, identified by its position
    /// among the model's tool calls.
    ToolCallDelta {
        /// Position of this call among the model's tool-call drafts — how
        /// deltas are matched to calls.
        call_index: usize,
        /// Provider-issued id, bound to the stable kernel `ToolCallId` at
        /// first sight by the gateway assembling `Done`.
        provider_call_id: Option<String>,
        /// Incremental fragment of the tool name, when the provider streams it.
        name_delta: Option<String>,
        /// Incremental fragment of the call's JSON arguments.
        arguments_delta: Option<String>,
    },
    /// A mid-stream token-usage observation.
    Usage(ModelUsage),
    /// Terminal delta: generation ended. Carries the fully-assembled output —
    /// the source of truth per this enum's contract.
    Done {
        /// The terminal [`ModelStopReason`].
        stop_reason: ModelStopReason,
        /// The fully-assembled [`ModelOutput`] — the authoritative result.
        final_output: ModelOutput,
    },
    /// In-stream failure observation, shaped like [`ModelInvokeError`].
    Error {
        /// The failure classification.
        kind: ModelInvokeErrorKind,
        /// Human-readable failure detail.
        message: String,
    },
}
