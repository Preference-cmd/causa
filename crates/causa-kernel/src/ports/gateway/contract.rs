//! `ModelGateway` port — the model-invocation seam drivers call into.
//! Transport-free; concrete gateways live outside the kernel. The port is
//! self-contained: its request parameters, result envelope, and transport
//! error all live here. Fact vocabulary (`ModelResponse`, `ModelStopReason`,
//! ids) comes from `crate::context`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::context::ids::InvocationId;
use crate::context::material::ContextFrame;
use crate::context::model::{ModelResponse, ModelStopReason};
use crate::ports::control::CallControl;
use crate::ports::tool::ToolSurface;

/// Opaque model identifier handed to the gateway. The kernel never parses
/// it; resolution is the gateway's job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef(pub String);
impl ModelRef {
    /// Wraps a raw model-id string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Requested reasoning effort. Availability and meaning depend on the model;
/// hosts must validate the selected model's supported values before admission.
/// This contract does not enable thinking replay or retain provider state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Explicitly request no reasoning effort, where supported.
    None,
    /// Minimal reasoning effort, where supported.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort, where supported.
    #[serde(rename = "xhigh")]
    XHigh,
    /// Maximum reasoning effort, where supported.
    Max,
}

/// Generation knobs a caller may pin for the invocation. `None` leaves the
/// provider default in force.
/// Protocol adapters reject options they cannot faithfully render. Hosts own
/// model-specific capability validation, defaults, and aggregate budgets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// Sampling temperature; `None` leaves the provider default in force.
    pub temperature: Option<f32>,
    /// Upper bound on generated tokens, including reasoning tokens where the
    /// provider counts them. `None` leaves the limit unconstrained by the caller.
    pub max_tokens: Option<u32>,
    /// Requested reasoning effort. `None` leaves the provider default in force;
    /// `Some(ReasoningEffort::None)` explicitly requests no reasoning.
    /// An adapter may reject an active effort combined with a temperature.
    /// Model-specific support remains the host's responsibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// JSON Schema the final response should satisfy, rendered natively
    /// where the underlying protocol supports it. Schema validation and
    /// corrective retry stay host-side. `None` omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<serde_json::Value>,
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

/// Structured reasoning content: the model's readable thinking text.
/// Recorded as-is by callers; the kernel does not interpret or persist it.
///
/// Opaque provider items that must be echoed back on later turns are carried
/// as context material (such as [`crate::ContentPart::Sealed`] parts). This
/// observation is display-only.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningPayload {
    /// The model's thinking text.
    pub text: String,
}

/// The gateway result envelope. Pure response conversion consumes neither
/// this value nor its usage and reasoning; execution consumers retain them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOutput {
    /// What the model said and asked: the [`ModelResponse`] fact.
    pub response: ModelResponse,
    /// Token accounting, when the provider disclosed it.
    pub usage: Option<ModelUsage>,
    /// Why generation ended: the [`ModelStopReason`] fact.
    pub stop_reason: ModelStopReason,
    /// The stop value exactly as the upstream provider or model sent it,
    /// so a host can inspect what an unrecognized stop condition was. `None`
    /// only for gateways with no wire value, such as test doubles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_stop_reason: Option<String>,
    /// Structured reasoning payload, when the provider emitted one.
    pub reasoning: Option<ReasoningPayload>,
}

/// Classification of a logical model invocation failure.
///
/// More kinds may be added; a match needs a wildcard arm, and an unknown kind
/// is best treated like [`Self::Permanent`]: not retried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelInvokeErrorKind {
    /// Recoverable failure a retry policy may allow: transport hiccups,
    /// rate limits, server-side errors.
    Transient,
    /// The request exceeded its deadline.
    TimedOut,
    /// The invocation was cancelled through its [`CallControl`], before send
    /// or while in flight.
    Cancelled,
    /// The provider refused and another attempt will not help: a rejected
    /// credential, an unknown model or path, or any other refusal not
    /// classified below.
    Permanent,
    /// The provider answered and the adapter could not use the answer: a
    /// success body or stream that is malformed or breaks the protocol's own
    /// rules, or any response over a configured size limit. Not retryable. Unlike
    /// [`Self::Permanent`], changing the credential, model or endpoint
    /// setting is not the remedy; the adapter and the server disagree about
    /// the wire.
    Undecodable,
    /// The provider rejected the request itself as malformed.
    InvalidRequest,
    /// No observable outcome — e.g. a stream that ended without `Done`.
    UnknownOutcome,
}

/// A model-invocation failure: a [`ModelInvokeErrorKind`] plus human-readable
/// detail (provider text included where available).
///
/// `status` and `code` repeat, as data, what the message says in words, so a
/// host can choose what to tell its user without parsing the message.
///
/// `retry_delay` carries a server-requested wait (an HTTP `retry-after` or
/// `retry-after-ms` header) when the transport filled one. The kernel does
/// not act on it; a caller-side retry policy decides whether to honor it.
///
/// `Display` prints the message alone; the kind is data, not prose.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
#[non_exhaustive]
pub struct ModelInvokeError {
    /// The failure classification.
    pub kind: ModelInvokeErrorKind,
    /// Human-readable failure detail.
    pub message: String,
    /// The HTTP status of a non-success response. `None` when the failure
    /// did not come from one: a transport fault, a local check, or an error
    /// reported inside a successful response.
    pub status: Option<u16>,
    /// The provider's own error code or type, such as `rate_limit_error`.
    /// `None` when the provider named none or the value was not a short
    /// identifier.
    pub code: Option<String>,
    /// Server-requested delay before another attempt, when the provider sent
    /// one. `None` when no usable delay was disclosed.
    pub retry_delay: Option<Duration>,
}
impl ModelInvokeError {
    /// Assembles an error from its kind and message, with no status, code or
    /// retry delay.
    pub fn new(kind: ModelInvokeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            status: None,
            code: None,
            retry_delay: None,
        }
    }
    /// Records the HTTP status of the non-success response behind this error.
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }
    /// Records the provider's error code or type. The caller passes only a
    /// short identifier; the kernel stores it as given.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
    /// Sets the server-requested retry delay. A retry policy may wait this
    /// long instead of using its own backoff.
    pub fn with_retry_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = Some(delay);
        self
    }
}

/// Prompt-cache instruction carried on a [`ModelRequest`]. Adapters map this
/// directive to protocol-specific cache mechanisms or treat it as a no-op where
/// caching is automatic or unsupported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheDirective {
    /// Mark no explicit cache breakpoints (the default).
    #[default]
    None,
    /// Mark the stable-prefix anchors so provider caches cover the tool
    /// surface, the system prefix, and the latest stable conversation
    /// message.
    StablePrefix,
}

/// Everything one logical invocation needs. All fields stay unchanged across
/// internal retries; the gateway owns any attempt numbering and retry policy.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// Identifies the turn + round this logical invocation belongs to; the same id
    /// recurs on every retry.
    pub invocation_id: InvocationId,
    /// The [`ContextFrame`] the gateway renders for the invocation.
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
/// honoring the invocation's control plane.
#[async_trait]
pub trait ModelGateway: Send + Sync {
    /// Runs one logical invocation, returning the
    /// assembled [`ModelOutput`], or a classified [`ModelInvokeError`].
    /// Cancellation and the deadline arrive through `control`.
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError>;

    /// Streams one logical output. Once any delta has been yielded, transparent
    /// retries are prohibited. The final Done alone supplies material for
    /// submission. Default implementation: degenerates to a
    /// single [`StreamDelta::Done`] after `invoke` — providers that never
    /// implemented streaming keep working through the same port; streaming
    /// providers override to expose real token increments.
    async fn stream(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        let output = self.invoke(request, control).await?;
        Ok(completed_model_stream(output))
    }
}

/// Bounded stream of provider deltas. Cancellation is observable through
/// the same `CallControl` the `invoke` path uses — drivers race
/// `next()` against the cancellation token, never poll.
pub type ModelStream = std::pin::Pin<Box<dyn futures_core::Stream<Item = StreamDelta> + Send>>;

struct OnceModelStream {
    item: Option<ModelOutput>,
}

impl futures_core::Stream for OnceModelStream {
    type Item = StreamDelta;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let delta = self
            .item
            .take()
            .map(|final_output| StreamDelta::Done { final_output });
        std::task::Poll::Ready(delta)
    }
}

/// Wrap an already-complete output as a single-`Done` stream — the
/// degenerate form the default `stream` implementation and tests use.
pub fn completed_model_stream(output: ModelOutput) -> ModelStream {
    Box::pin(OnceModelStream { item: Some(output) })
}

/// One incremental observation from a streaming provider.
///
/// Contract: `Done` MUST carry the fully-assembled [`ModelOutput`] (ordered
/// items, tool-call drafts with stable ids, usage) — deltas are advisory
/// observations for hosts, never the source of truth. A stream that ends
/// without `Done` is an `UnknownOutcome`-shaped failure at the driver.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
        /// Provider-issued id retained verbatim when the gateway assembles
        /// the complete tool declaration in `Done`.
        provider_call_id: Option<String>,
        /// Incremental fragment of the tool name, when the provider streams it.
        name_delta: Option<String>,
        /// Incremental fragment of the call's JSON arguments.
        arguments_delta: Option<String>,
    },
    /// A mid-stream token-usage observation.
    Usage(ModelUsage),
    /// Terminal delta: generation ended. Carries the fully-assembled output —
    /// the source of truth per this enum's contract, including its
    /// [`ModelStopReason`].
    Done {
        /// The fully-assembled [`ModelOutput`] — the authoritative result.
        final_output: ModelOutput,
    },
    /// In-stream failure observation carrying the gateway's own
    /// [`ModelInvokeError`], so an error before any delta keeps its
    /// classification and any server-requested retry delay.
    Error {
        /// The failure the gateway produced.
        error: ModelInvokeError,
    },
}
