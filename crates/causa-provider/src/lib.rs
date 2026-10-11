//! Concrete provider adapters for the context kernel's `ModelGateway`
//! seam — `AnthropicMessagesGateway`, `OpenAiChatCompletionsGateway`,
//! and `OpenAiResponsesGateway` compose the kernel-native translation in
//! `causa-protocol::translation` with reqwest transport, the shared
//! error mapping table, and read-only `CallControl` wiring.
//!
//! This crate is the transport + adapter layer: it owns reqwest HTTP
//! plumbing and the adapter implementations. Wire-protocol translation
//! and the `Protocol` discriminator live in `causa-protocol`.
//!
//! Media: the host injects a [`MediaResolver`] so the gateway can turn
//! fact-level references into inline render payloads; the kernel never
//! touches bytes.
//!
//! # Generation options
//!
//! [`causa_kernel::GenerationOptions`] carries explicit sampling, output-token
//! limits and optional [`causa_kernel::ReasoningEffort`]. Both call entrances
//! render the same options. Unsupported protocol combinations return
//! `InvalidRequest` before HTTP; fields are never silently dropped.
//!
//! Anthropic requires `max_tokens`: a missing limit is rejected before HTTP,
//! never defaulted. Effort renders natively on all three wires — Anthropic as
//! `thinking` plus `output_config.effort`, where `ReasoningEffort::None`
//! disables thinking and `ReasoningEffort::Minimal` has no wire level and
//! fails before HTTP — and `output_schema` remains unsupported by the
//! Anthropic adapter. Reasoning state replays as sealed context material: the
//! Anthropic and Responses parsers keep provider reasoning items whole and the
//! Chat renderer replays sealed `reasoning_content`. Hosts validate
//! model-specific effort values, sampling/default compatibility and
//! protocol/tool availability; model identifiers remain opaque to Causa.
//!
//! # Render options
//!
//! The renderer-specific choices live in
//! [`causa_protocol::translation::render_options`] and are hosted by the
//! matching gateway configuration: [`KernelHttpGateway::with_beta`] and
//! [`KernelHttpGateway::with_prefix_mismatch_behavior`] for Anthropic,
//! [`KernelHttpGateway::with_encrypted_reasoning`] and
//! [`KernelHttpGateway::with_store`] for Responses, and
//! [`KernelHttpGateway::with_reasoning_content_replay`] for Chat. They apply
//! to the gateway value, not to one request; the defaults are documented on
//! each option type.
//!
//! # Streaming
//!
//! All built-in gateways implement real SSE through
//! [`causa_kernel::ModelGateway::stream`]. Observations are provisional; only
//! [`causa_kernel::StreamDelta::Done`] carries the authoritative complete
//! output. Invalid/truncated streams fail without producing partial material.
//! [`StreamLimits`] bounds wire, event and retained-output budgets; configure
//! it with [`KernelHttpGateway::with_stream_limits`].
//!
//! Consumption drives reads, with no detached reader or automatic fallback.
//! Cancellation and deadlines apply during establishment and body processing;
//! parsing one complete JSON event is synchronous. Dropping the stream or
//! accepting a terminal releases the local HTTP response. Default clients
//! disable retries and redirects; host-injected clients retain their policies.
//! Session history, event delivery, display and persistence remain host-owned.
//!
//! # Retries
//!
//! Every non-2xx response is classified: 408, 409, 429 and 5xx are
//! `Transient`, 400 and 422 are `InvalidRequest`, and the rest
//! `Permanent`. An `x-should-retry` response header overrides that rule
//! when it is `true` or `false`. A `retry-after-ms` or `retry-after`
//! value (seconds or an HTTP date) fills
//! [`causa_kernel::ModelInvokeError::retry_delay`]. The adapters never retry
//! themselves — default clients disable transport retries — so bounded
//! retries stay a caller-side choice, such as the runtime's
//! `RetryingGateway`. See the shared transport module for the full table.
//!
//! # Error bodies
//!
//! Non-2xx responses from both `invoke` and `stream` follow one rule: when the
//! response body contains the documented error envelope (`{"error": {"message": "..."}}`),
//! the error message is formatted as `"{provider} error {status}: {error.message}"`,
//! with `error.message` truncated to 200 characters (appended with `…` if longer).
//! When the body is not that envelope, there is no fallback to raw body text and
//! the message contains the provider label and status only: `"{provider} error {status}"`.

#![deny(unsafe_code)]
#![deny(missing_docs)]

mod gateway_transport;
mod kernel_gateway;
pub mod media;

pub use kernel_gateway::{
    AnthropicGatewayConfig, AnthropicMessagesGateway, KernelGatewayConfig, KernelHttpGateway,
    OpenAiChatCompletionsGateway, OpenAiChatGatewayConfig, OpenAiResponsesGateway,
    OpenAiResponsesGatewayConfig,
};
pub use media::MediaResolver;

/// Resource limits for the transport-free streaming decoder.
pub use causa_protocol::translation::streaming::StreamLimits;
