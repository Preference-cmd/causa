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
//! limits, an optional reasoning effort and extra request fields. Both call
//! entrances render the same options; fields are never silently dropped.
//!
//! The reasoning effort is a string sent as given: Chat `reasoning_effort`,
//! Responses `reasoning.effort`, Anthropic `output_config.effort`.
//! `extra_body` is merged into the request, beside what the adapter wrote
//! and never in place of it, which is how
//! a host sends an Anthropic `thinking` object or any setting only one
//! server knows. Neither is checked here. A value the provider refuses comes
//! back as its error, with its message, status and code.
//!
//! Anthropic requires `max_tokens`: a missing limit is rejected before HTTP,
//! never defaulted, and `output_schema` remains unsupported by the Anthropic
//! adapter. Reasoning state replays as sealed context material: the
//! Anthropic and Responses parsers keep provider reasoning items whole and the
//! Chat renderer replays sealed `reasoning_content`. Hosts validate
//! model-specific effort values, sampling/default compatibility and
//! protocol/tool availability; model identifiers remain opaque to Causa.
//!
//! # Render options
//!
//! The renderer-specific choices live in
//! [`causa_protocol::translation::render_options`] and are hosted by the
//! matching gateway configuration: [`KernelHttpGateway::with_beta`] for
//! Anthropic,
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
//! `Permanent`. A response the adapter cannot read is `Undecodable`, which
//! is not retried either: a 2xx body or stream it cannot parse, or a body of
//! any status over the size cap. A failure before any response arrived,
//! including a connection the peer closed, is `Transient`; the request may
//! have reached the server. An `x-should-retry` response header overrides
//! the status rule when it is `true` or `false`. A `retry-after-ms` or `retry-after`
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
//!
//! The status is also in [`causa_kernel::ModelInvokeError::status`], and the
//! envelope's `code` (or `type`) in [`causa_kernel::ModelInvokeError::code`]
//! when it is a short identifier, so a host can tell a rejected credential
//! from an unknown model without parsing the message.
//!
//! An error inside a successful response (a stream event, or a 2xx body
//! that is an error envelope) is read by the same rule and formatted as
//! `"provider error ({code}): {error.message}"`; it has a code and no
//! status, and its kind comes from the envelope's `code` and `type`. Provider text is always one line: control characters
//! are replaced by spaces.

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
