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
