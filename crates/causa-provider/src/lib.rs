//! Concrete provider adapters for the context kernel's `ModelGateway`
//! seam — `AnthropicMessagesGateway`, `OpenAiChatCompletionsGateway`,
//! and `OpenAiResponsesGateway` compose the kernel-native translation in
//! `causa-protocol::translation` with reqwest transport, the shared
//! error mapping table, and read-only `AttemptControl` wiring.
//!
//! This crate is the transport + adapter layer: it owns reqwest HTTP
//! plumbing and the adapter implementations. Wire-protocol translation
//! and the `Protocol` discriminator live in `causa-protocol`.
//!
//! Media: the host injects a [`MediaResolver`] so the gateway can turn
//! fact-level references into inline render payloads; the kernel never
//! touches bytes.

#![deny(unsafe_code)]

mod gateway_transport;
mod kernel_gateway;
pub mod media;

pub use kernel_gateway::{
    AnthropicGatewayConfig, AnthropicMessagesGateway, KernelGatewayConfig, KernelHttpGateway,
    OpenAiChatCompletionsGateway, OpenAiChatGatewayConfig, OpenAiResponsesGateway,
    OpenAiResponsesGatewayConfig,
};
pub use media::MediaResolver;
