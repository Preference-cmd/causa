//! Concrete provider adapters for two consumer seams:
//!
//! - `reimagine_agent_harness::AgentProvider` (frozen harness stack) behind
//!   the `CompletionBackend` seam owned by `reimagine-ai-protocol`;
//! - `reimagine_context_kernel::ModelGateway` (context kernel, Slice 3) —
//!   `AnthropicMessagesGateway`, `OpenAiChatCompletionsGateway`, and
//!   `OpenAiResponsesGateway` compose the kernel-native translation in
//!   `ai-protocol::translation` with reqwest transport, the shared Slice 3
//!   error mapping table, and read-only `AttemptControl` wiring.
//!
//! This crate is the transport + adapter layer: it owns reqwest HTTP
//! plumbing and the adapter implementations. Wire-protocol translation,
//! the `Protocol` discriminator, adapter construction parameters, and the
//! backend seam live in `reimagine-ai-protocol`. Provider configuration
//! documents and adapter wiring belong to `reimagine-app-host` (the
//! application layer), mirroring the provider / protocol / harness / app
//! separation of the Pi agent toolkit.
//!
//! See `docs/architecture/modules/agent-provider.md` for the design source.

#![deny(unsafe_code)]

mod anthropic_gateway;
mod backend_provider;
mod gateway_transport;
mod openai_gateway;
pub mod reqwest_backend;

pub use anthropic_gateway::AnthropicMessagesGateway;
pub use backend_provider::{BackendProvider, ProviderConfig};
pub use openai_gateway::{OpenAiChatCompletionsGateway, OpenAiResponsesGateway};
pub use reqwest_backend::ReqwestBackend;

/// V1 adapter for OpenAI-compatible chat completion APIs
/// (delegation over [`BackendProvider`], AC-10).
pub type OpenAiChatCompletionsProvider = BackendProvider<OpenAiChatCompletionsConfig>;
/// V1 adapter for the Anthropic Messages API
/// (delegation over [`BackendProvider`], AC-10).
pub type AnthropicMessagesProvider = BackendProvider<AnthropicMessagesConfig>;
/// V1 adapter for the OpenAI Responses API
/// (delegation over [`BackendProvider`], AC-10).
pub type OpenAiResponsesProvider = BackendProvider<OpenAiResponsesConfig>;

/// Re-export the protocol-layer types so consumers can depend on the
/// adapter crate alone for the full provider stack surface.
pub use reimagine_ai_protocol::{
    AnthropicMessagesConfig, CompletionBackend, FakeCompletionBackend, OpenAiChatCompletionsConfig,
    OpenAiResponsesConfig, Protocol, ProviderAdapterError, ScriptedBackendStep, SseEvent,
    SseParser,
};
