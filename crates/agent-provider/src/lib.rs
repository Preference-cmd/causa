//! Concrete provider adapters for `reimagine_agent_harness::AgentProvider`.
//!
//! V1 supports reqwest-backed OpenAI-compatible, OpenAI Responses, and
//! Anthropic providers behind the `CompletionBackend` seam owned by
//! `reimagine-ai-protocol`.
//!
//! This crate is the transport + adapter layer: it owns reqwest HTTP
//! plumbing and the `AgentProvider` implementations. Wire-protocol
//! translation, the `Protocol` discriminator, adapter construction
//! parameters, and the backend seam live in `reimagine-ai-protocol`.
//! Provider configuration documents and adapter wiring belong to
//! `reimagine-app-host` (the application layer), mirroring the provider /
//! protocol / harness / app separation of the Pi agent toolkit.
//!
//! See `docs/architecture/modules/agent-provider.md` for the design source.

#![deny(unsafe_code)]

mod backend_provider;
pub mod reqwest_backend;

pub use backend_provider::{BackendProvider, ProviderConfig};
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
    OpenAiResponsesConfig, Protocol, ProviderAdapterError, ScriptedBackendStep, SseEvent, SseParser,
};
