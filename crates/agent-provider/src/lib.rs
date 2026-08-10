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

mod anthropic;
mod openai_compatible;
mod openai_responses;
pub mod reqwest_backend;

pub use anthropic::AnthropicMessagesProvider;
pub use openai_compatible::OpenAiChatCompletionsProvider;
pub use openai_responses::OpenAiResponsesProvider;
pub use reqwest_backend::{
    ReqwestBackend, arc_real_anthropic_messages_backend,
    arc_real_anthropic_messages_backend_with_http_client, arc_real_openai_chat_completions_backend,
    arc_real_openai_chat_completions_backend_with_http_client, arc_real_openai_responses_backend,
    arc_real_openai_responses_backend_with_http_client,
};

/// Re-export the protocol-layer types so consumers can depend on the
/// adapter crate alone for the full provider stack surface.
pub use reimagine_ai_protocol::{
    AnthropicMessagesConfig, CompletionBackend, FakeCompletionBackend, OpenAiChatCompletionsConfig,
    OpenAiResponsesConfig, Protocol, ProviderAdapterError, ScriptedBackendStep, SseEvent, SseParser,
};
