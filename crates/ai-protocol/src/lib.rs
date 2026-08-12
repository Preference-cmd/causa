//! Wire-protocol translation for LLM providers.
//!
//! `ai-protocol` owns the *protocol* layer of the Pi-style provider
//! stack: the `Protocol` discriminator, typed adapter construction
//! parameters, the `CompletionBackend` seam, and the pure DTO
//! translation between Reimagine-owned shapes
//! (`reimagine_agent_harness`) and provider-native wire payloads. It is
//! transport-free — no reqwest, no SDKs — so the same translation logic
//! serves any concrete adapter. Streaming delta translation lives here
//! too (`translation::streaming`): transports keep HTTP + SSE byte
//! parsing and route parsed events through the accumulators.
//!
//! Layering (mirrors the Pi agent toolkit):
//!
//! ```text
//! agent-harness (loop, tools, policy, model catalog)
//!   <- ai-protocol (Protocol, translation, CompletionBackend seam)
//!   <- agent-provider (reqwest transport + concrete adapters)
//!   <- app-host (provider config documents, adapter wiring)
//! ```
//!
//! `ai-protocol` must not depend on reqwest, Tauri, Axum, app-host, or
//! any concrete provider SDK.

#![deny(unsafe_code)]

pub mod adapter_config;
pub mod backend;
pub mod error;
pub mod protocol;
pub mod translation;

pub use adapter_config::{
    AnthropicMessagesConfig, OpenAiChatCompletionsConfig, OpenAiResponsesConfig,
};
pub use backend::{CompletionBackend, FakeCompletionBackend, ScriptedBackendStep};
pub use error::ProviderAdapterError;
pub use protocol::Protocol;
pub use translation::sse_parser::{SseEvent, SseParser};
