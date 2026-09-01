//! Wire-protocol translation for LLM providers.
//!
//! `ai-protocol` carries **two translation faces** over the same three
//! provider wires:
//!
//! - **Kernel-native face** (canonical, Slice 3): `ContextFrame` → wire
//!   body rendering and wire response → kernel `ModelOutput` parsing.
//!   `translation::context_frame` is the shared policy walk; the three
//!   renderers (`translation::anthropic`, `openai_chat`,
//!   `openai_responses`) are thin emitters over it. Serves the
//!   `reimagine_context_kernel::ModelGateway` seam.
//! - **Frozen harness-shaped face** (legacy): `reimagine_agent_harness`
//!   DTOs ↔ provider wire payloads (`translation::{request, response,
//!   streaming, params, tools, listing, files}`, `backend`, `error`,
//!   `adapter_config`). Each of these modules carries a `⚠️ FROZEN`
//!   header: no new production semantics; the face dies with
//!   `reimagine-agent-harness` (Slice 9, harness dissolution). The only
//!   shared module between the faces is `translation::usage` (both
//!   faces read the same provider usage JSON into their own types).
//!
//! Layering:
//!
//! ```text
//! reimagine-context-kernel (facts, ModelGateway seam)
//!   <- ai-protocol (kernel-native render/parse)   <- agent-provider
//!
//! agent-harness (frozen legacy: loop, tools, policy, catalog)
//!   <- ai-protocol (frozen harness-shaped DTO translation)
//!   <- agent-provider (BackendProvider, legacy reqwest backend)
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
