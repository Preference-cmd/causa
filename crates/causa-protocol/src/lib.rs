//! Wire-protocol translation for LLM providers (kernel-native face).
//!
//! `ai-protocol` renders `ContextFrame` → provider wire bodies and parses
//! wire responses → kernel `ModelOutput`. `translation::context_frame` is
//! the shared policy walk; the three renderers (`translation::anthropic`,
//! `openai_chat`, `openai_responses`) are thin emitters over it. It serves
//! the `causa_kernel::ModelGateway` seam and is consumed by
//! `causa-provider`'s kernel gateways.
//!
//! The former frozen harness-shaped face (`translation::{request,
//! response, streaming, params, tools, listing, files}`, `backend`,
//! `error`, `adapter_config`) now lives in
//! `reimagine-agent-legacy-stack` (Reimagine-side, dies with Slice 9
//! harness dissolution); this crate is transport-free and
//! harness-free.
//!
//! Layering:
//!
//! ```text
//! causa-kernel (facts, ModelGateway seam)
//!   <- ai-protocol (kernel-native render/parse)   <- agent-provider
//! ```
//!
//! `ai-protocol` must not depend on reqwest, Tauri, Axum, app-host,
//! `reimagine-agent-harness`, or any concrete provider SDK.

#![deny(unsafe_code)]

pub mod protocol;
pub mod translation;

pub use protocol::Protocol;
pub use translation::sse_parser::{SseEvent, SseParser};
