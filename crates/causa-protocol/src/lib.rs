//! Wire-protocol translation for LLM providers.
//!
//! `causa-protocol` renders `ContextFrame` → provider wire bodies and
//! parses wire responses → kernel `ModelOutput`.
//! `translation::context_frame` is the shared policy walk; the three
//! renderers (`translation::anthropic`, `openai_chat`,
//! `openai_responses`) are thin emitters over it. The crate serves the
//! `causa_kernel::ModelGateway` seam and is consumed by `causa-provider`'s
//! gateways. It is transport-free.
//!
//! Layering:
//!
//! ```text
//! causa-kernel (facts, ModelGateway seam)
//!   <- causa-protocol (render/parse)   <- causa-provider
//! ```
//!
//! `causa-protocol` must not depend on reqwest, Axum, or any concrete
//! provider SDK.

#![deny(unsafe_code)]

pub mod protocol;
pub mod translation;

pub use protocol::Protocol;
pub use translation::sse_parser::{SseEvent, SseParser};
