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
//! [`translation::streaming::StreamDecoder`] incrementally decodes raw SSE bytes
//! into kernel observations and a complete terminal output. [`SseParser`] is
//! the lower-level bounded byte framer. Neither owns HTTP, execution control,
//! retries, event history or a background task.
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
#![deny(missing_docs)]

pub mod protocol;
pub mod translation;

pub use protocol::Protocol;
pub use translation::sse_parser::{SseEvent, SseParser};
