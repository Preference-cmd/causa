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
//!
//! # Protocol options and wire rendering
//!
//! - **Generation options**: OpenAI Chat renders `max_tokens` as
//!   `max_completion_tokens`, Responses renders it as `max_output_tokens`,
//!   and Anthropic renders it as `max_tokens`. `None` omits the limit on the
//!   two OpenAI wires; Anthropic requires the field, so that renderer rejects
//!   a missing value before any HTTP. An active reasoning effort combined
//!   with a temperature is rejected by all three renderers.
//! - **Reasoning effort**: OpenAI Chat renders `reasoning_effort`, Responses
//!   renders `reasoning.effort`, and Anthropic renders its `thinking` block plus
//!   `output_config.effort` (where [`causa_kernel::ReasoningEffort::Minimal`]
//!   has no wire representation and fails with `InvalidRequest`).
//! - **Structured outputs**: `output_schema` renders as `response_format` on
//!   Chat and `text.format` on Responses; the Anthropic adapter rejects it.
//! - **Prompt cache directives**: [`causa_kernel::CacheDirective::StablePrefix`]
//!   renders explicit `cache_control` breakpoints on Anthropic; OpenAI-family
//!   providers cache automatically server-side, treating the directive as a no-op.
//! - **Stop values**: Anthropic maps wire `stop_reason`, Chat maps `finish_reason`,
//!   and Responses maps `status` (or `status.reason` when status is `incomplete`
//!   and carries a reason).

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod protocol;
pub mod translation;

pub use protocol::Protocol;
pub use translation::sse_parser::{SseEvent, SseParser};
