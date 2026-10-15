//! Translation between kernel DTOs and provider-native DTOs.
//!
//! The translation layer is intentionally provider-SDK-free. It operates on
//! kernel facts and `serde_json::Value` payloads so concrete adapters can use
//! direct HTTP without leaking provider-native types into the kernel.
//!
//! `ContextFrame → provider wire body` rendering and `wire response →
//! ModelOutput` parsing for the context kernel's `ModelGateway` seam
//! (`anthropic`, `openai_chat`, `openai_responses`, sharing one policy
//! walk in `context_frame`).
//!
//! # `Role` mapping
//!
//! Each message block carries its [`causa_kernel::Role`]; `BlockMeta::source`
//! is provenance only and is not read by this layer. The three protocols map
//! roles to wire structures as follows:
//!
//! | `Role`            | Anthropic Messages          | OpenAI Chat Completions     | OpenAI Responses            |
//! |-------------------|-----------------------------|-----------------------------|-----------------------------|
//! | `Role::System`    | Leading run -> top-level `system` parameter; subsequent -> inline `system` message | Inline `system` message | Leading run -> top-level `instructions` parameter; subsequent -> inline `system` message |
//! | `Role::User`      | `user` message              | `user` message              | `user` message              |
//! | `Role::Assistant` | `assistant` message         | `assistant` message         | `assistant` message         |
//!
//! Shared structural policy (applied once in `context_frame`, so the
//! three renderers cannot drift): zero-length text is skipped
//! (mirroring the kernel model door; whitespace-only text is content, and
//! the walk keeps it); one listed exception at emission — the Anthropic and
//! Chat renderers omit a whitespace-only assistant text part from the
//! request while the Responses renderer keeps it, and the fact stays in the
//! `Context` in every case (a Chat assistant message left with neither
//! `content` nor `tool_calls` is omitted with it);
//! no text is joined across block seams, for any role, and no renderer
//! inserts a character the model did not produce; tool ids come from
//! `meta.provider_call_id` with the declaration `BlockId` UUID as fallback,
//! and tool result ids resolve through their declaration block identity
//! (unpaired, duplicate, incomplete or ambiguously identified exchanges
//! fail with InvalidRequest before HTTP); non-string tool observations
//! serialize to a string. Ordered result notes render with the same tool
//! result, while remaining separate from the stored output payload.
//!
//! # Wire fidelity
//!
//! Model output the host has not edited is rendered back value-equal to what
//! the provider returned, item for item and in order, on the format it came
//! from. Each parser splits a wire item into the facts it understands (text,
//! tool name, arguments, call id) and keeps every other field in
//! `BlockMeta::wire` (`WireMeta { format, fields }`, the format being
//! `Protocol::as_str()`); message-level Chat fields such as
//! `reasoning_content` travel as a `ContentPart::Sealed` part. A renderer
//! starts from `fields` only when `wire.format` is its own and writes the
//! fact-derived fields over them; a block of another format, a block without
//! `wire` and an edited block render from facts alone. A sealed part of the
//! renderer's own format is written verbatim, one of another format is
//! skipped, and one outside assistant position is an `InvalidRequest`.
//! OpenAI-family tool calls keep the provider's `arguments` string in
//! `fields`; it is sent unchanged only while it parses to the fact's value.
//! Reasoning replay is material, not an observation: an Anthropic `thinking`
//! (signature included), `redacted_thinking` or unknown content block, a
//! Responses `reasoning` item and any other unknown output item, and a Chat
//! message-level residue such as `reasoning_content` are each kept as one
//! sealed item in place, so the turn is echoed back as returned. The readable
//! thinking text additionally feeds the reasoning observation, which is
//! display-only.
//!
//! Each renderer takes one renderer-specific argument, defined in
//! [`render_options`]: Anthropic native tool changes, Responses
//! `include`/`store` and `additional_tools`, and the Chat
//! `reasoning_content` replay opt-out.

/// The schema name rendered into OpenAI-family structured-output envelopes
/// (Chat `response_format.json_schema.name`, Responses `text.format.name`).
/// The kernel carries no schema name; this placeholder keeps the wire shape
/// complete without widening the `GenerationOptions` contract.
pub(crate) const OUTPUT_SCHEMA_NAME: &str = "response";

pub mod anthropic;
pub(crate) mod context_frame;
pub mod error_envelope;
mod generation;
pub mod media;
pub mod openai_chat;
pub mod openai_responses;
pub mod render_options;
pub mod sse_parser;
mod stop_reason;
pub mod streaming;
mod tool_history;
pub mod usage;

#[cfg(test)]
pub(crate) mod test_support;
