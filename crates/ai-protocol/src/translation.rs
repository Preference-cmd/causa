//! Translation between Reimagine DTOs and provider-native DTOs.
//!
//! The translation layer is intentionally provider-SDK-free. It operates on
//! Reimagine DTOs and `serde_json::Value` payloads so concrete adapters can use
//! direct HTTP without leaking provider-native types into
//! `reimagine_agent_harness`.
//!
//! Two faces share this layer: the frozen harness-shaped translation
//! (`request` / `response` / `tools` / `listing` / `usage` / `streaming`,
//! operating on `reimagine_agent_harness` DTOs), and the kernel-native face
//! (Slice 3): `ContextFrame → provider wire body` rendering and
//! `wire response → ModelOutput` parsing for the context kernel's
//! `ModelGateway` seam (`anthropic`, `openai_chat`, `openai_responses`,
//! sharing one policy walk in `context_frame`).
//!
//! # `BlockMeta::source` vocabulary
//!
//! The kernel records `source` verbatim and never interprets it; this
//! layer is its interpreter. The vocabulary is open — unknown tags fall
//! back to the user role rather than failing, because renderers must be
//! total over frames that outlive any single vocabulary revision.
//!
//! | `BlockMeta::source`       | Rendering semantics                                    |
//! |---------------------------|--------------------------------------------------------|
//! | `None`                    | assistant — the model door never stamps `source`        |
//! | `"assistant"`             | assistant — host-prepared model turn (history replay)   |
//! | `"system"`                | system — Anthropic/Responses: top-level parameter; Chat: inline message |
//! | `"user"`                  | user role                                               |
//! | `"inject"` / `"inject:…"` | user role (rendering policy for injected context)       |
//! | anything else             | user role (documented open-vocabulary fallback)         |
//!
//! Shared structural policy (applied once in `context_frame`, so the
//! three renderers cannot drift): empty text blocks are skipped
//! (mirroring the kernel model door); adjacent same-role text blocks
//! join into one text with `\n`; tool ids come from
//! `meta.provider_call_id` with the kernel `call_id` as fallback, and
//! tool result ids resolve through the frame's `call_id → wire id` map
//! (an unpaired result falls back to its own `call_id` — the provider
//! rejects the orphan at HTTP time, the loud failure path); non-string
//! tool observations serialize to a string.

pub mod anthropic;
pub(crate) mod context_frame;
pub mod files;
pub mod listing;
pub mod openai_chat;
pub mod openai_responses;
pub mod params;
pub mod request;
pub mod response;
pub mod sse_parser;
pub mod streaming;
pub mod tools;
pub mod usage;
