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
//! (`anthropic`, Slice 3): `ContextFrame → provider wire body` rendering and
//! `wire response → ModelOutput` parsing for the context kernel's
//! `ModelGateway` seam.

pub mod anthropic;
pub mod files;
pub mod listing;
pub mod params;
pub mod request;
pub mod response;
pub mod sse_parser;
pub mod streaming;
pub mod tools;
pub mod usage;
