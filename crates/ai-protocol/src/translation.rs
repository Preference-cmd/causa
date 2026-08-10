//! Translation between Reimagine DTOs and provider-native DTOs.
//!
//! The translation layer is intentionally provider-SDK-free. It operates on
//! Reimagine DTOs and `serde_json::Value` payloads so concrete adapters can use
//! direct HTTP without leaking provider-native types into
//! `reimagine_agent_harness`.

pub mod listing;
pub mod params;
pub mod request;
pub mod response;
pub mod sse_parser;
pub mod streaming;
pub mod tools;
