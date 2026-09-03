//! causa-extension — `DynamicToolSource` adapters over the kernel ports.
//!
//! Each capability lives behind its own Cargo feature so offline hosts pay
//! for nothing they do not use. The first (and currently only) adapter is
//! the first-class MCP client (`mcp` feature, on by default).
//!
//! # Adding an adapter
//!
//! A new adapter is a new module gated on a new feature (`dep:`-gated
//! heavy dependencies), following the `mcp` module's shape: implement the
//! kernel's `DynamicToolSource` port, expose only kernel port vocabulary,
//! and keep transport details inside the module.

#![deny(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "mcp")]
pub use mcp::McpToolSource;
