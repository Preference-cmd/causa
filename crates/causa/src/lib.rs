//! `causa` — facade over the `causa-*` family.
//!
//! The kernel is always present; every other layer is a Cargo feature so
//! hosts pay only for what they use:
//!
//! | feature | re-export | pulls |
//! |---|---|---|
//! | _(always on)_ | [`kernel`] | facts + ports, no I/O |
//! | `runtime` | [`runtime`] | reference driver (turn loop, dispatch) |
//! | `protocol` | [`protocol`] | wire-protocol translation (no transport) |
//! | `providers` | [`providers`] | reqwest gateways (implies `protocol`) |
//! | `extension-mcp` | `extension` | MCP client (implies `mcp` adapter) |
//!
//! The default is the runnable stack (`runtime` + `providers`); `full`
//! adds the extensions. `--no-default-features` is the bare kernel for
//! offline audit or minimal embedding:
//!
//! ```toml
//! causa = "0.1" # default: runtime + providers
//! causa = { version = "0.1", features = ["full"] } # + MCP extensions
//! causa = { version = "0.1", default-features = false } # kernel only
//! causa = { version = "0.1", default-features = false, features = ["runtime"] }
//! ```
//!
//! The fine-grained crates (`causa-kernel`, `causa-runtime`, …) remain
//! published and usable directly; this facade is a convenience alias, not
//! a replacement.

#![deny(unsafe_code)]
#![deny(missing_docs)]

/// Facts + contracts. Always available; the only layer with no I/O.
pub use causa_kernel as kernel;

#[cfg(feature = "runtime")]
/// Reference driver over the kernel ports.
pub use causa_runtime as runtime;

#[cfg(feature = "protocol")]
/// Kernel-native wire-protocol translation (transport-free).
pub use causa_protocol as protocol;

#[cfg(feature = "providers")]
/// Reqwest adapters for the kernel `ModelGateway` seam.
pub use causa_provider as providers;

#[cfg(feature = "extension-mcp")]
/// Extension adapters (`DynamicToolSource` implementors).
pub use causa_extension as extension;
