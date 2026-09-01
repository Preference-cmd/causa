//! reimagine-agent-runtime — framework primitives for the agent stack.
//!
//! Slice 4 (选项 B):
//!
//! - User-facing tool-use filters live here: `ToolUseFilter` trait,
//!   `DedupFilter`, `AllowAllFilter`, `DenyAllFilter`, `FilterChain`,
//!   `FilterContext`, `FilterResult` (the last two are type aliases
//!   over the kernel-side adapter types).
//! - `AgentRuntime` is the consumer-side wrapper that wires a
//!   `TurnRunner` together with a `FilterChain`. It is what callers
//!   construct in place of `TurnRunner::new(...)` when they want
//!   framework-level filter composition.
//!
//! The kernel stays zero new user-facing types; the bridge
//! `impl ToolUseHook for FilterChain` is the only seam.

#![deny(unsafe_code)]

pub mod filter;
pub mod runtime;

pub use filter::{
    AllowAllFilter, DedupFilter, DenyAllFilter, FilterChain, FilterContext, FilterResult,
    ToolUseFilter,
};
// Re-export the kernel-side adapter trait so consumers can name it
// from a single crate path without reaching into context-kernel.
pub use reimagine_context_kernel::ToolUseHook;
pub use runtime::AgentRuntime;

// Re-export `ToolExecutor` as `ToolRegistry` (B1):
// `ToolRegistry` is the management surface; `ToolExecutor` already
// implements it via `from_vec / from_map / execute_with_limits`.
pub use reimagine_context_kernel::ToolExecutor;
pub type ToolRegistry = ToolExecutor;
