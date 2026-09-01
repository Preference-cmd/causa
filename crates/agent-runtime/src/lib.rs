//! reimagine-agent-runtime — framework policies over the context kernel.
//!
//! - Tool-use filters (`DedupFilter`, `AllowAllFilter`, `DenyAllFilter`,
//!   `FilterChain`) implement the kernel's `ToolUseHook` directly — the
//!   one seam `TurnRunner::with_hook` consumes. The kernel ships only
//!   `PassthroughHook` (no opinion); composition and policy are opt-in.
//! - `ContextEvent` / `project_turn` project a finished turn's facts
//!   (`TurnContext` + `TurnResult` + `TurnTrace`) into an IPC-ready
//!   event sequence for UI / observability / audit consumers.
//!
//! The kernel stays zero new user-facing extension types beyond its own
//! `ToolUseHook` adapter; this crate adds policies, not seams.

#![deny(unsafe_code)]

pub mod event;
pub mod filter;

pub use event::{ContextEvent, ContextEventKind, project_turn};
pub use filter::{AllowAllFilter, DedupFilter, DenyAllFilter, FilterChain};
// Re-exported so a custom filter's `impl ToolUseHook` signature can be
// written entirely against this crate.
pub use reimagine_context_kernel::{HookCtx, HookOutcome, ToolExecutor, ToolUseHook};
