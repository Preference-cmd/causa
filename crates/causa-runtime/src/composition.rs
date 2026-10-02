//! Composition layer — adapters bridging kernel ports into each other.
//!
//! [`ToolBridge`] adapts a `(DynamicToolSource, ToolDefinition)`
//! pair into a plain [`Tool`](causa_kernel::Tool). It is the executor's dynamic-dispatch
//! adapter (an invocation binding wraps a dynamic definition in a
//! bridge so it runs the same path as a local tool) and the
//! canonical home of the `ToolExecutionError` → status mapping. Hosts that
//! prefer snapshot semantics (list once, register as static tools) reuse
//! the same bridge directly; for live catalogs prefer
//! `ToolExecutor::register_dynamic`.

mod bridge;

pub use bridge::ToolBridge;
