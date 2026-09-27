//! `ContextEvent` projection — a turn's facts projected into a sequence for
//! UI, observability, and audit consumers.
//!
//! Events are a **projection** of a turn's facts (`TurnContext` +
//! `TurnResult` + `TurnTrace`) into a sequence that consumers (UI,
//! observability, audit) can subscribe to. They are not facts themselves:
//! they are derived, not persisted, and never written back into a
//! `TurnContext`.
//!
//! ## Layering
//!
//! ```text
//! causa-kernel
//!   └─ TurnContext / facts vocabulary                          ← facts + contracts
//!             ^
//!             │ project_turn(...)
//!             |
//! causa-runtime
//!   └─ driver (TurnResult / TurnTrace / ModelRoundTrace)       ← outcome vocabulary
//!   └─ ContextEvent / project_turn                             ← projection
//!             ^
//!             |
//! app-host / external consumer   ← observers (UI, audit, metrics)
//! ```
//!
//! ## Boundaries
//!
//! - **Not facts**: `ContextEvent` instances are constructed on demand
//!   by `project_turn`; they never appear in a kernel snapshot.
//! - **Not persistent**: nothing in this module touches the workspace
//!   store. Consumers persist what they need.
//! - **No harness dependency**: this module is `Send + Sync`-pure and
//!   depends on no harness crate.
//! - **No `AgentEvent` reuse**: `AgentEvent` is out of scope; a host that
//!   needs to bridge to it does so with a one-off
//!   `From<ContextEvent> for AgentEvent` adapter in the host crate, not
//!   here.
//!
//! ## Serialization
//!
//! `ContextEvent` is serde-derived for IPC delivery to host UIs and audit
//! pipelines. The embedded driver outcome types (`TurnResult`, `TurnTrace`)
//! carry their own serde derives; their serde **shapes** are a load-bearing
//! wire contract for this module — see the wire-contract note on
//! `crate::driver::TurnOutcome`.

mod projection;

pub use projection::{
    ContextEvent, ContextEventKind, StreamEventCollector, project_streaming_turn, project_turn,
};
