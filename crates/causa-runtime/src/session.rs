//! One owner per conversation: the session aggregate that accepts work and
//! drives it through the reference driver.
//!
//! A [`Session`] owns one
//! [`ConversationState`](crate::conversation::ConversationState) plus an
//! already-assembled [`TurnRunner`](crate::driver::TurnRunner) /
//! [`TurnRunOptions`](crate::config::TurnRunOptions); cloneable
//! [`SessionHandle`]s submit, observe, and wait. The session is the
//! coordination layer around the driver (accept → begin/append → run →
//! commit/abort/retain → notify); it neither duplicates the model loop nor
//! exposes a writable
//! [`ConversationState`](crate::conversation::ConversationState).
//!
//! # Observable behaviour
//!
//! - One active work, no queue: while a work is `Accepted` / `Running` /
//!   `Paused`, `submit` returns [`SessionError::Busy`] naming it.
//! - Stable identity: each work gets a fresh `TurnId` at acceptance, never
//!   reused, even after an interrupt.
//! - Local request-key dedup, checked before the busy guard: same key and parts
//!   returns the original [`WorkReceipt`], same key with different parts is
//!   [`SessionError::Conflict`], and a rejected submit keeps the key free.
//! - Bounded retention: [`SessionConfig::retained_work_capacity`] bounds the
//!   registry; finished results stay observable across later works.
//! - One writable state: it moves into the worker task while a work runs.
//! - `Completed` commits into history; `Interrupted` keeps the real facts plus
//!   cause and stays out of history; `Paused` keeps the complete outcome for an
//!   explicit `resume`.
//! - A panic or runner rejection publishes [`WorkState::Faulted`] with the
//!   reason and refuses new work — never a work stuck `Running`.
//! - Dropping the owner stops acceptance and fires the active work's
//!   cancellation token; retained observations stay readable.
//!
//! `resume`, `cancel`, `shutdown`, checkpoint/restore, multi-session
//! coordination, materials, and model tools are not here.
//!
//! # Runtime
//!
//! [`SessionHandle::submit`] spawns the worker on the ambient Tokio runtime.
//!
//! # Layout
//!
//! This root file is the index only: `types` holds the public value vocabulary,
//! `owner` the [`Session`] anchor, `handle` the [`SessionHandle`] entry, and the
//! private `execution` core the registry and worker. Convention is `foo.rs` +
//! `foo/`, never `mod.rs`.

mod execution;
mod handle;
mod owner;
mod types;

pub use handle::SessionHandle;
pub use owner::{Session, SessionBuildRejection};
pub use types::{
    FinishedKind, SessionConfig, SessionError, SubmitRequest, WaitEnd, WaitOutcome,
    WorkObservation, WorkReceipt, WorkRef, WorkState,
};
