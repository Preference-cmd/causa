//! One owner per conversation: the session aggregate that accepts work and
//! drives it through the reference driver.
//!
//! A [`Session`] owns one
//! [`ConversationState`](crate::conversation::ConversationState) plus an
//! already-assembled [`TurnRunner`](crate::driver::TurnRunner) /
//! [`TurnRunOptions`](crate::config::TurnRunOptions); cloneable
//! [`SessionHandle`]s submit, observe, wait, resume, and cancel. The session is
//! the coordination layer around the driver (accept → begin/append → run →
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
//! - Local request-key dedup, checked before the busy and closed guards: same
//!   key and parts returns the original [`WorkReceipt`] (even after the owner
//!   closed), same key with different parts is [`SessionError::Conflict`], and
//!   a rejected submit keeps the key free.
//! - Bounded retention: [`SessionConfig::retained_work_capacity`] bounds the
//!   registry; finished results stay observable across later works.
//! - One writable state: it moves into the worker task while a work runs.
//! - `Completed` commits into history; `Interrupted` keeps the real facts plus
//!   cause and stays out of history; `Paused` keeps the complete outcome for an
//!   explicit `resume`. Observations include an independent copy of its
//!   [`crate::driver::PausePoint`] and the revision needed to decide it.
//! - A panic, runner rejection, or dropped worker publishes
//!   [`WorkState::Faulted`] with the reason and refuses new work.
//! - `resume` continues a paused work after validating the work's paused
//!   revision and the request; a rejected request executes nothing and leaves
//!   the paused material untouched. `cancel` fires the work's own control token
//!   (a racing completion keeps its result, a paused work is terminated in
//!   place with its continuation retained, and a terminal work is not
//!   rewritten). `shutdown` stops acceptance, winds down the running work, and
//!   retains material for inspection.
//! - Dropping the owner stops acceptance and fires the active work's
//!   cancellation token; retained observations stay readable.
//! - `checkpoint` exports a versioned [`SessionCheckpoint`] only from an
//!   idle or paused session (`Busy` while a work is accepted / running,
//!   refused once faulted; a closed session still exports, recording
//!   `closed`). `SessionCheckpoint::restore` validates the version, the
//!   assembled configuration description, and the material's internal
//!   consistency before registering — a rejection hands back the original
//!   envelope and the by-value inputs, and a successful restore calls no
//!   model or tool. Saved request keys replay their original receipts
//!   (different arguments still conflict), identity allocation continues
//!   from the saved progress, and a restored pause still needs an explicit
//!   `resume` under its remaining — never re-granted — deadline. Opening a
//!   conversation from history stays a different operation: it rebuilds no
//!   dedup table.
//!
//! Multi-session coordination, materials, and model tools are not here.
//!
//! # Runtime
//!
//! [`SessionHandle::submit`] spawns the worker on the ambient Tokio runtime.
//! If that runtime drops the task, even before its first poll, surviving
//! handles observe `Faulted` and the owner can still shut down. The lost
//! task's complete execution material cannot be recovered by this mechanism.
//!
//! # Layout
//!
//! This root file is the index only: `types` holds the public value vocabulary,
//! `owner` the [`Session`] anchor, `handle` the [`SessionHandle`] entry, the
//! private `execution` core the registry and worker, and `checkpoint` the
//! versioned save envelope (`SessionCheckpoint`) its restore consumes.
//! Convention is `foo.rs` + `foo/`, never `mod.rs`.

mod checkpoint;
mod execution;
mod handle;
mod owner;
mod types;

pub use checkpoint::{
    CheckpointPhase, SESSION_CHECKPOINT_VERSION, SavedCancelKey, SavedResumeKey, SavedSubmitKey,
    SavedWork, SessionCheckpoint, SessionConfigDescription, SessionRestoreRejection,
};
pub use handle::SessionHandle;
pub use owner::{Session, SessionBuildRejection};
pub use types::{
    CancelOutcome, CancelReceipt, FinishedKind, SessionConfig, SessionError, SubmitRequest,
    WaitEnd, WaitOutcome, WorkObservation, WorkReceipt, WorkRef, WorkState,
};
