//! One owner per conversation: a session accepts work and drives it through
//! the execution stack. Cloneable [`SessionHandle`]s submit, observe, wait,
//! and cancel; the session does not duplicate the model loop or expose a
//! writable [`ConversationState`](crate::conversation::ConversationState).
//!
//! A session runs one work at a time. Accepted request keys replay their
//! original receipts, while reusing a key with different input is a conflict.
//! Completed work commits into history. Interrupted work remains observable
//! with its fact snapshot, cause, and any uncommitted tool batch. The session
//! retains finished work up to its configured capacity.
//!
//! Shutdown stops admission, signals active work, and waits for it to finish.
//! Dropping the owner stops admission and signals work without waiting; handles
//! can still inspect already-published work observations.
//!
//! # Layout
//!
//! `types` holds the public values, `owner` the [`Session`] lifecycle anchor,
//! `handle` the [`SessionHandle`] operations, and `execution` the private
//! registry and worker.

mod execution;
mod handle;
mod owner;
mod types;

pub use handle::SessionHandle;
pub use owner::{Session, SessionBuildRejection};
pub use types::{
    CancelOutcome, CancelReceipt, FinishedKind, SessionConfig, SessionError, SubmitRequest,
    WaitEnd, WaitOutcome, WorkObservation, WorkReceipt, WorkRef, WorkState,
};
