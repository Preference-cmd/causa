//! The session's public value vocabulary: work identity and state, the
//! terminal-result shapes, the submit / observe / wait data carriers, the
//! coordination config, and the error surface.
//!
//! Plain data plus the error enum — the coordination behaviour lives in
//! [`Session`](crate::session::Session) /
//! [`SessionHandle`](crate::session::SessionHandle) and the private
//! `execution` core. The identifiers reuse the kernel's `ConversationId` /
//! `TurnId`; the terminal result reuses the driver's `TurnResult` vocabulary.

use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, ModelOutput, TurnId, TurnSnapshot};

use crate::driver::TurnInterruption;

/// Identity of one accepted work: the conversation plus the turn id the
/// session assigned at acceptance.
///
/// The pair of existing ids only — no separate agent/work id is introduced.
/// An `observe` / `wait` for a ref that does not belong to the session is
/// [`SessionError::NotFound`] (it is never routed to another conversation).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkRef {
    /// The conversation the work belongs to.
    pub conversation_id: ConversationId,
    /// The turn id assigned at acceptance; never reused by a later work.
    pub turn_id: TurnId,
}

/// The session's management view of a work.
///
/// This is the runtime's coordination vocabulary, not a kernel state machine:
/// the actual execution result is carried by [`FinishedKind`], which reuses
/// the driver's existing `TurnResult` vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkState {
    /// The session holds the work and its execution resources; the worker has
    /// not yet begun advancing the runner. `Accepted` does **not** mean the
    /// model has been called.
    Accepted,
    /// The runner is advancing the work. `submit` returns
    /// [`SessionError::Busy`]; `observe` reads the published view only.
    Running,
    /// The complete paused outcome came back to the session. The conversation
    /// is still busy; only an explicit `resume` continues it.
    Paused,
    /// Terminal: the work completed or was interrupted; see
    /// [`WorkObservation::finished`]. The conversation accepts a next work.
    Finished,
    /// Terminal: the worker exited abnormally (panic or runner rejection), so
    /// no complete outcome exists. The fault reason is retained (see
    /// [`WorkObservation::fault`]); no recovery scene is fabricated and no
    /// complete snapshot is invented, because the material may be
    /// unrecoverable. The conversation refuses new work with
    /// [`SessionError::Faulted`] until the harness handles it.
    Faulted,
}

/// How a terminal work ended — the two terminal `TurnResult` shapes, kept as
/// (or inside) the driver's own vocabulary rather than copied into new types.
#[derive(Debug, Clone)]
pub enum FinishedKind {
    /// The model ended the turn; the turn was committed into history.
    Completed {
        /// The turn's final model output.
        final_output: ModelOutput,
    },
    /// The turn stopped early; the real aborted facts and the cause were
    /// retained, and the turn stayed out of completed history.
    Interrupted {
        /// Why the turn was interrupted.
        cause: TurnInterruption,
        /// The real facts of the aborted turn — the [`TurnSnapshot`] taken
        /// when the session aborted it. Carried publicly so an interrupted
        /// work's material is verifiable without reading the
        /// (completed-only) history, which deliberately excludes it.
        facts: TurnSnapshot,
    },
}

/// One submission: the caller-scoped idempotency key plus the new task's
/// content parts.
///
/// The parts commit as one fact block through
/// [`TurnContext::append_parts`](causa_kernel::TurnContext::append_parts)
/// (source label `"user"`), exactly as the reference driver expects. There is
/// no model/tool hot-swapping and no per-submit materials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmitRequest {
    /// Caller-scoped idempotency key. Repeating it with the same parts returns
    /// the original [`WorkReceipt`]; reusing it with different parts is
    /// [`SessionError::Conflict`].
    pub request_key: String,
    /// The new task's content; must be non-empty.
    pub parts: Vec<ContentPart>,
}

/// Acceptance receipt for a submit — the assigned work and the revision it was
/// accepted at.
///
/// A receipt means *accepted*, not *completed*: the worker may not have called
/// the model yet. It exists so a lost receipt can be recovered by retrying the
/// same `request_key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkReceipt {
    /// The work the session accepted.
    pub work: WorkRef,
    /// The work's revision at acceptance (always `0`: the `Accepted` state).
    pub accepted_revision: u64,
}

/// A snapshot of one work's published state.
///
/// The revision increments on every observable state change, so a caller can
/// tell a fresh snapshot from a stale one without holding the session lock.
#[derive(Debug, Clone)]
pub struct WorkObservation {
    /// The observed work.
    pub work: WorkRef,
    /// Monotonic revision of this work's published state, starting at `0` for
    /// `Accepted`.
    pub revision: u64,
    /// The current management state.
    pub state: WorkState,
    /// The terminal result when `state` is [`WorkState::Finished`]; `None`
    /// otherwise.
    pub finished: Option<FinishedKind>,
    /// The fault reason when `state` is [`WorkState::Faulted`]; `None`
    /// otherwise.
    pub fault: Option<String>,
}

/// Why a [`SessionHandle::wait`](crate::session::SessionHandle::wait) returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitEnd {
    /// The wait observed a returnable state (`Paused` / `Finished` /
    /// `Faulted`).
    ReachedState,
    /// The wait's finite timeout elapsed first. The observed state is
    /// unchanged by the timeout.
    TimedOut,
}

/// The result of a finite wait: one consistent snapshot plus why the wait
/// ended.
///
/// Observing `Paused` / `Finished` / `Faulted`, or timing out, is a
/// *successful* observation — it never mutates the observed work's state.
#[derive(Debug, Clone)]
pub struct WaitOutcome {
    /// The snapshot taken when the wait ended.
    pub observation: WorkObservation,
    /// Whether the wait reached a returnable state or timed out.
    pub end: WaitEnd,
}

/// Session coordination configuration — not a mirror of `TurnRunOptions`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionConfig {
    /// Maximum number of works the session retains. `submit` is rejected with
    /// [`SessionError::CapacityExceeded`] at the ceiling; already-retained
    /// works stay observable.
    pub retained_work_capacity: usize,
    /// Per-work deadline, measured from acceptance (pauses included). `None`
    /// leaves the existing turn / tool limits as the only bounds. This is wired
    /// into the work's `RunControl`; deadline enforcement itself stays with the
    /// driver.
    pub work_deadline: Option<Duration>,
}

impl Default for SessionConfig {
    /// The offline reference defaults: retain 256 works, no work deadline.
    fn default() -> Self {
        Self {
            retained_work_capacity: 256,
            work_deadline: None,
        }
    }
}

/// Rejections of the session operations.
///
/// Request rejections (busy, input, conflict, capacity) are distinct from
/// internal faults: they leave no accepted work behind, and they do not
/// consume a request key.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SessionError {
    /// The referenced work does not belong to this session (unknown or
    /// foreign).
    #[error("no such work in this session: {0:?}")]
    NotFound(WorkRef),
    /// The session already holds an active work; no queue and no steering.
    #[error("session is busy with an active work: {active:?}")]
    Busy {
        /// The active work the caller must observe, resume, or wait for.
        active: WorkRef,
    },
    /// The request key was reused with different arguments.
    #[error("request key reused with different arguments")]
    Conflict,
    /// The retained-work capacity is exhausted.
    #[error("session retained-work capacity is exhausted")]
    CapacityExceeded,
    /// The submitted input is unusable (e.g. empty parts).
    #[error("invalid submit input: {0}")]
    InvalidInput(String),
    /// The owner has been dropped (or the session closed); no new work is
    /// accepted. Retained results remain readable.
    #[error("session is closed and accepts no new work")]
    Closed,
    /// The worker faulted; the session refuses new work until the harness
    /// handles it.
    #[error("session worker faulted: {reason}")]
    Faulted {
        /// The retained fault reason.
        reason: String,
    },
}
