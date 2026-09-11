//! The cloneable operation entry — [`SessionHandle`] — for one session.
//!
//! A handle shares the session's private execution core but never owns the
//! lifecycle and never exposes the writable conversation state.

use std::sync::Arc;
use std::time::Duration;

use causa_kernel::ConversationId;

use crate::resume::ResumeRequest;

use super::execution::SessionCore;
use super::types::{
    CancelReceipt, SessionError, SubmitRequest, WaitOutcome, WorkObservation, WorkReceipt, WorkRef,
};

/// A cloneable operation entry for one [`Session`](crate::session::Session).
///
/// Handles never own the lifecycle: dropping the last handle does not close
/// the session, and after the owner is dropped they can still read retained
/// observations while new submissions are refused with
/// [`SessionError::Closed`].
#[derive(Clone)]
pub struct SessionHandle {
    pub(super) core: Arc<SessionCore>,
}

impl std::fmt::Debug for SessionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionHandle")
            .field("conversation_id", &self.core.conversation_id())
            .finish_non_exhaustive()
    }
}

impl SessionHandle {
    /// The conversation this handle operates on.
    pub fn id(&self) -> ConversationId {
        self.core.conversation_id()
    }

    /// Submit one new work.
    ///
    /// Accepts on an idle session, atomically allocates a fresh turn id,
    /// writes the request parts as the turn's first fact block, and spawns the
    /// worker. Rejections (busy, conflict, capacity, invalid input, closed,
    /// faulted) leave no accepted work behind and do not consume the request
    /// key. Idempotent on `request_key` + parts: a repeat returns the original
    /// receipt without executing again.
    ///
    /// Synchronous: admission never blocks on the runner. Spawning the worker
    /// still requires a Tokio runtime context.
    ///
    /// # Errors
    ///
    /// See [`SessionError`].
    pub fn submit(&self, request: SubmitRequest) -> Result<WorkReceipt, SessionError> {
        self.core.submit(request)
    }

    /// Read one work's current published state.
    ///
    /// Never blocks on the runner and never mutates the work.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotFound`] if the ref does not belong to this session.
    pub fn observe(&self, work: &WorkRef) -> Result<WorkObservation, SessionError> {
        self.core.observe(work)
    }

    /// Wait finitely for a work to reach `Paused` / `Finished` / `Faulted`.
    ///
    /// Returns as soon as one of those states is observed, or `TimedOut` with
    /// a consistent snapshot when `timeout` elapses first; a wakeup is never
    /// lost, and timing out does not change the work's state.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotFound`] if the ref does not belong to this session.
    pub async fn wait(
        &self,
        work: &WorkRef,
        timeout: Duration,
    ) -> Result<WaitOutcome, SessionError> {
        self.core.wait(work, timeout).await
    }

    /// Continue a paused work.
    ///
    /// Validates the ref, the work's current paused revision, and the request
    /// against the paused continuation before anything runs: a rejected request
    /// executes nothing and leaves the paused material untouched. On acceptance
    /// the work returns to `Running` in the same turn id and continues under the
    /// deadline measured from its original acceptance. Idempotent on
    /// `request_key` + `(work, expected_revision, request)`: a same-key repeat
    /// returns the original receipt without re-executing, while the same key
    /// with a different decision or injection is [`SessionError::Conflict`].
    ///
    /// Synchronous: admission never blocks on the runner.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotFound`] for a foreign or unknown ref,
    /// [`SessionError::NotPaused`] when the work is not paused,
    /// [`SessionError::StaleRevision`] when `expected_revision` is not the
    /// work's current paused revision, [`SessionError::Busy`] while another work
    /// owns the conversation, [`SessionError::InvalidResume`] when the driver
    /// rejects the request, [`SessionError::Conflict`] on a key reuse with
    /// different arguments, and [`SessionError::Closed`] after shutdown.
    pub fn resume(
        &self,
        work: &WorkRef,
        expected_revision: u64,
        request_key: String,
        request: ResumeRequest,
    ) -> Result<WorkReceipt, SessionError> {
        self.core
            .resume(work, expected_revision, request_key, request)
    }

    /// Cancel one work.
    ///
    /// Fires the work's own control token while it runs — a completion that
    /// races the cancel keeps its result — or terminates a paused work in
    /// place, retaining its committed facts and continuation. A work
    /// that already finished is never rewritten. Idempotent on `request_key`: a
    /// same-key retry returns the original receipt.
    ///
    /// Synchronous: admission never blocks on the runner.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotFound`] for a foreign, unknown, or already-cleared
    /// ref, [`SessionError::Busy`] while another work owns the conversation,
    /// [`SessionError::Conflict`] on a key reuse with a different work, and
    /// [`SessionError::Closed`] after shutdown.
    pub fn cancel(
        &self,
        work: &WorkRef,
        request_key: String,
    ) -> Result<CancelReceipt, SessionError> {
        self.core.cancel(work, request_key)
    }
}
