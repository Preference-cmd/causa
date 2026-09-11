//! The cloneable operation entry — [`SessionHandle`] — for one session.
//!
//! A handle shares the session's private execution core but never owns the
//! lifecycle and never exposes the writable conversation state.

use std::sync::Arc;
use std::time::Duration;

use causa_kernel::ConversationId;

use super::execution::SessionCore;
use super::types::{
    SessionError, SubmitRequest, WaitOutcome, WorkObservation, WorkReceipt, WorkRef,
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
    /// # Errors
    ///
    /// See [`SessionError`]. Requires a Tokio runtime (the worker is spawned).
    pub async fn submit(&self, request: SubmitRequest) -> Result<WorkReceipt, SessionError> {
        self.core.submit(request)
    }

    /// Read one work's current published state.
    ///
    /// Never blocks on the runner and never mutates the work.
    ///
    /// # Errors
    ///
    /// [`SessionError::NotFound`] if the ref does not belong to this session.
    pub async fn observe(&self, work: &WorkRef) -> Result<WorkObservation, SessionError> {
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
}
