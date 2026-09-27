//! Cloneable submit, observe, wait, and cancel entry into a session.

use std::sync::Arc;
use std::time::Duration;

use causa_kernel::ConversationId;

use super::execution::SessionCore;
use super::types::{
    CancelReceipt, SessionError, SubmitRequest, WaitOutcome, WorkObservation, WorkReceipt, WorkRef,
};

/// A cloneable operation entry for one [`Session`](crate::session::Session).
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

    /// Submit one new work. Accepted keys replay the original receipt; a key
    /// reused with different parts returns [`SessionError::Conflict`].
    pub fn submit(&self, request: SubmitRequest) -> Result<WorkReceipt, SessionError> {
        self.core.submit(request)
    }

    /// Read one work's current published state.
    pub fn observe(&self, work: &WorkRef) -> Result<WorkObservation, SessionError> {
        self.core.observe(work)
    }

    /// Wait finitely for a work to finish or fault. A timeout returns the
    /// current consistent observation without changing the work.
    pub async fn wait(
        &self,
        work: &WorkRef,
        timeout: Duration,
    ) -> Result<WaitOutcome, SessionError> {
        self.core.wait(work, timeout).await
    }

    /// Cancel one work. A running work receives its own cancellation signal;
    /// an already terminal result is left unchanged. Request keys replay the
    /// original cancel receipt.
    pub fn cancel(
        &self,
        work: &WorkRef,
        request_key: String,
    ) -> Result<CancelReceipt, SessionError> {
        self.core.cancel(work, request_key)
    }
}
