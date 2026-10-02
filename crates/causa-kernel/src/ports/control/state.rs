//! Cooperative cancellation and absolute deadline control for ports.

use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Cooperative control for a logical model invocation, preparation or tool work.
#[derive(Debug, Clone)]
pub struct CallControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

/// Why a control check failed.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ControlError {
    /// The turn was cancelled.
    #[error("cancelled")]
    Cancelled,
    /// The effective deadline has passed.
    #[error("deadline exceeded")]
    TimedOut,
}
impl CallControl {
    /// Builds control from a cancellation token and an absolute deadline.
    pub fn new(cancellation: CancellationToken, deadline: Option<Instant>) -> Self {
        Self {
            cancellation,
            deadline,
        }
    }

    /// Inherits cancellation and narrows the deadline by a local timeout.
    ///
    /// The local clock starts now. An unrepresentably distant timeout adds no
    /// earlier constraint, and this operation never cancels the parent token.
    pub fn with_timeout(&self, timeout: Duration) -> Self {
        let local = Instant::now().checked_add(timeout);
        let deadline = match (self.deadline, local) {
            (Some(parent), Some(local)) => Some(parent.min(local)),
            (parent, None) => parent,
            (None, local) => local,
        };
        Self::new(self.cancellation.clone(), deadline)
    }

    /// Whether the turn's cancellation has fired.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    /// The effective call deadline, if any.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    /// The turn-shared primitive — race it against in-flight tool work
    /// instead of polling.
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
    /// Sync guard for tool loops: `Err` when the turn is cancelled or the
    /// effective deadline has passed.
    pub fn check(&self) -> Result<(), ControlError> {
        if self.cancellation.is_cancelled() {
            return Err(ControlError::Cancelled);
        }
        if self.deadline.map(|d| Instant::now() >= d).unwrap_or(false) {
            return Err(ControlError::TimedOut);
        }
        Ok(())
    }
}
