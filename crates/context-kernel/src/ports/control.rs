//! Control planes — the cooperative-cancellation kit handed to port
//! implementors. `AttemptControl` rides on `ModelGateway::invoke`,
//! `CallControl` on `Tool::execute`; both carry the turn-shared
//! `CancellationToken` plus a deadline chain (turn → attempt → call). The
//! turn-level bundling (`RunControl`) is staged driver vocabulary and lives
//! in `crate::internal::control`, not here: no port signature consumes it.

use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Fold a parent deadline and a timeout into the effective deadline: the
/// earlier of the two, where "no parent"/"no timeout" each leave the other
/// side in force.
pub(crate) fn effective_deadline(
    parent: Option<Instant>,
    timeout: Option<Duration>,
) -> Option<Instant> {
    let from_timeout = timeout.map(|t| Instant::now() + t);
    match (parent, from_timeout) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[derive(Debug, Clone)]
pub struct AttemptControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}
impl AttemptControl {
    /// Staged constructors only (`internal::control::RunControl`); external
    /// drivers build attempt controls through the root-exported
    /// `RunControl::for_attempt` chain.
    pub(crate) fn new(cancellation: CancellationToken, deadline: Option<Instant>) -> Self {
        Self {
            cancellation,
            deadline,
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    /// The turn-shared primitive — race it against in-flight provider work
    /// (`select!`) instead of polling.
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
    pub fn for_call(&self, call_timeout: Option<Duration>) -> CallControl {
        CallControl {
            cancellation: self.cancellation.clone(),
            deadline: effective_deadline(self.deadline, call_timeout),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CallControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ControlError {
    #[error("cancelled")]
    Cancelled,
    #[error("deadline exceeded")]
    TimedOut,
}
impl CallControl {
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
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
