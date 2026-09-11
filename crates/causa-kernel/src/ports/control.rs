//! Control planes — the cooperative-cancellation kit handed to port
//! implementors. `AttemptControl` rides on `ModelGateway::invoke`,
//! `CallControl` on `Tool::execute`; both carry the turn-shared
//! `CancellationToken` plus a deadline chain (turn → attempt → call). The
//! turn-level bundling (`RunControl`) is reference-driver vocabulary and
//! lives in
//! `causa_runtime::control`, not here: no port signature consumes it.

use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Fold a parent deadline and a timeout into the effective deadline: the
/// earlier of the two, where "no parent"/"no timeout" each leave the other
/// side in force. External drivers narrow turn deadlines into attempt
/// deadlines through this fold.
pub fn effective_deadline(parent: Option<Instant>, timeout: Option<Duration>) -> Option<Instant> {
    let from_timeout = timeout.map(|t| Instant::now() + t);
    match (parent, from_timeout) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// One model attempt's control plane: the turn-shared [`CancellationToken`]
/// plus the effective attempt deadline (turn deadline folded with the
/// attempt timeout).
#[derive(Debug, Clone)]
pub struct AttemptControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}
impl AttemptControl {
    /// Builds an attempt control. External drivers construct these directly
    /// — e.g. through the `causa_runtime::control::RunControl::for_attempt`
    /// chain or from a bare token.
    pub fn new(cancellation: CancellationToken, deadline: Option<Instant>) -> Self {
        Self {
            cancellation,
            deadline,
        }
    }
    /// Whether the turn's cancellation has fired.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    /// The effective attempt deadline, if any.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    /// The turn-shared primitive — race it against in-flight provider work
    /// (`select!`) instead of polling.
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
    /// Narrows this attempt control into a [`CallControl`] for one tool
    /// call, folding `call_timeout` into the deadline chain.
    pub fn for_call(&self, call_timeout: Option<Duration>) -> CallControl {
        CallControl {
            cancellation: self.cancellation.clone(),
            deadline: effective_deadline(self.deadline, call_timeout),
        }
    }
}

/// One tool call's control plane: the same turn-shared [`CancellationToken`]
/// with the deadline narrowed to the call.
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
    /// Build a `CallControl` from a cancellation token and an optional
    /// duration-based deadline. Prefer `AttemptControl::for_call`, which
    /// folds the parent attempt deadline in; this constructor is for tests
    /// and callers that cannot go through it.
    pub fn new(cancellation: CancellationToken, call_timeout: Option<Duration>) -> Self {
        let deadline = call_timeout.map(|d| Instant::now() + d);
        Self {
            cancellation,
            deadline,
        }
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
