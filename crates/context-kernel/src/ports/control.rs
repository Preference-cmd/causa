use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct RunControl {
    cancellation: CancellationToken,
    turn_deadline: Option<Instant>,
}
impl RunControl {
    pub fn new(cancellation: CancellationToken, turn_deadline: Option<Instant>) -> Self {
        Self {
            cancellation,
            turn_deadline,
        }
    }
    pub fn with_deadline(cancellation: CancellationToken, deadline: Instant) -> Self {
        Self {
            cancellation,
            turn_deadline: Some(deadline),
        }
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    pub fn turn_deadline(&self) -> Option<Instant> {
        self.turn_deadline
    }
    pub fn is_deadline_exceeded(&self) -> bool {
        self.turn_deadline
            .map(|d| Instant::now() >= d)
            .unwrap_or(false)
    }
    pub fn should_stop(&self) -> bool {
        self.is_cancelled() || self.is_deadline_exceeded()
    }
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
    pub fn for_attempt(&self, attempt_timeout: Option<Duration>) -> AttemptControl {
        let deadline = Self::effective_deadline(self.turn_deadline, attempt_timeout);
        AttemptControl {
            cancellation: self.cancellation.clone(),
            deadline,
        }
    }
    fn effective_deadline(parent: Option<Instant>, timeout: Option<Duration>) -> Option<Instant> {
        let from_timeout = timeout.map(|t| Instant::now() + t);
        match (parent, from_timeout) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AttemptControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}
impl AttemptControl {
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
    pub fn is_deadline_exceeded(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
    pub fn for_call(&self, call_timeout: Option<Duration>) -> CallControl {
        let deadline = RunControl::effective_deadline(self.deadline, call_timeout);
        CallControl {
            cancellation: self.cancellation.clone(),
            deadline,
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
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
    pub fn is_deadline_exceeded(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }
    pub fn check(&self) -> Result<(), ControlError> {
        if self.is_cancelled() {
            return Err(ControlError::Cancelled);
        }
        if self.is_deadline_exceeded() {
            return Err(ControlError::TimedOut);
        }
        Ok(())
    }
    pub async fn check_cancelled(&self) -> Result<(), ControlError> {
        tokio::select! {
            _ = self.cancellation.cancelled() => Err(ControlError::Cancelled),
            _ = async {
                if let Some(d) = self.deadline { tokio::time::sleep_until(d.into()).await; } else { std::future::pending::<()>().await; }
            } => Err(ControlError::TimedOut),
        }
    }
}
