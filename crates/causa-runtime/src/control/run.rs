use causa_kernel::CallControl;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// Shared cancellation and an optional absolute execution deadline.
#[derive(Debug, Clone)]
pub struct RunControl {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}
impl RunControl {
    /// Combine a caller-owned cancellation token and absolute deadline.
    pub fn new(cancellation: CancellationToken, deadline: Option<Instant>) -> Self {
        Self {
            cancellation,
            deadline,
        }
    }
    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    /// The caller-owned cancellation token.
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }
    /// The absolute execution deadline.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    /// Whether cancellation or the absolute deadline requires stopping.
    pub fn should_stop(&self) -> bool {
        self.is_cancelled() || self.deadline.is_some_and(|d| Instant::now() >= d)
    }
    /// Inherit the exact token and deadline without starting a local timeout.
    pub fn call_control(&self) -> CallControl {
        CallControl::new(self.cancellation.clone(), self.deadline)
    }
    /// Remaining execution time, saturating at zero.
    pub fn remaining_turn_time(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
}
