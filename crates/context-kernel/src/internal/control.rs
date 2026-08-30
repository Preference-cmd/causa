//! Turn-level run control — the staged driver's bundling of the shared
//! cancellation token and the turn deadline. No port signature consumes this
//! type (the gateway takes `AttemptControl`, tools take `CallControl`), which
//! is why it lives in the staged perimeter rather than `ports`.

use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use crate::ports::control::{AttemptControl, effective_deadline};

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
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
    /// True when the turn is cancelled or the turn deadline has passed — the
    /// driver checks this at every loop top.
    pub fn should_stop(&self) -> bool {
        self.is_cancelled()
            || self
                .turn_deadline
                .map(|d| Instant::now() >= d)
                .unwrap_or(false)
    }
    /// Narrow the turn-level control into an attempt-scoped one: the shared
    /// token plus the attempt deadline (earlier of turn deadline and
    /// attempt timeout).
    pub fn for_attempt(&self, attempt_timeout: Option<Duration>) -> AttemptControl {
        AttemptControl::new(
            self.cancellation.clone(),
            effective_deadline(self.turn_deadline, attempt_timeout),
        )
    }
}
