//! Retry policy for the optional gateway wrapper.

use std::time::Duration;

/// Bounded retry policy for [`RetryingGateway`](super::RetryingGateway).
///
/// There is no `Default` and no recommended attempt count: retry cost is
/// host policy, so the host states every value. The durations a fresh policy
/// starts from are the ones the compared references use — a 500 ms base
/// delay, an 8 s backoff cap and a 60 s server-delay ceiling (`pi-ai`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total calls including the first; at least 1.
    pub max_attempts: u32,
    /// Backoff before the second attempt; it doubles per failed attempt.
    pub base_delay: Duration,
    /// Ceiling on the wrapper's own backoff. A server-requested delay is not
    /// capped by this, only by [`RetryPolicy::max_server_delay`].
    pub max_delay: Duration,
    /// Ceiling on a server-requested delay. A longer request fails the call
    /// at once instead of waiting.
    pub max_server_delay: Duration,
}

impl RetryPolicy {
    /// A policy allowing `max_attempts` calls in total — the first included,
    /// so `new(1)` never retries — starting from a 500 ms base delay, an 8 s
    /// backoff cap and a 60 s server-delay ceiling. Values below 1 are raised
    /// to 1.
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(8),
            max_server_delay: Duration::from_secs(60),
        }
    }

    /// Sets [`RetryPolicy::base_delay`].
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Sets [`RetryPolicy::max_delay`].
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Sets [`RetryPolicy::max_server_delay`].
    pub fn with_max_server_delay(mut self, delay: Duration) -> Self {
        self.max_server_delay = delay;
        self
    }

    /// The wrapper's own backoff after the `attempt`-th failure (`attempt`
    /// from 1): `min(base_delay * 2^(attempt - 1), max_delay)` shortened by a
    /// uniform random factor in `[0.75, 1.0]`. Jitter is fixed, not
    /// configurable, and never applies to a server-requested delay.
    pub(super) fn backoff(&self, attempt: u32) -> Duration {
        let doubling = 1u32 << attempt.saturating_sub(1).min(31);
        let capped = self.base_delay.saturating_mul(doubling).min(self.max_delay);
        jittered(capped, 0.75 + 0.25 * fastrand::f64())
    }
}

/// Shorten `delay` by `factor` without a panic on extreme host-set caps.
fn jittered(delay: Duration, factor: f64) -> Duration {
    Duration::from_nanos(((delay.as_nanos() as f64) * factor) as u64)
}
