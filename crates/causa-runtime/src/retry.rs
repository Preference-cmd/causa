//! Optional retrying wrapper around any `ModelGateway`.
//!
//! Off by default: the runner still makes one logical gateway call per
//! invocation, and a host that wants bounded retries hands the runner a
//! [`RetryingGateway`] wrapping its gateway. Nothing else changes — no new
//! trait, no runner option, no implicit wrapper.
//!
//! The wrapper follows the kernel contract and the caller's control plane:
//! every attempt sends the same request value; only `Transient` and
//! `TimedOut` failures are retried; a stream is never retried once any
//! delta has been handed to the caller; backoff stays inside the caller's
//! cancellation and deadline; and only the final error reaches the runner.

mod gateway;
mod policy;

pub use gateway::RetryingGateway;
pub use policy::RetryPolicy;
