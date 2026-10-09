//! Optional execution components over the Causa material kernel.
//!
//! [`TurnRunner`] owns only a gateway and [`ToolExecutor`]. Each run consumes
//! a context and returns it in [`TurnOutcome`], with any uncommitted tool batch.
//! Model identity, generation, limits, optional preparation and observation
//! are explicit per-run choices. Both entry points share the same loop.
//!
//! [`ToolExecutor::bind`] fixes the actual tool surface and targets before
//! preparation. Processing owns controlled collection of started calls;
//! the runner validates and atomically appends results only after success.
//! Unknown results are committed and stop further model rounds.
//!
//! # Policy surface
//!
//! | Choice | Default and owner |
//! |---|---|
//! | Model rounds / tool declarations | [`TurnLimits`]: 10 / 64 |
//! | Model retries | None; optional [`RetryingGateway`] around the chosen gateway |
//! | Material preparation | None; use the current context frame |
//! | Observation | None; an optional synchronous borrowed callback |
//! | Tool processing | Executor-owned ordered before/after processors |
//! | Tool dispatch | Concurrent pending calls; panic becomes failed result |
//! | Tool timeout | None; local timeout becomes unknown outcome |
//! | Unknown outcomes | Commit the successful batch, then stop |
//! | Output retention and persistence | Caller-owned policies and storage |
//!
//! # Optional retries
//!
//! Retries are off by default: the runner makes one logical gateway call per
//! round and the first error ends it. A host that wants bounded retries wraps
//! its gateway in [`RetryingGateway`] with a [`RetryPolicy`]. The wrapper
//! is an ordinary [`causa_kernel::ModelGateway`]: every attempt sends the
//! same request, only `Transient` and `TimedOut` failures are retried, waits
//! stay inside the caller's cancellation and deadline, and the call is never
//! retried once any stream delta has been yielded. Only the final error
//! reaches the runner.
//!
//! The runner does not interpret imported tool declarations as work to resume.
//! It creates work only from the current model output. Session admission,
//! history, input queues, retries, budget strategies and observation storage
//! belong to callers. Dropping a run future or panicking bypasses the normal
//! material-return guarantee; stopping local waits cannot undo remote effects.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod composition;
pub mod config;
pub mod control;
pub mod driver;
pub mod event;
pub mod executor;
pub mod ids;
mod processors;
mod retry;

pub use composition::ToolBridge;
pub use config::{TurnLimits, TurnRunOptions};
pub use control::RunControl;
pub use driver::{TurnInterruption, TurnOutcome, TurnResult, TurnRunner};
pub use event::{RunEvent, RunObserver};
pub use executor::{
    BoundTools, ToolCatalogError, ToolExecutor, ToolExecutorOptions, ToolProcessingError,
    ToolProcessorPhase, ToolRegistryError,
};
pub use ids::new_block_id;
pub use retry::{RetryPolicy, RetryingGateway};
