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
//! | Model retries | Gateway implementation or caller wrapper |
//! | Material preparation | None; use the current context frame |
//! | Observation | None; an optional synchronous borrowed callback |
//! | Tool processing | Executor-owned ordered before/after processors |
//! | Tool dispatch | Concurrent pending calls; panic becomes failed result |
//! | Tool timeout | None; local timeout becomes unknown outcome |
//! | Unknown outcomes | Commit the successful batch, then stop |
//! | Output retention and persistence | Caller-owned policies and storage |
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
