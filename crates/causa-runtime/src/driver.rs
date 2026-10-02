//! Single-run material ownership and logical model execution.
mod model;
mod outcome;
mod runner;
pub use outcome::{TurnInterruption, TurnOutcome, TurnResult};
pub use runner::TurnRunner;
