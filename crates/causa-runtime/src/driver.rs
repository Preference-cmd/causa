//! Single-run material ownership and logical model execution.
mod commit;
mod cycle;
mod model;
mod outcome;
mod request;
mod runner;
pub use outcome::{TurnInterruption, TurnOutcome, TurnResult};
pub use runner::TurnRunner;
