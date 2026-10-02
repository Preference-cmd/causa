//! Controlled material and lossless request projections.

mod state;
mod tool_results;

pub use state::{Context, ContextEdit, ContextFrame, EditError, EditFailure, Replacement};
pub use tool_results::{ToolResultError, validate_tool_result_append};
