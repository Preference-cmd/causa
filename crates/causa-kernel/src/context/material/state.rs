//! Controlled context material and independent request frames.

mod edit;
mod serde_impl;
mod value;

pub use edit::{ContextEdit, EditError, EditFailure, Replacement};
pub use value::{Context, ContextFrame};
