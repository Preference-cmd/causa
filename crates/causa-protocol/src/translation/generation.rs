//! Shared generation-option validation before wire rendering.

mod options;

pub(super) use options::{validate_anthropic, validate_openai};
