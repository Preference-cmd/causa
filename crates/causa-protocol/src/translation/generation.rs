//! Shared generation-option checks and pass-through before wire rendering.
//!
//! The checks cover what the adapter itself needs to render a request. The
//! reasoning effort and the extra request fields are not checked: they are
//! written as the host gave them.

mod options;

pub(super) use options::{merge_extra_body, validate_anthropic, validate_openai};
