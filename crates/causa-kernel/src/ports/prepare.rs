//! Context preparation contract for each logical model round.
//!
//! Exclusive material preparation and its actual invocation configuration.

use crate::{
    CallControl, Context, ContextFrame, GenerationOptions, InvocationId, ModelRef, ToolSurface,
};
use async_trait::async_trait;

/// Configuration visible while preparing this invocation's material.
#[non_exhaustive]
pub struct RoundInfo<'a> {
    /// The actual logical model invocation being prepared.
    pub invocation_id: &'a InvocationId,
    /// The actual model selected by the caller.
    pub model: &'a ModelRef,
    /// The bound tools offered for this invocation.
    pub tool_surface: &'a ToolSurface,
    /// The generation settings selected by the caller.
    pub generation: &'a GenerationOptions,
}
impl<'a> RoundInfo<'a> {
    /// The facts of the round about to be invoked.
    pub fn new(
        invocation_id: &'a InvocationId,
        model: &'a ModelRef,
        tool_surface: &'a ToolSurface,
        generation: &'a GenerationOptions,
    ) -> Self {
        Self {
            invocation_id,
            model,
            tool_surface,
            generation,
        }
    }
}

/// Failure to prepare a request's material.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
#[non_exhaustive]
pub struct PrepareError {
    /// Human-readable failure detail.
    pub message: String,
}
impl PrepareError {
    /// A preparation failure with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Prepares an independent request frame with exclusive access to context.
///
/// Each context edit commits independently; edits already committed remain if
/// preparation subsequently fails. The returned frame is not written back to
/// context. Input consumption and recovery belong to the implementation.
#[async_trait]
pub trait ContextPreparer: Send + Sync {
    /// Selects and optionally edits material using actual invocation settings.
    async fn prepare(
        &self,
        context: &mut Context,
        round: &RoundInfo<'_>,
        control: &CallControl,
    ) -> Result<ContextFrame, PrepareError>;
}
