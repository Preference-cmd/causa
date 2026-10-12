use crate::RunObserver;
use causa_kernel::{CacheDirective, ContextPreparer, GenerationOptions, ModelRef};
use std::sync::Arc;

/// Numeric limits for one execution.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TurnLimits {
    /// Maximum number of logical model invocations, including the first.
    pub max_model_rounds: u32,
    /// Maximum cumulative number of model tool declarations.
    pub max_tool_calls: u32,
}
impl TurnLimits {
    /// Limits on model rounds and on tool calls for one run.
    pub fn new(max_model_rounds: u32, max_tool_calls: u32) -> Self {
        Self {
            max_model_rounds,
            max_tool_calls,
        }
    }
}
impl Default for TurnLimits {
    fn default() -> Self {
        Self {
            max_model_rounds: 10,
            max_tool_calls: 64,
        }
    }
}

/// Choices fixed for one run; the model must be supplied explicitly.
#[derive(Clone)]
#[non_exhaustive]
pub struct TurnRunOptions {
    /// Model used by every logical invocation.
    pub model: ModelRef,
    /// Generation parameters used by every logical invocation.
    pub generation: GenerationOptions,
    /// Provider cache instruction.
    pub cache: CacheDirective,
    /// Execution limits.
    pub limits: TurnLimits,
    /// Optional exclusive material preparation before each invocation.
    pub preparer: Option<Arc<dyn ContextPreparer>>,
    /// Optional synchronous borrowed observation callback.
    pub observer: Option<Arc<RunObserver>>,
}
impl TurnRunOptions {
    /// Select a model with default generation, limits and no preparation or observer.
    pub fn new(model: ModelRef) -> Self {
        Self {
            model,
            generation: GenerationOptions::default(),
            cache: CacheDirective::None,
            limits: TurnLimits::default(),
            preparer: None,
            observer: None,
        }
    }
}
