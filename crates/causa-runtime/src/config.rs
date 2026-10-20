//! Per-run choices for the minimal runner.

use crate::RunObserver;
use causa_kernel::{CacheDirective, ContextPreparer, GenerationOptions, ModelRef, ProcessorData};
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
    /// Optional host data handed to every tool processor of the run, through
    /// [`causa_kernel::ProcessorContext::data`]. Passed through unchanged:
    /// state that changes during the run belongs in a shared handle.
    pub processor_data: Option<ProcessorData>,
}
impl TurnRunOptions {
    /// Select a model with default generation, limits and no preparation,
    /// observer or processor data.
    pub fn new(model: ModelRef) -> Self {
        Self {
            model,
            generation: GenerationOptions::default(),
            cache: CacheDirective::None,
            limits: TurnLimits::default(),
            preparer: None,
            observer: None,
            processor_data: None,
        }
    }

    /// Replaces the generation parameters used by every invocation.
    pub fn with_generation(mut self, generation: GenerationOptions) -> Self {
        self.generation = generation;
        self
    }

    /// Replaces the provider cache instruction.
    pub fn with_cache(mut self, cache: CacheDirective) -> Self {
        self.cache = cache;
        self
    }

    /// Replaces the execution limits.
    pub fn with_limits(mut self, limits: TurnLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets the exclusive material preparation before each invocation.
    pub fn with_preparer(mut self, preparer: Arc<dyn ContextPreparer>) -> Self {
        self.preparer = Some(preparer);
        self
    }

    /// Sets the synchronous borrowed observation callback.
    pub fn with_observer(mut self, observer: Arc<RunObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Sets the host data handed to every tool processor of the run.
    pub fn with_processor_data(mut self, processor_data: ProcessorData) -> Self {
        self.processor_data = Some(processor_data);
        self
    }
}
