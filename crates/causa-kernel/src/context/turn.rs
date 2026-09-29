//! Turn fact state and lossless model-context projections.

mod state;

pub use state::{
    AppliedModelOutput, ContextError, ContextFrame, ModelContext, TurnContext, TurnLifecycle,
    merged_frame, model_output_block_count,
};
