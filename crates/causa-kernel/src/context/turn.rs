//! Turn fact state and lossless model-context projections.

mod state;

pub use state::{
    AppliedModelOutput, ContextEdit, ContextError, ContextFrame, EditError, EditFailure,
    ModelContext, Replacement, TurnContext, TurnLifecycle, merged_frame, model_output_block_count,
};
