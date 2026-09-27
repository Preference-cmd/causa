//! Turn fact state and lossless model-context projections.

mod state;

pub use state::{
    AppliedModelOutput, ContextError, ContextFrame, ModelContext, OrderedBlocks, TurnContext,
    TurnLifecycle, TurnSnapshot, merged_frame, model_output_block_count,
    option_turn_context_as_snapshot, turn_context_as_snapshot,
};
