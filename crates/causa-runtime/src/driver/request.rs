//! Prepare request material using the invocation's already-bound tool surface.
//!
//! The bound tools reach the model through the frame: the runner records a
//! difference as a `ToolChange` block before preparation, and a prepared
//! frame must still add up to the bound tools.

use super::TurnInterruption;
use super::model::{control_cause, stopped};
use crate::TurnRunOptions;
use crate::control::wait_for_stop;
use causa_kernel::{
    CallControl, Context, InvocationId, ModelRequest, PrepareError, RoundInfo, ToolChange,
    ToolSurface, current_tools,
};

// TurnInterruption intentionally owns failed inputs for caller recovery.
pub(super) async fn prepare(
    context: &mut Context,
    invocation_id: &InvocationId,
    tool_surface: &ToolSurface,
    options: &TurnRunOptions,
    control: &CallControl,
) -> Result<ModelRequest, TurnInterruption> {
    let frame = if let Some(preparer) = &options.preparer {
        let round_info = RoundInfo::new(
            invocation_id,
            &options.model,
            tool_surface,
            &options.generation,
        );
        let call_control = control.clone();

        let prepared = tokio::select! {
            biased;
            error = wait_for_stop(control) => return Err(control_cause(error, Some(invocation_id.clone()))),
            result = preparer.prepare(context, &round_info, &call_control) => result,
        };
        if let Some(cause) = stopped(control, Some(invocation_id)) {
            return Err(cause);
        }
        let frame = prepared.map_err(|error| TurnInterruption::PrepareFailed {
            invocation_id: invocation_id.clone(),
            error,
        })?;
        // The frame is what renders, tools included: a frame that dropped or
        // altered the context's tool changes would offer the model other
        // tools than the ones this round can execute.
        if !ToolChange::between(&current_tools(&frame.blocks), &tool_surface.definitions).is_empty()
        {
            return Err(TurnInterruption::PrepareFailed {
                invocation_id: invocation_id.clone(),
                error: PrepareError::new(
                    "the prepared frame's tool changes do not add up to the round's bound tools",
                ),
            });
        }
        frame
    } else {
        context.frame()
    };
    Ok({
        let mut request = ModelRequest::new(invocation_id.clone(), frame, options.model.clone());
        request.generation = options.generation.clone();
        request.cache = options.cache;
        request
    })
}
