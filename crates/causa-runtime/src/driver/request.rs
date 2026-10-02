//! Prepare request material using the invocation's already-bound tool surface.

use super::TurnInterruption;
use super::model::{control_cause, stopped, wait_for_stop};
use crate::{RunControl, TurnRunOptions};
use causa_kernel::{Context, InvocationId, ModelRequest, RoundInfo, ToolSurface};

// TurnInterruption intentionally owns failed inputs for caller recovery.
#[allow(clippy::result_large_err)]
pub(super) async fn prepare(
    context: &mut Context,
    invocation_id: &InvocationId,
    tool_surface: &ToolSurface,
    options: &TurnRunOptions,
    control: &RunControl,
) -> Result<ModelRequest, TurnInterruption> {
    let frame = if let Some(preparer) = &options.preparer {
        let round_info = RoundInfo {
            invocation_id,
            model: &options.model,
            tool_surface,
            generation: &options.generation,
        };
        let call_control = control.call_control();
        let prepared = tokio::select! {
            biased;
            error = wait_for_stop(control) => return Err(control_cause(error, Some(invocation_id.clone()))),
            result = preparer.prepare(context, &round_info, &call_control) => result,
        };
        if let Some(cause) = stopped(control, Some(invocation_id)) {
            return Err(cause);
        }
        prepared.map_err(|error| TurnInterruption::PrepareFailed {
            invocation_id: invocation_id.clone(),
            error,
        })?
    } else {
        context.frame()
    };
    Ok(ModelRequest {
        invocation_id: invocation_id.clone(),
        frame,
        model: options.model.clone(),
        tool_surface: tool_surface.clone(),
        generation: options.generation.clone(),
        cache: options.cache,
    })
}
