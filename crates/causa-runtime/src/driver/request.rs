//! Prepare request material using the invocation's already-bound tool surface.

use super::TurnInterruption;
use super::model::{control_cause, stopped};
use crate::TurnRunOptions;
use crate::control::wait_for_stop;
use causa_kernel::{CallControl, Context, InvocationId, ModelRequest, RoundInfo, ToolSurface};

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
        prepared.map_err(|error| TurnInterruption::PrepareFailed {
            invocation_id: invocation_id.clone(),
            error,
        })?
    } else {
        context.frame()
    };
    Ok({
        let mut request = ModelRequest::new(invocation_id.clone(), frame, options.model.clone());
        request.tool_surface = tool_surface.clone();
        request.generation = options.generation.clone();
        request.cache = options.cache;
        request
    })
}
