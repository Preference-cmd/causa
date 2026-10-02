use super::TurnInterruption;
use crate::{RunControl, RunEvent, TurnRunOptions};
use causa_kernel::{
    ControlError, InvocationId, ModelGateway, ModelInvokeError, ModelInvokeErrorKind, ModelOutput,
    ModelRequest, StreamDelta,
};
use futures_util::StreamExt;

pub(super) fn notify(options: &TurnRunOptions, event: RunEvent<'_>) {
    if let Some(observer) = &options.observer {
        observer(&event);
    }
}

pub(super) fn control_cause(
    error: ControlError,
    invocation_id: Option<InvocationId>,
) -> TurnInterruption {
    match error {
        ControlError::Cancelled => TurnInterruption::Cancelled { invocation_id },
        ControlError::TimedOut => TurnInterruption::DeadlineExceeded { invocation_id },
    }
}
pub(super) fn stopped(
    control: &RunControl,
    invocation: Option<&InvocationId>,
) -> Option<TurnInterruption> {
    control
        .call_control()
        .check()
        .err()
        .map(|e| control_cause(e, invocation.cloned()))
}
pub(super) async fn wait_for_stop(control: &RunControl) -> ControlError {
    tokio::select! {
        biased;
        _ = control.cancellation_token().cancelled() => ControlError::Cancelled,
        _ = async {
            match control.deadline() {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        } => ControlError::TimedOut,
    }
}

// TurnInterruption intentionally owns failed inputs for caller recovery.
#[allow(clippy::result_large_err)]
#[tracing::instrument(name = "agent.round", skip_all, fields(turn_id = %request.invocation_id.turn_id.0, round_id = request.invocation_id.round_id.0, model = %request.model.0))]
pub(super) async fn invoke(
    gateway: &dyn ModelGateway,
    request: &ModelRequest,
    options: &TurnRunOptions,
    control: &RunControl,
    streaming: bool,
) -> Result<ModelOutput, TurnInterruption> {
    let call_control = control.call_control();
    let call = async {
        if !streaming {
            return gateway.invoke(request, &call_control).await;
        }
        let mut stream = gateway.stream(request, &call_control).await?;
        loop {
            // A synchronous observer may cancel while this future is still
            // ready. Do not advance the upstream stream after that boundary.
            call_control.check().map_err(stream_control_error)?;
            let Some(delta) = stream.next().await else {
                break;
            };
            call_control.check().map_err(stream_control_error)?;
            match delta {
                StreamDelta::Done { final_output, .. } => return Ok(final_output),
                StreamDelta::Error { kind, message } => {
                    return Err(ModelInvokeError::new(kind, message));
                }
                delta => notify(
                    options,
                    RunEvent::ModelDelta {
                        invocation_id: &request.invocation_id,
                        delta: &delta,
                    },
                ),
            }
        }
        Err(ModelInvokeError::new(
            ModelInvokeErrorKind::UnknownOutcome,
            "model stream ended without Done",
        ))
    };
    let output = tokio::select! {
        biased;
        error = wait_for_stop(control) => return Err(control_cause(error, Some(request.invocation_id.clone()))),
        result = call => result,
    };
    // A cooperative implementation may synchronously cancel while returning.
    if let Some(cause) = stopped(control, Some(&request.invocation_id)) {
        return Err(cause);
    }
    output.map_err(|error| TurnInterruption::ModelCallFailed {
        invocation_id: request.invocation_id.clone(),
        error,
    })
}

fn stream_control_error(error: ControlError) -> ModelInvokeError {
    ModelInvokeError::new(
        match error {
            ControlError::Cancelled => ModelInvokeErrorKind::Cancelled,
            ControlError::TimedOut => ModelInvokeErrorKind::TimedOut,
        },
        "execution control stopped stream",
    )
}
