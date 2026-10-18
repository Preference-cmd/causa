//! The direct stream consumer on the streaming page.

// ANCHOR: consume
use causa::kernel::{
    CallControl, ModelGateway, ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRequest,
    StreamDelta,
};
use futures_util::StreamExt;

pub async fn consume(
    gateway: &dyn ModelGateway,
    request: &ModelRequest,
    control: &CallControl,
) -> Result<ModelOutput, ModelInvokeError> {
    let mut stream = gateway.stream(request, control).await?;
    while let Some(delta) = stream.next().await {
        match delta {
            StreamDelta::TextDelta { delta } => print!("{delta}"),
            StreamDelta::Done { final_output } => return Ok(final_output),
            // The in-band failure carries the gateway's own error, so a retry
            // wrapper can see its classification and any server-requested delay.
            StreamDelta::Error { error } => return Err(error),
            _ => {}
        }
    }
    Err(ModelInvokeError::new(
        ModelInvokeErrorKind::UnknownOutcome,
        "stream ended without a complete output",
    ))
}
// ANCHOR_END: consume
