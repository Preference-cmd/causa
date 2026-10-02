use causa_kernel::{
    ArtifactStore, CallControl, ControlError, Tool, ToolCallContext, ToolOutput, ToolOutputMeta,
    ToolResultPayload, ToolResultStatus, Truncation,
};
use futures_util::FutureExt;
use std::sync::Arc;
use tracing::Instrument;

pub(crate) async fn wait_for_stop(control: &CallControl) -> ControlError {
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

pub(super) async fn execute(
    tool: Arc<dyn Tool>,
    call: ToolCallContext,
    control: CallControl,
    store: Option<&dyn ArtifactStore>,
) -> ToolResultPayload {
    let span = tracing::info_span!(
        "agent.tool",
        tool_name = %call.input.tool_name,
        call_block_id = ?call.call_block_id
    );
    execute_inner(tool, &call, &control, store)
        .instrument(span)
        .await
}

async fn execute_inner(
    tool: Arc<dyn Tool>,
    call: &ToolCallContext,
    control: &CallControl,
    store: Option<&dyn ArtifactStore>,
) -> ToolResultPayload {
    // timeout_at can poll a ready future before an already-expired timer.
    // Check the newly tightened control before constructing or polling tools.
    if let Err(error) = control.check() {
        return unknown(call, &error.to_string(), "call_control_stopped");
    }
    // Construct the port future inside the catch as well: manually implemented
    // async-trait signatures can panic before they return a future.
    let future =
        std::panic::AssertUnwindSafe(async { tool.execute_with_store(call, control, store).await })
            .catch_unwind();
    let result = match control.deadline() {
        Some(deadline) => match tokio::time::timeout_at(deadline.into(), future).await {
            Ok(result) => result,
            Err(_) => {
                return unknown(
                    call,
                    "tool did not return before call deadline",
                    "call_deadline_backstop",
                );
            }
        },
        None => future.await,
    };
    match result {
        Ok(result) => result,
        Err(_) => ToolResultPayload {
            call_block_id: call.call_block_id,
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(serde_json::json!({"error": "tool panicked"})),
            media: Vec::new(),
            notes: Vec::new(),
        },
    }
}

fn unknown(call: &ToolCallContext, message: &str, reason: &str) -> ToolResultPayload {
    ToolResultPayload {
        call_block_id: call.call_block_id,
        status: ToolResultStatus::UnknownOutcome,
        output: ToolOutput {
            content: serde_json::json!({"error": message}),
            truncation: Truncation::None,
            meta: Some(ToolOutputMeta {
                duration_ms: None,
                original_tokens: None,
                extra: Some(serde_json::json!({"reason": reason})),
            }),
            artifact: None,
        },
        media: Vec::new(),
        notes: Vec::new(),
    }
}
