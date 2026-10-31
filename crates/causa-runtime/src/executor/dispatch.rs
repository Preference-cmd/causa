use causa_kernel::{
    ArtifactStore, BlockId, CallControl, ControlError, Tool, ToolCallContext, ToolOutput,
    ToolResultPayload, ToolResultStatus,
};
use futures_util::FutureExt;
use std::sync::Arc;
use tracing::Instrument;

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
        // A deadline that expired before the first poll belongs to the
        // deadline backstop, not to a turn stop.
        let reason = match &error {
            ControlError::Cancelled => "call_control_stopped",
            ControlError::TimedOut => "call_deadline_backstop",
        };
        return unknown(call.call_block_id, &error.to_string(), reason);
    }
    // Construct the port future inside the catch as well: manually implemented
    // async-trait signatures can panic before they return a future.
    let future = std::panic::AssertUnwindSafe(async { tool.execute(call, control, store).await })
        .catch_unwind();
    let result = match control.deadline() {
        Some(deadline) => match tokio::time::timeout_at(deadline.into(), future).await {
            Ok(result) => result,
            Err(_) => {
                return unknown(
                    call.call_block_id,
                    "tool did not return before call deadline",
                    "call_deadline_backstop",
                );
            }
        },
        None => future.await,
    };
    match result {
        Ok(result) => result,
        Err(_) => ToolResultPayload::new(
            call.call_block_id,
            ToolResultStatus::Failed,
            ToolOutput::new(serde_json::json!({"error": "tool panicked"})),
        ),
    }
}

/// Synthesizes an unknown outcome for a call whose result was not observed,
/// tagging `meta.extra.reason` with `reason` so hosts can tell causes apart.
/// Shared by dispatch backstops and batch-stop settlement.
pub(super) fn unknown(call_block_id: BlockId, message: &str, reason: &str) -> ToolResultPayload {
    let mut output = ToolOutput::new(serde_json::json!({"error": message}));
    output.meta.get_or_insert_default().extra = Some(serde_json::json!({"reason": reason}));
    ToolResultPayload::new(call_block_id, ToolResultStatus::UnknownOutcome, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use causa_kernel::{CancellationToken, ToolCallPayload, ToolDefinition};

    /// Panics if polled: a stopped control must settle the call before the
    /// tool future is constructed.
    struct PanicsIfPolled;

    #[async_trait::async_trait]
    impl Tool for PanicsIfPolled {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "never-polled",
                "stub",
                serde_json::json!({"type": "object"}),
            )
        }

        async fn execute(
            &self,
            _call: &ToolCallContext,
            _control: &CallControl,
            _store: Option<&dyn ArtifactStore>,
        ) -> ToolResultPayload {
            panic!("a stopped call must not be polled");
        }
    }

    #[tokio::test]
    async fn stopped_control_records_its_reason_without_polling_the_tool() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let call_block_id = crate::new_block_id();
        let call = ToolCallContext::new(
            call_block_id,
            ToolCallPayload::new("never-polled", serde_json::json!({})),
        );
        let result = execute(
            Arc::new(PanicsIfPolled),
            call,
            CallControl::new(cancellation, None),
            None,
        )
        .await;
        assert_eq!(result.call_block_id, call_block_id);
        assert_eq!(result.status, ToolResultStatus::UnknownOutcome);
        assert_eq!(
            result
                .output
                .meta
                .as_ref()
                .and_then(|meta| meta.extra.as_ref()),
            Some(&serde_json::json!({"reason": "call_control_stopped"}))
        );
    }
}
