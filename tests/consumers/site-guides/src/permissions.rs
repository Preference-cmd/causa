//! Deciding calls by host-defined tool metadata and the run's mode, from the
//! "Process tool calls" page.

use async_trait::async_trait;
use causa::{
    kernel::{
        ArtifactStore, CallControl, ModelRef, ProcessorContext, ProcessorData, ProcessorError,
        Tool, ToolBatch, ToolBatchProcessor, ToolCallContext, ToolDefinition, ToolMetadata,
        ToolOutput, ToolResultPayload, ToolResultStatus,
    },
    runtime::{TurnRunOptions, new_block_id},
};
use serde_json::json;
use std::sync::{Arc, Mutex};

// ANCHOR: metadata
/// Your application's own tool classification. Causa carries it without
/// inspecting or persisting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolEffect {
    /// Reads only; safe in read-only mode.
    ReadOnly,
}
// ANCHOR_END: metadata

// ANCHOR: mode
/// Your application's permission modes. Causa has none of its own: it only
/// hands this value to the processors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Reads run; every other call is refused.
    ReadOnly,
    /// Every call runs.
    FullAccess,
}

/// Shared, so the user can switch modes while a run is going.
pub type ModeSwitch = Arc<Mutex<Mode>>;
// ANCHOR_END: mode

// ANCHOR: options
/// Every processor of the run can read `mode` from its context.
pub fn run_options(mode: &ModeSwitch) -> TurnRunOptions {
    TurnRunOptions::new(ModelRef::new("your-model"))
        .with_processor_data(ProcessorData::new(mode.clone()))
}
// ANCHOR_END: options

// ANCHOR: processor
/// Lets reads through and decides every other call by the current mode.
pub struct ByMode;

#[async_trait]
impl ToolBatchProcessor for ByMode {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        // A run without a mode is a mistake in the application: fail
        // rather than let every call through.
        let mode = ctx
            .data::<ModeSwitch>()
            .ok_or_else(|| ProcessorError::Failed("the run has no mode".into()))?;
        let mut offset = 0;
        while offset < batch.calls().len() {
            let call = batch.calls()[offset].call();
            let declares_read_only = ctx.tool_metadata::<ToolEffect>(&call.input.tool_name)
                == Some(&ToolEffect::ReadOnly);
            // Read the mode for each call: it may have changed since the last.
            if declares_read_only || *mode.lock().unwrap() == Mode::FullAccess {
                offset += 1;
                continue;
            }
            let rejected = ToolResultPayload::new(
                call.call_block_id,
                ToolResultStatus::Rejected,
                ToolOutput::new(json!({ "error": "read-only mode: only reads can run" })),
            );
            batch
                .resolve_at(batch.completed_len() + offset, new_block_id(), rejected)
                .map_err(|refused| ProcessorError::Failed(refused.error.to_string()))?;
        }
        Ok(())
    }
}
// ANCHOR_END: processor

/// A tool that only reads, and says so.
pub struct ListNotes;

#[async_trait]
impl Tool for ListNotes {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("list_notes", "List the notes.", json!({ "type": "object" }))
    }

    // ANCHOR: tool_metadata
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::new().with(ToolEffect::ReadOnly)
    }
    // ANCHOR_END: tool_metadata

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        ToolResultPayload::new(
            ctx.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!({ "notes": ["todo.txt"] })),
        )
    }
}

/// A tool that changes what exists, and carries no metadata.
pub struct DeleteNote;

#[async_trait]
impl Tool for DeleteNote {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("delete_note", "Delete a note.", json!({ "type": "object" }))
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        ToolResultPayload::new(
            ctx.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!({ "deleted": true })),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causa::{
        kernel::{CancellationToken, InvocationId, RoundId, ToolCallPayload, TurnId},
        runtime::{ToolExecutor, ToolExecutorOptions},
    };

    async fn statuses(mode: &ModeSwitch) -> Vec<(String, ToolResultStatus)> {
        let executor = ToolExecutor::new(
            vec![Arc::new(ListNotes), Arc::new(DeleteNote)],
            ToolExecutorOptions::default().with_before(vec![Arc::new(ByMode)]),
        )
        .unwrap();
        let options = run_options(mode);
        let calls = ["list_notes", "delete_note"].map(|name| {
            ToolCallContext::new(new_block_id(), ToolCallPayload::new(name, json!({})))
        });
        let mut batch = ToolBatch::new(calls.to_vec()).unwrap();
        let invocation = InvocationId {
            turn_id: TurnId::new("modes"),
            round_id: RoundId(0),
        };
        executor
            .bind(invocation, CallControl::new(CancellationToken::new(), None))
            .await
            .unwrap()
            .with_processor_data(options.processor_data.unwrap())
            .process(&mut batch)
            .await
            .unwrap();
        let mut statuses: Vec<_> = batch
            .results()
            .iter()
            .map(|entry| {
                let (_, result) = entry.result().unwrap();
                (entry.call().input.tool_name.clone(), result.status.clone())
            })
            .collect();
        statuses.sort_by(|a, b| a.0.cmp(&b.0));
        statuses
    }

    #[tokio::test]
    async fn read_only_refuses_the_tool_that_does_not_declare_a_read() {
        let mode: ModeSwitch = Arc::new(Mutex::new(Mode::ReadOnly));
        assert_eq!(
            statuses(&mode).await,
            vec![
                ("delete_note".into(), ToolResultStatus::Rejected),
                ("list_notes".into(), ToolResultStatus::Succeeded),
            ]
        );
        *mode.lock().unwrap() = Mode::FullAccess;
        assert_eq!(
            statuses(&mode).await,
            vec![
                ("delete_note".into(), ToolResultStatus::Succeeded),
                ("list_notes".into(), ToolResultStatus::Succeeded),
            ]
        );
    }
}
