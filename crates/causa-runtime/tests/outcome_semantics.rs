//! A tool-reported timeout is an ordinary result: the batch commits it with
//! its output unchanged and the run continues.
mod common;

use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::TurnResult;
use common::*;
use serde_json::json;
use std::sync::Arc;

struct TimedOutTool;

#[async_trait]
impl Tool for TimedOutTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("timed-out", "stub", json!({"type": "object"}))
    }
    async fn execute(
        &self,
        call: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        ToolResultPayload::new(
            call.call_block_id,
            ToolResultStatus::TimedOut,
            ToolOutput::new(json!({"partial": "half"})),
        )
    }
}

#[tokio::test]
async fn tool_reported_timeout_commits_its_partial_output_and_the_run_continues() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("reading", "timed-out", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gateway.clone(), vec![Arc::new(TimedOutTool)]);
    let outcome = runner
        .run(
            TurnId::new("reported-timeout"),
            input("Read with a timeout."),
            options(),
            ctrl(),
        )
        .await;
    assert!(matches!(&outcome.result, TurnResult::Completed { .. }));
    assert!(outcome.uncommitted_tool_batch.is_none());
    assert_eq!(gateway.recorded.lock().unwrap().len(), 2);
    let results = outcome
        .context
        .blocks()
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].status, ToolResultStatus::TimedOut);
    assert_eq!(
        results[0].output,
        ToolOutput::new(json!({"partial": "half"}))
    );
}
