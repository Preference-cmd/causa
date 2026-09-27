//! Minimal session submit / observe / wait / shutdown flow.

mod common;

use std::time::Duration;

use async_trait::async_trait;
use causa_kernel::TextPayload;
use causa_kernel::{
    ConversationId, ProcessorContext, ProcessorError, ToolBatch, ToolBatchProcessor, ToolCallDraft,
};
use causa_runtime::{
    ConversationState, FinishedKind, Session, SessionConfig, SubmitRequest, ToolExecutor,
    ToolProcessingBuilder, TurnRunOptions, TurnRunner, WaitEnd, WorkState,
};
use common::{EchoTool, RecordingGateway, endturn_output, runner_with, tooluse_calls_output};
use std::sync::Arc;

fn text_part(value: &str) -> causa_kernel::ContentPart {
    causa_kernel::ContentPart::Text(TextPayload::new(value))
}

#[tokio::test]
async fn submit_runs_and_publishes_completed_work() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("done"))]);
    let session = Session::new(
        ConversationState::new(ConversationId("smoke".into())),
        std::sync::Arc::new(runner_with(gateway, vec![])),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("an empty state is accepted");
    let handle = session.handle();
    let receipt = handle
        .submit(SubmitRequest {
            request_key: "first".into(),
            parts: vec![text_part("hello")],
        })
        .expect("work is accepted");

    let waited = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("work is observable");
    assert_eq!(waited.end, WaitEnd::ReachedState);
    assert_eq!(waited.observation.state, WorkState::Finished);
    assert!(matches!(
        waited.observation.finished,
        Some(FinishedKind::Completed { ref final_output }) if final_output.response.text.0 == "done"
    ));
}

struct NoteThenFail;

#[async_trait]
impl ToolBatchProcessor for NoteThenFail {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        batch.calls_mut()[0].push_note(TextPayload::new("kept note"));
        Err(ProcessorError::Failed("stop before dispatch".into()))
    }
}

#[tokio::test]
async fn interrupted_batch_and_its_notes_remain_observable() {
    let gateway = RecordingGateway::scripted(vec![Ok(tooluse_calls_output(
        "thinking",
        vec![ToolCallDraft {
            tool_name: "echo".into(),
            arguments: serde_json::json!({"value": 1}),
            provider_call_id: Some("provider-call-7".into()),
        }],
    ))]);
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(EchoTool)])),
        ToolProcessingBuilder::default()
            .before(Arc::new(NoteThenFail))
            .build(),
    );
    let session = Session::new(
        ConversationState::new(ConversationId("interrupted-batch".into())),
        Arc::new(runner),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("an empty state is accepted");
    let handle = session.handle();
    let receipt = handle
        .submit(SubmitRequest {
            request_key: "tool-use".into(),
            parts: vec![text_part("run")],
        })
        .expect("work is accepted");

    let waited = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("work is observable");
    let Some(FinishedKind::Interrupted {
        uncommitted_tool_batch: Some(batch),
        ..
    }) = waited.observation.finished
    else {
        panic!("the interrupted work retains its pending batch");
    };
    assert_eq!(batch.completed_len(), 0);
    assert_eq!(batch.calls().len(), 1);
    assert_eq!(batch.calls()[0].call().input.tool_name, "echo");
    assert_eq!(
        batch.calls()[0].call().result_notes,
        [TextPayload::new("kept note")]
    );
}
