//! Unknown results complete the batch and stop the runner after one commit.
mod tool_fixtures;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{
    RunControl, ToolExecutorOptions, TurnInterruption, TurnResult, TurnRunOptions, TurnRunner,
};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;
use tool_fixtures::*;

struct StatusTool {
    name: &'static str,
    status: ToolResultStatus,
    count: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for StatusTool {
    fn definition(&self) -> ToolDefinition {
        definition(self.name)
    }
    async fn execute(&self, call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        self.count.fetch_add(1, Ordering::SeqCst);
        result(
            call,
            self.status.clone(),
            json!({"state": format!("{:?}", self.status)}),
        )
    }
}
struct Post(Arc<AtomicUsize>);
#[async_trait]
impl ToolBatchProcessor for Post {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        for entry in batch.results_mut() {
            entry.push_note(TextPayload::new("after note"));
        }
        batch
            .results_mut()
            .sort_by_key(|entry| entry.call().input.tool_name.clone());
        Ok(())
    }
}
struct Gateway {
    names: Vec<&'static str>,
    count: Arc<AtomicUsize>,
}
#[async_trait]
impl ModelGateway for Gateway {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let first = self.count.fetch_add(1, Ordering::SeqCst) == 0;
        Ok(ModelOutput {
            response: ModelResponse {
                text: TextPayload::new(if first { "" } else { "done" }),
                tool_calls: if first {
                    self.names
                        .iter()
                        .map(|name| ToolCallDraft {
                            tool_name: (*name).into(),
                            arguments: json!({}),
                            provider_call_id: Some(format!("provider-{name}")),
                        })
                        .collect()
                } else {
                    vec![]
                },
            },
            usage: None,
            reasoning: None,
            stop_reason: if first {
                ModelStopReason::ToolUse
            } else {
                ModelStopReason::EndTurn
            },
        })
    }
}
#[tokio::test]
async fn unknown_success_and_failure_complete_after_and_commit_once_before_fixed_stop() {
    let calls = Arc::new(AtomicUsize::new(0));
    let after = Arc::new(AtomicUsize::new(0));
    let models = Arc::new(AtomicUsize::new(0));
    let executor = Arc::new(executor(
        vec![
            Arc::new(StatusTool {
                name: "z-unknown",
                status: ToolResultStatus::UnknownOutcome,
                count: calls.clone(),
            }),
            Arc::new(StatusTool {
                name: "a-success",
                status: ToolResultStatus::Succeeded,
                count: calls.clone(),
            }),
            Arc::new(StatusTool {
                name: "b-failed",
                status: ToolResultStatus::Failed,
                count: calls.clone(),
            }),
            Arc::new(StatusTool {
                name: "c-unknown",
                status: ToolResultStatus::UnknownOutcome,
                count: calls.clone(),
            }),
        ],
        ToolExecutorOptions {
            after: vec![Arc::new(Post(after.clone()))],
            ..Default::default()
        },
    ));
    let runner = TurnRunner::new(
        Arc::new(Gateway {
            names: vec!["z-unknown", "a-success", "b-failed", "c-unknown"],
            count: models.clone(),
        }),
        executor,
    );
    let outcome = runner
        .run(
            TurnId::new("unknown"),
            Context::new(),
            TurnRunOptions::new(ModelRef("model".into())),
            RunControl::new(CancellationToken::new(), None),
        )
        .await;
    let call_block_id = match outcome.result {
        TurnResult::Interrupted {
            cause: TurnInterruption::UnknownToolOutcome { call_block_id, .. },
        } => call_block_id,
        result => panic!("unexpected {result:?}"),
    };
    assert!(outcome.uncommitted_tool_batch.is_none());
    assert_eq!(models.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(after.load(Ordering::SeqCst), 1);
    let results = outcome
        .context
        .blocks()
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(results.len(), 4);
    assert_eq!(
        results.iter().map(|r| r.status.clone()).collect::<Vec<_>>(),
        [
            ToolResultStatus::Succeeded,
            ToolResultStatus::Failed,
            ToolResultStatus::UnknownOutcome,
            ToolResultStatus::UnknownOutcome
        ]
    );
    assert_eq!(results[2].call_block_id, call_block_id);
    assert!(
        results
            .iter()
            .all(|result| result.notes == [TextPayload::new("after note")])
    );
}
#[tokio::test]
async fn ordinary_failed_and_rejected_results_allow_the_next_model_round() {
    let count = Arc::new(AtomicUsize::new(0));
    let models = Arc::new(AtomicUsize::new(0));
    let runner = TurnRunner::new(
        Arc::new(Gateway {
            names: vec!["failed", "outside"],
            count: models.clone(),
        }),
        Arc::new(executor(
            vec![Arc::new(StatusTool {
                name: "failed",
                status: ToolResultStatus::Failed,
                count: count.clone(),
            })],
            ToolExecutorOptions::default(),
        )),
    );
    let outcome = runner
        .run(
            TurnId::new("ordinary"),
            Context::new(),
            TurnRunOptions::new(ModelRef("model".into())),
            RunControl::new(CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    assert_eq!(models.load(Ordering::SeqCst), 2);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
struct Alias(Arc<dyn Tool>);
#[async_trait]
impl Tool for Alias {
    fn definition(&self) -> ToolDefinition {
        let mut definition = self.0.definition();
        definition.name = "alias".into();
        definition
    }
    async fn execute(&self, call: &ToolCallContext, control: &CallControl) -> ToolResultPayload {
        let mut translated = call.clone();
        translated.input.tool_name = self.0.definition().name;
        self.0.execute(&translated, control).await
    }
}
#[tokio::test]
async fn explicitly_advertised_alias_wrapper_dispatches_and_keeps_unknown_fact() {
    let count = Arc::new(AtomicUsize::new(0));
    let target: Arc<dyn Tool> = Arc::new(StatusTool {
        name: "real",
        status: ToolResultStatus::UnknownOutcome,
        count: count.clone(),
    });
    let executor = executor(
        vec![Arc::new(Alias(target))],
        ToolExecutorOptions::default(),
    );
    let mut batch = batch(&["alias"]);
    let binding = executor.bind(invocation(), control()).await.unwrap();
    assert_eq!(binding.surface().definitions[0].name, "alias");
    binding.process(&mut batch).await.unwrap();
    assert_eq!(statuses(&batch), vec![ToolResultStatus::UnknownOutcome]);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
struct Slow;
#[async_trait]
impl Tool for Slow {
    fn definition(&self) -> ToolDefinition {
        definition("slow")
    }
    async fn execute(&self, _call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        std::future::pending().await
    }
}
#[tokio::test]
async fn local_timeout_is_unknown_without_cancelling_parent_siblings_or_after() {
    let parent = control();
    let after = Arc::new(AtomicUsize::new(0));
    let sibling = NamedTool::new("fast");
    let executor = executor(
        vec![Arc::new(Slow), sibling.clone()],
        ToolExecutorOptions {
            call_timeout: Some(Duration::from_millis(10)),
            after: vec![Arc::new(Post(after.clone()))],
            ..Default::default()
        },
    );
    let mut batch = batch(&["slow", "fast"]);
    executor
        .bind(invocation(), parent.clone())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    assert!(!parent.is_cancelled());
    assert_eq!(sibling.count.load(Ordering::SeqCst), 1);
    assert_eq!(after.load(Ordering::SeqCst), 1);
    assert_eq!(
        statuses(&batch),
        [
            ToolResultStatus::Succeeded,
            ToolResultStatus::UnknownOutcome
        ]
    );
}
struct Delay;
#[async_trait]
impl ToolBatchProcessor for Delay {
    async fn process(
        &self,
        _batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        tokio::time::sleep(Duration::from_millis(25)).await;
        Ok(())
    }
}
#[tokio::test]
async fn tool_timeout_starts_after_before_stage_instead_of_bind_time() {
    let tool = NamedTool::new("fast");
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            call_timeout: Some(Duration::from_millis(10)),
            before: vec![Arc::new(Delay)],
            ..Default::default()
        },
    );
    let binding = executor.bind(invocation(), control()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let mut batch = batch(&["fast"]);
    binding.process(&mut batch).await.unwrap();
    assert_eq!(statuses(&batch), vec![ToolResultStatus::Succeeded]);
    assert_eq!(tool.count.load(Ordering::SeqCst), 1);
}
struct Panics;
#[async_trait]
impl Tool for Panics {
    fn definition(&self) -> ToolDefinition {
        definition("panic")
    }
    async fn execute(&self, _call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        panic!("tool panic fixture")
    }
}
#[tokio::test]
async fn tool_panic_is_an_ordinary_failed_result_and_siblings_complete() {
    let sibling = NamedTool::new("fast");
    let executor = executor(
        vec![Arc::new(Panics), sibling.clone()],
        ToolExecutorOptions::default(),
    );
    let mut batch = batch(&["panic", "fast"]);
    executor
        .bind(invocation(), control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    assert_eq!(
        statuses(&batch),
        [ToolResultStatus::Failed, ToolResultStatus::Succeeded]
    );
    assert_eq!(sibling.count.load(Ordering::SeqCst), 1);
}

struct ChangeUnknown;
#[async_trait]
impl ToolBatchProcessor for ChangeUnknown {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let entry = &batch.results()[0];
        let call = entry.call().clone();
        let (id, result) = entry.result().unwrap();
        let id = *id;
        let mut result = result.clone();
        result.status = ToolResultStatus::Succeeded;
        let mut replacement = ToolBatch::new(vec![call]).unwrap();
        replacement.resolve_at(0, id, result).unwrap();
        *batch = replacement;
        Ok(())
    }
}
#[tokio::test]
async fn after_cannot_turn_unknown_into_success_to_allow_continuation() {
    let count = Arc::new(AtomicUsize::new(0));
    let parent = control();
    let executor = executor(
        vec![Arc::new(StatusTool {
            name: "unknown",
            status: ToolResultStatus::UnknownOutcome,
            count: count.clone(),
        })],
        ToolExecutorOptions {
            after: vec![Arc::new(ChangeUnknown)],
            ..Default::default()
        },
    );
    let mut batch = batch(&["unknown"]);
    assert!(matches!(
        executor
            .bind(invocation(), parent.clone())
            .await
            .unwrap()
            .process(&mut batch)
            .await,
        Err(causa_runtime::ToolProcessingError::Handoff {
            phase: causa_runtime::ToolProcessorPhase::After,
            index: 0,
            ..
        })
    ));
    assert!(!parent.is_cancelled());
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn zero_local_timeout_does_not_poll_an_immediately_ready_uncooperative_tool() {
    let tool = NamedTool::new("instant");
    let parent = control();
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            call_timeout: Some(Duration::ZERO),
            ..Default::default()
        },
    );
    let mut batch = batch(&["instant"]);
    executor
        .bind(invocation(), parent.clone())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
    assert_eq!(statuses(&batch), vec![ToolResultStatus::UnknownOutcome]);
    assert!(!parent.is_cancelled());
}
