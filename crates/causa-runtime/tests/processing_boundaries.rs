//! Controlled returns preserve known/unknown/pending state and error payloads.
mod tool_fixtures;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{ToolExecutorOptions, ToolProcessingError, ToolProcessorPhase, new_block_id};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, Semaphore};
use tool_fixtures::*;

struct CancelProcessor;
#[async_trait]
impl ToolBatchProcessor for CancelProcessor {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        for entry in batch.calls_mut() {
            entry.push_note(TextPayload::new("pre note"));
        }
        for entry in batch.results_mut() {
            entry.output_mut().unwrap().content = json!("post edit");
        }
        context.control.cancellation_token().cancel();
        Ok(())
    }
}
struct Count(Arc<AtomicUsize>);
#[async_trait]
impl ToolBatchProcessor for Count {
    async fn process(
        &self,
        _batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
#[tokio::test]
async fn synchronous_before_cancel_preserves_mutations_and_stops_remaining_stages() {
    let count = Arc::new(AtomicUsize::new(0));
    let tool = NamedTool::new("work");
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(CancelProcessor), Arc::new(Count(count.clone()))],
            ..Default::default()
        },
    );
    let mut batch = batch(&["work", "work"]);
    assert!(matches!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .process(&mut batch)
            .await,
        Err(ToolProcessingError::Control(ControlError::Cancelled))
    ));
    assert_eq!(batch.completed_len(), 0);
    assert_eq!(
        batch.calls()[0].call().result_notes,
        [TextPayload::new("pre note")]
    );
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn synchronous_after_cancel_returns_error_even_with_all_results_and_preserves_edits() {
    let count = Arc::new(AtomicUsize::new(0));
    let tool = NamedTool::new("work");
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            after: vec![Arc::new(CancelProcessor), Arc::new(Count(count.clone()))],
            ..Default::default()
        },
    );
    let mut batch = batch(&["work"]);
    assert!(matches!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .process(&mut batch)
            .await,
        Err(ToolProcessingError::Control(ControlError::Cancelled))
    ));
    assert_eq!(statuses(&batch), vec![ToolResultStatus::Succeeded]);
    assert_eq!(
        batch.results()[0].result().unwrap().1.output.content,
        json!("post edit")
    );
    assert_eq!(
        batch.results()[0].result().unwrap().1.notes,
        [TextPayload::new("tool note")]
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
struct CancelsOnStart(Arc<AtomicUsize>);
#[async_trait]
impl Tool for CancelsOnStart {
    fn definition(&self) -> ToolDefinition {
        definition("cancel")
    }
    async fn execute(&self, _call: &ToolCallContext, control: &CallControl) -> ToolResultPayload {
        self.0.fetch_add(1, Ordering::SeqCst);
        control.cancellation_token().cancel();
        std::future::pending().await
    }
}
#[tokio::test]
async fn first_dispatch_cancel_marks_started_unknown_and_unstarted_pending() {
    let count = Arc::new(AtomicUsize::new(0));
    let executor = executor(
        vec![Arc::new(CancelsOnStart(count.clone()))],
        ToolExecutorOptions::default(),
    );
    let mut batch = batch(&["cancel", "cancel", "cancel", "cancel"]);
    let returned = tokio::time::timeout(
        Duration::from_secs(2),
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .process(&mut batch),
    )
    .await
    .unwrap();
    assert!(matches!(
        returned,
        Err(ToolProcessingError::Control(ControlError::Cancelled))
    ));
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(statuses(&batch), vec![ToolResultStatus::UnknownOutcome]);
    assert_eq!(batch.calls().len(), 3);
}
#[tokio::test]
async fn already_accepted_result_survives_later_sibling_cancel() {
    let count = Arc::new(AtomicUsize::new(0));
    let known = NamedTool::new("known");
    let executor = executor(
        vec![known.clone(), Arc::new(CancelsOnStart(count))],
        ToolExecutorOptions::default(),
    );
    let mut batch = batch(&["known", "cancel", "known"]);
    let known_id = batch.declaration_ids()[0];
    assert!(matches!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .process(&mut batch)
            .await,
        Err(ToolProcessingError::Control(ControlError::Cancelled))
    ));
    assert_eq!(known.count.load(Ordering::SeqCst), 1);
    assert_eq!(
        batch
            .results()
            .iter()
            .find(|entry| entry.call().call_block_id == known_id)
            .unwrap()
            .result()
            .unwrap()
            .1
            .status,
        ToolResultStatus::Succeeded
    );
    assert_eq!(batch.completed_len(), 2);
    assert_eq!(batch.calls().len(), 1);
}
struct WaitTool {
    name: &'static str,
    entered: Arc<Notify>,
    count: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for WaitTool {
    fn definition(&self) -> ToolDefinition {
        definition(self.name)
    }
    async fn execute(&self, _call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        self.count.fetch_add(1, Ordering::SeqCst);
        self.entered.notify_one();
        std::future::pending().await
    }
}
#[tokio::test]
async fn external_cancel_settles_every_started_future_without_waiting_for_tool_cooperation() {
    let entered = Arc::new(Notify::new());
    let count = Arc::new(AtomicUsize::new(0));
    let parent = control();
    let caller = parent.clone();
    let executor = Arc::new(executor(
        vec![Arc::new(WaitTool {
            name: "wait",
            entered: entered.clone(),
            count: count.clone(),
        })],
        ToolExecutorOptions::default(),
    ));
    let task = tokio::spawn(async move {
        let mut batch = batch(&["wait", "wait"]);
        let result = executor
            .bind(invocation(), parent)
            .await
            .unwrap()
            .process(&mut batch)
            .await;
        (result, batch)
    });
    while count.load(Ordering::SeqCst) != 2 {
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
    }
    caller.cancellation_token().cancel();
    let (result, batch) = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        result,
        Err(ToolProcessingError::Control(ControlError::Cancelled))
    ));
    assert_eq!(statuses(&batch), vec![ToolResultStatus::UnknownOutcome; 2]);
}
struct WaitingProcessor {
    entered: Arc<Notify>,
}
#[async_trait]
impl ToolBatchProcessor for WaitingProcessor {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        for entry in batch.calls_mut() {
            entry.push_note(TextPayload::new("before wait"));
        }
        for entry in batch.results_mut() {
            entry.output_mut().unwrap().content = json!("before wait");
        }
        self.entered.notify_one();
        std::future::pending().await
    }
}
#[tokio::test]
async fn cancel_during_before_and_after_wait_keeps_current_materials() {
    for phase in [ToolProcessorPhase::Before, ToolProcessorPhase::After] {
        let entered = Arc::new(Notify::new());
        let tool = NamedTool::new("work");
        let parent = control();
        let caller = parent.clone();
        let processors: Vec<Arc<dyn ToolBatchProcessor>> = vec![Arc::new(WaitingProcessor {
            entered: entered.clone(),
        })];
        let options = match phase {
            ToolProcessorPhase::Before => ToolExecutorOptions {
                before: processors,
                ..Default::default()
            },
            ToolProcessorPhase::After => ToolExecutorOptions {
                after: processors,
                ..Default::default()
            },
        };
        let executor = Arc::new(executor(vec![tool.clone()], options));
        let task = tokio::spawn(async move {
            let mut batch = batch(&["work"]);
            let result = executor
                .bind(invocation(), parent)
                .await
                .unwrap()
                .process(&mut batch)
                .await;
            (result, batch)
        });
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        caller.cancellation_token().cancel();
        let (result, batch) = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            result,
            Err(ToolProcessingError::Control(ControlError::Cancelled))
        ));
        if phase == ToolProcessorPhase::Before {
            assert_eq!(
                batch.calls()[0].call().result_notes,
                [TextPayload::new("before wait")]
            );
            assert_eq!(tool.count.load(Ordering::SeqCst), 0);
        } else {
            assert_eq!(
                batch.results()[0].result().unwrap().1.output.content,
                json!("before wait")
            );
            assert_eq!(statuses(&batch), vec![ToolResultStatus::Succeeded]);
        }
    }
}
#[tokio::test]
async fn binding_does_not_restart_parent_deadline_when_processing_later() {
    let parent = CallControl::new(
        CancellationToken::new(),
        Some(Instant::now() + Duration::from_millis(15)),
    );
    let tool = NamedTool::new("work");
    let executor = executor(vec![tool.clone()], ToolExecutorOptions::default());
    let binding = executor.bind(invocation(), parent).await.unwrap();
    tokio::time::sleep(Duration::from_millis(25)).await;
    let mut batch = batch(&["work"]);
    assert!(matches!(
        binding.process(&mut batch).await,
        Err(ToolProcessingError::Control(ControlError::TimedOut))
    ));
    assert_eq!(batch.completed_len(), 0);
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn explicit_cancel_wins_when_deadline_is_also_expired() {
    let parent = CallControl::new(
        CancellationToken::new(),
        Some(Instant::now() - Duration::from_secs(1)),
    );
    parent.cancellation_token().cancel();
    let executor = executor(vec![], ToolExecutorOptions::default());
    assert!(matches!(
        executor.bind(invocation(), parent).await,
        Err(causa_runtime::ToolCatalogError::Control(
            ControlError::Cancelled
        ))
    ));
}
struct WrongResult {
    entered: Arc<Notify>,
    release: Arc<Semaphore>,
    foreign_id: BlockId,
}
#[async_trait]
impl Tool for WrongResult {
    fn definition(&self) -> ToolDefinition {
        definition("wrong")
    }
    async fn execute(&self, call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        self.entered.notify_one();
        self.release.acquire().await.unwrap().forget();
        let mut result = result(
            call,
            ToolResultStatus::Succeeded,
            json!({"diagnostic": "actual payload"}),
        );
        result.call_block_id = self.foreign_id;
        result
            .media
            .push(MediaRef::new("image/png", "foreign-media"));
        result.notes.push(TextPayload::new("foreign note"));
        result
    }
}
#[tokio::test]
async fn rejected_tool_payload_is_retained_and_module_error_never_cancels_shared_parent() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let waiting = Arc::new(Notify::new());
    let count = Arc::new(AtomicUsize::new(0));
    let foreign_id = new_block_id();
    let parent = control();
    let caller = parent.clone();
    let known = NamedTool::new("known");
    let executor = Arc::new(executor(
        vec![
            known.clone(),
            Arc::new(WrongResult {
                entered: entered.clone(),
                release: release.clone(),
                foreign_id,
            }),
            Arc::new(WaitTool {
                name: "wait",
                entered: waiting.clone(),
                count: count.clone(),
            }),
        ],
        ToolExecutorOptions::default(),
    ));
    let executing = executor.clone();
    let task = tokio::spawn(async move {
        let mut batch = batch(&["known", "wrong", "wait"]);
        let wrong_id = batch.declaration_ids()[1];
        let result = executing
            .bind(invocation(), parent)
            .await
            .unwrap()
            .process(&mut batch)
            .await;
        (result, batch, wrong_id)
    });
    tokio::time::timeout(Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), waiting.notified())
        .await
        .unwrap();
    release.add_permits(1);
    let (error, returned, wrong_id) = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    match error.unwrap_err() {
        ToolProcessingError::ResultRejected {
            call_block_id,
            error: BatchError::MismatchedCall { expected, actual },
            result,
        } => {
            assert_eq!(call_block_id, wrong_id);
            assert_eq!(expected, wrong_id);
            assert_eq!(actual, foreign_id);
            assert_eq!(
                result.output.content,
                json!({"diagnostic": "actual payload"})
            );
            assert_eq!(
                result.media,
                vec![MediaRef::new("image/png", "foreign-media")]
            );
            assert_eq!(result.notes, [TextPayload::new("foreign note")]);
        }
        error => panic!("unexpected error {error:?}"),
    }
    assert!(!caller.is_cancelled());
    assert_eq!(returned.completed_len(), 3);
    assert_eq!(
        statuses(&returned)
            .iter()
            .filter(|status| **status == ToolResultStatus::UnknownOutcome)
            .count(),
        2
    );
    let mut separate = batch(&["known"]);
    executor
        .bind(invocation(), caller)
        .await
        .unwrap()
        .process(&mut separate)
        .await
        .unwrap();
    assert_eq!(statuses(&separate), vec![ToolResultStatus::Succeeded]);
}

struct CancelOnDrop(CancellationToken);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
struct CancelWhenDropped(Arc<Notify>);
#[async_trait]
impl Tool for CancelWhenDropped {
    fn definition(&self) -> ToolDefinition {
        definition("drop")
    }
    async fn execute(&self, _call: &ToolCallContext, control: &CallControl) -> ToolResultPayload {
        let _guard = CancelOnDrop(control.cancellation_token().clone());
        self.0.notify_one();
        std::future::pending().await
    }
}
#[tokio::test]
async fn selected_parent_deadline_survives_later_cancel_while_call_future_is_dropped() {
    let entered = Arc::new(Notify::new());
    let parent = CallControl::new(
        CancellationToken::new(),
        Some(Instant::now() + Duration::from_millis(25)),
    );
    let caller = parent.clone();
    let executor = executor(
        vec![Arc::new(CancelWhenDropped(entered))],
        ToolExecutorOptions::default(),
    );
    let mut batch = batch(&["drop"]);
    let result = executor
        .bind(invocation(), parent)
        .await
        .unwrap()
        .process(&mut batch)
        .await;
    assert!(matches!(
        result,
        Err(ToolProcessingError::Control(ControlError::TimedOut))
    ));
    assert!(
        caller.is_cancelled(),
        "the tool's Drop runs after the cause was selected"
    );
    assert_eq!(statuses(&batch), vec![ToolResultStatus::UnknownOutcome]);
}
