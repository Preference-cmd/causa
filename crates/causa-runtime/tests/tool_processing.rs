//! Processor policy is supplied by callers; the executor enforces each handoff.
mod tool_fixtures;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{ToolExecutorOptions, ToolProcessingError, ToolProcessorPhase, new_block_id};
use serde_json::json;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::{Notify, Semaphore};
use tool_fixtures::*;

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
struct Gate {
    entered: Arc<Notify>,
    release: Arc<Semaphore>,
}
#[async_trait]
impl ToolBatchProcessor for Gate {
    async fn process(
        &self,
        _batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        self.entered.notify_one();
        self.release.acquire().await.unwrap().forget();
        Ok(())
    }
}
#[tokio::test]
async fn asynchronous_before_holds_dispatch_and_after_until_released() {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let tool = NamedTool::new("work");
    let after = Arc::new(AtomicUsize::new(0));
    let executor = Arc::new(executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(Gate {
                entered: entered.clone(),
                release: release.clone(),
            })],
            after: vec![Arc::new(Count(after.clone()))],
            ..Default::default()
        },
    ));
    let task = tokio::spawn(async move {
        let mut batch = batch(&["work"]);
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .process(&mut batch)
            .await
            .unwrap();
        batch
    });
    entered.notified().await;
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
    assert_eq!(after.load(Ordering::SeqCst), 0);
    release.add_permits(1);
    assert_eq!(task.await.unwrap().completed_len(), 1);
    assert_eq!(tool.count.load(Ordering::SeqCst), 1);
    assert_eq!(after.load(Ordering::SeqCst), 1);
}
struct Reject;
#[async_trait]
impl ToolBatchProcessor for Reject {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        while let Some(entry) = batch.calls().first() {
            let output = result(entry.call(), ToolResultStatus::Rejected, json!("denied"));
            batch
                .resolve_at(batch.completed_len(), new_block_id(), output)
                .unwrap();
        }
        Ok(())
    }
}
#[tokio::test]
async fn before_can_complete_all_calls_and_after_still_runs() {
    let tool = NamedTool::new("work");
    let after = Arc::new(AtomicUsize::new(0));
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(Reject)],
            after: vec![Arc::new(Count(after.clone()))],
            ..Default::default()
        },
    );
    let mut batch = batch(&["work", "work"]);
    executor
        .bind(invocation(), control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    assert_eq!(statuses(&batch), vec![ToolResultStatus::Rejected; 2]);
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
    assert_eq!(after.load(Ordering::SeqCst), 1);
}
struct Rewrite {
    name: &'static str,
}
#[async_trait]
impl ToolBatchProcessor for Rewrite {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        assert_eq!(context.turn_id, &invocation().turn_id);
        assert_eq!(context.round_id, invocation().round_id);
        for entry in batch.calls_mut() {
            assert!(
                context
                    .declaration_order
                    .contains(&entry.call().call_block_id)
            );
            entry.input_mut().unwrap().tool_name = self.name.into();
            entry.input_mut().unwrap().arguments = json!({"changed": true});
            entry.push_note(TextPayload::new("before note"));
        }
        Ok(())
    }
}
#[tokio::test]
async fn original_unadvertised_name_cannot_be_rescued_by_before() {
    let tool = NamedTool::new("real");
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(Rewrite { name: "real" })],
            ..Default::default()
        },
    );
    let mut batch = batch(&["unadvertised alias"]);
    executor
        .bind(invocation(), control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    assert_eq!(statuses(&batch), vec![ToolResultStatus::Rejected]);
    assert_eq!(
        batch.results()[0].call().input.tool_name,
        "unadvertised alias"
    );
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn legal_name_rewritten_outside_binding_is_rejected() {
    let tool = NamedTool::new("real");
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(Rewrite { name: "outside" })],
            ..Default::default()
        },
    );
    let mut batch = batch(&["real"]);
    executor
        .bind(invocation(), control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    assert_eq!(statuses(&batch), vec![ToolResultStatus::Rejected]);
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
    assert_eq!(
        batch.results()[0].result().unwrap().1.notes,
        vec![TextPayload::new("before note")]
    );
}
#[tokio::test]
async fn legal_name_rewritten_inside_binding_dispatches_the_fixed_target() {
    let first = NamedTool::labeled("first", "first object");
    let second = NamedTool::labeled("second", "second object");
    let executor = executor(
        vec![first.clone(), second.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(Rewrite { name: "second" })],
            ..Default::default()
        },
    );
    let mut batch = batch(&["first"]);
    let id = batch.declaration_ids()[0];
    executor
        .bind(invocation(), control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    let output = batch.results()[0].result().unwrap().1;
    assert_eq!(output.call_block_id, id);
    assert_eq!(
        output.output.content,
        json!({"object": "second object", "arguments": {"changed": true}})
    );
    assert_eq!(
        output.notes,
        [
            TextPayload::new("before note"),
            TextPayload::new("tool note")
        ]
    );
    assert_eq!(first.count.load(Ordering::SeqCst), 0);
    assert_eq!(second.count.load(Ordering::SeqCst), 1);
}
struct Reverse(Mutex<Vec<BlockId>>);
#[async_trait]
impl ToolBatchProcessor for Reverse {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        *self.0.lock().unwrap() = batch.declaration_ids();
        batch.results_mut().reverse();
        Ok(())
    }
}
#[tokio::test]
async fn after_reorder_preserves_results_and_is_the_returned_order() {
    let reverse = Arc::new(Reverse(Mutex::new(vec![])));
    let executor = executor(
        vec![NamedTool::new("a"), NamedTool::new("b")],
        ToolExecutorOptions {
            after: vec![reverse.clone()],
            ..Default::default()
        },
    );
    let mut batch = batch(&["a", "b"]);
    executor
        .bind(invocation(), control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    let mut prior = reverse.0.lock().unwrap().clone();
    prior.reverse();
    assert_eq!(batch.declaration_ids(), prior);
    assert_eq!(statuses(&batch), vec![ToolResultStatus::Succeeded; 2]);
}
struct ModifyThenFail;
#[async_trait]
impl ToolBatchProcessor for ModifyThenFail {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        for entry in batch.calls_mut() {
            entry.push_note(TextPayload::new("kept mutation"));
        }
        for entry in batch.results_mut() {
            entry.output_mut().unwrap().content = json!("kept mutation");
        }
        Err(ProcessorError::Failed("original processor failure".into()))
    }
}
#[tokio::test]
async fn before_and_after_errors_keep_edits_original_error_and_skip_later_processors() {
    for phase in [ToolProcessorPhase::Before, ToolProcessorPhase::After] {
        let tool = NamedTool::new("work");
        let count = Arc::new(AtomicUsize::new(0));
        let sequence: Vec<Arc<dyn ToolBatchProcessor>> = vec![
            Arc::new(Count(Arc::new(AtomicUsize::new(0)))),
            Arc::new(ModifyThenFail),
            Arc::new(Count(count.clone())),
        ];
        let options = match phase {
            ToolProcessorPhase::Before => ToolExecutorOptions {
                before: sequence,
                ..Default::default()
            },
            ToolProcessorPhase::After => ToolExecutorOptions {
                after: sequence,
                ..Default::default()
            },
        };
        let executor = executor(vec![tool.clone()], options);
        let mut batch = batch(&["work"]);
        let parent = control();
        let error = executor
            .bind(invocation(), parent.clone())
            .await
            .unwrap()
            .process(&mut batch)
            .await
            .unwrap_err();
        assert!(
            matches!(error, ToolProcessingError::Processor { phase: actual, index: 1, error: ProcessorError::Failed(message) } if actual == phase && message == "original processor failure")
        );
        assert!(!parent.is_cancelled());
        assert_eq!(count.load(Ordering::SeqCst), 0);
        if phase == ToolProcessorPhase::Before {
            assert_eq!(
                batch.calls()[0].call().result_notes,
                [TextPayload::new("kept mutation")]
            );
            assert_eq!(tool.count.load(Ordering::SeqCst), 0);
        } else {
            assert_eq!(
                batch.results()[0].result().unwrap().1.output.content,
                json!("kept mutation")
            );
            assert_eq!(tool.count.load(Ordering::SeqCst), 1);
        }
    }
}
#[tokio::test]
async fn precompleted_input_is_rejected_unchanged_without_processors_or_calls() {
    let tool = NamedTool::new("work");
    let count = Arc::new(AtomicUsize::new(0));
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(Count(count.clone()))],
            ..Default::default()
        },
    );
    let mut batch = batch(&["work"]);
    let call = batch.calls()[0].call().clone();
    batch
        .resolve_at(
            0,
            new_block_id(),
            result(&call, ToolResultStatus::Succeeded, json!("already")),
        )
        .unwrap();
    let input = format!("{batch:?}");
    assert!(matches!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .process(&mut batch)
            .await,
        Err(ToolProcessingError::NotFreshBatch { completed: 1 })
    ));
    assert_eq!(format!("{batch:?}"), input);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
}
struct Replace {
    reopen: bool,
    remove_notes: bool,
    change_status: bool,
}
#[async_trait]
impl ToolBatchProcessor for Replace {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let original: Vec<_> = batch
            .results()
            .iter()
            .chain(batch.calls())
            .map(|entry| {
                (
                    entry.call().clone(),
                    entry.result().map(|(id, value)| (*id, value.clone())),
                )
            })
            .collect();
        let calls = original
            .iter()
            .map(|(call, _)| {
                let mut call = call.clone();
                if self.remove_notes {
                    call.result_notes.clear();
                }
                call
            })
            .collect();
        let mut replacement = ToolBatch::new(calls).unwrap();
        if !self.reopen {
            for (call, output) in original {
                if let Some((id, mut output)) = output {
                    if self.change_status {
                        output.status = ToolResultStatus::Succeeded;
                    }
                    if self.remove_notes {
                        output.notes.clear();
                    }
                    let index = replacement.completed_len()
                        + replacement
                            .calls()
                            .iter()
                            .position(|entry| entry.call().call_block_id == call.call_block_id)
                            .unwrap();
                    replacement.resolve_at(index, id, output).unwrap();
                }
            }
        }
        *batch = replacement;
        Ok(())
    }
}
#[tokio::test]
async fn rejected_results_cannot_reopen_change_status_or_lose_notes() {
    for replacement in [
        Replace {
            reopen: true,
            remove_notes: false,
            change_status: false,
        },
        Replace {
            reopen: false,
            remove_notes: false,
            change_status: true,
        },
        Replace {
            reopen: false,
            remove_notes: true,
            change_status: false,
        },
    ] {
        let tool = NamedTool::new("work");
        let executor = executor(
            vec![tool.clone()],
            ToolExecutorOptions {
                before: vec![
                    Arc::new(Rewrite { name: "work" }),
                    Arc::new(Reject),
                    Arc::new(replacement),
                ],
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
            Err(ToolProcessingError::Handoff {
                phase: ToolProcessorPhase::Before,
                index: 2,
                ..
            })
        ));
        assert_eq!(tool.count.load(Ordering::SeqCst), 0);
    }
}
struct Foreign;
#[async_trait]
impl ToolBatchProcessor for Foreign {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        *batch = tool_fixtures::batch(&["work"]);
        Ok(())
    }
}
#[tokio::test]
async fn foreign_membership_is_rejected_at_handoff_without_dispatch() {
    let tool = NamedTool::new("work");
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(Foreign)],
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
        Err(ToolProcessingError::Handoff {
            phase: ToolProcessorPhase::Before,
            index: 0,
            ..
        })
    ));
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
}

struct ChangeCompletedFact {
    identity: bool,
}
#[async_trait]
impl ToolBatchProcessor for ChangeCompletedFact {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let entry = &batch.results()[0];
        let mut call = entry.call().clone();
        let (result_id, result) = entry.result().unwrap();
        let result = result.clone();
        let result_id = if self.identity {
            new_block_id()
        } else {
            call.input.arguments = json!({"changed after execution": true});
            *result_id
        };
        let mut replacement = ToolBatch::new(vec![call]).unwrap();
        replacement.resolve_at(0, result_id, result).unwrap();
        *batch = replacement;
        Ok(())
    }
}
#[tokio::test]
async fn after_cannot_change_result_identity_or_completed_input() {
    for identity in [false, true] {
        let tool = NamedTool::new("work");
        let executor = executor(
            vec![tool.clone()],
            ToolExecutorOptions {
                after: vec![Arc::new(ChangeCompletedFact { identity })],
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
            Err(ToolProcessingError::Handoff {
                phase: ToolProcessorPhase::After,
                index: 0,
                ..
            })
        ));
        assert_eq!(tool.count.load(Ordering::SeqCst), 1);
    }
}
#[tokio::test]
async fn pending_notes_cannot_be_removed_by_recreating_the_same_declaration() {
    let tool = NamedTool::new("work");
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![
                Arc::new(Rewrite { name: "work" }),
                Arc::new(Replace {
                    reopen: false,
                    remove_notes: true,
                    change_status: false,
                }),
            ],
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
        Err(ToolProcessingError::Handoff {
            phase: ToolProcessorPhase::Before,
            index: 1,
            ..
        })
    ));
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn original_out_of_bounds_rejection_cannot_be_reopened() {
    let tool = NamedTool::new("work");
    let executor = executor(
        vec![tool.clone()],
        ToolExecutorOptions {
            before: vec![Arc::new(Replace {
                reopen: true,
                remove_notes: false,
                change_status: false,
            })],
            ..Default::default()
        },
    );
    let mut batch = batch(&["outside"]);
    assert!(matches!(
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .process(&mut batch)
            .await,
        Err(ToolProcessingError::Handoff {
            phase: ToolProcessorPhase::Before,
            index: 0,
            ..
        })
    ));
    assert_eq!(tool.count.load(Ordering::SeqCst), 0);
}
