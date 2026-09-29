//! Contract probes for interruption, commit boundaries, and slice handoffs.

use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::*;
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Gateway(Arc<AtomicUsize>, usize);
#[async_trait]
impl ModelGateway for Gateway {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let first = self.0.fetch_add(1, Ordering::SeqCst) == 0;
        Ok(ModelOutput {
            response: ModelResponse {
                text: TextPayload::new(if first { "" } else { "done" }),
                tool_calls: if first {
                    (0..self.1)
                        .map(|i| ToolCallDraft {
                            tool_name: "work".into(),
                            arguments: json!({"i": i}),
                            provider_call_id: None,
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
struct Work {
    invocations: Arc<AtomicUsize>,
    cancel_when_started: bool,
}
#[async_trait]
impl Tool for Work {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "work".into(),
            description: "fixture".into(),
            parameters: json!({"type":"object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, control: &CallControl) -> ToolResultPayload {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        if self.cancel_when_started {
            control.cancellation_token().cancel();
            std::future::pending::<()>().await;
        }
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("known")),
            media: vec![],
            notes: vec![TextPayload::new("tool note")],
        }
    }
}
fn runner(
    chain: ToolProcessingChain,
    cancel_when_started: bool,
    calls: usize,
) -> (TurnRunner, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let model_count = Arc::new(AtomicUsize::new(0));
    let tool_count = Arc::new(AtomicUsize::new(0));
    (
        TurnRunner::with_tool_processors(
            Arc::new(Gateway(model_count.clone(), calls)),
            Arc::new(ToolExecutor::from_vec(vec![Arc::new(Work {
                invocations: tool_count.clone(),
                cancel_when_started,
            })])),
            chain,
        ),
        model_count,
        tool_count,
    )
}
async fn run(runner: &TurnRunner) -> TurnOutcome {
    let mut context = TurnContext::new(TurnId::new("boundary"));
    context
        .append_input(new_block_id(), TextPayload::new("input"), "user")
        .unwrap();
    // A Continue policy must not override whole-turn cancellation.
    let mut options = TurnRunOptions::default();
    options.policy.unknown_outcome.default = UnknownOutcomePolicy::Continue;
    runner
        .run(
            context,
            options,
            RunControl::new(CancellationToken::new(), None),
        )
        .await
}
fn assert_no_committed_results(outcome: &TurnOutcome) {
    assert!(
        !outcome
            .context
            .blocks()
            .iter()
            .any(|block| matches!(block.content(), BlockContent::ToolResult(_)))
    );
}
struct CancelProcessor;
#[async_trait]
impl ToolBatchProcessor for CancelProcessor {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        for entry in batch.calls_mut() {
            entry.push_note(TextPayload::new("pre note"));
        }
        for entry in batch.results_mut() {
            entry.output_mut().unwrap().content = json!("post edit");
        }
        ctx.control.cancellation_token().cancel();
        Ok(())
    }
}
struct CountProcessor(Arc<AtomicUsize>);
#[async_trait]
impl ToolBatchProcessor for CountProcessor {
    async fn process(
        &self,
        _batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
#[tokio::test]
async fn synchronous_pre_cancel_stops_the_next_processor_and_dispatch() {
    let count = Arc::new(AtomicUsize::new(0));
    let chain = ToolProcessingChain::builder()
        .before(Arc::new(CancelProcessor))
        .before(Arc::new(CountProcessor(count.clone())))
        .build();
    let (runner, models, tools) = runner(chain, false, 2);
    let outcome = run(&runner).await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ExplicitCancellation
        }
    ));
    assert_no_committed_results(&outcome);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert_eq!(tools.load(Ordering::SeqCst), 0);
    assert_eq!(models.load(Ordering::SeqCst), 1);
    let batch = outcome.uncommitted_tool_batch.unwrap();
    assert_eq!(batch.calls().len(), 2);
    assert_eq!(batch.calls()[0].call().result_notes[0].0, "pre note");
}
#[tokio::test]
async fn synchronous_post_cancel_keeps_known_result_and_edit_without_commit() {
    let chain = ToolProcessingChain::builder()
        .after(Arc::new(CancelProcessor))
        .build();
    let (runner, models, tools) = runner(chain, false, 1);
    let outcome = run(&runner).await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ExplicitCancellation
        }
    ));
    assert_no_committed_results(&outcome);
    assert_eq!(tools.load(Ordering::SeqCst), 1);
    assert_eq!(models.load(Ordering::SeqCst), 1);
    let batch = outcome.uncommitted_tool_batch.unwrap();
    let (_, result) = batch.results()[0].result().unwrap();
    assert_eq!(result.status, ToolResultStatus::Succeeded);
    assert_eq!(result.output.content, json!("post edit"));
    assert_eq!(result.notes[0].0, "tool note");
}
#[tokio::test]
async fn cancellation_during_first_dispatch_does_not_start_remaining_calls() {
    let (runner, models, tools) = runner(ToolProcessingChain::default(), true, 4);
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(2), run(&runner))
        .await
        .unwrap();
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ExplicitCancellation
        }
    ));
    assert_no_committed_results(&outcome);
    assert_eq!(tools.load(Ordering::SeqCst), 1);
    assert_eq!(models.load(Ordering::SeqCst), 1);
    let batch = outcome.uncommitted_tool_batch.unwrap();
    assert_eq!(batch.completed_len(), 1);
    assert_eq!(
        batch.results()[0].result().unwrap().1.status,
        ToolResultStatus::UnknownOutcome
    );
    assert_eq!(batch.calls().len(), 3);
}
struct ForeignEntry;
#[async_trait]
impl ToolBatchProcessor for ForeignEntry {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let mut donor = ToolBatch::new(vec![ToolCallContext::from_declaration(
            new_block_id(),
            &batch.calls()[0].call().input,
        )])
        .unwrap();
        std::mem::swap(&mut batch.calls_mut()[0], &mut donor.calls_mut()[0]);
        Ok(())
    }
}
#[tokio::test]
async fn swapping_in_a_foreign_entry_is_rejected_at_handoff() {
    let chain = ToolProcessingChain::builder()
        .before(Arc::new(ForeignEntry))
        .build();
    let (runner, models, tools) = runner(chain, false, 1);
    let outcome = run(&runner).await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ProcessorFailed { .. }
        }
    ));
    assert_no_committed_results(&outcome);
    assert!(outcome.uncommitted_tool_batch.is_some());
    assert_eq!(tools.load(Ordering::SeqCst), 0);
    assert_eq!(models.load(Ordering::SeqCst), 1);
}
struct RemoveNotesByReplacement;
#[async_trait]
impl ToolBatchProcessor for RemoveNotesByReplacement {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let entry = &batch.results()[0];
        let (result_id, result) = entry.result().unwrap();
        let mut altered = result.clone();
        altered.notes.clear();
        let mut donor = ToolBatch::new(vec![entry.call().clone()]).unwrap();
        donor.resolve_at(0, *result_id, altered).unwrap();
        std::mem::swap(&mut batch.results_mut()[0], &mut donor.results_mut()[0]);
        Ok(())
    }
}
#[tokio::test]
async fn recreating_same_identity_cannot_remove_required_notes() {
    let chain = ToolProcessingChain::builder()
        .after(Arc::new(RemoveNotesByReplacement))
        .build();
    let (runner, models, tools) = runner(chain, false, 1);
    let outcome = run(&runner).await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ProcessorFailed { .. }
        }
    ));
    assert_no_committed_results(&outcome);
    assert!(outcome.uncommitted_tool_batch.is_some());
    assert_eq!(tools.load(Ordering::SeqCst), 1);
    assert_eq!(models.load(Ordering::SeqCst), 1);
}

struct KnownThenCancel;
#[async_trait]
impl Tool for KnownThenCancel {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "work".into(),
            description: "fixture".into(),
            parameters: json!({"type":"object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, control: &CallControl) -> ToolResultPayload {
        if ctx.input.arguments["i"] == 1 {
            // Let the other ready result be received before cancellation.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            control.cancellation_token().cancel();
            std::future::pending::<()>().await;
        }
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("already received")),
            media: vec![],
            notes: vec![],
        }
    }
}
#[tokio::test]
async fn a_received_result_survives_sibling_cancellation_without_becoming_unknown() {
    let model_count = Arc::new(AtomicUsize::new(0));
    let runner = TurnRunner::new(
        Arc::new(Gateway(model_count.clone(), 2)),
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(KnownThenCancel)])),
    );
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(2), run(&runner))
        .await
        .unwrap();
    assert_no_committed_results(&outcome);
    assert_eq!(model_count.load(Ordering::SeqCst), 1);
    let batch = outcome.uncommitted_tool_batch.unwrap();
    let results = batch
        .results()
        .iter()
        .map(|entry| entry.result().unwrap().1)
        .collect::<Vec<_>>();
    assert_eq!(results.len(), 2);
    let known = results
        .iter()
        .find(|result| result.status == ToolResultStatus::Succeeded)
        .unwrap();
    assert_eq!(known.output.content, json!("already received"));
    assert_eq!(
        results
            .iter()
            .filter(|result| result.status == ToolResultStatus::UnknownOutcome)
            .count(),
        1
    );
    assert_ne!(results[0].call_block_id, results[1].call_block_id);
}

#[tokio::test]
async fn explicit_cancel_wins_when_turn_deadline_is_also_expired() {
    let token = CancellationToken::new();
    token.cancel();
    let (runner, models, tools) = runner(ToolProcessingChain::default(), false, 1);
    let outcome = runner
        .run(
            TurnContext::new(TurnId::new("both-stopped")),
            TurnRunOptions::default(),
            RunControl::new(token, Some(std::time::Instant::now())),
        )
        .await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ExplicitCancellation
        }
    ));
    assert_eq!(models.load(Ordering::SeqCst), 0);
    assert_eq!(tools.load(Ordering::SeqCst), 0);
}
