//! End-to-end coverage for the opt-in tool-batch processor chain.

mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use async_trait::async_trait;
use causa_kernel::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, BlockContent, CallControl, ContentPart,
    ConversationId, MediaRef, ProcessorContext, ProcessorError, StoreError, TextPayload, Tool,
    ToolBatch, ToolBatchProcessor, ToolCallContext, ToolDefinition, ToolOutput, ToolResultPayload,
    ToolResultStatus, Truncation, TurnId,
};
use causa_runtime::{
    ConversationState, DeduplicateProcessor, RejectAllProcessor, Session, SessionConfig,
    SubmitRequest, TokenCounter, ToolExecutor, ToolOutputBudgetProcessor, ToolProcessingChain,
    TurnLimits, TurnPolicy, TurnResult, TurnRunOptions, TurnRunner, WaitEnd, WorkState,
    new_block_id,
};
use common::{EchoTool, RecordingGateway, ctrl, ctx, draft, endturn_output, tooluse_calls_output};
use serde_json::json;
use tokio::sync::{Notify, Semaphore};

struct NamedTool(&'static str, Arc<AtomicUsize>);

#[async_trait]
impl Tool for NamedTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.0.into(),
            description: self.0.into(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn execute(&self, call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        self.1.fetch_add(1, Ordering::SeqCst);
        ToolResultPayload {
            call_block_id: call.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!(self.0)),
            media: Vec::new(),
            notes: Vec::new(),
        }
    }
}

struct CountPost(Arc<AtomicUsize>);

#[async_trait]
impl ToolBatchProcessor for CountPost {
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
async fn asynchronous_preprocessor_holds_batch_before_execution_and_postprocessing() {
    struct Gate {
        entered: Arc<Notify>,
        release: Arc<Semaphore>,
    }
    #[async_trait]
    impl ToolBatchProcessor for Gate {
        async fn process(
            &self,
            _batch: &mut ToolBatch,
            _ctx: &ProcessorContext<'_>,
        ) -> Result<(), ProcessorError> {
            self.entered.notify_one();
            self.release.acquire().await.expect("gate is open").forget();
            Ok(())
        }
    }

    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let executions = Arc::new(AtomicUsize::new(0));
    let post_calls = Arc::new(AtomicUsize::new(0));
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output("", vec![draft("slow", json!({}))])),
        Ok(endturn_output("done")),
    ]);
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(NamedTool(
            "slow",
            executions.clone(),
        ))])),
        ToolProcessingChain::builder()
            .before(Arc::new(Gate {
                entered: entered.clone(),
                release: release.clone(),
            }))
            .after(Arc::new(CountPost(post_calls.clone())))
            .build(),
    );
    let task = tokio::spawn(async move {
        runner
            .run(ctx("async-chain"), TurnRunOptions::default(), ctrl())
            .await
    });
    entered.notified().await;
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert_eq!(post_calls.load(Ordering::SeqCst), 0);
    release.add_permits(1);
    let outcome = task.await.unwrap();
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(post_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn all_pre_rejections_still_reach_post_processor_without_execution() {
    let executions = Arc::new(AtomicUsize::new(0));
    let post_calls = Arc::new(AtomicUsize::new(0));
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "",
            vec![draft("first", json!({})), draft("second", json!({}))],
        )),
        Ok(endturn_output("done")),
    ]);
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![
            Arc::new(NamedTool("first", executions.clone())),
            Arc::new(NamedTool("second", executions.clone())),
        ])),
        ToolProcessingChain::builder()
            .before(Arc::new(RejectAllProcessor::new("approval denied")))
            .after(Arc::new(CountPost(post_calls.clone())))
            .build(),
    );
    let outcome = runner
        .run(ctx("all-rejected"), TurnRunOptions::default(), ctrl())
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert_eq!(post_calls.load(Ordering::SeqCst), 1);
    let statuses = outcome
        .context
        .blocks()
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result.status.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(statuses, vec![ToolResultStatus::Rejected; 2]);
}

struct ReverseResults;

#[async_trait]
impl ToolBatchProcessor for ReverseResults {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        batch.results_mut().reverse();
        Ok(())
    }
}

#[tokio::test]
async fn postprocessor_order_is_the_committed_result_order() {
    let executions = Arc::new(AtomicUsize::new(0));
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "",
            vec![draft("first", json!({})), draft("second", json!({}))],
        )),
        Ok(endturn_output("done")),
    ]);
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![
            Arc::new(NamedTool("first", executions.clone())),
            Arc::new(NamedTool("second", executions)),
        ])),
        ToolProcessingChain::builder()
            .after(Arc::new(ReverseResults))
            .build(),
    );
    let outcome = runner
        .run(ctx("post-order"), TurnRunOptions::default(), ctrl())
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    let results = outcome
        .context
        .blocks()
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolResult(result) => {
                Some((result.call_block_id, result.output.content.clone()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let calls = outcome
        .context
        .blocks()
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolCall(call) => Some((block.id(), call.tool_name.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls[0].1, "first");
    assert_eq!(calls[1].1, "second");
    assert_eq!(results[0].0, calls[1].0);
    assert_eq!(results[0].1, json!("second"));
    assert_eq!(results[1].0, calls[0].0);
    assert_eq!(results[1].1, json!("first"));
}

#[tokio::test]
async fn session_submit_returns_running_receipt_and_later_observes_same_new_id() {
    struct Gate {
        entered: Arc<Notify>,
        release: Arc<Semaphore>,
    }
    #[async_trait]
    impl ToolBatchProcessor for Gate {
        async fn process(
            &self,
            _batch: &mut ToolBatch,
            _ctx: &ProcessorContext<'_>,
        ) -> Result<(), ProcessorError> {
            self.entered.notify_one();
            self.release.acquire().await.expect("gate is open").forget();
            Ok(())
        }
    }

    let entered = Arc::new(Notify::new());
    let release = Arc::new(Semaphore::new(0));
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output("", vec![draft("echo", json!({}))])),
        Ok(endturn_output("done")),
    ]);
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(EchoTool)])),
        ToolProcessingChain::builder()
            .before(Arc::new(Gate {
                entered: entered.clone(),
                release: release.clone(),
            }))
            .build(),
    );
    let session = Session::new(
        ConversationState::new(ConversationId("long-task".into())),
        Arc::new(runner),
        TurnRunOptions {
            policy: TurnPolicy {
                limits: TurnLimits {
                    max_model_rounds: 3,
                    max_tool_calls: 3,
                },
                ..Default::default()
            },
            ..Default::default()
        },
        SessionConfig::default(),
    )
    .expect("session initializes");
    let handle = session.handle();
    let receipt = handle
        .submit(SubmitRequest {
            request_key: "long-work".into(),
            parts: vec![ContentPart::Text(TextPayload::new("go"))],
        })
        .expect("submit returns without waiting for runner");
    entered.notified().await;
    let running = handle
        .wait(&receipt.work, Duration::from_millis(5))
        .await
        .unwrap();
    assert_eq!(running.end, WaitEnd::TimedOut);
    assert_eq!(running.observation.state, WorkState::Running);
    assert_eq!(running.observation.work, receipt.work);
    release.add_permits(1);
    let finished = handle
        .wait(&receipt.work, Duration::from_secs(2))
        .await
        .unwrap();
    assert_eq!(finished.end, WaitEnd::ReachedState);
    assert_eq!(finished.observation.state, WorkState::Finished);
    assert_eq!(finished.observation.work.turn_id, receipt.work.turn_id);
}

fn resolved_batch(
    tool_name: &str,
    content: serde_json::Value,
    notes: Vec<TextPayload>,
    media: Vec<MediaRef>,
) -> (ToolBatch, causa_kernel::BlockId) {
    let call_id = new_block_id();
    let mut batch = ToolBatch::new(vec![ToolCallContext::from_declaration(
        call_id,
        &causa_kernel::ToolCallPayload {
            tool_name: tool_name.to_owned(),
            arguments: json!({}),
        },
    )])
    .unwrap();
    batch
        .resolve_at(
            0,
            new_block_id(),
            ToolResultPayload {
                call_block_id: call_id,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(content),
                media,
                notes,
            },
        )
        .unwrap();
    (batch, call_id)
}

struct CountTokens;
impl TokenCounter for CountTokens {
    fn estimate(&self, _blocks: &[causa_kernel::ContextBlock]) -> usize {
        0
    }

    fn estimate_value(&self, value: &serde_json::Value) -> usize {
        value
            .as_str()
            .map_or_else(|| value.to_string().len(), str::len)
    }

    fn estimate_media(&self, _media: &MediaRef) -> Option<usize> {
        Some(10)
    }
}

fn processor_context<'a>(
    turn_id: &'a TurnId,
    declaration_order: &'a [causa_kernel::BlockId],
    control: &'a CallControl,
) -> ProcessorContext<'a> {
    ProcessorContext {
        conversation_id: None,
        turn_id,
        round_id: causa_kernel::RoundId(0),
        declaration_order,
        control,
    }
}

#[tokio::test]
async fn output_budget_counts_notes_and_media_and_preserves_result_facts() {
    let media = MediaRef::new("image/png", "chart-1");
    let notes = vec![TextPayload::new("audit note")];
    let (mut batch, call_id) = resolved_batch(
        "bounded",
        json!("x".repeat(400)),
        notes.clone(),
        vec![media.clone()],
    );
    let control = CallControl::new(tokio_util::sync::CancellationToken::new(), None);
    let turn_id = TurnId::new("budget");
    let ids = batch.declaration_ids();
    let ctx = processor_context(&turn_id, &ids, &control);
    let processor = ToolOutputBudgetProcessor::new(usize::MAX)
        .for_tool("bounded", 80)
        .with_token_counter(Arc::new(CountTokens));
    processor.process(&mut batch, &ctx).await.unwrap();

    let (result_id, result) = batch.results()[0].result().unwrap();
    assert_ne!(*result_id, call_id);
    assert_eq!(result.call_block_id, call_id);
    assert_eq!(result.status, ToolResultStatus::Succeeded);
    assert_eq!(result.notes, notes);
    assert_eq!(result.media, vec![media]);
    assert_eq!(result.output.truncation, Truncation::Middle);
    let text = result.output.content.as_str().unwrap();
    assert!(text.len() < 400);
    assert!(text.contains("output truncated"));
    assert_eq!(
        result.output.meta.as_ref().unwrap().original_tokens,
        Some(400)
    );
}

#[tokio::test]
async fn output_budget_rejects_unestimated_media_and_preserves_current_result() {
    let (mut batch, _) = resolved_batch(
        "bounded",
        json!("large text"),
        vec![],
        vec![MediaRef::new("image/png", "image-1")],
    );
    let before = batch.results()[0].result().unwrap().1.clone();
    let control = CallControl::new(tokio_util::sync::CancellationToken::new(), None);
    let turn_id = TurnId::new("media");
    let ids = batch.declaration_ids();
    let ctx = processor_context(&turn_id, &ids, &control);
    let error = ToolOutputBudgetProcessor::new(1)
        .process(&mut batch, &ctx)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("media token estimator"));
    assert_eq!(batch.results()[0].result().unwrap().1, &before);
}

#[tokio::test]
async fn output_budget_keeps_unlimited_media_results_without_estimator() {
    let media = MediaRef::new("image/png", "image-1");
    let (mut batch, _) =
        resolved_batch("image", json!("x".repeat(100)), vec![], vec![media.clone()]);
    let control = CallControl::new(tokio_util::sync::CancellationToken::new(), None);
    let turn_id = TurnId::new("unlimited");
    let ids = batch.declaration_ids();
    let ctx = processor_context(&turn_id, &ids, &control);
    ToolOutputBudgetProcessor::new(usize::MAX)
        .for_tool("bounded", 10)
        .process(&mut batch, &ctx)
        .await
        .unwrap();
    let result = batch.results()[0].result().unwrap().1;
    assert_eq!(result.output.truncation, Truncation::None);
    assert_eq!(result.media, vec![media]);
}

#[tokio::test]
async fn output_budget_keeps_notes_and_fails_when_irreducible_material_exceeds_limit() {
    let notes = vec![TextPayload::new(
        "mandatory audit note that cannot be dropped",
    )];
    let (mut batch, _) = resolved_batch("bounded", json!("large output"), notes.clone(), vec![]);
    let before = batch.results()[0].result().unwrap().1.clone();
    let control = CallControl::new(tokio_util::sync::CancellationToken::new(), None);
    let turn_id = TurnId::new("irreducible");
    let ids = batch.declaration_ids();
    let ctx = processor_context(&turn_id, &ids, &control);
    let error = ToolOutputBudgetProcessor::new(8)
        .with_token_counter(Arc::new(CountTokens))
        .process(&mut batch, &ctx)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("notes and media"));
    let after = batch.results()[0].result().unwrap().1;
    assert_eq!(after.status, ToolResultStatus::Succeeded);
    assert_eq!(after.notes, notes);
    assert_eq!(after.output.content, before.output.content);
}

#[tokio::test]
async fn output_budget_spills_exact_original_and_replaces_stale_artifact() {
    struct CaptureStore(std::sync::Mutex<Vec<u8>>);
    #[async_trait]
    impl ArtifactStore for CaptureStore {
        async fn persist(
            &self,
            data: &[u8],
            _hint: ArtifactHint,
        ) -> Result<ArtifactRef, StoreError> {
            *self.0.lock().unwrap() = data.to_vec();
            Ok(ArtifactRef {
                id: "fresh-artifact".into(),
                size_bytes: data.len(),
                kind: ArtifactKind::FullOutput,
                persisted: true,
            })
        }
        async fn read(
            &self,
            _id: &str,
            _range: Option<std::ops::Range<u64>>,
        ) -> Result<Vec<u8>, StoreError> {
            Ok(Vec::new())
        }
    }

    let original = json!({"text":"original text ".repeat(50)});
    let original_bytes = serde_json::to_vec(&original).unwrap();
    let (mut batch, _) = resolved_batch("bounded", original.clone(), vec![], vec![]);
    batch.results_mut()[0].output_mut().unwrap().artifact = Some(ArtifactRef {
        id: "stale-artifact".into(),
        size_bytes: 1,
        kind: ArtifactKind::FullOutput,
        persisted: true,
    });
    let store = Arc::new(CaptureStore(std::sync::Mutex::new(Vec::new())));
    let control = CallControl::new(tokio_util::sync::CancellationToken::new(), None);
    let turn_id = TurnId::new("spill");
    let ids = batch.declaration_ids();
    let ctx = processor_context(&turn_id, &ids, &control);
    ToolOutputBudgetProcessor::new(60)
        .with_token_counter(Arc::new(CountTokens))
        .with_artifact_store(store.clone())
        .process(&mut batch, &ctx)
        .await
        .unwrap();
    assert_eq!(*store.0.lock().unwrap(), original_bytes);
    let result = batch.results()[0].result().unwrap().1;
    assert_eq!(
        result.output.artifact.as_ref().unwrap().id,
        "fresh-artifact"
    );
    assert_ne!(
        result.output.artifact.as_ref().unwrap().id,
        "stale-artifact"
    );
    assert!(
        result
            .output
            .content
            .as_str()
            .unwrap()
            .contains("truncated")
    );
}

#[tokio::test]
async fn dedup_rejects_later_declaration_but_allows_same_content_next_round() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_calls_output(
            "",
            vec![draft("echo", json!({"x":1})), draft("echo", json!({"x":1}))],
        )),
        Ok(tooluse_calls_output(
            "",
            vec![draft("echo", json!({"x":1}))],
        )),
        Ok(endturn_output("done")),
    ]);
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(EchoTool)])),
        ToolProcessingChain::builder()
            .before(Arc::new(DeduplicateProcessor))
            .build(),
    );
    let outcome = runner
        .run(ctx("dedup"), TurnRunOptions::default(), ctrl())
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    let statuses = outcome
        .context
        .blocks()
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result.status.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(statuses.len(), 3);
    assert_eq!(
        statuses
            .iter()
            .filter(|status| **status == ToolResultStatus::Rejected)
            .count(),
        1
    );
    assert_eq!(
        statuses
            .iter()
            .filter(|status| **status == ToolResultStatus::Succeeded)
            .count(),
        2
    );
}
