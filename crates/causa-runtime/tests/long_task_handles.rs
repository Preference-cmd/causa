//! Long-running work is represented by ordinary tool results. The model
//! chooses when to query a tool-owned handle; the runner never polls or waits
//! for the background task on its own.

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, BlockContent, CallControl, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelOutput, ModelRequest, ModelResponse, ModelStopReason, ModelStream,
    ProcessorContext, ProcessorError, TextPayload, Tool, ToolBatch, ToolBatchProcessor,
    ToolCallContext, ToolCallDraft, ToolDefinition, ToolOutput, ToolResultPayload,
    ToolResultStatus,
};
use causa_runtime::{
    RunControl, ToolExecutor, ToolProcessingChain, TurnInterruption, TurnResult, TurnRunOptions,
    TurnRunner,
};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

const HANDLE: &str = "build-17";

struct BackgroundJob {
    started: AtomicBool,
    finished: AtomicBool,
    polls: AtomicUsize,
    query_started: AtomicBool,
    start_notice: Notify,
    query_notice: Notify,
    finish_notice: Notify,
    finish_signal: Notify,
}

impl BackgroundJob {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            started: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            polls: AtomicUsize::new(0),
            query_started: AtomicBool::new(false),
            start_notice: Notify::new(),
            query_notice: Notify::new(),
            finish_notice: Notify::new(),
            finish_signal: Notify::new(),
        })
    }

    async fn start(self: &Arc<Self>) {
        let job = self.clone();
        tokio::spawn(async move {
            job.started.store(true, Ordering::SeqCst);
            job.start_notice.notify_one();
            job.finish_signal.notified().await;
            job.finished.store(true, Ordering::SeqCst);
            job.finish_notice.notify_one();
        });
        self.wait_started().await;
    }

    async fn wait_started(&self) {
        if self.started.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::timeout(Duration::from_secs(1), self.start_notice.notified())
            .await
            .expect("background worker starts")
    }

    async fn wait_query_started(&self) {
        if self.query_started.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::timeout(Duration::from_secs(1), self.query_notice.notified())
            .await
            .expect("model-directed query starts")
    }

    async fn finish_from_fixture(&self) {
        self.finish_signal.notify_one();
        if self.finished.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::timeout(Duration::from_secs(1), self.finish_notice.notified())
            .await
            .expect("background worker receives the fixture completion signal");
        assert!(self.finished.load(Ordering::SeqCst));
    }
}

struct ScriptedGateway {
    outputs: Mutex<VecDeque<ModelOutput>>,
    calls: AtomicUsize,
    finish_job_before_call: Option<(usize, Arc<BackgroundJob>)>,
}

impl ScriptedGateway {
    fn new(outputs: Vec<ModelOutput>) -> Arc<Self> {
        Arc::new(Self {
            outputs: Mutex::new(outputs.into()),
            calls: AtomicUsize::new(0),
            finish_job_before_call: None,
        })
    }

    fn finishing_job_on_call(
        outputs: Vec<ModelOutput>,
        call_index: usize,
        job: Arc<BackgroundJob>,
    ) -> Arc<Self> {
        Arc::new(Self {
            outputs: Mutex::new(outputs.into()),
            calls: AtomicUsize::new(0),
            finish_job_before_call: Some((call_index, job)),
        })
    }
}

#[async_trait]
impl ModelGateway for ScriptedGateway {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some((finish_index, job)) = &self.finish_job_before_call
            && call_index == *finish_index
        {
            job.finish_from_fixture().await;
        }
        self.outputs
            .lock()
            .expect("scripted outputs lock")
            .pop_front()
            .ok_or_else(|| {
                ModelInvokeError::new(
                    ModelInvokeErrorKind::Permanent,
                    "scripted gateway has no remaining output",
                )
            })
    }

    async fn stream(
        &self,
        _request: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Permanent,
            "long-task fixture exercises invoke only",
        ))
    }
}

struct JobTool(Arc<BackgroundJob>);

#[async_trait]
impl Tool for JobTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "job".into(),
            description: "Start, query, or stop querying a background job".into(),
            parameters: json!({"type":"object"}),
        }
    }

    async fn execute(&self, call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        let action = call.input.arguments["action"].as_str().unwrap_or_default();
        let handle = call.input.arguments["handle"].as_str();
        let output = match action {
            "start" => {
                self.0.start().await;
                json!({"status":"running", "handle":HANDLE})
            }
            "poll" if handle == Some(HANDLE) => {
                self.0.polls.fetch_add(1, Ordering::SeqCst);
                let status = if self.0.finished.load(Ordering::SeqCst) {
                    "complete"
                } else {
                    "running"
                };
                json!({"status":status, "handle":HANDLE})
            }
            "query" if handle == Some(HANDLE) => {
                self.0.polls.fetch_add(1, Ordering::SeqCst);
                self.0.query_started.store(true, Ordering::SeqCst);
                self.0.query_notice.notify_one();
                // Wait for a real status response. The runner's shared
                // cancellation path drops this future and records an
                // UnknownOutcome in the uncommitted batch.
                std::future::pending::<Value>().await
            }
            _ => json!({"error":"unknown action or handle"}),
        };
        ToolResultPayload {
            call_block_id: call.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(output),
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
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn tool_call(action: &str, handle: Option<&str>) -> ModelOutput {
    let mut arguments = serde_json::Map::new();
    arguments.insert("action".into(), Value::String(action.into()));
    if let Some(handle) = handle {
        arguments.insert("handle".into(), Value::String(handle.into()));
    }
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(""),
            tool_calls: vec![ToolCallDraft {
                tool_name: "job".into(),
                arguments: Value::Object(arguments),
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    }
}

fn end_turn(text: &str) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: Vec::new(),
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    }
}

fn start_id_and_results(
    outcome: &causa_runtime::TurnOutcome,
) -> (Vec<causa_kernel::BlockId>, Vec<ToolResultPayload>) {
    let mut declarations = Vec::new();
    let mut results = Vec::new();
    for block in outcome.context.blocks() {
        match block.content() {
            BlockContent::ToolCall(_) => declarations.push(block.id()),
            BlockContent::ToolResult(result) => results.push(result.clone()),
            _ => {}
        }
    }
    (declarations, results)
}

#[tokio::test]
async fn model_queries_long_task_handle_until_complete_without_framework_polling() {
    let job = BackgroundJob::new();
    let post_calls = Arc::new(AtomicUsize::new(0));
    let gateway = ScriptedGateway::finishing_job_on_call(
        vec![
            tool_call("start", None),
            tool_call("poll", Some(HANDLE)),
            tool_call("poll", Some(HANDLE)),
            end_turn("build completed"),
        ],
        2,
        job.clone(),
    );
    let runner = TurnRunner::with_tool_processors(
        gateway.clone(),
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(JobTool(job.clone()))])),
        ToolProcessingChain::builder()
            .after(Arc::new(CountPost(post_calls.clone())))
            .build(),
    );

    let outcome = tokio::time::timeout(
        Duration::from_secs(2),
        runner.run(
            causa_kernel::TurnContext::new(causa_kernel::TurnId::new("long-job")),
            TurnRunOptions::default(),
            causa_runtime::RunControl::new(tokio_util::sync::CancellationToken::new(), None),
        ),
    )
    .await
    .expect("the turn completes without awaiting a background process");

    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    assert_eq!(gateway.calls.load(Ordering::SeqCst), 4);
    assert_eq!(job.polls.load(Ordering::SeqCst), 2);
    assert_eq!(post_calls.load(Ordering::SeqCst), 3);

    let (declarations, results) = start_id_and_results(&outcome);
    assert_eq!(declarations.len(), 3);
    assert_eq!(results.len(), 3);
    assert_eq!(
        declarations
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert_eq!(
        results
            .iter()
            .map(|result| result.call_block_id)
            .collect::<Vec<_>>(),
        declarations
    );
    assert_eq!(results[0].status, ToolResultStatus::Succeeded);
    assert_eq!(
        results[0].output.content,
        json!({"status":"running", "handle":HANDLE})
    );
    assert_eq!(
        results[1].output.content,
        json!({"status":"running", "handle":HANDLE})
    );
    assert_eq!(
        results[2].output.content,
        json!({"status":"complete", "handle":HANDLE})
    );
}

#[tokio::test]
async fn cancelling_a_waiting_query_preserves_start_and_leaves_job_running() {
    let job = BackgroundJob::new();
    let post_calls = Arc::new(AtomicUsize::new(0));
    let gateway = ScriptedGateway::new(vec![
        tool_call("start", None),
        tool_call("query", Some(HANDLE)),
    ]);
    let runner = TurnRunner::with_tool_processors(
        gateway.clone(),
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(JobTool(job.clone()))])),
        ToolProcessingChain::builder()
            .after(Arc::new(CountPost(post_calls.clone())))
            .build(),
    );

    let token = tokio_util::sync::CancellationToken::new();
    let run_token = token.clone();
    let task = tokio::spawn(async move {
        runner
            .run(
                causa_kernel::TurnContext::new(causa_kernel::TurnId::new("cancel-query")),
                TurnRunOptions::default(),
                RunControl::new(run_token, None),
            )
            .await
    });
    job.wait_query_started().await;
    token.cancel();
    let outcome = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .expect("cancelling the pending query returns the interrupted turn")
        .unwrap();

    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ExplicitCancellation
        }
    ));
    assert_eq!(gateway.calls.load(Ordering::SeqCst), 2);
    assert_eq!(job.polls.load(Ordering::SeqCst), 1);
    assert_eq!(post_calls.load(Ordering::SeqCst), 1);
    assert!(job.started.load(Ordering::SeqCst));
    assert!(!job.finished.load(Ordering::SeqCst));

    let (declarations, results) = start_id_and_results(&outcome);
    assert_eq!(declarations.len(), 2);
    assert_ne!(declarations[0], declarations[1]);
    assert_eq!(
        results.len(),
        1,
        "the query result remains uncommitted in U"
    );
    assert_eq!(results[0].status, ToolResultStatus::Succeeded);
    assert_eq!(
        results[0].output.content,
        json!({"status":"running", "handle":HANDLE})
    );
    assert_eq!(results[0].call_block_id, declarations[0]);

    let pending = outcome
        .uncommitted_tool_batch
        .as_ref()
        .expect("the interrupted query returns its uncommitted batch");
    assert_eq!(pending.completed_len(), 1);
    assert!(pending.calls().is_empty());
    let (result_id, query_result) = pending.results()[0]
        .result()
        .expect("the started query is settled into U");
    assert_ne!(*result_id, declarations[1]);
    assert_eq!(query_result.call_block_id, declarations[1]);
    assert_eq!(query_result.status, ToolResultStatus::UnknownOutcome);
    assert_eq!(
        query_result.output.content,
        json!({"error":"tool was started but no result was observed before the batch stopped"})
    );
    // The worker can still receive its own completion signal after the turn
    // stops; cancellation did not abort or take ownership of it.
    job.finish_from_fixture().await;
}
