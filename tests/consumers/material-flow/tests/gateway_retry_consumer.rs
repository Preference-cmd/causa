#![cfg(feature = "runtime")]

//! Consumer-form acceptance for the optional retry wrapper: a tool loop whose
//! second round is rate limited twice still completes, with preparation,
//! binding and the final commit each happening once per logical round; and
//! without a wrapper the first failure is the single call's result.

use async_trait::async_trait;
use causa::kernel::*;
use causa::runtime::*;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct State {
    calls: AtomicUsize,
    requests: Mutex<Vec<ModelRequest>>,
    script: Mutex<VecDeque<Result<ModelOutput, ModelInvokeError>>>,
}

struct Scripted {
    state: Arc<State>,
}

fn gateway_state(script: Vec<Result<ModelOutput, ModelInvokeError>>) -> Arc<State> {
    Arc::new(State {
        calls: AtomicUsize::new(0),
        requests: Mutex::new(Vec::new()),
        script: Mutex::new(script.into()),
    })
}

fn output(items: Vec<ModelItem>, stop_reason: ModelStopReason) -> ModelOutput {
    ModelOutput::new(ModelResponse { items }, stop_reason)
}

fn tool_round() -> ModelOutput {
    output(
        vec![ModelItem::tool_call(
            ToolCallDraft::new("local", json!({})).with_provider_call_id("call-1"),
        )],
        ModelStopReason::ToolUse,
    )
}

fn rate_limited() -> ModelInvokeError {
    ModelInvokeError::new(ModelInvokeErrorKind::Transient, "429 rate limited")
}

#[async_trait]
impl ModelGateway for Scripted {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.state.calls.fetch_add(1, Ordering::SeqCst);
        self.state.requests.lock().unwrap().push(request.clone());
        self.state
            .script
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected invoke")
    }
}

struct CountingTool {
    definitions: Arc<AtomicUsize>,
    executions: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CountingTool {
    fn definition(&self) -> ToolDefinition {
        self.definitions.fetch_add(1, Ordering::SeqCst);
        ToolDefinition::new("local", "host tool", json!({"type":"object"}))
    }

    async fn execute(
        &self,
        call: &ToolCallContext,
        _: &CallControl,
        _store: Option<&dyn causa::kernel::ArtifactStore>,
    ) -> ToolResultPayload {
        self.executions.fetch_add(1, Ordering::SeqCst);
        ToolResultPayload::new(
            call.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!("tool output")),
        )
    }
}

struct CountingPreparer(Arc<AtomicUsize>);

#[async_trait]
impl ContextPreparer for CountingPreparer {
    async fn prepare(
        &self,
        context: &mut Context,
        _round: &RoundInfo<'_>,
        _: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(context.frame())
    }
}

fn input() -> ContextBlock {
    ContextBlock::text(new_block_id(), Role::User, "go")
}

fn is_final_answer(block: &ContextBlock) -> bool {
    matches!(
        block.content(),
        BlockContent::Message { parts, .. }
            if parts.iter().any(|part| matches!(part, ContentPart::Text(text) if text.0 == "final answer"))
    )
}

#[tokio::test]
async fn a_three_attempt_wrapper_completes_a_rate_limited_second_round() {
    let state = gateway_state(vec![
        Ok(tool_round()),
        Err(rate_limited()),
        Err(rate_limited()),
        Ok(output(
            vec![ModelItem::text("final answer")],
            ModelStopReason::EndTurn,
        )),
    ]);
    let definitions = Arc::new(AtomicUsize::new(0));
    let executions = Arc::new(AtomicUsize::new(0));
    let preparations = Arc::new(AtomicUsize::new(0));
    let tool = Arc::new(CountingTool {
        definitions: definitions.clone(),
        executions: executions.clone(),
    });
    let executor = ToolExecutor::new(vec![tool], ToolExecutorOptions::default()).unwrap();
    // The executor's own registry lookup happens once, before any round.
    let definitions_before = definitions.load(Ordering::SeqCst);
    let gateway = RetryingGateway::new(
        Scripted {
            state: state.clone(),
        },
        RetryPolicy::new(3).with_base_delay(Duration::from_millis(1)),
    );
    let runner = TurnRunner::new(Arc::new(gateway), Arc::new(executor));
    let committed = Arc::new(Mutex::new(Vec::new()));
    let sink = committed.clone();
    let options = TurnRunOptions::new(ModelRef::new("host-model"))
        .with_preparer(Arc::new(CountingPreparer(preparations.clone())))
        .with_observer(Arc::new(move |event| {
            if let RunEvent::BlocksCommitted { blocks, .. } = event {
                sink.lock().unwrap().extend_from_slice(blocks);
            }
        }));
    let out = runner
        .run(
            TurnId::new("retry-loop"),
            Context::from_blocks(vec![input()]).unwrap(),
            options,
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));

    // One logical call in round 1, three attempts in round 2.
    assert_eq!(state.calls.load(Ordering::SeqCst), 4);
    let requests = state.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert_ne!(requests[0].invocation_id, requests[1].invocation_id);
    assert_eq!(requests[1].invocation_id, requests[2].invocation_id);
    assert_eq!(requests[2].invocation_id, requests[3].invocation_id);
    assert_eq!(requests[1].model.0, requests[3].model.0);
    assert_eq!(requests[1].frame.blocks, requests[3].frame.blocks);
    drop(requests);

    // Preparation and binding run once per round, not once per attempt.
    assert_eq!(preparations.load(Ordering::SeqCst), 2);
    assert_eq!(
        definitions.load(Ordering::SeqCst) - definitions_before,
        2,
        "one bind per round"
    );
    // The tool executes once, after the tool-call round commits.
    assert_eq!(executions.load(Ordering::SeqCst), 1);

    // The successful round-2 output is committed exactly once.
    assert_eq!(
        out.context
            .blocks()
            .iter()
            .filter(|b| is_final_answer(b))
            .count(),
        1
    );
    assert_eq!(
        committed
            .lock()
            .unwrap()
            .iter()
            .filter(|b| is_final_answer(b))
            .count(),
        1
    );
}

#[tokio::test]
async fn without_a_wrapper_one_failure_is_one_gateway_call() {
    let state = gateway_state(vec![
        Err(rate_limited()),
        Ok(output(
            vec![ModelItem::text("never returned")],
            ModelStopReason::EndTurn,
        )),
    ]);
    let executor = ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap();
    let runner = TurnRunner::new(
        Arc::new(Scripted {
            state: state.clone(),
        }),
        Arc::new(executor),
    );
    let out = runner
        .run(
            TurnId::new("unwrapped"),
            Context::from_blocks(vec![input()]).unwrap(),
            TurnRunOptions::new(ModelRef::new("host-model")),
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ModelCallFailed { .. }
        }
    ));
    assert_eq!(state.calls.load(Ordering::SeqCst), 1);
    // No partial material is invented; the admitted input stays.
    assert_eq!(out.context.blocks().len(), 1);
}
