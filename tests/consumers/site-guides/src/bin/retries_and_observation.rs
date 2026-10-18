// The "Retries and observation" guide. Runs offline: a stand-in provider
// refuses its first call as rate-limited, the retry wrapper calls it again,
// and an observer prints what the runner does.

// ANCHOR: imports
use causa::{
    kernel::{CallControl, CancellationToken, Context, ContextBlock, ModelRef, Role, TurnId},
    runtime::{
        RetryPolicy, RetryingGateway, RunEvent, RunObserver, ToolExecutor, ToolExecutorOptions,
        TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use std::{sync::Arc, time::Duration};
// ANCHOR_END: imports

use async_trait::async_trait;
use causa::kernel::{
    ArtifactStore, ModelGateway, ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRequest,
    Tool, ToolCallContext, ToolDefinition, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use causa_site_guides::scripted::{ScriptedModel, call, reply, tool_results};
use serde_json::json;
use std::sync::atomic::{AtomicU32, Ordering};

/// Stands in for a provider: the first call is refused as rate-limited,
/// later calls go to a scripted model.
struct RateLimitedOnce<G> {
    inner: G,
    calls: AtomicU32,
}

#[async_trait]
impl<G: ModelGateway> ModelGateway for RateLimitedOnce<G> {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 1 {
            println!("  provider: call {call} refused (429)");
            return Err(
                ModelInvokeError::new(ModelInvokeErrorKind::Transient, "rate limited")
                    .with_status(429),
            );
        }
        println!("  provider: call {call} answered");
        self.inner.invoke(request, control).await
    }
}

/// Tells the time, always the same.
struct Clock;

#[async_trait]
impl Tool for Clock {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("clock", "The current time.", json!({"type": "object"}))
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
            ToolOutput::new(json!({ "time": "09:30" })),
        )
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = RateLimitedOnce {
        inner: ScriptedModel(
            |request: &ModelRequest| match tool_results(request).last() {
                None => call("clock", json!({})),
                Some(result) => reply(format!(
                    "It is {}.",
                    result.output.content["time"].as_str().unwrap_or_default()
                )),
            },
        ),
        calls: AtomicU32::new(0),
    };

    // ANCHOR: retry
    // Three calls in all: the first and two retries. The base delay is
    // 500 ms unless set; it is short here so the program is quick.
    let policy = RetryPolicy::new(3).with_base_delay(Duration::from_millis(10));
    let gateway = Arc::new(RetryingGateway::new(provider, policy));
    // ANCHOR_END: retry

    // ANCHOR: observer
    // Called synchronously, in order, for each step of the run. It borrows
    // what it is shown: copy what you keep, and return quickly.
    let observer: Arc<RunObserver> = Arc::new(|event: &RunEvent<'_>| match event {
        RunEvent::ModelRequestReady { request, .. } => {
            println!("request: {} blocks", request.frame.blocks.len())
        }
        RunEvent::ModelOutput { output, .. } => println!("output: {:?}", output.stop_reason),
        RunEvent::BlocksCommitted { blocks, .. } => println!("committed: {} blocks", blocks.len()),
        RunEvent::ToolBatchReady { batch, .. } => println!("tools: {} calls", batch.calls().len()),
        RunEvent::ToolBatchReturned { batch, .. } => {
            println!("tools: {} results", batch.results().len())
        }
        _ => {}
    });
    // ANCHOR_END: observer

    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Clock)];
    let executor = Arc::new(ToolExecutor::new(tools, ToolExecutorOptions::default())?);
    let runner = TurnRunner::new(gateway, executor);
    let context = Context::from_blocks(vec![ContextBlock::text(
        new_block_id(),
        Role::User,
        "What time is it?",
    )])?;

    // ANCHOR: run
    let options = TurnRunOptions::new(ModelRef::new("scripted")).with_observer(observer);
    let outcome = runner
        .run(
            TurnId::new("time"),
            context,
            options,
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    // ANCHOR_END: run

    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        result => return Err(format!("turn did not complete: {result:?}").into()),
    }
    Ok(())
}
