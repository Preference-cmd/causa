// The "Handle interrupted runs" guide. Runs offline: the model calls a quick
// tool and a slow one, the run is cancelled while the slow one works, and
// the program closes the batch it gets back and runs again.

// ANCHOR: imports
use causa::{
    kernel::{CallControl, CancellationToken, Context, ContextBlock, ModelRef, Role, Tool, TurnId},
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnInterruption, TurnResult, TurnRunOptions,
        TurnRunner, new_block_id,
    },
};
use std::{sync::Arc, time::Duration};
// ANCHOR_END: imports

use async_trait::async_trait;
use causa::kernel::{
    ArtifactStore, ModelItem, ModelOutput, ModelRequest, ModelResponse, ModelStopReason,
    ToolCallContext, ToolCallDraft, ToolDefinition, ToolOutput, ToolResultPayload,
    ToolResultStatus,
};
use causa_site_guides::interrupted_runs::close_batch;
use causa_site_guides::scripted::{ScriptedModel, reply, tool_results};
use serde_json::json;

/// Waits `delay`, then reports that it ran.
struct Wait {
    name: &'static str,
    delay: Duration,
}

#[async_trait]
impl Tool for Wait {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(self.name, "Do some work.", json!({"type": "object"}))
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        tokio::time::sleep(self.delay).await;
        ToolResultPayload::new(
            ctx.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!("done")),
        )
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Stands in for a provider: calls both tools, then lists the results.
    let gateway = Arc::new(ScriptedModel(|request: &ModelRequest| {
        let results = tool_results(request);
        if results.is_empty() {
            let call = |name: &str| ModelItem::tool_call(ToolCallDraft::new(name, json!({})));
            return ModelOutput::new(
                ModelResponse {
                    items: vec![call("quick"), call("slow")],
                },
                ModelStopReason::ToolUse,
            );
        }
        let statuses: Vec<String> = results.iter().map(|r| format!("{:?}", r.status)).collect();
        reply(format!("Results so far: {}.", statuses.join(", ")))
    }));
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(Wait {
            name: "quick",
            delay: Duration::ZERO,
        }),
        Arc::new(Wait {
            name: "slow",
            delay: Duration::from_secs(30),
        }),
    ];
    let executor = Arc::new(ToolExecutor::new(tools, ToolExecutorOptions::default())?);
    let runner = TurnRunner::new(gateway, executor);
    let context = Context::from_blocks(vec![ContextBlock::text(
        new_block_id(),
        Role::User,
        "Run both jobs.",
    )])?;

    // The user presses stop while the slow tool is working.
    let stop = CancellationToken::new();
    let pressed = stop.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        pressed.cancel();
    });
    let outcome = runner
        .run(
            TurnId::new("jobs"),
            context,
            TurnRunOptions::new(ModelRef::new("scripted")),
            CallControl::new(stop, None),
        )
        .await;

    // ANCHOR: outcome
    let mut context = outcome.context;
    if let TurnResult::Interrupted {
        cause: TurnInterruption::Cancelled { .. },
    } = &outcome.result
    {
        println!("stopped: cancelled");
    }
    if let Some(batch) = outcome.uncommitted_tool_batch {
        for entry in batch.results() {
            if let Some((_, result)) = entry.result() {
                println!("{}: {:?}", entry.call().input.tool_name, result.status);
            }
        }
        println!("not started: {}", batch.calls().len());
        // Answer what never started, then commit every result.
        close_batch(&mut context, batch, "stopped by the user")?;
    }
    // ANCHOR_END: outcome

    // ANCHOR: again
    let next = runner
        .run(
            TurnId::new("jobs-2"),
            context,
            TurnRunOptions::new(ModelRef::new("scripted")),
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    // ANCHOR_END: again
    match next.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        result => return Err(format!("turn did not complete: {result:?}").into()),
    }
    Ok(())
}
