//! Close a tool batch that an interrupted run handed back. Runnable offline:
//! `cargo run -p causa-runtime --example interrupted_batch`
//!
//! The model declares two calls while the run allows one, so the run stops at
//! the tool-call limit before any tool starts. The declarations are already
//! in the context; their results are not, and a context in that state cannot
//! be sent to a model. The host answers each pending call with a result that
//! says it did not run, commits the batch, and runs again.

use async_trait::async_trait;
use causa_kernel::{
    BlockContent, BlockMeta, CallControl, Context, ContextBlock, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelItem, ModelOutput, ModelRef, ModelRequest, ModelResponse,
    ModelStopReason, Role, Tool, ToolBatch, ToolCallContext, ToolCallDraft, ToolDefinition,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnId, validate_tool_result_append,
};
use causa_runtime::{
    ToolExecutor, ToolExecutorOptions, TurnInterruption, TurnLimits, TurnResult, TurnRunOptions,
    TurnRunner, new_block_id,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

/// Scripted stub gateway: one round that declares two calls, then the answer.
struct ScriptedGateway(Mutex<Vec<ModelOutput>>);

#[async_trait]
impl ModelGateway for ScriptedGateway {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut outputs = self.0.lock().expect("script lock");
        if outputs.is_empty() {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                "script exhausted",
            ));
        }
        Ok(outputs.remove(0))
    }
}

struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "read_file",
            "Read a file.",
            json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn causa_kernel::ArtifactStore>,
    ) -> ToolResultPayload {
        ToolResultPayload::new(
            ctx.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!({"contents": "file contents"})),
        )
    }
}

/// Answers every call that has no result yet, then appends all results of the
/// batch to the context. Calls that already have a result keep it.
fn close_batch(
    context: &mut Context,
    mut batch: ToolBatch,
    reason: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Pending calls sit after the completed ones; resolving the first pending
    // entry moves it into the completed prefix.
    while let Some(entry) = batch.calls().first() {
        let call_block_id = entry.call().call_block_id;
        batch
            .resolve_at(
                batch.completed_len(),
                new_block_id(),
                ToolResultPayload::new(
                    call_block_id,
                    ToolResultStatus::Cancelled,
                    ToolOutput::new(json!({"error": reason})),
                ),
            )
            .map_err(|rejected| rejected.error)?;
    }
    let results = batch.into_results().map_err(|(error, _batch)| error)?;
    // The same check the runner makes before it commits a batch.
    validate_tool_result_append(context.blocks(), &results)?;
    let blocks = results
        .into_iter()
        .map(|(id, result)| {
            ContextBlock::new(id, BlockContent::ToolResult(result), BlockMeta::default())
        })
        .collect();
    context.apply(Vec::new(), blocks)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gateway = Arc::new(ScriptedGateway(Mutex::new(vec![
        ModelOutput::new(
            ModelResponse {
                items: ["a.txt", "b.txt"]
                    .into_iter()
                    .map(|path| {
                        ModelItem::tool_call(ToolCallDraft::new("read_file", json!({"path": path})))
                    })
                    .collect(),
            },
            ModelStopReason::ToolUse,
        ),
        ModelOutput::new(
            ModelResponse {
                items: vec![ModelItem::text("Neither file was read.")],
            },
            ModelStopReason::EndTurn,
        ),
    ])));
    let executor = ToolExecutor::new(vec![Arc::new(ReadFile)], ToolExecutorOptions::default())?;
    let runner = TurnRunner::new(gateway, Arc::new(executor));
    let model = ModelRef::new("offline-demo");

    let mut context = Context::new();
    context.apply(
        Vec::new(),
        vec![ContextBlock::text(
            new_block_id(),
            Role::User,
            "Read both files.",
        )],
    )?;

    // First run: one call allowed, two declared.
    let outcome = runner
        .run(
            TurnId::new("first"),
            context,
            TurnRunOptions::new(model.clone()).with_limits(TurnLimits::new(10, 1)),
            CallControl::new(Default::default(), None),
        )
        .await;
    let TurnResult::Interrupted {
        cause: cause @ TurnInterruption::MaxToolCalls { .. },
    } = &outcome.result
    else {
        return Err(format!("expected the tool-call limit, got {:?}", outcome.result).into());
    };
    println!("interrupted: {cause}");

    // The outcome returns everything: the context with the two declarations
    // and the batch that holds them.
    let mut context = outcome.context;
    let batch = outcome
        .uncommitted_tool_batch
        .ok_or("the interrupted run returned no batch")?;
    println!(
        "batch: {} completed, {} pending",
        batch.completed_len(),
        batch.calls().len()
    );
    close_batch(
        &mut context,
        batch,
        "not run: the tool-call limit was reached",
    )?;

    // Second run on the closed context.
    let outcome = runner
        .run(
            TurnId::new("second"),
            context,
            TurnRunOptions::new(model),
            CallControl::new(Default::default(), None),
        )
        .await;
    match outcome.result {
        TurnResult::Completed { final_output } => {
            println!("completed: {}", final_output.response.text());
            Ok(())
        }
        other => Err(format!("turn did not complete: {other:?}").into()),
    }
}
