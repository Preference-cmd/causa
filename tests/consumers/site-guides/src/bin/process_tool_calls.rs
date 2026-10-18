// The "Process tool calls" guide. Runs offline: a scripted model asks to
// read two notes in one round, an approver task allows one, and the
// program prints the decisions and the results in the order they were
// declared.

// ANCHOR: imports
use async_trait::async_trait;
use causa::{
    kernel::{
        BlockContent, CallControl, CancellationToken, Context, ContextBlock, ModelRef,
        ProcessorContext, ProcessorError, Role, Tool, ToolBatch, ToolBatchProcessor,
        ToolCallPayload, ToolOutput, ToolResultPayload, ToolResultStatus, TurnId,
    },
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
// ANCHOR_END: imports

use causa::kernel::{
    ArtifactStore, ModelItem, ModelOutput, ModelResponse, ModelStopReason, ToolCallContext,
    ToolCallDraft, ToolDefinition,
};
use causa_site_guides::scripted::{ScriptedModel, reply, tool_results};

// ANCHOR: approval
/// One question for the approver: may this call run?
struct Approval {
    call: ToolCallPayload,
    reply: oneshot::Sender<bool>,
}

/// Asks before each call runs. A call that is not approved gets a
/// `Rejected` result here, and never reaches its tool.
struct RequireApproval {
    approver: mpsc::Sender<Approval>,
}

#[async_trait]
impl ToolBatchProcessor for RequireApproval {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let mut offset = 0;
        while offset < batch.calls().len() {
            let call = batch.calls()[offset].call();
            let call_block_id = call.call_block_id;
            let (reply, decision) = oneshot::channel();
            let question = Approval {
                call: call.input.clone(),
                reply,
            };
            self.approver
                .send(question)
                .await
                .map_err(|error| ProcessorError::Failed(error.to_string()))?;
            // An approver that goes away without answering counts as a no.
            if decision.await.unwrap_or(false) {
                offset += 1;
                continue;
            }
            let rejected = ToolResultPayload::new(
                call_block_id,
                ToolResultStatus::Rejected,
                ToolOutput::new(json!({ "error": "the user did not allow this call" })),
            );
            // The resolved call leaves the pending list, so `offset` now
            // points at the next one.
            batch
                .resolve_at(batch.completed_len() + offset, new_block_id(), rejected)
                .map_err(|refused| ProcessorError::Failed(refused.error.to_string()))?;
        }
        Ok(())
    }
}
// ANCHOR_END: approval

// ANCHOR: order
/// Puts the results in the order the model declared the calls. Without it,
/// they are recorded in the order they finished.
struct DeclarationOrder;

#[async_trait]
impl ToolBatchProcessor for DeclarationOrder {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        batch.results_mut().sort_by_key(|entry| {
            ctx.declaration_order
                .iter()
                .position(|id| *id == entry.call().call_block_id)
        });
        Ok(())
    }
}
// ANCHOR_END: order

/// Reads a note from a fixed set.
struct ReadNote;

#[async_trait]
impl Tool for ReadNote {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "read_note",
            "Read one note by its file name.",
            json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
        )
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        let text = match ctx.input.arguments["name"].as_str() {
            Some("todo.txt") => "Buy milk.",
            _ => "Top secret.",
        };
        ToolResultPayload::new(
            ctx.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!({ "text": text })),
        )
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: approver
    // Stands in for your interface: a person would answer here.
    let (approver, mut questions) = mpsc::channel::<Approval>(1);
    let answering = tokio::spawn(async move {
        while let Some(question) = questions.recv().await {
            let name = question.call.arguments["name"].as_str().unwrap_or_default();
            let allowed = name == "todo.txt";
            println!("allow {} {name}? {allowed}", question.call.tool_name);
            let _ = question.reply.send(allowed);
        }
    });
    // ANCHOR_END: approver

    // ANCHOR: executor
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(ReadNote)];
    let options = ToolExecutorOptions::default()
        .with_before(vec![Arc::new(RequireApproval { approver })])
        .with_after(vec![Arc::new(DeclarationOrder)]);
    let executor = Arc::new(ToolExecutor::new(tools, options)?);
    // ANCHOR_END: executor

    // Stands in for a provider: asks for two notes at once, then answers.
    let gateway = Arc::new(ScriptedModel(|request: &_| {
        if tool_results(request).is_empty() {
            let read = |name: &str| {
                ModelItem::tool_call(ToolCallDraft::new("read_note", json!({ "name": name })))
            };
            ModelOutput::new(
                ModelResponse {
                    items: vec![read("todo.txt"), read("secret.txt")],
                },
                ModelStopReason::ToolUse,
            )
        } else {
            reply("I could read one of the two notes.")
        }
    }));
    let runner = TurnRunner::new(gateway, executor);
    let context = Context::from_blocks(vec![ContextBlock::text(
        new_block_id(),
        Role::User,
        "Read todo.txt and secret.txt.",
    )])?;
    let outcome = runner
        .run(
            TurnId::new("notes"),
            context,
            TurnRunOptions::new(ModelRef::new("scripted")),
            CallControl::new(CancellationToken::new(), None),
        )
        .await;

    for block in outcome.context.blocks() {
        if let BlockContent::ToolResult(result) = block.content() {
            println!("{:?} {}", result.status, result.output.content);
        }
    }
    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        result => return Err(format!("turn did not complete: {result:?}").into()),
    }
    // The runner held the last sender; with it gone, the approver ends.
    drop(runner);
    answering.await?;
    Ok(())
}
