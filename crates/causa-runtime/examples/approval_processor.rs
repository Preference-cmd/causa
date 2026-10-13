//! Await approval inside a tool batch processor. Runnable offline:
//! `cargo run -p causa-runtime --example approval_processor`
//!
//! A host task receives each approval request and replies through a channel.
//! The processor exclusively borrows U while waiting, so the entire batch
//! stays in this stage until all decisions are available. Rejection writes a
//! normal result; it is not a processor failure or a paused-turn protocol.

use async_trait::async_trait;
use causa_kernel::{
    CallControl, ModelGateway, ModelInvokeError, ModelItem, ModelOutput, ModelRequest,
    ModelResponse, ModelStopReason, ProcessorContext, ProcessorError, TextPayload, Tool, ToolBatch,
    ToolBatchProcessor, ToolCallContext, ToolCallDraft, ToolDefinition, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnId,
};
use causa_runtime::{
    ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
};

use std::sync::Arc;
use tokio::sync::{Mutex, mpsc, oneshot};

/// Scripted stub gateway: one tool-use round, then the closing round.
struct ScriptedGateway(Mutex<Vec<ModelOutput>>);

impl ScriptedGateway {
    fn new(outputs: Vec<ModelOutput>) -> Arc<Self> {
        Arc::new(Self(Mutex::new(outputs)))
    }
}

#[async_trait]
impl ModelGateway for ScriptedGateway {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &causa_kernel::CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut outputs = self.0.lock().await;
        if outputs.is_empty() {
            return Err(ModelInvokeError::new(
                causa_kernel::ModelInvokeErrorKind::Permanent,
                "script exhausted",
            ));
        }
        Ok(outputs.remove(0))
    }
}

/// A deliberately sensitive local tool — the kind a host gates behind
/// approval.
struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "read_file",
            "Read a file from the host filesystem.",
            serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        )
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &causa_kernel::CallControl,
        _store: Option<&dyn causa_kernel::ArtifactStore>,
    ) -> ToolResultPayload {
        ToolResultPayload::new(
            ctx.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(serde_json::json!(
                { "contents": "the promised file contents" }
            )),
        )
    }
}

struct ApprovalRequest {
    path: String,
    reply: oneshot::Sender<bool>,
}

struct ApprovalProcessor(mpsc::Sender<ApprovalRequest>);

#[async_trait]
impl ToolBatchProcessor for ApprovalProcessor {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let mut offset = 0;
        while offset < batch.calls().len() {
            let call = batch.calls()[offset].call();
            let id = call.call_block_id;
            let path = call.input.arguments["path"]
                .as_str()
                .unwrap_or("")
                .to_owned();
            let (reply, decision) = oneshot::channel();
            self.0
                .send(ApprovalRequest { path, reply })
                .await
                .map_err(|error| ProcessorError::Failed(error.to_string()))?;
            let approved = decision
                .await
                .map_err(|error| ProcessorError::Failed(error.to_string()))?;
            if approved {
                batch.calls_mut()[offset]
                    .push_note(TextPayload::new("Approved by the host policy."));
                offset += 1;
            } else {
                batch
                    .resolve_at(
                        batch.completed_len() + offset,
                        new_block_id(),
                        ToolResultPayload::new(
                            id,
                            ToolResultStatus::Rejected,
                            ToolOutput::new(serde_json::json!({"error": "file access denied"})),
                        )
                        .with_notes(vec![TextPayload::new(
                            "Only notes.txt is allowed in this session.",
                        )]),
                    )
                    .map_err(|error| ProcessorError::Failed(error.to_string()))?;
                // Completion swaps the entry to the prefix; inspect the new
                // element at this suffix offset before advancing.
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gateway = ScriptedGateway::new(vec![
        ModelOutput::new(
            ModelResponse {
                items: std::iter::once(ModelItem::text("Read the requested files."))
                    .chain(["notes.txt", "private.txt"].into_iter().map(|path| {
                        ModelItem::tool_call(ToolCallDraft::new(
                            "read_file",
                            serde_json::json!({"path": path}),
                        ))
                    }))
                    .collect(),
            },
            ModelStopReason::ToolUse,
        ),
        ModelOutput::new(
            ModelResponse {
                items: vec![ModelItem::text("Read notes.txt; private.txt was denied.")],
            },
            ModelStopReason::EndTurn,
        ),
    ]);
    let (requests, mut receiver) = mpsc::channel::<ApprovalRequest>(1);
    let reviewer = tokio::spawn(async move {
        while let Some(request) = receiver.recv().await {
            println!("approval request: {}", request.path);
            let _ = request.reply.send(request.path == "notes.txt");
        }
    });
    let executor = ToolExecutor::new(
        vec![Arc::new(ReadFile)],
        ToolExecutorOptions::default().with_before(vec![Arc::new(ApprovalProcessor(requests))]),
    )?;
    let runner = TurnRunner::new(gateway, Arc::new(executor));
    let mut context = causa_kernel::Context::new();
    context
        .edit()
        .append([causa_kernel::ContextBlock::new(
            new_block_id(),
            causa_kernel::BlockContent::Parts(vec![causa_kernel::ContentPart::Text(
                TextPayload::new("Read the two files."),
            )]),
            causa_kernel::BlockMeta::from_source("user"),
        )])
        .commit()?;
    let outcome = runner
        .run(
            TurnId::new("approval_processor"),
            context,
            TurnRunOptions::new(causa_kernel::ModelRef::new("offline-demo")),
            CallControl::new(Default::default(), None),
        )
        .await;
    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        other => return Err(format!("turn did not complete: {other:?}").into()),
    }
    drop(runner);
    reviewer.await?;
    Ok(())
}
