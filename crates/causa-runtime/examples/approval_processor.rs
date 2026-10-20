//! Await approval inside a tool batch processor. Runnable offline:
//! `cargo run -p causa-runtime --example approval_processor`
//!
//! The processor asks only about tools that do not declare themselves
//! read-only, and reads the host's permission mode from the run's processor
//! data each time it decides a call. A host task receives each approval
//! request and replies through a channel; when it declines a write it also
//! switches the run to read-only, and the remaining writes of the same batch
//! are refused without asking. The processor exclusively borrows the batch
//! while waiting, so the entire batch stays in this stage until all decisions
//! are available. Rejection writes a normal result; it is not a processor
//! failure or a paused-turn protocol.

use async_trait::async_trait;
use causa_kernel::{
    CallControl, ModelGateway, ModelInvokeError, ModelItem, ModelOutput, ModelRequest,
    ModelResponse, ModelStopReason, ProcessorContext, ProcessorData, ProcessorError, TextPayload,
    Tool, ToolBatch, ToolBatchProcessor, ToolCallContext, ToolCallDraft, ToolDefinition, ToolHints,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnId,
};
use causa_runtime::{
    ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
};

use std::sync::{Arc, Mutex as StdMutex};
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

fn path_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": { "path": { "type": "string" } },
        "required": ["path"]
    })
}

/// A local tool that declares itself read-only, so it runs without asking.
struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("read_file", "Read a file.", path_schema())
    }

    fn hints(&self) -> ToolHints {
        ToolHints::default().with_read_only(true)
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
            ToolOutput::new(serde_json::json!({ "contents": "the promised file contents" })),
        )
    }
}

/// A deliberately sensitive local tool — the kind a host gates behind
/// approval. It declares that it changes what exists.
struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("write_file", "Replace a file's contents.", path_schema())
    }

    fn hints(&self) -> ToolHints {
        ToolHints::default()
            .with_read_only(false)
            .with_destructive(true)
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
            ToolOutput::new(serde_json::json!({ "written": true })),
        )
    }
}

/// The host's permission mode. The host owns the vocabulary; Causa only
/// passes the handle to the processors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Ask before any call that is not declared read-only.
    Ask,
    /// Refuse every call that is not declared read-only.
    ReadOnly,
}

/// Shared so the host can switch the mode while a batch is being decided.
type ModeHandle = Arc<StdMutex<Mode>>;

struct ApprovalRequest {
    tool: String,
    path: String,
    reply: oneshot::Sender<bool>,
}

struct ApprovalProcessor(mpsc::Sender<ApprovalRequest>);

#[async_trait]
impl ToolBatchProcessor for ApprovalProcessor {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let modes = ctx
            .data::<ModeHandle>()
            .ok_or_else(|| ProcessorError::Failed("the run sets no permission mode".into()))?;
        let mut offset = 0;
        while offset < batch.calls().len() {
            let call = batch.calls()[offset].call();
            let id = call.call_block_id;
            let tool = call.input.tool_name.clone();
            // Hints are the tool's own claim; this host trusts its local tools.
            if ctx.hints(&tool).and_then(|hints| hints.read_only) == Some(true) {
                offset += 1;
                continue;
            }
            // Read the mode now, not once per batch: the host may have
            // switched it while an earlier call was being decided.
            let mode = *modes.lock().expect("mode lock");
            let refusal = match mode {
                Mode::ReadOnly => Some("the session is read-only"),
                Mode::Ask => {
                    let path = call.input.arguments["path"]
                        .as_str()
                        .unwrap_or("")
                        .to_owned();
                    let (reply, decision) = oneshot::channel();
                    self.0
                        .send(ApprovalRequest { tool, path, reply })
                        .await
                        .map_err(|error| ProcessorError::Failed(error.to_string()))?;
                    let approved = decision
                        .await
                        .map_err(|error| ProcessorError::Failed(error.to_string()))?;
                    (!approved).then_some("the reviewer declined this call")
                }
            };
            match refusal {
                None => {
                    batch.calls_mut()[offset]
                        .push_note(TextPayload::new("Approved by the host policy."));
                    offset += 1;
                }
                Some(reason) => {
                    batch
                        .resolve_at(
                            batch.completed_len() + offset,
                            new_block_id(),
                            ToolResultPayload::new(
                                id,
                                ToolResultStatus::Rejected,
                                ToolOutput::new(serde_json::json!({ "error": reason })),
                            ),
                        )
                        .map_err(|error| ProcessorError::Failed(error.to_string()))?;
                    // Completion swaps the entry to the prefix; inspect the
                    // new element at this suffix offset before advancing.
                }
            }
        }
        Ok(())
    }
}

fn call(tool: &str, path: &str) -> ModelItem {
    ModelItem::tool_call(ToolCallDraft::new(
        tool,
        serde_json::json!({ "path": path }),
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gateway = ScriptedGateway::new(vec![
        ModelOutput::new(
            ModelResponse {
                items: vec![
                    ModelItem::text("Read the notes, then update three files."),
                    call("read_file", "notes.txt"),
                    call("write_file", "notes.txt"),
                    call("write_file", "private.txt"),
                    call("write_file", "secrets.txt"),
                ],
            },
            ModelStopReason::ToolUse,
        ),
        ModelOutput::new(
            ModelResponse {
                items: vec![ModelItem::text(
                    "Read and updated notes.txt; the other two writes were refused.",
                )],
            },
            ModelStopReason::EndTurn,
        ),
    ]);
    let mode: ModeHandle = Arc::new(StdMutex::new(Mode::Ask));
    let (requests, mut receiver) = mpsc::channel::<ApprovalRequest>(1);
    let reviewer = {
        let mode = mode.clone();
        tokio::spawn(async move {
            while let Some(request) = receiver.recv().await {
                let approved = request.path == "notes.txt";
                println!(
                    "approval request: {} {} -> {}",
                    request.tool,
                    request.path,
                    if approved { "approved" } else { "declined" }
                );
                if !approved {
                    // Decline, and stop asking for the rest of the session.
                    *mode.lock().expect("mode lock") = Mode::ReadOnly;
                    println!("mode switched to {:?}", Mode::ReadOnly);
                }
                let _ = request.reply.send(approved);
            }
        })
    };
    let executor = ToolExecutor::new(
        vec![Arc::new(ReadFile), Arc::new(WriteFile)],
        ToolExecutorOptions::default().with_before(vec![Arc::new(ApprovalProcessor(requests))]),
    )?;
    let runner = TurnRunner::new(gateway, Arc::new(executor));
    let mut context = causa_kernel::Context::new();
    context
        .edit()
        .append([causa_kernel::ContextBlock::text(
            new_block_id(),
            causa_kernel::Role::User,
            "Read the notes and update the files.",
        )])
        .commit()?;
    let outcome = runner
        .run(
            TurnId::new("approval_processor"),
            context,
            TurnRunOptions::new(causa_kernel::ModelRef::new("offline-demo"))
                .with_processor_data(ProcessorData::new(mode.clone())),
            CallControl::new(Default::default(), None),
        )
        .await;
    for block in outcome.context.blocks() {
        if let causa_kernel::BlockContent::ToolResult(result) = block.content() {
            println!("result: {:?} {}", result.status, result.output.content);
        }
    }
    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        other => return Err(format!("turn did not complete: {other:?}").into()),
    }
    drop(runner);
    reviewer.await?;
    Ok(())
}
