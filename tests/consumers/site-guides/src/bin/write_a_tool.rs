// The "Write a tool" guide. Runs offline: a scripted model asks for a note,
// and the program prints the recorded result and the model's answer.

// ANCHOR: imports
use async_trait::async_trait;
use causa::{
    kernel::{
        ArtifactStore, BlockContent, CallControl, CancellationToken, Context, ContextBlock,
        ModelRef, Role, Tool, ToolCallContext, ToolDefinition, ToolOutput, ToolResultPayload,
        ToolResultStatus, TurnId,
    },
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use serde::Deserialize;
use serde_json::json;
use std::{path::PathBuf, sync::Arc};
// ANCHOR_END: imports

use causa_site_guides::scripted::{ScriptedModel, call, reply, tool_results};

// ANCHOR: tool
/// Reads a note from one directory.
struct ReadNote {
    root: PathBuf,
}

#[derive(Deserialize)]
struct ReadNoteArgs {
    name: String,
}

#[async_trait]
impl Tool for ReadNote {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "read_note",
            "Read one note by its file name, for example `todo.txt`.",
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
        control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        // Bad arguments are a result the model can read and correct.
        let args: ReadNoteArgs = match serde_json::from_value(ctx.input.arguments.clone()) {
            Ok(args) => args,
            Err(error) => return result(ctx, ToolResultStatus::Failed, error.to_string()),
        };
        if args.name.contains(['/', '\\']) || args.name.starts_with('.') {
            return result(
                ctx,
                ToolResultStatus::Failed,
                "name must be a plain file name",
            );
        }

        // Past the call's deadline, report a timeout the model can read.
        // Without this, the executor records an unknown outcome and the run
        // stops.
        let deadline = async {
            match control.deadline() {
                Some(at) => tokio::time::sleep_until(at.into()).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = deadline => result(ctx, ToolResultStatus::TimedOut, "timed out"),
            read = tokio::fs::read_to_string(self.root.join(&args.name)) => match read {
                Ok(text) => ToolResultPayload::new(
                    ctx.call_block_id,
                    ToolResultStatus::Succeeded,
                    ToolOutput::new(json!({ "text": text })),
                ),
                Err(error) => result(ctx, ToolResultStatus::Failed, error.to_string()),
            },
        }
    }
}

fn result(
    ctx: &ToolCallContext,
    status: ToolResultStatus,
    error: impl Into<String>,
) -> ToolResultPayload {
    ToolResultPayload::new(
        ctx.call_block_id,
        status,
        ToolOutput::new(json!({ "error": error.into() })),
    )
}
// ANCHOR_END: tool

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::temp_dir().join(format!("causa-notes-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    std::fs::write(root.join("todo.txt"), "Buy milk.")?;

    // Stands in for a provider: asks for the note, then reads the result.
    let gateway = Arc::new(ScriptedModel(|request: &_| {
        match tool_results(request).last() {
            None => call("read_note", json!({ "name": "todo.txt" })),
            Some(result) => reply(format!("The note says: {}", result.output.content["text"])),
        }
    }));

    // ANCHOR: run
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(ReadNote { root: root.clone() })];
    let executor = Arc::new(ToolExecutor::new(tools, ToolExecutorOptions::default())?);
    let runner = TurnRunner::new(gateway, executor);

    let context = Context::from_blocks(vec![ContextBlock::text(
        new_block_id(),
        Role::User,
        "What is on my todo list?",
    )])?;
    let outcome = runner
        .run(
            TurnId::new("notes"),
            context,
            TurnRunOptions::new(ModelRef::new("scripted")),
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    // ANCHOR_END: run

    std::fs::remove_dir_all(&root)?;
    for block in outcome.context.blocks() {
        if let BlockContent::ToolResult(result) = block.content() {
            println!("read_note: {:?} {}", result.status, result.output.content);
        }
    }
    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        result => return Err(format!("turn did not complete: {result:?}").into()),
    }
    Ok(())
}
