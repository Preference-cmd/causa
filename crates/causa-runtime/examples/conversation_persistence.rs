//! Save caller-owned Context material as JSON and import it before another run.
//! Runnable offline: `cargo run -p causa-runtime --example conversation_persistence`.
//!
//! The caller chooses the file and retention unit. Context has no execution
//! identity or closed state; importing prior tool material never restarts it.
//! Media remains a reference; bytes are owned by the caller's asset store.

use async_trait::async_trait;
use causa_kernel::{
    BlockContent, BlockMeta, CallControl, ContentPart, Context, ContextBlock, MediaRef,
    ModelGateway, ModelInvokeError, ModelOutput, ModelRef, ModelRequest, ModelResponse,
    ModelStopReason, TextPayload, TurnId,
};
use causa_runtime::{
    RunControl, ToolExecutor, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
};
use std::sync::Arc;

struct AckGateway;
#[async_trait]
impl ModelGateway for AckGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        Ok(ModelOutput {
            response: ModelResponse {
                text: TextPayload::new(format!(
                    "ack: {} material blocks",
                    request.frame.blocks.len()
                )),
                tool_calls: vec![],
            },
            usage: None,
            stop_reason: ModelStopReason::EndTurn,
            reasoning: None,
        })
    }
}
fn input(text: &str) -> ContextBlock {
    ContextBlock::new(
        new_block_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
        BlockMeta {
            source: Some("user".into()),
            ..Default::default()
        },
    )
}
async fn run(
    runner: &TurnRunner,
    mut context: Context,
    id: &str,
    text: &str,
) -> Result<Context, Box<dyn std::error::Error>> {
    context.edit().append([input(text)]).commit()?;
    let outcome = runner
        .run(
            TurnId::new(id),
            context,
            TurnRunOptions::new(ModelRef::new("offline-ack")),
            RunControl::new(Default::default(), None),
        )
        .await;
    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text.0),
        other => return Err(format!("execution failed: {other:?}").into()),
    }
    Ok(outcome.context)
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runner = TurnRunner::new(
        Arc::new(AckGateway),
        Arc::new(ToolExecutor::new(vec![], Default::default())?),
    );
    let context = run(&runner, Context::new(), "first", "hello").await?;
    let mut context = run(&runner, context, "second", "and again").await?;
    context
        .edit()
        .append([ContextBlock::new(
            new_block_id(),
            BlockContent::Parts(vec![
                ContentPart::Text(TextPayload::new("chart")),
                ContentPart::Media(MediaRef::new("image/png", "asset-chart")),
            ]),
            BlockMeta {
                source: Some("user".into()),
                ..Default::default()
            },
        )])
        .commit()?;
    let scratch = tempfile::tempdir()?;
    let path = scratch.path().join("materials.json");
    tokio::fs::write(&path, serde_json::to_vec_pretty(&context)?).await?;
    let restored: Context = serde_json::from_slice(&tokio::fs::read(&path).await?)?;
    assert_eq!(restored.blocks(), context.blocks());
    let resumed = run(&runner, restored, "third", "back after reload").await?;
    assert_eq!(
        &resumed.blocks()[..context.blocks().len()],
        context.blocks()
    );
    println!(
        "saved, imported and continued {} material blocks",
        resumed.blocks().len()
    );
    Ok(())
}
