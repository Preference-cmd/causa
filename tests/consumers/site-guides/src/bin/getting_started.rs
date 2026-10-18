// The first program on the getting-started page. `README.md` carries a
// copy; change both together. It needs a key and a network, so CI only
// compiles it.

// ANCHOR: program
use causa::{
    kernel::{CallControl, CancellationToken, Context, ContextBlock, ModelRef, Role, TurnId},
    providers::AnthropicMessagesGateway,
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gateway = Arc::new(AnthropicMessagesGateway::new(std::env::var(
        "ANTHROPIC_API_KEY",
    )?));
    let executor = Arc::new(ToolExecutor::new(
        Vec::new(),
        ToolExecutorOptions::default(),
    )?);
    let runner = TurnRunner::new(gateway, executor);
    let context = Context::from_blocks(vec![ContextBlock::text(
        new_block_id(),
        Role::User,
        "Say hello in one sentence.",
    )])?;
    let options = TurnRunOptions::new(ModelRef::new(std::env::var("ANTHROPIC_MODEL")?));
    let outcome = runner
        .run(
            TurnId::new("hello"),
            context,
            options,
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        result => return Err(format!("turn did not complete: {result:?}").into()),
    }
    Ok(())
}
// ANCHOR_END: program
