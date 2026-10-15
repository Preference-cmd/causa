//! The first program from `README.md` and the website's getting-started page,
//! formatted by rustfmt. It needs a key and a network to run, so the test only
//! compiles it; change it together with those two copies.

use causa::{
    kernel::{CallControl, CancellationToken, Context, ContextBlock, ModelRef, Role, TurnId},
    providers::AnthropicMessagesGateway,
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use std::sync::Arc;

async fn readme_first_program() -> Result<(), Box<dyn std::error::Error>> {
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

#[test]
fn readme_first_program_compiles() {
    let _ = readme_first_program;
}
