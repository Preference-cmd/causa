// The "Stream output" guide. Runs offline: a local server streams a Chat
// Completions answer in three pieces. The program reads it twice: through
// the runner with an observer, and from the gateway directly.

// ANCHOR: imports
use causa::{
    kernel::{
        CallControl, CancellationToken, Context, ContextBlock, InvocationId, ModelRef,
        ModelRequest, Role, RoundId, StreamDelta, TurnId,
    },
    providers::OpenAiChatCompletionsGateway,
    runtime::{
        RunEvent, RunObserver, ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions,
        TurnRunner, new_block_id,
    },
};
use std::sync::Arc;
// ANCHOR_END: imports

use causa_site_guides::{fake_provider::FakeProvider, streaming::consume};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = FakeProvider::start().await?;
    let gateway =
        Arc::new(OpenAiChatCompletionsGateway::new("test-key").with_base_url(&server.base_url));
    let context = Context::from_blocks(vec![ContextBlock::text(
        new_block_id(),
        Role::User,
        "Say hello.",
    )])?;

    // ANCHOR: observer
    // Prints each piece of text as it arrives. The pieces are for display;
    // the run's result holds the complete answer.
    let observer: Arc<RunObserver> = Arc::new(|event: &RunEvent<'_>| {
        if let RunEvent::ModelDelta {
            delta: StreamDelta::TextDelta { delta },
            ..
        } = event
        {
            print!("[{delta}]");
        }
    });
    // ANCHOR_END: observer

    // ANCHOR: run
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(ToolExecutor::new(
            Vec::new(),
            ToolExecutorOptions::default(),
        )?),
    );
    let options = TurnRunOptions::new(ModelRef::new("model")).with_observer(observer);
    let outcome = runner
        .run_streaming(
            TurnId::new("hello"),
            context.clone(),
            options,
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    if let TurnResult::Completed { final_output } = &outcome.result {
        println!("\nrunner: {}", final_output.response.text());
    }
    // ANCHOR_END: run

    // ANCHOR: direct
    // Without the runner: build the request yourself and read the stream.
    let request = ModelRequest::new(
        InvocationId {
            turn_id: TurnId::new("hello"),
            round_id: RoundId(0),
        },
        context.frame(),
        ModelRef::new("model"),
    );
    let control = CallControl::new(CancellationToken::new(), None);
    let output = consume(gateway.as_ref(), &request, &control).await?;
    println!("\ndirect: {}", output.response.text());
    // ANCHOR_END: direct

    if !matches!(outcome.result, TurnResult::Completed { .. }) {
        return Err(format!("turn did not complete: {:?}", outcome.result).into());
    }
    Ok(())
}
