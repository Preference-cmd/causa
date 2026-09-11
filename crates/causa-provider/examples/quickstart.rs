//! Quickstart — one turn, one local tool, one real provider round-trip.
//!
//! This example lives in `causa-provider` (not `causa-runtime`) for one
//! reason: the dependency direction. The runtime must never depend on
//! the edge adapters it drives, so an example that wires *both* sides
//! belongs to the adapter crate — which dev-depends on the runtime for
//! exactly this kind of demonstration. In your own host you compose all
//! five crates freely.
//!
//! ```text
//! ANTHROPIC_API_KEY=sk-ant-... cargo run --example quickstart -p causa-provider
//! ```
//!
//! The shape is the whole framework: fill the kernel's ports (a
//! `ModelGateway` adapter from `causa-provider`, a local `Tool`), hand
//! them to the reference driver, and drive one turn. The driver loops
//! model rounds and tool dispatch until the model ends the turn.

use async_trait::async_trait;
use causa_kernel::{
    CallControl, CancellationToken, ModelRef, TextPayload, Tool, ToolCallContext, ToolDefinition,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnContext, TurnId,
};
use causa_provider::AnthropicMessagesGateway;
use causa_runtime::{
    RunControl, ToolExecutor, TurnInvocation, TurnResult, TurnRunOptions, TurnRunner,
};
use std::sync::Arc;

/// A local Rust tool the model may call — one `ToolDefinition` for the
/// surface, one `execute` for the effect.
struct WordCount;

#[async_trait]
impl Tool for WordCount {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "word_count".into(),
            description: "Count whitespace-separated words in a text.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            }),
        }
    }

    async fn execute(&self, ctx: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        let text = ctx
            .arguments
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let count = text.split_whitespace().count();
        ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(serde_json::json!({ "words": count })),
            media: Vec::new(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "set ANTHROPIC_API_KEY to run this example against the real API")?;

    // Ports out, driver in: the executor holds the tools, the gateway
    // adapter talks to the provider, the runner loops the turn.
    let executor = Arc::new(ToolExecutor::from_vec(vec![Arc::new(WordCount)]));
    // Media: inject the host's asset table so fact-level
    // media references resolve to inline payloads at render time. The
    // table is prefetched before the turn; misses degrade to a
    // deterministic text placeholder.
    let asset_table: std::collections::HashMap<
        String,
        causa_protocol::translation::media::MediaPayload,
    > = std::collections::HashMap::new();
    let gateway = Arc::new(
        AnthropicMessagesGateway::new(api_key)
            .with_media_resolver(std::sync::Arc::new(asset_table)),
    );
    let runner = TurnRunner::new(gateway, executor.clone());

    let mut context = TurnContext::new(TurnId::new("quickstart"));
    context.append_input(
        TextPayload::new(
            "How many words are in \"the quick brown fox jumps over the lazy dog\"? \
             Use the word_count tool, then answer with just the number.",
        ),
        "user",
    )?;

    let options = TurnRunOptions {
        invocation: TurnInvocation {
            model: ModelRef::new("claude-sonnet-4-5"),
            tool_surface: executor.tool_surface().await,
            ..Default::default()
        },
        ..Default::default()
    };

    let outcome = runner
        .run(
            context,
            options,
            RunControl::new(CancellationToken::new(), None),
        )
        .await;
    match outcome.result {
        TurnResult::Completed { final_output } => {
            println!("answer: {}", final_output.response.text.0);
        }
        TurnResult::Interrupted { cause } => {
            return Err(format!("turn interrupted: {cause:?}").into());
        }
        TurnResult::Paused { .. } => unreachable!("no interaction gate installed"),
    }
    println!("rounds: {}", outcome.trace.rounds.len());
    Ok(())
}
