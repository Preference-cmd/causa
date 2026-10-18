// The "Prepare each request" guide. Runs offline: a conversation of three
// turns is run with a preparer that sends only the last two, and a scripted
// model prints what it received.

// ANCHOR: imports
use async_trait::async_trait;
use causa::{
    kernel::{
        BlockContent, CallControl, CancellationToken, Context, ContextBlock, ContextFrame,
        ContextPreparer, ModelRef, PrepareError, Role, RoundInfo, TurnId,
    },
    runtime::{TurnResult, TurnRunOptions, TurnRunner, new_block_id},
};
use std::sync::Arc;
// ANCHOR_END: imports

use causa::kernel::{
    ArtifactStore, BlockMeta, ContentPart, ModelRequest, Tool, ToolCallContext, ToolChange,
    ToolDefinition, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use causa::runtime::{ToolExecutor, ToolExecutorOptions};
use causa_site_guides::scripted::{ScriptedModel, reply};

// ANCHOR: preparer
/// Sends the last `turns` user turns, with every system message and every
/// tool change. A turn starts at a user message, so a tool call is never
/// cut off from its result.
struct RecentTurns {
    turns: usize,
}

#[async_trait]
impl ContextPreparer for RecentTurns {
    async fn prepare(
        &self,
        context: &mut Context,
        _round: &RoundInfo<'_>,
        _control: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        let blocks = context.blocks();
        let start = blocks
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, block)| is_message(block, Role::User))
            .nth(self.turns.saturating_sub(1))
            .map_or(0, |(index, _)| index);
        let blocks = blocks
            .iter()
            .enumerate()
            .filter(|(index, block)| {
                *index >= start
                    || is_message(block, Role::System)
                    || matches!(block.content(), BlockContent::ToolChange(_))
            })
            .map(|(_, block)| block.clone())
            .collect();
        Ok(ContextFrame { blocks })
    }
}

fn is_message(block: &ContextBlock, role: Role) -> bool {
    matches!(block.content(), BlockContent::Message { role: r, .. } if *r == role)
}
// ANCHOR_END: preparer

/// A tool the model is offered; this program never calls it.
struct Lookup;

#[async_trait]
impl Tool for Lookup {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "lookup",
            "Look a word up.",
            serde_json::json!({"type": "object"}),
        )
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        ToolResultPayload::new(
            ctx.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(serde_json::json!({})),
        )
    }
}

/// One line per block of the request, as the model received it.
fn describe(request: &ModelRequest) -> String {
    let lines: Vec<String> = request
        .frame
        .blocks
        .iter()
        .map(|block| match block.content() {
            BlockContent::Message { role, parts } => {
                let text: String = parts
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text(text) => Some(text.0.as_str()),
                        _ => None,
                    })
                    .collect();
                format!("{role:?}: {text}")
            }
            BlockContent::ToolChange(change) => {
                let added: Vec<&str> = change.added.iter().map(|tool| tool.name.as_str()).collect();
                format!("ToolChange: +{}", added.join(" +"))
            }
            _ => "other".to_owned(),
        })
        .collect();
    lines.join("\n")
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = |role, text: &str| ContextBlock::text(new_block_id(), role, text);
    let context = Context::from_blocks(vec![
        text(Role::System, "Answer in one word."),
        text(Role::User, "First?"),
        // The runner recorded the tools offered in the first turn.
        ContextBlock::new(
            new_block_id(),
            BlockContent::ToolChange(ToolChange::between(&[], &[Lookup.definition()])),
            BlockMeta::default(),
        ),
        text(Role::Assistant, "One."),
        text(Role::User, "Second?"),
        text(Role::Assistant, "Two."),
        text(Role::User, "Third?"),
    ])?;

    // Stands in for a provider: answers with what it was sent.
    let gateway = Arc::new(ScriptedModel(|request: &ModelRequest| {
        reply(describe(request))
    }));
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Lookup)];
    let executor = Arc::new(ToolExecutor::new(tools, ToolExecutorOptions::default())?);
    let runner = TurnRunner::new(gateway, executor);

    // ANCHOR: run
    let mut options = TurnRunOptions::new(ModelRef::new("scripted"));
    options.preparer = Some(Arc::new(RecentTurns { turns: 2 }));
    let outcome = runner
        .run(
            TurnId::new("third"),
            context,
            options,
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    // ANCHOR_END: run

    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        result => return Err(format!("turn did not complete: {result:?}").into()),
    }
    println!(
        "The context holds {} blocks.",
        outcome.context.blocks().len()
    );
    Ok(())
}
