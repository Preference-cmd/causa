// The "Save and restore context" guide. Runs offline: each pass of the loop
// stands for one process that loads the conversation, runs a turn with a
// scripted model and saves it again.

// ANCHOR: imports
use causa::{
    kernel::{CallControl, CancellationToken, Context, ContextBlock, ModelRef, Role, TurnId},
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use std::{error::Error, path::Path, sync::Arc};
// ANCHOR_END: imports

use causa_site_guides::scripted::{ScriptedModel, reply};

// ANCHOR: save
async fn save(context: &Context, path: &Path) -> Result<(), Box<dyn Error>> {
    let json = serde_json::to_vec_pretty(context)?;
    // Write a new file, then rename it over the old one, so a crash
    // never leaves half a conversation behind.
    let temp = path.with_extension("tmp");
    tokio::fs::write(&temp, json).await?;
    tokio::fs::rename(&temp, path).await?;
    Ok(())
}
// ANCHOR_END: save

// ANCHOR: load
async fn load(path: &Path) -> Result<Context, Box<dyn Error>> {
    match tokio::fs::read(path).await {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Context::new()),
        Err(error) => Err(error.into()),
    }
}
// ANCHOR_END: load

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::temp_dir().join(format!("causa-conversation-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("conversation.json");

    // Stands in for a provider: counts the user messages it was sent.
    let gateway = Arc::new(ScriptedModel(|request: &causa::kernel::ModelRequest| {
        let seen = request
            .frame
            .blocks
            .iter()
            .filter(|block| {
                matches!(
                    block.content(),
                    causa::kernel::BlockContent::Message {
                        role: Role::User,
                        ..
                    }
                )
            })
            .count();
        reply(format!("Message {seen} received."))
    }));
    let runner = TurnRunner::new(
        gateway,
        Arc::new(ToolExecutor::new(
            Vec::new(),
            ToolExecutorOptions::default(),
        )?),
    );

    for (turn, text) in ["Hello.", "Still there?"].into_iter().enumerate() {
        // ANCHOR: turn
        let mut context = load(&path).await?;
        context
            .edit()
            .append([ContextBlock::text(new_block_id(), Role::User, text)])
            .commit()?;

        let outcome = runner
            .run(
                TurnId::new(format!("turn-{turn}")),
                context,
                TurnRunOptions::new(ModelRef::new("scripted")),
                CallControl::new(CancellationToken::new(), None),
            )
            .await;
        // The context is valid whatever the result; save it either way.
        save(&outcome.context, &path).await?;
        // ANCHOR_END: turn

        match outcome.result {
            TurnResult::Completed { final_output } => println!(
                "{} ({} blocks saved)",
                final_output.response.text(),
                outcome.context.blocks().len()
            ),
            result => return Err(format!("turn did not complete: {result:?}").into()),
        }
    }

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
