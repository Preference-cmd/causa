// The code on the site's home page. Runs offline: a scripted model answers,
// and the conversation is loaded from and saved to a JSON file, twice, as
// two separate runs of an application would.

use causa::{
    kernel::{CallControl, CancellationToken, Context, ContextBlock, ModelRef, Role, TurnId},
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use std::{error::Error, path::Path, sync::Arc};

use causa_site_guides::scripted::{ScriptedModel, reply};

async fn load(path: &Path) -> Result<Context, Box<dyn Error>> {
    match tokio::fs::read(path).await {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Context::new()),
        Err(error) => Err(error.into()),
    }
}

async fn save(context: &Context, path: &Path) -> Result<(), Box<dyn Error>> {
    tokio::fs::write(path, serde_json::to_vec_pretty(context)?).await?;
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::temp_dir().join(format!("causa-home-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("chat.json");

    let gateway = Arc::new(ScriptedModel(|request: &causa::kernel::ModelRequest| {
        reply(format!("Blocks sent: {}.", request.frame.blocks.len()))
    }));
    let tools = ToolExecutor::new(Vec::new(), ToolExecutorOptions::default())?;
    let runner = TurnRunner::new(gateway, Arc::new(tools));

    for question in ["What is Causa?", "And the kernel?"] {
        let options = TurnRunOptions::new(ModelRef::new("scripted"));
        let control = CallControl::new(CancellationToken::new(), None);

        // ANCHOR: hero
        let mut context = load(&path).await?;
        context
            .edit()
            .append([ContextBlock::text(new_block_id(), Role::User, question)])
            .commit()?;

        let outcome = runner
            .run(TurnId::new("t1"), context, options, control)
            .await;

        // The context comes back whatever happened. Save it as JSON.
        save(&outcome.context, &path).await?;

        if let TurnResult::Interrupted { cause } = &outcome.result {
            eprintln!("stopped: {cause}");
        }
        // ANCHOR_END: hero

        if let TurnResult::Completed { final_output } = &outcome.result {
            println!(
                "{} ({} blocks saved)",
                final_output.response.text(),
                outcome.context.blocks().len()
            );
        }
    }

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
