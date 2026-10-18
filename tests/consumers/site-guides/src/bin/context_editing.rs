// The "Edit context" guide. Runs offline: three old blocks are replaced by
// a summary, a note is appended, and an edit that reuses a block id is
// refused without changing anything.

// ANCHOR: imports
use causa::{
    kernel::{BlockContent, ContentPart, Context, ContextBlock, EditError, Role},
    runtime::new_block_id,
};
// ANCHOR_END: imports

fn print(context: &Context) {
    for block in context.blocks() {
        if let BlockContent::Message { role, parts } = block.content() {
            let text: String = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(text.0.as_str()),
                    _ => None,
                })
                .collect();
            println!("{role:?}: {text}");
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = |role, text: &str| ContextBlock::text(new_block_id(), role, text);
    let mut context = Context::from_blocks(vec![
        text(Role::System, "Answer briefly."),
        text(Role::User, "I am building a Rust agent."),
        text(Role::Assistant, "What does it do?"),
        text(Role::User, "It reads my notes. I like short answers."),
        text(Role::User, "Which crate should I use for HTTP?"),
    ])?;

    // ANCHOR: edit
    // Keep a copy of what the edit removes, if you need it later: the
    // context does not keep it.
    let replaced = context.blocks()[1..4].to_vec();

    let summary = ContextBlock::text(
        new_block_id(),
        Role::System,
        "Earlier: the user is building a Rust agent that reads notes, and likes short answers.",
    );
    let reminder = ContextBlock::text(new_block_id(), Role::System, "Name one crate.");
    context
        .edit()
        .replace(1..4, [summary])
        .append([reminder])
        .commit()?;
    // ANCHOR_END: edit

    print(&context);
    println!("{} blocks replaced", replaced.len());

    // ANCHOR: refused
    // A new block with an id already in the context: the whole edit is
    // refused, and the blocks it carried are handed back.
    let reused_id = context.blocks()[0].id();
    let clash = ContextBlock::text(reused_id, Role::User, "Another message.");
    let before = context.blocks().len();
    match context.edit().append([clash]).commit() {
        Ok(()) => println!("committed"),
        Err(failure) => {
            let reason = match failure.reason {
                EditError::BlockIdentityMismatch(_) => "id reused for other content",
                EditError::DuplicateBlockId(_) => "duplicate id",
                _ => "other",
            };
            println!(
                "refused ({reason}): {} block handed back, {} blocks before and after",
                failure.appended.len(),
                context.blocks().len()
            );
            assert_eq!(context.blocks().len(), before);
        }
    }
    // ANCHOR_END: refused
    Ok(())
}
