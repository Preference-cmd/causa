// The program on the context-editing page. It runs offline.

// ANCHOR: program
use causa::{
    kernel::{Context, ContextBlock, Role},
    runtime::new_block_id,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::new();
    let initial = [
        ("system note", Role::System),
        ("old detail one", Role::User),
        ("old detail two", Role::User),
        ("old detail three", Role::User),
        ("recent request", Role::User),
    ];
    for (text, role) in initial {
        context
            .edit()
            .append([ContextBlock::text(new_block_id(), role, text)])
            .commit()?;
    }

    let summary = ContextBlock::text(
        new_block_id(),
        Role::System,
        "The user prefers short answers and is building a Rust agent.",
    );
    let note = ContextBlock::text(
        new_block_id(),
        Role::System,
        "The current request concerns context editing.",
    );

    context
        .edit()
        .replace(1..4, [summary])
        .append([note])
        .commit()?;

    println!("{} blocks", context.blocks().len());
    Ok(())
}
// ANCHOR_END: program
