//! Demo shell for `FsConversationStore` — since Slice 12 the store lives
//! in `reimagine-agent-extension` (its layer question is registered in
//! that crate's charter); this example only proves the trait wiring.

use reimagine_agent_extension::FsConversationStore;
use reimagine_context_kernel::{ConversationId, ConversationStore, ConversationStoreError};

#[tokio::main]
async fn main() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = FsConversationStore::new(tmp.path());
    let conv = ConversationId("conv-demo".into());
    let empty = store.load_snapshots(&conv).await;
    match empty {
        Err(ConversationStoreError::NotFound(id)) => println!("empty conv ok: {id}"),
        other => println!("unexpected: {other:?}"),
    }
    println!(
        "FsConversationStore example compiled and ran at {}.",
        store.root().display()
    );
}
