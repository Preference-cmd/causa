//! `FsConversationStore` -- filesystem-backed [`ConversationStore`].

//! Reference implementation of the Slice 5A `ConversationStore` port.
//! Lives in `examples/` because (a) it pulls in `tokio::fs`, and (b)
//! the kernel ships no storage implementations -- hosts pick their
//! own store (FS / Sled / Sqlite / ...).

//!
//! ## Layout
//!
//! ```text
//! <root>/
//!   <conversation_id>/
//!     <turn_sequence>.json
//!     ...
//! ```
//!
//! One directory per conversation, one file per completed turn
//! snapshot. Filenames are the zero-padded `turn_sequence` at width
//! 20 -- lex order matches commit order without an explicit sort.
//! Width 20 covers `u64::MAX` (20 digits). Hosts adopting this
//! layout must keep the same width to preserve the lex-order
//! invariant.
//!
//! ## Atomicity
//!
//! `save_snapshot` writes to `<conversation_id>/<seq>.json.tmp` and
//! `tokio::fs::rename`s the temp file into place. `rename` is atomic
//! on POSIX (and replaces existing files on Windows via
//! `MoveFileExW(MOVEFILE_REPLACE_EXISTING)`). Concurrent writers
//! land on distinct filenames (one per `turn_sequence`), so two
//! hosts racing on the same conversation each win their own file
//! rather than corrupt a sibling.
//!
//! No file-level lock is held -- the host's harness serializes
//! `commit` per `ConversationId` via its own channel.
//!
//! ## Why not in `agent-runtime` lib?
//!
//! Keeping it out of `lib.rs` honors the framework's "minimum
//! invariant not a policy" rule: `FsConversationStore` is one of many
//! possible stores. Promoting it into the lib would advertise it as
//! canonical. The example demonstrates the port is sufficient.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use reimagine_context_kernel::{
    ConversationId, ConversationStore, ConversationStoreError, TurnSequence, TurnSnapshot,
};
use tokio::fs;

/// Filesystem-backed [`ConversationStore`]. Layout:
/// `<root>/<conversation_id>/<turn_sequence_zero_padded>.json`.
pub struct FsConversationStore {
    root: PathBuf,
}

impl FsConversationStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Root directory passed at construction.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn conv_dir(&self, conversation_id: &ConversationId) -> PathBuf {
        self.root.join(conversation_id.0.as_str())
    }

    fn snapshot_path(
        &self,
        conversation_id: &ConversationId,
        turn_sequence: TurnSequence,
    ) -> PathBuf {
        self.conv_dir(conversation_id)
            .join(format!("{:020}.json", turn_sequence.0))
    }
}

#[async_trait]
impl ConversationStore for FsConversationStore {
    async fn save_snapshot(
        &self,
        conversation_id: &ConversationId,
        snapshot: &TurnSnapshot,
    ) -> Result<(), ConversationStoreError> {
        let final_path = self.snapshot_path(conversation_id, snapshot.turn_sequence);
        let dir = final_path.parent().expect("path has parent").to_path_buf();
        let tmp_path = dir.join(format!("{:020}.json.tmp", snapshot.turn_sequence.0));

        let json = serde_json::to_vec_pretty(snapshot)
            .map_err(|e| ConversationStoreError::Serialization(e.to_string()))?;

        fs::create_dir_all(&dir)
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))?;
        fs::write(&tmp_path, &json)
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))?;
        fs::rename(&tmp_path, &final_path)
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))?;
        Ok(())
    }

    async fn load_snapshots(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<TurnSnapshot>, ConversationStoreError> {
        let dir = self.conv_dir(conversation_id);
        let mut entries = match fs::read_dir(&dir).await {
            Ok(it) => it,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ConversationStoreError::NotFound(conversation_id.0.clone()));
            }
            Err(e) => return Err(ConversationStoreError::Io(e.to_string())),
        };
        let mut paths: Vec<PathBuf> = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))?
        {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("json") {
                paths.push(p);
            }
        }
        // Filenames are zero-padded `turn_sequence.json` so lex order
        // already matches commit order. Belt-and-braces: sort again.
        paths.sort();

        let mut snapshots = Vec::with_capacity(paths.len());
        for p in paths {
            let bytes = fs::read(&p)
                .await
                .map_err(|e| ConversationStoreError::Io(e.to_string()))?;
            let snap: TurnSnapshot = serde_json::from_slice(&bytes)
                .map_err(|e| ConversationStoreError::Serialization(e.to_string()))?;
            snapshots.push(snap);
        }
        // Defensive monotonicity check (kernel validates again via
        // `ConversationState::from_snapshots`, but a corrupted file
        // surfaces here too).
        let mut last: Option<TurnSequence> = None;
        for s in &snapshots {
            if let Some(prev) = last
                && s.turn_sequence <= prev
            {
                return Err(ConversationStoreError::Corrupted(format!(
                    "turn_sequence not strictly increasing: {:?} <= {:?}",
                    s.turn_sequence, prev,
                )));
            }
            last = Some(s.turn_sequence);
        }
        Ok(snapshots)
    }
}

// `TurnSnapshot` does not carry its `ConversationId` -- that lives on
// the `ConversationState` the host commits. The `ConversationStore`
// trait takes the conversation id explicitly so the kernel never has
// to read it back from the snapshot. The example mirrors the host
// pattern: the harness pairs each `TurnSnapshot` with its
// `ConversationId` at the call site.

#[tokio::main]
async fn main() {
    // Demo: prove the example compiles and the trait wiring is right.
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

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_context_kernel::{BlockContent, TextPayload, TurnContext, TurnId};

    /// `(tempdir, store, conversation_id)` ready for a single test.
    /// The tempdir cleans up on drop. Each test gets a distinct
    /// conversation id (system-time nanos suffix) so filesystem
    /// state cannot leak between tests.
    fn new_store() -> (tempfile::TempDir, FsConversationStore, ConversationId) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store = FsConversationStore::new(tmp.path());
        let conv = ConversationId(format!("conv-{}", rand_suffix()));
        (tmp, store, conv)
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }

    fn snapshot(turn_seq: u64, turn_label: &str, text: &str) -> TurnSnapshot {
        // Build via the canonical `TurnContext` snapshot path so the
        // example does not depend on `OrderedBlocks`'s private inner
        // constructor. `TurnSnapshot::with_turn_sequence` sets the
        // sequence that `TurnContext::snapshot` leaves as a
        // placeholder (Slice 1.5: the conversation owns the sequence,
        // not the turn).
        let mut ctx = TurnContext::new(TurnId::new(turn_label));
        ctx.append_input(TextPayload::new(text), "user")
            .expect("append_input");
        ctx.snapshot().with_turn_sequence(TurnSequence(turn_seq))
    }

    #[tokio::test]
    async fn save_then_load_returns_same_snapshots_in_order() {
        let (_tmp, store, conv) = new_store();

        let s1 = snapshot(0, "t-1", "first");
        let s2 = snapshot(1, "t-2", "second");
        let s3 = snapshot(2, "t-3", "third");

        store.save_snapshot(&conv, &s1).await.expect("save 1");
        store.save_snapshot(&conv, &s2).await.expect("save 2");
        store.save_snapshot(&conv, &s3).await.expect("save 3");

        let loaded = store.load_snapshots(&conv).await.expect("load");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].turn_sequence, TurnSequence(0));
        assert_eq!(loaded[1].turn_sequence, TurnSequence(1));
        assert_eq!(loaded[2].turn_sequence, TurnSequence(2));
        assert_eq!(loaded[0].blocks.as_slice().len(), 1);
        let text_match = matches!(
            &loaded[0].blocks.as_slice()[0].content,
            BlockContent::Text(TextPayload(t)) if t == "first"
        );
        assert!(text_match, "loaded snapshot 0 text mismatch");
    }

    #[tokio::test]
    async fn upsert_overwrites_same_sequence_idempotently() {
        let (_tmp, store, conv) = new_store();

        let s = snapshot(5, "t-5", "original");
        store.save_snapshot(&conv, &s).await.expect("save original");
        // Overwrite: same turn_sequence, fresh empty context -> same
        // filename, atomic rename replaces the prior file.
        let ctx = TurnContext::new(TurnId::new("t-5"));
        let overwrite = ctx.snapshot().with_turn_sequence(TurnSequence(5));
        store
            .save_snapshot(&conv, &overwrite)
            .await
            .expect("save overwrite");

        let loaded = store.load_snapshots(&conv).await.expect("load");
        assert_eq!(loaded.len(), 1, "upsert must not duplicate");
        assert!(loaded[0].blocks.as_slice().is_empty());
    }

    #[tokio::test]
    async fn save_then_more_save_then_load_returns_all_in_order() {
        let (_tmp, store, conv) = new_store();

        store
            .save_snapshot(&conv, &snapshot(0, "t-a", "a"))
            .await
            .expect("save 0");
        store
            .save_snapshot(&conv, &snapshot(1, "t-b", "b"))
            .await
            .expect("save 1");
        // Gap in turn_sequence is allowed -- replay is the kernel's
        // concern, the store only orders lexicographically.
        store
            .save_snapshot(&conv, &snapshot(3, "t-d", "d"))
            .await
            .expect("save 3");

        let loaded = store.load_snapshots(&conv).await.expect("load");
        assert_eq!(
            loaded.iter().map(|s| s.turn_sequence.0).collect::<Vec<_>>(),
            vec![0, 1, 3],
        );
    }

    #[tokio::test]
    async fn unknown_conversation_returns_not_found() {
        let (_tmp, store, _conv) = new_store();
        let other = ConversationId("nope".into());
        let err = store.load_snapshots(&other).await.expect_err("NotFound");
        assert!(matches!(err, ConversationStoreError::NotFound(_)));
    }
}
