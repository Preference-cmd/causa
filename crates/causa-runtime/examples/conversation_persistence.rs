//! Multi-turn conversation with session entries persisted to disk and
//! reloaded across a "restart" — the host-side persistence loop the
//! `ConversationStore` port exists for.
//!
//! Runnable offline: the model gateway is a scripted stub that answers
//! `"ack."` every round. In production you would plug a provider gateway
//! from `causa-provider` here — persistence is gateway-agnostic.
//!
//! ```text
//! cargo run --example conversation_persistence -p causa-runtime
//! ```
//!
//! The dance the example performs is the canonical commit loop:
//! `begin_turn → append_parts → run_in_conversation → commit →
//! save_entry`, and on reload `load_entries →
//! ConversationState::from_history`. The store writes one JSON file per
//! committed turn; `from_history` validates strict sequence monotonicity
//! on load.
//!
//! Slice 6.5 shapes:
//! - the store persists **`HistoryEntry`** records (`sequence` + `snapshot`)
//!   — the snapshot itself carries no session order anymore;
//! - the load path migrates **pre-6.5 files** that embedded
//!   `turn_sequence` inside the snapshot by extracting it into the entry;
//! - one turn's message mixes text and a `MediaRef`: facts carry the
//!   reference only, never bytes.

use async_trait::async_trait;
use causa_kernel::{
    ContentPart, ConversationId, MediaRef, ModelGateway, ModelInvokeError, ModelOutput,
    ModelRequest, ModelResponse, ModelStopReason, TextPayload, ToolCallDraft, TurnId, TurnSnapshot,
};
use causa_runtime::{
    ConversationOutcome, ConversationState, ConversationStore, ConversationStoreError,
    HistoryEntry, RunControl, ToolExecutor, TurnResult, TurnRunner, TurnSequence,
};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Scripted stub gateway: pops one canned output per model round, then
/// fails permanently — the offline stand-in for a provider adapter.
struct ScriptedGateway(Mutex<Vec<ModelOutput>>);

impl ScriptedGateway {
    fn new(outputs: Vec<ModelOutput>) -> Arc<Self> {
        Arc::new(Self(Mutex::new(outputs)))
    }
}

#[async_trait]
impl ModelGateway for ScriptedGateway {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &causa_kernel::AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut outputs = self.0.lock().await;
        if outputs.is_empty() {
            return Err(ModelInvokeError::new(
                causa_kernel::ModelInvokeErrorKind::Permanent,
                "script exhausted",
            ));
        }
        Ok(outputs.remove(0))
    }
}

fn ack(text: &str) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: Vec::<ToolCallDraft>::new(),
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    }
}

/// Reference `ConversationStore` over the local filesystem — one JSON
/// file per committed turn under `<root>/<conversation_id>/`. Reference
/// implementations of ports live in examples and hosts, never in the
/// publish set.
struct FsConversationStore {
    root: PathBuf,
}

impl FsConversationStore {
    fn turn_path(&self, conversation_id: &ConversationId, sequence: TurnSequence) -> PathBuf {
        self.root
            .join(&conversation_id.0)
            .join(format!("turn-{:04}.json", sequence.0))
    }
}

#[async_trait]
impl ConversationStore for FsConversationStore {
    async fn save_entry(
        &self,
        conversation_id: &ConversationId,
        entry: &HistoryEntry,
    ) -> Result<(), ConversationStoreError> {
        let dir = self.root.join(&conversation_id.0);
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))?;
        let bytes = serde_json::to_vec_pretty(entry)
            .map_err(|e| ConversationStoreError::Serialization(e.to_string()))?;
        tokio::fs::write(self.turn_path(conversation_id, entry.sequence), bytes)
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))
    }

    async fn load_entries(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<HistoryEntry>, ConversationStoreError> {
        let dir = self.root.join(&conversation_id.0);
        let mut entries = Vec::new();
        let mut files = tokio::fs::read_dir(&dir).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConversationStoreError::NotFound(conversation_id.0.clone())
            } else {
                ConversationStoreError::Io(e.to_string())
            }
        })?;
        while let Some(file) = files
            .next_entry()
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))?
        {
            let bytes = tokio::fs::read(file.path())
                .await
                .map_err(|e| ConversationStoreError::Io(e.to_string()))?;
            entries.push(load_entry(&bytes)?);
        }
        entries.sort_by_key(|e| e.sequence);
        Ok(entries)
    }
}

/// Load one history file, accepting both wire shapes:
///
/// - **current** (Slice 6.5): `{"sequence": N, "snapshot": {...}}`;
/// - **pre-6.5 legacy**: the old DTO embedded `turn_sequence` inside the
///   snapshot itself — extract it into the entry, never drop the order.
fn load_entry(bytes: &[u8]) -> Result<HistoryEntry, ConversationStoreError> {
    if let Ok(entry) = serde_json::from_slice::<HistoryEntry>(bytes) {
        return Ok(entry);
    }
    #[derive(Deserialize)]
    struct LegacySnapshot {
        #[serde(flatten)]
        snapshot: TurnSnapshot,
        turn_sequence: u64,
    }
    let legacy: LegacySnapshot = serde_json::from_slice(bytes).map_err(|e| {
        ConversationStoreError::Corrupted(format!(
            "not a history entry nor a legacy turn snapshot: {e}"
        ))
    })?;
    Ok(HistoryEntry {
        sequence: TurnSequence(legacy.turn_sequence),
        snapshot: legacy.snapshot,
    })
}

/// Drive one input through the conversation entry, print the outcome,
/// and return the state for the next stage. `run_in_conversation`
/// consumes and returns the state with the active turn sealed; the host
/// then commits (or aborts) it.
async fn run_turn(
    runner: &TurnRunner,
    mut state: ConversationState,
    turn_label: &str,
    input: &str,
) -> Result<(ConversationState, String), Box<dyn std::error::Error>> {
    state.begin_turn(TurnId::new(turn_label))?;
    state
        .active_turn_mut()
        .expect("begin_turn just opened it")
        .append_input(TextPayload::new(input), "user")?;

    let ConversationOutcome {
        state: returned_state,
        result,
        ..
    } = runner
        .run_in_conversation(
            state,
            Default::default(),
            RunControl::new(Default::default(), None),
        )
        .await?;
    let mut state = returned_state;

    let text = match result {
        TurnResult::Completed { final_output } => final_output.response.text.0,
        other => return Err(format!("turn {turn_label} did not complete: {other:?}").into()),
    };
    let entry = state.commit(TurnId::new(turn_label))?;
    println!(
        "turn {turn_label} committed (seq {}): {text}",
        entry.sequence.0
    );
    Ok((state, text))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scratch = tempfile::tempdir()?;
    let store = Arc::new(FsConversationStore {
        root: scratch.path().to_path_buf(),
    });
    let conversation_id = ConversationId("demo".into());

    let gateway = ScriptedGateway::new(vec![
        ack("ack — first turn sealed"),
        ack("ack — second turn, with history"),
        ack("ack — third turn, reloaded from disk"),
    ]);
    let runner = TurnRunner::new(gateway, Arc::new(ToolExecutor::from_vec(Vec::new())));

    // --- session one: two turns, each persisted at commit -------------------
    let (state, _) = run_turn(
        &runner,
        ConversationState::new(conversation_id.clone()),
        "turn-1",
        "hello",
    )
    .await?;
    let (mut state, _) = run_turn(&runner, state, "turn-2", "and again").await?;

    // Slice 6.5: a turn whose message mixes text and a media reference.
    // Facts carry the reference only; the bytes live in the host's asset
    // store and resolve provider-side at render time.
    state.begin_turn(TurnId::new("turn-media"))?;
    state
        .active_turn_mut()
        .expect("begin_turn just opened it")
        .append_parts(
            vec![
                ContentPart::Text(TextPayload::new("the chart you asked for")),
                ContentPart::Media(MediaRef::new("image/png", "blake3-demo-asset")),
            ],
            "user",
        )?;
    state.seal_turn(
        TurnId::new("turn-media"),
        causa_runtime::SealedResult::Completed,
    )?;
    state.commit(TurnId::new("turn-media"))?;

    for entry in state.history() {
        store.save_entry(&conversation_id, entry).await?;
    }

    // --- "restart": rebuild the session from disk and continue -------------
    let history = store.load_entries(&conversation_id).await?;
    println!("reloaded {} session entry(s) from disk", history.len());
    let state = ConversationState::from_history(conversation_id.clone(), history)?;
    let (state, text) = run_turn(&runner, state, "turn-3", "back after the restart").await?;
    if let Some(entry) = state.history().last() {
        store.save_entry(&conversation_id, entry).await?;
    }
    assert!(text.contains("reloaded"), "third script entry served");

    println!("conversation persisted and resumed cleanly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The legacy migration path: a pre-6.5 file (turn_sequence embedded
    /// in the snapshot) loads into an entry that keeps both the order and
    /// the facts.
    #[test]
    fn legacy_snapshot_files_migrate_into_history_entries() {
        let legacy = r#"{
            "turn_id": "t-old",
            "turn_sequence": 3,
            "blocks": [],
            "source_version": 2,
            "sealed": true
        }"#;
        let entry = load_entry(legacy.as_bytes()).expect("legacy file migrates");
        assert_eq!(entry.sequence.0, 3);
        assert_eq!(entry.snapshot.turn_id.0, "t-old");
        assert_eq!(entry.snapshot.source_version.0, 2);

        // The current shape loads directly.
        let current = r#"{
            "sequence": 4,
            "snapshot": {
                "turn_id": "t-new",
                "blocks": [],
                "source_version": 1,
                "sealed": true
            }
        }"#;
        let entry = load_entry(current.as_bytes()).expect("current file loads");
        assert_eq!(entry.sequence.0, 4);
    }
}
