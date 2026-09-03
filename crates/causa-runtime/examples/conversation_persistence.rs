//! Multi-turn conversation with snapshots persisted to disk and reloaded
//! across a "restart" — the host-side persistence loop the kernel's
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
//! `begin_turn → append_input → run_in_conversation → commit →
//! save_snapshot`, and on reload `load_snapshots →
//! ConversationState::from_snapshots`. The store writes one JSON file
//! per committed turn; `from_snapshots` validates strict sequence
//! monotonicity on load.

use async_trait::async_trait;
use causa_kernel::{
    ConversationId, ConversationState, ConversationStore, ConversationStoreError, ModelGateway,
    ModelInvokeError, ModelOutput, ModelRequest, ModelResponse, ModelStopReason, TextPayload,
    ToolCallDraft, TurnId, TurnSnapshot,
};
use causa_runtime::{ConversationOutcome, RunControl, ToolExecutor, TurnResult, TurnRunner};
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
    fn turn_path(&self, conversation_id: &ConversationId, snapshot: &TurnSnapshot) -> PathBuf {
        self.root
            .join(&conversation_id.0)
            .join(format!("turn-{:04}.json", snapshot.turn_sequence.0))
    }
}

#[async_trait]
impl ConversationStore for FsConversationStore {
    async fn save_snapshot(
        &self,
        conversation_id: &ConversationId,
        snapshot: &TurnSnapshot,
    ) -> Result<(), ConversationStoreError> {
        let dir = self.root.join(&conversation_id.0);
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))?;
        let bytes = serde_json::to_vec_pretty(snapshot)
            .map_err(|e| ConversationStoreError::Serialization(e.to_string()))?;
        tokio::fs::write(self.turn_path(conversation_id, snapshot), bytes)
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))
    }

    async fn load_snapshots(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<TurnSnapshot>, ConversationStoreError> {
        let dir = self.root.join(&conversation_id.0);
        let mut snapshots = Vec::new();
        let mut entries = tokio::fs::read_dir(&dir).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConversationStoreError::NotFound(conversation_id.0.clone())
            } else {
                ConversationStoreError::Io(e.to_string())
            }
        })?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| ConversationStoreError::Io(e.to_string()))?
        {
            let bytes = tokio::fs::read(entry.path())
                .await
                .map_err(|e| ConversationStoreError::Io(e.to_string()))?;
            let snapshot: TurnSnapshot = serde_json::from_slice(&bytes)
                .map_err(|e| ConversationStoreError::Serialization(e.to_string()))?;
            snapshots.push(snapshot);
        }
        snapshots.sort_by_key(|s| s.turn_sequence);
        Ok(snapshots)
    }
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
    let snapshot = state.commit(TurnId::new(turn_label))?;
    println!(
        "turn {turn_label} committed (seq {}): {text}",
        snapshot.turn_sequence.0
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
    let (state, _) = run_turn(&runner, state, "turn-2", "and again").await?;
    for snapshot in state.completed_turns() {
        store.save_snapshot(&conversation_id, snapshot).await?;
    }

    // --- "restart": rebuild the conversation from disk and continue -------
    let history = store.load_snapshots(&conversation_id).await?;
    println!("reloaded {} snapshot(s) from disk", history.len());
    let state = ConversationState::from_snapshots(conversation_id.clone(), history)?;
    let (state, text) = run_turn(&runner, state, "turn-3", "back after the restart").await?;
    if let Some(snapshot) = state.completed_turns().last() {
        store.save_snapshot(&conversation_id, snapshot).await?;
    }
    assert!(text.contains("reloaded"), "third script entry served");

    println!("conversation persisted and resumed cleanly");
    Ok(())
}
