//! Output retention is a host policy, exercised through the public batch contract.
#[path = "../examples/media_feedback/output_budget.rs"]
mod output_budget;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::new_block_id;
use output_budget::{TokenCounter, ToolOutputBudgetProcessor};
use serde_json::json;
use std::sync::{Arc, Mutex};

struct Counter;
impl TokenCounter for Counter {
    fn estimate_value(&self, value: &serde_json::Value) -> usize {
        value.as_str().unwrap().len()
    }
    fn estimate_media(&self, media: &MediaRef) -> Option<usize> {
        (media.media_type == "image/png").then_some(20)
    }
}
#[derive(Default)]
struct Store(Mutex<Vec<u8>>, bool);
#[async_trait]
impl ArtifactStore for Store {
    async fn persist(&self, bytes: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        if self.1 {
            return Err(StoreError::Persist("offline".into()));
        }
        *self.0.lock().unwrap() = bytes.to_vec();
        Ok(ArtifactRef {
            id: "original".into(),
            size_bytes: bytes.len(),
            kind: hint.kind,
            persisted: true,
        })
    }
    async fn read(&self, _: &str, _: Option<std::ops::Range<u64>>) -> Result<Vec<u8>, StoreError> {
        Ok(self.0.lock().unwrap().clone())
    }
}
fn batch() -> ToolBatch {
    let call = ToolCallContext::from_declaration(
        new_block_id(),
        &ToolCallPayload {
            tool_name: "echo".into(),
            arguments: json!({}),
        },
    );
    let id = call.call_block_id;
    let mut batch = ToolBatch::new(vec![call]).unwrap();
    batch
        .resolve_at(
            0,
            new_block_id(),
            ToolResultPayload {
                call_block_id: id,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("前缀 Unicode 🦀 tail ".repeat(100))),
                media: vec![MediaRef::new("image/png", "image")],
                notes: vec![TextPayload::new("retain this note")],
            },
        )
        .unwrap();
    batch
}
async fn process(
    policy: &ToolOutputBudgetProcessor,
    batch: &mut ToolBatch,
) -> Result<(), ProcessorError> {
    let ids = batch.declaration_ids();
    let turn = TurnId::new("retention");
    let control = CallControl::new(Default::default(), None);
    policy
        .process(
            batch,
            &ProcessorContext {
                turn_id: &turn,
                round_id: RoundId(0),
                declaration_order: &ids,
                control: &control,
            },
        )
        .await
}
#[tokio::test]
async fn spill_keeps_exact_original_utf8_and_respects_budget_including_notes_and_media() {
    let store = Arc::new(Store::default());
    let mut batch = batch();
    let before = batch.results()[0].result().unwrap().1.clone();
    let policy = ToolOutputBudgetProcessor::new(1)
        .for_tool("echo", 140)
        .with_token_counter(Arc::new(Counter))
        .with_artifact_store(store.clone());
    process(&policy, &mut batch).await.unwrap();
    let after = batch.results()[0].result().unwrap().1;
    assert_eq!(after.call_block_id, before.call_block_id);
    assert_eq!(after.status, before.status);
    assert_eq!(after.notes, before.notes);
    assert_eq!(after.media, before.media);
    assert_eq!(after.output.truncation, Truncation::Middle);
    let visible = format!(
        "{}\n\nNotes:\n- retain this note",
        after.output.content.as_str().unwrap()
    );
    assert!(visible.len() + 20 <= 140);
    assert!(visible.contains("… output truncated …"));
    let artifact = after.output.artifact.as_ref().unwrap();
    assert_eq!(artifact.kind, ArtifactKind::FullOutput);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&store.read(&artifact.id, None).await.unwrap())
            .unwrap(),
        before.output.content
    );
    assert_eq!(
        after.output.meta.as_ref().unwrap().original_tokens,
        Some(before.output.content.as_str().unwrap().len())
    );
}
#[tokio::test]
async fn missing_media_estimate_or_unrepresentable_floor_rejects_without_changing_entry() {
    for policy in [
        ToolOutputBudgetProcessor::new(140),
        ToolOutputBudgetProcessor::new(1).with_token_counter(Arc::new(Counter)),
    ] {
        let mut batch = batch();
        let before = batch.results()[0].result().unwrap().1.clone();
        assert!(process(&policy, &mut batch).await.is_err());
        assert_eq!(batch.results()[0].result().unwrap().1, &before);
    }
}
#[tokio::test]
async fn artifact_failure_keeps_full_output_and_existing_material() {
    let policy = ToolOutputBudgetProcessor::new(140)
        .with_token_counter(Arc::new(Counter))
        .with_artifact_store(Arc::new(Store(Mutex::new(vec![]), true)));
    let mut batch = batch();
    let before = batch.results()[0].result().unwrap().1.clone();
    assert!(process(&policy, &mut batch).await.is_err());
    assert_eq!(batch.results()[0].result().unwrap().1, &before);
}
#[tokio::test]
async fn unlimited_policy_preserves_unsupported_media_without_estimating() {
    let policy = ToolOutputBudgetProcessor::new(usize::MAX);
    let mut batch = batch();
    let before = batch.results()[0].result().unwrap().1.clone();
    process(&policy, &mut batch).await.unwrap();
    assert_eq!(batch.results()[0].result().unwrap().1, &before);
}
