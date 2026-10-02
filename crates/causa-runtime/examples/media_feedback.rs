//! Offline media closed loop: a tool produces an image, the
//! host's asset store ingests the bytes, and only a `MediaRef` rides the
//! facts — the next model round's frame carries the reference, and a
//! Context round-trip preserves it. Rendering-side resolution (bytes →
//! wire image blocks) is the provider's `MediaResolver`, shown in
//! `causa-provider`'s quickstart; this example runs fully offline.
//!
//! Run: `cargo run -p causa-runtime --example media_feedback`

#[path = "media_feedback/output_budget.rs"]
mod output_budget;
use output_budget::{TokenCounter, ToolOutputBudgetProcessor};

use async_trait::async_trait;
use causa_kernel::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, BlockContent, MediaRef, ModelGateway,
    ModelInvokeError, ModelOutput, ModelRequest, ModelResponse, ModelStopReason, StoreError,
    TextPayload, Tool, ToolCallContext, ToolDefinition, ToolOutput, ToolResultPayload,
    ToolResultStatus,
};
use causa_runtime::{
    RunControl, ToolExecutor, ToolExecutorOptions, TurnRunOptions, TurnRunner, new_block_id,
};
use std::sync::{Arc, Mutex};

/// The host's in-memory asset table: bytes keyed by content hash. Facts
/// only ever see the id.
#[derive(Default)]
struct MemoryAssets(Mutex<Vec<u8>>);
#[async_trait]
impl ArtifactStore for MemoryAssets {
    async fn persist(&self, data: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        *self.0.lock().unwrap() = data.to_vec();
        Ok(ArtifactRef {
            id: format!("blake3-{}", &blake3::hash(data).to_hex()[..8]),
            size_bytes: data.len(),
            kind: hint.kind,
            persisted: true,
        })
    }
    async fn read(
        &self,
        _id: &str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError> {
        Ok(self.0.lock().unwrap().clone())
    }
}

/// A "chart renderer": its bytes go to the store, its result carries only
/// the media reference.
struct RenderChart(Arc<MemoryAssets>);
#[async_trait]
impl Tool for RenderChart {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "render_chart".into(),
            description: "render a chart image".into(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }
    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _c: &causa_kernel::CallControl,
    ) -> ToolResultPayload {
        let bytes = b"fake-png-bytes".to_vec();
        let artifact = self
            .0
            .persist(
                &bytes,
                ArtifactHint {
                    tool_name: ctx.input.tool_name.clone(),
                    call_block_id: ctx.call_block_id,
                    kind: ArtifactKind::Binary,
                },
            )
            .await
            .expect("ingest");
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(serde_json::json!(
                "chart ready with explanatory details. ".repeat(100)
            )),
            media: vec![causa_kernel::MediaRef::new("image/png", artifact.id)],
            notes: vec![TextPayload::new("Image reference preserved.")],
        }
    }
}

/// A scripted gateway that records the frames it was shown.
struct Recorder(Mutex<Vec<ModelRequest>>);
#[async_trait]
impl ModelGateway for Recorder {
    async fn invoke(
        &self,
        req: &ModelRequest,
        _ctrl: &causa_kernel::CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.0.lock().unwrap().push(req.clone());
        let n = self.0.lock().unwrap().len();
        Ok(if n == 1 {
            ModelOutput {
                response: ModelResponse {
                    text: TextPayload::new(""),
                    tool_calls: vec![causa_kernel::ToolCallDraft {
                        tool_name: "render_chart".into(),
                        arguments: serde_json::json!({}),
                        provider_call_id: None,
                    }],
                },
                usage: None,
                stop_reason: ModelStopReason::ToolUse,
                reasoning: None,
            }
        } else {
            ModelOutput {
                response: ModelResponse {
                    text: TextPayload::new("I see the chart"),
                    tool_calls: vec![],
                },
                usage: None,
                stop_reason: ModelStopReason::EndTurn,
                reasoning: None,
            }
        })
    }
}

/// Deliberately simple demo estimate; production hosts choose their tokenizer.
struct DemoCounter;
impl TokenCounter for DemoCounter {
    fn estimate_value(&self, value: &serde_json::Value) -> usize {
        value
            .as_str()
            .map_or_else(|| value.to_string().len(), str::len)
            .div_ceil(4)
    }
    fn estimate_media(&self, _: &MediaRef) -> Option<usize> {
        Some(8)
    }
}

#[tokio::main]
async fn main() {
    let assets = Arc::new(MemoryAssets::default());
    let outputs = Arc::new(MemoryAssets::default());
    let budget = ToolOutputBudgetProcessor::new(64)
        .for_tool("render_chart", 64)
        .with_token_counter(Arc::new(DemoCounter))
        .with_artifact_store(outputs.clone());
    let executor = Arc::new(
        ToolExecutor::new(
            vec![Arc::new(RenderChart(assets.clone()))],
            ToolExecutorOptions {
                after: vec![Arc::new(budget)],
                ..Default::default()
            },
        )
        .expect("unique tools"),
    );

    let gateway = Arc::new(Recorder(Mutex::new(vec![])));
    let runner = TurnRunner::new(gateway.clone(), executor);

    let mut ctx = causa_kernel::Context::new();
    ctx.edit()
        .append([causa_kernel::ContextBlock::new(
            new_block_id(),
            causa_kernel::BlockContent::Parts(vec![causa_kernel::ContentPart::Text(
                TextPayload::new("chart the numbers"),
            )]),
            causa_kernel::BlockMeta {
                source: Some("user".into()),
                ..Default::default()
            },
        )])
        .commit()
        .unwrap();

    let outcome = runner
        .run(
            causa_kernel::TurnId::new("media_feedback"),
            ctx,
            TurnRunOptions::new(causa_kernel::ModelRef::new("offline-demo")),
            RunControl::new(Default::default(), None),
        )
        .await;
    assert!(matches!(
        outcome.result,
        causa_runtime::TurnResult::Completed { .. }
    ));

    // Round 2's frame carried the tool result's media reference.
    let round2 = gateway.0.lock().unwrap()[1].clone();
    let media_refs: Vec<MediaRef> = round2
        .frame
        .blocks
        .iter()
        .filter_map(|b| match b.content() {
            BlockContent::ToolResult(r) => r.media.first().cloned(),
            _ => None,
        })
        .collect();
    assert_eq!(media_refs.len(), 1);
    let reference = media_refs[0].reference.clone();
    println!("round 2 saw media reference: {reference}");

    // The asset id is content-addressed and the bytes live only in the store.
    assert!(reference.starts_with("blake3-"));
    assert_eq!(
        assets.read(&reference, None).await.unwrap(),
        b"fake-png-bytes"
    );

    // This consumer selected result-local retention. It retains media/notes,
    // spills the exact original JSON bytes, and verifies the retained estimate.
    let result = outcome
        .context
        .blocks()
        .iter()
        .find_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .expect("tool result committed");
    assert_eq!(result.output.truncation, causa_kernel::Truncation::Middle);
    assert_eq!(
        result.notes,
        vec![TextPayload::new("Image reference preserved.")]
    );
    assert_eq!(result.media, media_refs);
    let mut visible = result.output.content.as_str().unwrap().to_owned();
    visible.push_str("\n\nNotes:\n- Image reference preserved.");
    assert!(visible.len().div_ceil(4) + 8 <= 64);
    let artifact = result.output.artifact.as_ref().expect("full output stored");
    assert_eq!(artifact.kind, ArtifactKind::FullOutput);
    let full: serde_json::Value =
        serde_json::from_slice(&outputs.read(&artifact.id, None).await.unwrap()).unwrap();
    assert_eq!(
        full,
        serde_json::json!("chart ready with explanatory details. ".repeat(100))
    );

    // Context serde preserves the reference — and nothing else.
    let restored = serde_json::to_string(&outcome.context).unwrap();
    assert!(restored.contains(&reference));
    assert!(!restored.contains("fake-png-bytes"));
    println!("turn round-trip kept the reference, never the bytes");
}
