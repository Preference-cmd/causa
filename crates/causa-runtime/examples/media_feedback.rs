//! Offline media closed loop (Slice 6.5): a tool produces an image, the
//! host's asset store ingests the bytes, and only a `MediaRef` rides the
//! facts — the next model round's frame carries the reference, and a
//! snapshot round-trip preserves it. Rendering-side resolution (bytes →
//! wire image blocks) is the provider's `MediaResolver`, shown in
//! `causa-provider`'s quickstart; this example runs fully offline.
//!
//! Run: `cargo run -p causa-runtime --example media_feedback`

use async_trait::async_trait;
use causa_kernel::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, BlockContent, MediaRef, ModelGateway,
    ModelInvokeError, ModelOutput, ModelRequest, ModelResponse, ModelStopReason, StoreError,
    TextPayload, Tool, ToolCallContext, ToolDefinition, ToolExecutionOutcome, ToolOutput,
    ToolResultPayload, ToolResultStatus,
};
use causa_runtime::{RunControl, ToolExecutor, TurnRunOptions, TurnRunner};
use std::sync::{Arc, Mutex};

/// The host's in-memory asset table: bytes keyed by content hash. Facts
/// only ever see the id.
#[derive(Default)]
struct MemoryAssets(Mutex<Vec<u8>>);
#[async_trait]
impl ArtifactStore for MemoryAssets {
    async fn persist(&self, data: &[u8], _hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        *self.0.lock().unwrap() = data.to_vec();
        Ok(ArtifactRef {
            id: format!("blake3-{}", &blake3::hash(data).to_hex()[..8]),
            size_bytes: data.len(),
            kind: ArtifactKind::Binary,
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
    ) -> ToolExecutionOutcome {
        let bytes = b"fake-png-bytes".to_vec();
        let artifact = self
            .0
            .persist(
                &bytes,
                ArtifactHint {
                    tool_name: ctx.tool_name.clone(),
                    call_id: ctx.call_id.clone(),
                    kind: ArtifactKind::Binary,
                },
            )
            .await
            .expect("ingest");
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(serde_json::json!("chart ready")),
            media: vec![causa_kernel::MediaRef::new("image/png", artifact.id)],
        })
    }
}

/// A scripted gateway that records the frames it was shown.
struct Recorder(Mutex<Vec<ModelRequest>>);
#[async_trait]
impl ModelGateway for Recorder {
    async fn invoke(
        &self,
        req: &ModelRequest,
        _ctrl: &causa_kernel::AttemptControl,
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

#[tokio::main]
async fn main() {
    let assets = Arc::new(MemoryAssets::default());
    let executor = Arc::new(ToolExecutor::from_vec(vec![Arc::new(RenderChart(
        assets.clone(),
    ))]));

    let gateway = Arc::new(Recorder(Mutex::new(vec![])));
    let runner = TurnRunner::new(gateway.clone(), executor);

    let mut ctx = causa_kernel::TurnContext::new(causa_kernel::TurnId::new("media-turn"));
    ctx.append_input(TextPayload::new("chart the numbers"), "user")
        .unwrap();

    let outcome = runner
        .run(
            ctx,
            TurnRunOptions::default(),
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
        .model_context
        .blocks
        .iter()
        .filter_map(|b| match &b.content {
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

    // A snapshot round-trip preserves the reference — and nothing else.
    let snap = outcome.context.snapshot();
    let restored = serde_json::to_string(&snap).unwrap();
    assert!(restored.contains(&reference));
    assert!(!restored.contains("fake-png-bytes"));
    println!("snapshot round-trip kept the reference, never the bytes");
}
