//! Standalone tool-dispatch evidence: the executor accepts declaration
//! context with explicit block identity and returns a result paired to it.

use async_trait::async_trait;
use causa_kernel::{
    ArtifactHint, ArtifactRef, ArtifactStore, BlockId, CallControl, DynamicToolSource, MediaRef,
    ProcessorContext, RoundId, SourceError, StoreError, TextPayload, Tool, ToolBatch,
    ToolBatchProcessor, ToolCallContext, ToolCallPayload, ToolDefinition, ToolExecutionError,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnId,
};
use causa_runtime::{TokenCounter, ToolExecutor, ToolOutputBudgetProcessor, new_block_id};
use serde_json::json;
use std::sync::Arc;

fn call_context(name: &str, args: serde_json::Value) -> ToolCallContext {
    ToolCallContext::from_declaration(
        new_block_id(),
        &ToolCallPayload {
            tool_name: name.into(),
            arguments: args,
        },
    )
}

fn ctrl() -> CallControl {
    CallControl::new(tokio_util::sync::CancellationToken::new(), None)
}

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echo arguments".into(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn execute(&self, ctx: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(ctx.input.arguments.clone()),
            media: Vec::new(),
            notes: Vec::new(),
        }
    }
}

struct StubSource;

#[async_trait]
impl DynamicToolSource for StubSource {
    fn id(&self) -> &str {
        "stub"
    }

    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        Ok(vec![ToolDefinition {
            name: "mcp_echo".into(),
            description: "dynamic echo".into(),
            parameters: json!({"type": "object"}),
        }])
    }

    async fn invoke(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        Ok(ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(ctx.input.arguments.clone()),
            media: Vec::new(),
            notes: Vec::new(),
        })
    }
}

#[tokio::test]
async fn standalone_executor_dispatches_with_declaration_identity() {
    let executor = ToolExecutor::from_vec(vec![Arc::new(EchoTool)]);
    let context = call_context("echo", json!({"q": 1}));
    let call_block_id = context.call_block_id;

    let result = executor.execute(context, ctrl(), None).await.unwrap();

    assert_eq!(result.call_block_id, call_block_id);
    assert_eq!(result.status, ToolResultStatus::Succeeded);
    assert_eq!(result.output.content, json!({"q": 1}));
}

#[tokio::test]
async fn dynamic_dispatch_uses_the_same_identity_bearing_context() {
    let executor = ToolExecutor::from_vec(vec![]);
    executor.register_dynamic(Arc::new(StubSource)).unwrap();
    let surface = executor.tool_surface().await;
    assert!(surface.definitions.iter().any(|d| d.name == "mcp_echo"));

    let context = call_context("mcp_echo", json!({"hello": "world"}));
    let call_block_id = context.call_block_id;
    let result = executor.execute(context, ctrl(), None).await.unwrap();

    assert_eq!(result.call_block_id, call_block_id);
    assert_eq!(result.output.content, json!({"hello": "world"}));
}

struct EmptyStore;

#[async_trait]
impl ArtifactStore for EmptyStore {
    async fn persist(&self, _data: &[u8], _hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        Err(StoreError::Persist("unused".into()))
    }

    async fn read(
        &self,
        _id: &str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError> {
        Err(StoreError::Read("unused".into()))
    }
}

struct StoreAwareTool;

#[async_trait]
impl Tool for StoreAwareTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "store-aware".into(),
            description: "reports whether the configured store was forwarded".into(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn execute(&self, ctx: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({"store": false})),
            media: Vec::new(),
            notes: Vec::new(),
        }
    }

    async fn execute_with_store(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({"store": store.is_some()})),
            media: Vec::new(),
            notes: Vec::new(),
        }
    }
}

#[tokio::test]
async fn executor_forwards_the_configured_artifact_store_to_tools() {
    let executor = ToolExecutor::from_vec(vec![Arc::new(StoreAwareTool)]);
    let result = executor
        .execute(
            call_context("store-aware", json!({})),
            ctrl(),
            Some(Arc::new(EmptyStore)),
        )
        .await
        .unwrap();
    assert_eq!(result.output.content, json!({"store": true}));
}

struct Counter;

impl TokenCounter for Counter {
    fn estimate(&self, _blocks: &[causa_kernel::ContextBlock]) -> usize {
        0
    }

    fn estimate_value(&self, value: &serde_json::Value) -> usize {
        value.as_str().map_or(0, |text| text.len().div_ceil(4))
    }

    fn estimate_media(&self, _media: &MediaRef) -> Option<usize> {
        Some(2)
    }
}

#[tokio::test]
async fn output_budget_retains_media_sidecars_while_truncating_text() {
    let declaration = call_context("echo", json!({}));
    let call_block_id = declaration.call_block_id;
    let mut batch = ToolBatch::new(vec![declaration]).unwrap();
    let media = vec![MediaRef::new("image/png", "asset-1")];
    batch
        .resolve_at(
            0,
            new_block_id(),
            ToolResultPayload {
                call_block_id,
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!("x".repeat(4_000))),
                media: media.clone(),
                notes: vec![TextPayload::new("kept note")],
            },
        )
        .unwrap();

    let ids: [BlockId; 1] = [call_block_id];
    let control = ctrl();
    let turn_id = TurnId::new("budget");
    let context = ProcessorContext {
        conversation_id: None,
        turn_id: &turn_id,
        round_id: RoundId(0),
        declaration_order: &ids,
        control: &control,
    };
    let processor = ToolOutputBudgetProcessor::new(100).with_token_counter(Arc::new(Counter));
    processor.process(&mut batch, &context).await.unwrap();

    let (_, result) = batch.results()[0].result().unwrap();
    assert_eq!(result.output.truncation, causa_kernel::Truncation::Middle);
    assert_eq!(result.media, media, "media references remain untouched");
    assert_eq!(result.notes, [TextPayload::new("kept note")]);
}
