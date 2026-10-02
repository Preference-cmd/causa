//! Compose tool processors around fixed executor dispatch. Runnable offline:
//! `cargo run -p causa-runtime --example tool_processors`
//!
//! This example rewrites a limit, explains it in result notes, deduplicates
//! effective inputs, then restores declaration order after execution. Each
//! policy is an ordinary processor; the empty chain applies none of them.

#[path = "tool_processors/policy.rs"]
mod policy;
use policy::DeduplicateProcessor;

use async_trait::async_trait;
use causa_kernel::{
    BlockContent, CallControl, ModelGateway, ModelInvokeError, ModelOutput, ModelRequest,
    ModelResponse, ModelStopReason, ProcessorContext, ProcessorError, TextPayload, Tool, ToolBatch,
    ToolBatchProcessor, ToolCallContext, ToolCallDraft, ToolDefinition, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnId,
};
use causa_runtime::{
    RunControl, ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner,
    new_block_id,
};
use std::sync::Arc;

struct LimitResults;
#[async_trait]
impl ToolBatchProcessor for LimitResults {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        for entry in batch.calls_mut() {
            let input = entry.input_mut().expect("pending entry");
            if input.tool_name == "search" && input.arguments["limit"].as_u64().unwrap_or(0) > 5 {
                input.arguments["limit"] = serde_json::json!(5);
                entry.push_note(TextPayload::new(
                    "This session limits search results to five.",
                ));
            }
        }
        Ok(())
    }
}

struct DeclarationOrder;
#[async_trait]
impl ToolBatchProcessor for DeclarationOrder {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        // Sorting moves each result together with both of its identities.
        batch.results_mut().sort_by_key(|entry| {
            ctx.declaration_order
                .iter()
                .position(|id| *id == entry.call().call_block_id)
        });
        Ok(())
    }
}

struct Search;
#[async_trait]
impl Tool for Search {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search".into(),
            description: "Offline research fixture".into(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(serde_json::json!({"count": ctx.input.arguments["limit"]})),
            media: Vec::new(),
            notes: Vec::new(),
        }
    }
}

struct ResearchGateway;
#[async_trait]
impl ModelGateway for ResearchGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let has_results = request
            .frame
            .blocks
            .iter()
            .any(|block| matches!(block.content(), BlockContent::ToolResult(_)));
        Ok(ModelOutput {
            response: ModelResponse {
                text: TextPayload::new(if has_results {
                    "Research complete."
                } else {
                    ""
                }),
                tool_calls: if has_results {
                    Vec::new()
                } else {
                    // Distinct declarations become identical effective calls
                    // after LimitResults, so only the first reaches Search.
                    [20, 10].into_iter().map(|limit| ToolCallDraft {
                        tool_name: "search".into(),
                        arguments: serde_json::json!({"query": "video generation", "limit": limit}),
                        provider_call_id: None,
                    }).collect()
                },
            },
            usage: None,
            stop_reason: if has_results {
                ModelStopReason::EndTurn
            } else {
                ModelStopReason::ToolUse
            },
            reasoning: None,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let executor = ToolExecutor::new(
        vec![Arc::new(Search)],
        ToolExecutorOptions {
            before: vec![Arc::new(LimitResults), Arc::new(DeduplicateProcessor)],
            after: vec![Arc::new(DeclarationOrder)],
            ..Default::default()
        },
    )?;
    let runner = TurnRunner::new(Arc::new(ResearchGateway), Arc::new(executor));
    let mut context = causa_kernel::Context::new();
    context
        .edit()
        .append([causa_kernel::ContextBlock::new(
            new_block_id(),
            causa_kernel::BlockContent::Parts(vec![causa_kernel::ContentPart::Text(
                TextPayload::new("Research video models."),
            )]),
            causa_kernel::BlockMeta {
                source: Some("user".into()),
                ..Default::default()
            },
        )])
        .commit()?;
    let outcome = runner
        .run(
            TurnId::new("tool_processors"),
            context,
            TurnRunOptions::new(causa_kernel::ModelRef::new("offline-demo")),
            RunControl::new(Default::default(), None),
        )
        .await;
    if !matches!(outcome.result, TurnResult::Completed { .. }) {
        return Err(format!("turn did not complete: {:?}", outcome.result).into());
    }
    for block in outcome.context.blocks() {
        if let BlockContent::ToolResult(result) = block.content() {
            println!(
                "{:?}: {} — {:?}",
                result.status, result.output.content, result.notes
            );
        }
    }
    Ok(())
}
