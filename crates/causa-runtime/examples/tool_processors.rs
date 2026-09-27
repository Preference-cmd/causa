//! Compose tool processors around fixed executor dispatch. Runnable offline:
//! `cargo run -p causa-runtime --example tool_processors`
//!
//! This example rewrites a limit, explains it in result notes, deduplicates
//! effective inputs, then restores declaration order after execution. Each
//! policy is an ordinary processor; the empty chain applies none of them.

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, BlockContent, CallControl, ModelGateway, ModelInvokeError, ModelOutput,
    ModelRequest, ModelResponse, ModelStopReason, ProcessorContext, ProcessorError, TextPayload,
    Tool, ToolBatch, ToolBatchProcessor, ToolCallContext, ToolCallDraft, ToolDefinition,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnContext, TurnId,
};
use causa_runtime::{
    DeduplicateProcessor, RunControl, ToolExecutor, ToolProcessingChain, TurnResult,
    TurnRunOptions, TurnRunner, new_block_id,
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
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let has_results = request
            .frame
            .model_context
            .blocks
            .iter()
            .any(|block| matches!(block.content, BlockContent::ToolResult(_)));
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
    let chain = ToolProcessingChain::builder()
        .before(Arc::new(LimitResults))
        .before(Arc::new(DeduplicateProcessor))
        .after(Arc::new(DeclarationOrder))
        .build();
    let runner = TurnRunner::with_tool_processors(
        Arc::new(ResearchGateway),
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(Search)])),
        chain,
    );
    let mut context = TurnContext::new(TurnId::new("research"));
    context.append_input(
        new_block_id(),
        TextPayload::new("Research video models."),
        "user",
    )?;
    let outcome = runner
        .run(
            context,
            TurnRunOptions::default(),
            RunControl::new(Default::default(), None),
        )
        .await;
    if !matches!(outcome.result, TurnResult::Completed { .. }) {
        return Err(format!("turn did not complete: {:?}", outcome.result).into());
    }
    for block in outcome.context.blocks() {
        if let BlockContent::ToolResult(result) = &block.content {
            println!(
                "{:?}: {} — {:?}",
                result.status, result.output.content, result.notes
            );
        }
    }
    Ok(())
}
