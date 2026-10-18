//! Closing a tool batch, from the interrupted-runs page.

use causa::{
    kernel::{
        BlockContent, BlockMeta, Context, ContextBlock, ToolBatch, ToolOutput, ToolResultPayload,
        ToolResultStatus, validate_tool_result_append,
    },
    runtime::new_block_id,
};
use serde_json::json;

// ANCHOR: close_batch
pub fn close_batch(
    context: &mut Context,
    mut batch: ToolBatch,
    reason: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Pending calls sit after the completed ones; resolving the first pending
    // entry moves it into the completed prefix.
    while let Some(entry) = batch.calls().first() {
        let call_block_id = entry.call().call_block_id;
        batch
            .resolve_at(
                batch.completed_len(),
                new_block_id(),
                ToolResultPayload::new(
                    call_block_id,
                    ToolResultStatus::Cancelled,
                    ToolOutput::new(json!({"error": reason})),
                ),
            )
            .map_err(|rejected| rejected.error)?;
    }
    let results = batch.into_results().map_err(|(error, _batch)| error)?;
    validate_tool_result_append(context.blocks(), &results)?;
    let blocks = results
        .into_iter()
        .map(|(id, result)| {
            ContextBlock::new(id, BlockContent::ToolResult(result), BlockMeta::default())
        })
        .collect();
    context.apply(Vec::new(), blocks)?;
    Ok(())
}
// ANCHOR_END: close_batch
