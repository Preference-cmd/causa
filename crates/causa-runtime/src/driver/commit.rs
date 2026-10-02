//! Synchronous material conversion and atomic appends, preserving failed inputs.
//!
//! Observations and terminal choices remain in the runner. These operations
//! neither check control nor notify callbacks between conversion and commit.

use super::TurnInterruption;
use crate::new_block_id;
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, Context, ContextBlock, InvocationId, ModelOutput, ToolBatch,
    ToolCallContext, ToolResultStatus, validate_tool_result_append,
};

pub(super) struct ModelCommit {
    pub(super) output: ModelOutput,
    pub(super) batch: ToolBatch,
    pub(super) start: usize,
}

// Keep the public interruption's owned failure inputs without boxing or loss.
#[allow(clippy::result_large_err)]
pub(super) fn model(
    context: &mut Context,
    invocation_id: &InvocationId,
    output: ModelOutput,
) -> Result<ModelCommit, TurnInterruption> {
    let block_ids = (0..output.response.block_count())
        .map(|_| new_block_id())
        .collect::<Vec<_>>();
    let blocks = match output.response.to_blocks(output.stop_reason, &block_ids) {
        Ok(blocks) => blocks,
        Err(error) => {
            return Err(TurnInterruption::ModelConversionFailed {
                invocation_id: invocation_id.clone(),
                error,
                output,
                block_ids,
            });
        }
    };
    let calls = blocks
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolCall(payload) => {
                Some(ToolCallContext::from_declaration(block.id(), payload))
            }
            _ => None,
        })
        .collect();
    let batch = match ToolBatch::new(calls) {
        Ok(batch) => batch,
        Err(error) => {
            return Err(TurnInterruption::ModelBatchFailed {
                invocation_id: invocation_id.clone(),
                error,
                output,
                blocks,
            });
        }
    };
    let start = context.blocks().len();
    if let Err(error) = context.apply(vec![], blocks) {
        return Err(TurnInterruption::ModelCommitFailed {
            invocation_id: invocation_id.clone(),
            error,
            output,
        });
    }
    Ok(ModelCommit {
        output,
        batch,
        start,
    })
}

pub(super) struct ToolCommit {
    pub(super) start: usize,
    pub(super) unknown: Option<BlockId>,
}

// The caller still owns the batch on failure; EditFailure retains append inputs.
#[allow(clippy::result_large_err)]
pub(super) fn tools(
    context: &mut Context,
    invocation_id: &InvocationId,
    batch: &ToolBatch,
) -> Result<ToolCommit, TurnInterruption> {
    let results = batch
        .results()
        .iter()
        .filter_map(|entry| entry.result().map(|(id, payload)| (*id, payload.clone())))
        .collect::<Vec<_>>();
    validate_tool_result_append(context.blocks(), &results).map_err(|error| {
        TurnInterruption::ToolResultValidationFailed {
            invocation_id: invocation_id.clone(),
            error,
        }
    })?;
    let unknown = results.iter().find_map(|(_, result)| {
        (result.status == ToolResultStatus::UnknownOutcome).then_some(result.call_block_id)
    });
    let blocks = results
        .into_iter()
        .map(|(id, result)| {
            ContextBlock::new(id, BlockContent::ToolResult(result), BlockMeta::default())
        })
        .collect();
    let start = context.blocks().len();
    context
        .apply(vec![], blocks)
        .map_err(|error| TurnInterruption::ToolCommitFailed {
            invocation_id: invocation_id.clone(),
            error,
        })?;
    Ok(ToolCommit { start, unknown })
}
