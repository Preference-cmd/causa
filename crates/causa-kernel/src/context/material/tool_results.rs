//! Execution-specific tool-result append validation.

use crate::context::block::{BlockContent, ContextBlock};
use crate::context::ids::BlockId;
use crate::context::tool_data::ToolResultPayload;
use std::collections::HashSet;

/// Why tool results cannot be appended to the supplied material scope.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolResultError {
    /// The reference does not identify a tool declaration in this scope.
    #[error("missing tool declaration {call_block_id:?}")]
    MissingDeclaration {
        /// The declaration referenced by the rejected result.
        call_block_id: BlockId,
    },
    /// This declaration already has a result in the supplied material.
    #[error("tool declaration already completed: {call_block_id:?}")]
    AlreadyCompleted {
        /// The declaration referenced by the rejected result.
        call_block_id: BlockId,
    },
    /// This append would complete the same declaration more than once.
    #[error("duplicate tool result for declaration {call_block_id:?}")]
    DuplicateResult {
        /// The declaration referenced by the rejected result.
        call_block_id: BlockId,
    },
}

/// Validates this result append without mutating or consuming its inputs.
///
/// Checks references in input order. Other local tool material need not form
/// complete exchanges. Result block identity remains the responsibility of
/// [`super::Context::apply`].
pub fn validate_tool_result_append(
    context_blocks: &[ContextBlock],
    results: &[(BlockId, ToolResultPayload)],
) -> Result<(), ToolResultError> {
    let declarations = context_blocks
        .iter()
        .filter_map(|block| {
            matches!(block.content(), BlockContent::ToolCall(_)).then_some(block.id())
        })
        .collect::<HashSet<_>>();
    let completed = context_blocks
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result.call_block_id),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut batch = HashSet::with_capacity(results.len());
    for (_, result) in results {
        let call_block_id = result.call_block_id;
        if !declarations.contains(&call_block_id) {
            return Err(ToolResultError::MissingDeclaration { call_block_id });
        }
        if completed.contains(&call_block_id) {
            return Err(ToolResultError::AlreadyCompleted { call_block_id });
        }
        if !batch.insert(call_block_id) {
            return Err(ToolResultError::DuplicateResult { call_block_id });
        }
    }
    Ok(())
}
