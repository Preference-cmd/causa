//! Pure model response validation and block construction.

use super::{ModelItemContent, ModelResponse, ModelStopReason};
use crate::context::block::{BlockContent, BlockMeta, ContentPart, ContextBlock, ToolCallPayload};
use crate::context::ids::BlockId;

/// Why a model response cannot be converted into blocks.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ModelBlockError {
    /// EndTurn cannot include tool declarations.
    #[error("EndTurn includes {count} tool calls")]
    EndTurnWithToolCalls {
        /// Number of tool declarations in the response.
        count: usize,
    },
    /// ToolUse requires at least one tool declaration.
    #[error("ToolUse has no tool calls")]
    ToolUseWithoutToolCalls,
    /// The supplied identities do not match the response's material count.
    #[error("expected {expected} block ids, received {actual}")]
    BlockIdCountMismatch {
        /// Number of identities required by the response.
        expected: usize,
        /// Number of supplied identities.
        actual: usize,
    },
    /// A tool declaration's name is blank.
    #[error("tool {tool_index} has an empty name")]
    EmptyToolName {
        /// Zero-based index in the response's tool declarations.
        tool_index: usize,
    },
    /// A tool declaration's arguments are not a JSON object.
    #[error("tool {tool_index} arguments are not an object")]
    ArgumentsNotObject {
        /// Zero-based index in the response's tool declarations.
        tool_index: usize,
    },
}

impl ModelResponse {
    /// Counts the blocks `to_blocks` produces: one per item, except that a
    /// `Parts` item whose parts are all zero-length text produces none.
    pub fn block_count(&self) -> usize {
        self.items
            .iter()
            .filter(|item| match &item.content {
                ModelItemContent::Parts(parts) => parts.iter().any(keeps_part),
                ModelItemContent::ToolCall(_) => true,
            })
            .count()
    }

    /// Validates and constructs material without changing or consuming inputs.
    ///
    /// One item becomes one block, in item order. A `Parts` item becomes a
    /// `Parts` block and a tool-call item a `ToolCall` block carrying its
    /// provider identifier. An item's `wire` is copied to the block's
    /// [`BlockMeta::wire`]. Only zero-length text parts are omitted
    /// (whitespace-only text is kept verbatim); a `Parts` item left with no
    /// parts produces no block and its `wire` is dropped. The kernel never
    /// reads `wire` or sealed items. Duplicate block identities are checked
    /// on context submission rather than here. MaxTokens and Refusal policy
    /// belongs to the caller.
    pub fn to_blocks(
        &self,
        stop_reason: ModelStopReason,
        block_ids: &[BlockId],
    ) -> Result<Vec<ContextBlock>, ModelBlockError> {
        let tool_count = self.tool_calls().count();
        match stop_reason {
            ModelStopReason::EndTurn if tool_count > 0 => {
                return Err(ModelBlockError::EndTurnWithToolCalls { count: tool_count });
            }
            ModelStopReason::ToolUse if tool_count == 0 => {
                return Err(ModelBlockError::ToolUseWithoutToolCalls);
            }
            // The runner commits only for EndTurn and ToolUse; the other four
            // reasons are interruptions, so conversion applies no policy.
            ModelStopReason::EndTurn
            | ModelStopReason::ToolUse
            | ModelStopReason::MaxTokens
            | ModelStopReason::Refusal
            | ModelStopReason::ContextWindowExceeded
            | ModelStopReason::Unrecognized => {}
        }
        let expected = self.block_count();
        if expected != block_ids.len() {
            return Err(ModelBlockError::BlockIdCountMismatch {
                expected,
                actual: block_ids.len(),
            });
        }
        for (tool_index, call) in self.tool_calls().enumerate() {
            if call.tool_name.trim().is_empty() {
                return Err(ModelBlockError::EmptyToolName { tool_index });
            }
            if !call.arguments.is_object() {
                return Err(ModelBlockError::ArgumentsNotObject { tool_index });
            }
        }
        let mut ids = block_ids.iter().copied();
        let mut blocks = Vec::with_capacity(expected);
        for item in &self.items {
            match &item.content {
                ModelItemContent::Parts(parts) => {
                    let parts: Vec<ContentPart> =
                        parts.iter().filter(|p| keeps_part(p)).cloned().collect();
                    if parts.is_empty() {
                        continue;
                    }
                    blocks.push(ContextBlock::new(
                        ids.next().expect("identity count validated"),
                        BlockContent::Parts(parts),
                        BlockMeta {
                            wire: item.wire.clone(),
                            ..BlockMeta::default()
                        },
                    ));
                }
                ModelItemContent::ToolCall(call) => {
                    blocks.push(ContextBlock::new(
                        ids.next().expect("identity count validated"),
                        BlockContent::ToolCall(ToolCallPayload {
                            tool_name: call.tool_name.clone(),
                            arguments: call.arguments.clone(),
                        }),
                        BlockMeta {
                            provider_call_id: call.provider_call_id.clone(),
                            wire: item.wire.clone(),
                            ..BlockMeta::default()
                        },
                    ));
                }
            }
        }
        Ok(blocks)
    }
}

/// Whether a part from a `Parts` item survives conversion: only text of zero
/// length is dropped.
fn keeps_part(part: &ContentPart) -> bool {
    !matches!(part, ContentPart::Text(t) if t.0.is_empty())
}
