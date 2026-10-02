//! Pure model response validation and block construction.

use super::{ModelResponse, ModelStopReason};
use crate::context::block::{BlockContent, BlockMeta, ContentPart, ContextBlock, ToolCallPayload};
use crate::context::ids::BlockId;

/// Why a model response cannot be converted into blocks.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
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
    /// Counts one block for nonblank text, plus one per tool declaration.
    pub fn block_count(&self) -> usize {
        usize::from(!self.text.0.trim().is_empty()) + self.tool_calls.len()
    }

    /// Validates and constructs material without changing or consuming inputs.
    ///
    /// Text precedes declarations, which retain their original order, arguments,
    /// names and provider identifiers. Blank text is omitted; other text is
    /// preserved verbatim. Duplicate block identities are checked on context
    /// submission rather than here. MaxTokens and Refusal policy belongs to the
    /// caller.
    pub fn to_blocks(
        &self,
        stop_reason: ModelStopReason,
        block_ids: &[BlockId],
    ) -> Result<Vec<ContextBlock>, ModelBlockError> {
        match stop_reason {
            ModelStopReason::EndTurn if !self.tool_calls.is_empty() => {
                return Err(ModelBlockError::EndTurnWithToolCalls {
                    count: self.tool_calls.len(),
                });
            }
            ModelStopReason::ToolUse if self.tool_calls.is_empty() => {
                return Err(ModelBlockError::ToolUseWithoutToolCalls);
            }
            _ => {}
        }
        let expected = self.block_count();
        if expected != block_ids.len() {
            return Err(ModelBlockError::BlockIdCountMismatch {
                expected,
                actual: block_ids.len(),
            });
        }
        for (tool_index, call) in self.tool_calls.iter().enumerate() {
            if call.tool_name.trim().is_empty() {
                return Err(ModelBlockError::EmptyToolName { tool_index });
            }
            if !call.arguments.is_object() {
                return Err(ModelBlockError::ArgumentsNotObject { tool_index });
            }
        }
        let mut ids = block_ids.iter().copied();
        let mut blocks = Vec::with_capacity(expected);
        if !self.text.0.trim().is_empty() {
            blocks.push(ContextBlock::new(
                ids.next().expect("identity count validated"),
                BlockContent::Parts(vec![ContentPart::Text(self.text.clone())]),
                BlockMeta::default(),
            ));
        }
        for call in &self.tool_calls {
            blocks.push(ContextBlock::new(
                ids.next().expect("identity count validated"),
                BlockContent::ToolCall(ToolCallPayload {
                    tool_name: call.tool_name.clone(),
                    arguments: call.arguments.clone(),
                }),
                BlockMeta {
                    provider_call_id: call.provider_call_id.clone(),
                    ..BlockMeta::default()
                },
            ));
        }
        Ok(blocks)
    }
}
