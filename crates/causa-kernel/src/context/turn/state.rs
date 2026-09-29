//! TurnContext / ContextFrame — the turn fact machine.

use crate::context::block::{
    BlockContent, BlockMeta, ContentPart, ContextBlock, TextPayload, ToolCallPayload,
};
use crate::context::ids::{BlockId, ConversationId, FrameScope, InvocationId, RoundId, TurnId};
use crate::context::model::{ModelResponse, ModelStopReason};
use crate::context::tool_data::ToolResultPayload;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

mod edit;
mod serde_impl;

pub use edit::{ContextEdit, EditError, EditFailure, Replacement};

/// Whether a turn accepts appends or is terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnLifecycle {
    /// The turn accepts appends through its controlled doors.
    Open,
    /// The turn is terminal and rejects appends.
    Sealed,
}

/// The ordered fact blocks a model round saw.
#[derive(Debug, Clone)]
pub struct ModelContext {
    /// The ordered fact blocks that make up the model context.
    pub blocks: Vec<ContextBlock>,
}

/// One projection of fact state handed to or received from a model round.
#[derive(Debug, Clone)]
pub struct ContextFrame {
    /// What this frame projects.
    pub scope: FrameScope,
    /// The model round this frame was built for.
    pub round_id: RoundId,
    /// The ordered blocks presented to the model.
    pub model_context: ModelContext,
}

/// Rejections for lifecycle, identity, pairing, or structural violations.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    /// A context edit violates lifecycle or block identity rules.
    #[error(transparent)]
    Edit(#[from] EditError),
    /// A model output was submitted under a different turn id.
    #[error("foreign invocation: expected turn {expected:?}, got {actual:?}")]
    ForeignInvocation {
        /// The turn id the context was built with.
        expected: TurnId,
        /// The turn id carried by the invocation.
        actual: TurnId,
    },
    /// A model response is structurally invalid.
    #[error("invalid model output: {0}")]
    InvalidModelOutput(String),
    /// Block identity or pairing state is invalid.
    #[error("invalid context: {0}")]
    InvalidContext(String),
    /// A result references no declared tool call.
    #[error("unpaired tool result for declaration {0:?}")]
    UnpairedToolResult(BlockId),
}

/// Receipt of the model door: committed block IDs and declaration payloads.
#[derive(Debug, Clone)]
pub struct AppliedModelOutput {
    /// IDs committed for this output, in commit order.
    pub block_ids: Vec<BlockId>,
    /// Tool declarations paired with IDs in model draft order.
    pub tool_calls: Vec<(BlockId, ToolCallPayload)>,
}

/// Returns how many block IDs an accepted response requires.
pub fn model_output_block_count(response: &ModelResponse) -> usize {
    usize::from(!response.text.0.trim().is_empty()) + response.tool_calls.len()
}

/// Current turn facts. Mutations go through explicit-ID append doors or atomic edits.
#[derive(Clone, Serialize)]
pub struct TurnContext {
    turn_id: TurnId,
    blocks: Vec<ContextBlock>,
    lifecycle: TurnLifecycle,
}

impl std::fmt::Debug for TurnContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnContext")
            .field("turn_id", &self.turn_id)
            .field("lifecycle", &self.lifecycle)
            .field("blocks_len", &self.blocks.len())
            .finish()
    }
}

impl TurnContext {
    /// Creates an empty, open turn.
    pub fn new(turn_id: TurnId) -> Self {
        Self {
            turn_id,
            blocks: Vec::new(),
            lifecycle: TurnLifecycle::Open,
        }
    }

    /// Returns whether the turn has been sealed.
    pub fn is_sealed(&self) -> bool {
        matches!(self.lifecycle, TurnLifecycle::Sealed)
    }
    /// Borrows the current facts in commit order.
    pub fn blocks(&self) -> &[ContextBlock] {
        &self.blocks
    }
    /// Returns the turn's ID.
    pub fn turn_id(&self) -> TurnId {
        self.turn_id.clone()
    }
    /// Returns the lifecycle state.
    pub fn lifecycle(&self) -> TurnLifecycle {
        self.lifecycle
    }
    /// Appends one text fact with an ID prepared by the caller.
    pub fn append_input(
        &mut self,
        block_id: BlockId,
        text: TextPayload,
        source: impl Into<String>,
    ) -> Result<BlockId, ContextError> {
        self.append_parts(block_id, vec![ContentPart::Text(text)], source)
    }

    /// Appends one logical message containing ordered text and media parts.
    pub fn append_parts(
        &mut self,
        block_id: BlockId,
        parts: Vec<ContentPart>,
        source: impl Into<String>,
    ) -> Result<BlockId, ContextError> {
        self.ensure_open()?;
        if parts.is_empty() {
            return Err(ContextError::InvalidContext(
                "append_parts: empty parts are not a fact".into(),
            ));
        }
        let meta = BlockMeta {
            source: Some(source.into()),
            ..BlockMeta::default()
        };
        let block = ContextBlock::new(block_id, BlockContent::Parts(parts), meta);
        self.apply(Vec::new(), vec![block])
            .map_err(|failure| ContextError::Edit(failure.reason))?;
        Ok(block_id)
    }

    /// Records model output as facts, binding IDs in text-then-call order.
    pub fn append_model_output(
        &mut self,
        invocation: InvocationId,
        response: &ModelResponse,
        stop_reason: ModelStopReason,
        block_ids: Vec<BlockId>,
    ) -> Result<AppliedModelOutput, ContextError> {
        self.ensure_open()?;
        if invocation.turn_id != self.turn_id {
            return Err(ContextError::ForeignInvocation {
                expected: self.turn_id.clone(),
                actual: invocation.turn_id.clone(),
            });
        }
        match stop_reason {
            ModelStopReason::EndTurn if !response.tool_calls.is_empty() => {
                return Err(ContextError::InvalidModelOutput(
                    "EndTurn must have empty tool_calls".into(),
                ));
            }
            ModelStopReason::ToolUse if response.tool_calls.is_empty() => {
                return Err(ContextError::InvalidModelOutput(
                    "ToolUse must have non-empty tool_calls".into(),
                ));
            }
            _ => {}
        }
        let expected = model_output_block_count(response);
        if block_ids.len() != expected {
            return Err(ContextError::InvalidContext(format!(
                "model output requires {expected} block ids, received {}",
                block_ids.len()
            )));
        }
        for draft in &response.tool_calls {
            if draft.tool_name.trim().is_empty() {
                return Err(ContextError::InvalidModelOutput("tool_name empty".into()));
            }
            if !draft.arguments.is_object() {
                return Err(ContextError::InvalidModelOutput(
                    "arguments must be object".into(),
                ));
            }
        }

        let mut supplied_ids = block_ids.into_iter();
        let mut prepared = Vec::with_capacity(expected);
        let mut tool_calls = Vec::with_capacity(response.tool_calls.len());
        if !response.text.0.trim().is_empty() {
            prepared.push(ContextBlock::new(
                supplied_ids.next().expect("ID count validated"),
                BlockContent::Parts(vec![ContentPart::Text(TextPayload(
                    response.text.0.clone(),
                ))]),
                BlockMeta::default(),
            ));
        }
        for draft in &response.tool_calls {
            let block_id = supplied_ids.next().expect("ID count validated");
            let payload = ToolCallPayload {
                tool_name: draft.tool_name.clone(),
                arguments: draft.arguments.clone(),
            };
            tool_calls.push((block_id, payload.clone()));
            prepared.push(ContextBlock::new(
                block_id,
                BlockContent::ToolCall(payload),
                BlockMeta {
                    provider_call_id: draft.provider_call_id.clone(),
                    ..BlockMeta::default()
                },
            ));
        }
        let committed_ids = prepared.iter().map(ContextBlock::id).collect();
        self.apply(Vec::new(), prepared)
            .map_err(|failure| ContextError::Edit(failure.reason))?;
        Ok(AppliedModelOutput {
            block_ids: committed_ids,
            tool_calls,
        })
    }

    /// Appends results atomically in the order of the supplied `(ID, result)`
    /// tuples. Each result retains its own identity and declaration reference.
    pub fn append_tool_results(
        &mut self,
        results: Vec<(BlockId, ToolResultPayload)>,
    ) -> Result<Vec<BlockId>, ContextError> {
        self.ensure_open()?;
        if results.is_empty() {
            return Ok(Vec::new());
        }
        let declarations = self
            .blocks
            .iter()
            .filter_map(|block| {
                matches!(block.content(), BlockContent::ToolCall(_)).then_some(block.id())
            })
            .collect::<HashSet<_>>();
        let already_paired = self
            .blocks
            .iter()
            .filter_map(|block| match block.content() {
                BlockContent::ToolResult(result) => Some(result.call_block_id),
                _ => None,
            })
            .collect::<HashSet<_>>();
        let mut batch = HashSet::with_capacity(results.len());
        for (_, result) in &results {
            let call_id = result.call_block_id;
            if !declarations.contains(&call_id) {
                return Err(ContextError::UnpairedToolResult(call_id));
            }
            if already_paired.contains(&call_id) || !batch.insert(call_id) {
                return Err(ContextError::InvalidContext(format!(
                    "tool declaration already has a result: {call_id:?}"
                )));
            }
        }
        let result_ids = results.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let prepared = results
            .into_iter()
            .map(|(id, result)| {
                ContextBlock::new(id, BlockContent::ToolResult(result), BlockMeta::default())
            })
            .collect();
        self.apply(Vec::new(), prepared)
            .map_err(|failure| ContextError::Edit(failure.reason))?;
        Ok(result_ids)
    }

    /// Returns the lossless projection of current facts for this round.
    pub fn frame(&self, round_id: RoundId) -> ContextFrame {
        self.frame_with(round_id, self.blocks.clone())
    }

    /// Projects an explicit block list with this turn's provenance.
    pub(crate) fn frame_with(&self, round_id: RoundId, blocks: Vec<ContextBlock>) -> ContextFrame {
        ContextFrame {
            scope: FrameScope::Turn {
                turn_id: self.turn_id.clone(),
            },
            round_id,
            model_context: ModelContext { blocks },
        }
    }

    /// Seals the turn; sealed turns reject every append or edit operation.
    pub fn seal(&mut self) {
        self.lifecycle = TurnLifecycle::Sealed;
    }

    fn ensure_open(&self) -> Result<(), ContextError> {
        if self.is_sealed() {
            Err(EditError::SealedTurn.into())
        } else {
            Ok(())
        }
    }

    /// Rebuilds an open turn from a validated block log.
    pub fn from_validated_blocks(
        turn_id: TurnId,
        blocks: Vec<ContextBlock>,
    ) -> Result<Self, ContextError> {
        Self::validate_blocks(&blocks)?;
        Ok(Self {
            turn_id,
            blocks,
            lifecycle: TurnLifecycle::Open,
        })
    }

    /// Validates uniqueness of block IDs within this turn.
    pub fn validate_blocks(blocks: &[ContextBlock]) -> Result<(), ContextError> {
        let mut block_ids = HashSet::with_capacity(blocks.len());
        for block in blocks {
            if !block_ids.insert(block.id()) {
                return Err(EditError::DuplicateBlockId(block.id()).into());
            }
        }
        Ok(())
    }
}

/// Creates a lossless merged frame from history and the active turn.
pub fn merged_frame<'a>(
    conversation_id: &ConversationId,
    history: impl IntoIterator<Item = &'a TurnContext>,
    active: &TurnContext,
    round_id: RoundId,
) -> ContextFrame {
    let mut blocks = Vec::new();
    for turn in history {
        blocks.extend(turn.blocks().iter().cloned());
    }
    blocks.extend(active.blocks().iter().cloned());
    ContextFrame {
        scope: FrameScope::Conversation {
            conversation_id: conversation_id.clone(),
            active_turn_id: active.turn_id(),
        },
        round_id,
        model_context: ModelContext { blocks },
    }
}
