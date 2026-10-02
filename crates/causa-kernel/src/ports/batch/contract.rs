//! Controlled tool-batch facts and the processor contract.

use crate::context::block::{MediaRef, TextPayload, ToolCallPayload};
use crate::context::ids::BlockId;
use crate::context::ids::{RoundId, TurnId};
use crate::context::tool_data::{ToolOutput, ToolResultPayload};
use crate::ports::control::CallControl;
use crate::ports::tool::ToolCallContext;
use async_trait::async_trait;
use std::collections::HashSet;

/// A declaration plus its optional completed result.
#[derive(Debug)]
pub struct ToolCallEntry {
    call: ToolCallContext,
    result: Option<(BlockId, ToolResultPayload)>,
}

impl ToolCallEntry {
    /// Borrows the declaration and effective execution input.
    pub fn call(&self) -> &crate::ports::tool::ToolCallContext {
        &self.call
    }

    /// Mutably borrows the input while this call is still pending.
    pub fn input_mut(&mut self) -> Option<&mut ToolCallPayload> {
        self.result.is_none().then_some(&mut self.call.input)
    }

    /// Borrows the result block ID and result payload when completed.
    pub fn result(&self) -> Option<(&BlockId, &ToolResultPayload)> {
        self.result.as_ref().map(|(id, result)| (id, result))
    }

    /// Mutably borrows output when this call has a result.
    pub fn output_mut(&mut self) -> Option<&mut ToolOutput> {
        self.result.as_mut().map(|(_, result)| &mut result.output)
    }

    /// Mutably borrows result media when this call has a result.
    pub fn media_mut(&mut self) -> Option<&mut Vec<MediaRef>> {
        self.result.as_mut().map(|(_, result)| &mut result.media)
    }

    /// Adds a note to the pending call or completed result as appropriate.
    pub fn push_note(&mut self, note: TextPayload) {
        match self.result.as_mut() {
            Some((_, result)) => result.notes.push(note),
            None => self.call.result_notes.push(note),
        }
    }
}

/// One tool batch with a completed prefix and pending suffix.
#[derive(Debug)]
pub struct ToolBatch {
    entries: Vec<ToolCallEntry>,
    completed: usize,
}

impl ToolBatch {
    /// Creates a batch from declarations in their original order.
    pub fn new(calls: Vec<ToolCallContext>) -> Result<Self, BatchError> {
        let mut declarations = HashSet::with_capacity(calls.len());
        for call in &calls {
            if !declarations.insert(call.call_block_id) {
                return Err(BatchError::DuplicateDeclaration(call.call_block_id));
            }
        }
        let batch = Self {
            entries: calls
                .into_iter()
                .map(|call| ToolCallEntry { call, result: None })
                .collect(),
            completed: 0,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Returns the pending suffix.
    pub fn calls(&self) -> &[ToolCallEntry] {
        &self.entries[self.completed..]
    }

    /// Mutably returns the pending suffix.
    pub fn calls_mut(&mut self) -> &mut [ToolCallEntry] {
        &mut self.entries[self.completed..]
    }

    /// Returns the completed prefix.
    pub fn results(&self) -> &[ToolCallEntry] {
        &self.entries[..self.completed]
    }

    /// Mutably returns the completed prefix for output editing and ordering.
    pub fn results_mut(&mut self) -> &mut [ToolCallEntry] {
        &mut self.entries[..self.completed]
    }

    /// Returns the number of completed entries.
    pub fn completed_len(&self) -> usize {
        self.completed
    }

    /// Returns declaration identities in the batch's current entry order.
    pub fn declaration_ids(&self) -> Vec<BlockId> {
        self.entries
            .iter()
            .map(|entry| entry.call.call_block_id)
            .collect()
    }

    /// Validates the completed-prefix/pending-suffix and identity invariants.
    pub fn validate(&self) -> Result<(), BatchError> {
        if self.completed > self.entries.len() {
            return Err(BatchError::InvalidPartition(format!(
                "completed count {} exceeds entry count {}",
                self.completed,
                self.entries.len()
            )));
        }
        let mut declarations = HashSet::with_capacity(self.entries.len());
        for entry in &self.entries {
            let declaration_id = entry.call.call_block_id;
            if !declarations.insert(declaration_id) {
                return Err(BatchError::DuplicateDeclaration(declaration_id));
            }
        }
        let mut all_ids = declarations.clone();
        for (index, entry) in self.entries.iter().enumerate() {
            let declaration_id = entry.call.call_block_id;
            match (index < self.completed, &entry.result) {
                (true, Some((result_id, result))) => {
                    if result.call_block_id != declaration_id {
                        return Err(BatchError::MismatchedCall {
                            expected: declaration_id,
                            actual: result.call_block_id,
                        });
                    }
                    if !all_ids.insert(*result_id) {
                        return Err(BatchError::DuplicateResultBlockId(*result_id));
                    }
                }
                (false, None) => {}
                (true, None) => {
                    return Err(BatchError::InvalidPartition(format!(
                        "completed entry {index} has no result"
                    )));
                }
                (false, Some(_)) => {
                    return Err(BatchError::InvalidPartition(format!(
                        "pending entry {index} already has a result"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Installs a result at an absolute entry index and advances the prefix.
    /// Validation failure leaves the batch unchanged.
    pub fn resolve_at(
        &mut self,
        index: usize,
        result_block_id: BlockId,
        mut result: ToolResultPayload,
    ) -> Result<(), BatchError> {
        self.validate()?;
        if index < self.completed || index >= self.entries.len() {
            return Err(BatchError::InvalidIndex {
                index,
                completed: self.completed,
                total: self.entries.len(),
            });
        }
        let expected = self.entries[index].call.call_block_id;
        if result.call_block_id != expected {
            return Err(BatchError::MismatchedCall {
                expected,
                actual: result.call_block_id,
            });
        }
        if self.entries[index].result.is_some() {
            return Err(BatchError::InvalidPartition(format!(
                "pending entry {index} already has a result"
            )));
        }
        if self.entries.iter().any(|entry| {
            entry.call.call_block_id == result_block_id
                || entry
                    .result
                    .as_ref()
                    .is_some_and(|(id, _)| *id == result_block_id)
        }) {
            return Err(BatchError::DuplicateResultBlockId(result_block_id));
        }
        let mut notes = std::mem::take(&mut self.entries[index].call.result_notes);
        notes.append(&mut result.notes);
        result.notes = notes;
        self.entries[index].result = Some((result_block_id, result));
        self.entries.swap(index, self.completed);
        self.completed += 1;
        Ok(())
    }

    /// Consumes a fully completed batch into result blocks in current order.
    pub fn into_results(self) -> Result<Vec<(BlockId, ToolResultPayload)>, (BatchError, Self)> {
        if let Err(error) = self.validate() {
            return Err((error, self));
        }
        if self.completed != self.entries.len() {
            let error = BatchError::Incomplete {
                completed: self.completed,
                total: self.entries.len(),
            };
            return Err((error, self));
        }
        Ok(self
            .entries
            .into_iter()
            .map(|entry| entry.result.expect("validated complete entry"))
            .collect())
    }
}

/// Context made available to tool batch processors.
pub struct ProcessorContext<'a> {
    /// Turn identity for the active tool batch.
    pub turn_id: &'a TurnId,
    /// Model round that declared this batch.
    pub round_id: RoundId,
    /// Original declaration order, represented by stable block IDs.
    pub declaration_order: &'a [BlockId],
    /// Cooperative cancellation and deadline control.
    pub control: &'a CallControl,
}

/// Failure to finish a processing phase. Per-call rejection is a result,
/// and should be recorded on the batch while returning `Ok(())`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProcessorError {
    /// The processor could not complete its phase.
    #[error("processor failed: {0}")]
    Failed(String),
}

/// A pre- or post-processing phase over one exclusively borrowed tool batch.
#[async_trait]
pub trait ToolBatchProcessor: Send + Sync {
    /// Processes the batch in place; errors leave its current materials with
    /// the caller and stop the chain.
    async fn process(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError>;
}

/// Invalid batch construction, completion, or partition state.
#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    /// A declaration appears more than once in a batch.
    #[error("duplicate declaration block id: {0:?}")]
    DuplicateDeclaration(BlockId),
    /// The requested absolute index is not in the pending suffix.
    #[error("index {index} is not pending (completed {completed}, total {total})")]
    InvalidIndex {
        /// Requested entry index.
        index: usize,
        /// Number of entries in the completed prefix.
        completed: usize,
        /// Total entry count.
        total: usize,
    },
    /// A result references a different declaration than its entry.
    #[error("result call ID mismatch: expected {expected:?}, got {actual:?}")]
    MismatchedCall {
        /// Declaration identity expected by the entry.
        expected: BlockId,
        /// Declaration identity carried by the result.
        actual: BlockId,
    },
    /// A result's own block identity collides within the batch.
    #[error("duplicate result block ID: {0:?}")]
    DuplicateResultBlockId(BlockId),
    /// The completed prefix and pending suffix disagree with entry contents.
    #[error("invalid batch partition: {0}")]
    InvalidPartition(String),
    /// Not all entries have results.
    #[error("batch is incomplete: {completed} of {total} entries completed")]
    Incomplete {
        /// Number of completed entries.
        completed: usize,
        /// Total number of entries.
        total: usize,
    },
}
