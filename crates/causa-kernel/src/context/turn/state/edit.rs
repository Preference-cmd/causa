//! Atomic, identity-preserving edits to an open turn.

use super::{TurnContext, TurnLifecycle};
use crate::context::block::ContextBlock;
use crate::context::ids::BlockId;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// One replacement expressed in coordinates of the original block sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Replacement {
    /// The original half-open block range to replace.
    pub range: Range<usize>,
    /// Blocks inserted at the replacement range.
    pub with: Vec<ContextBlock>,
}

/// Why an atomic context edit could not be committed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EditError {
    /// The turn is sealed.
    #[error("sealed turn")]
    SealedTurn,
    /// A replacement range is outside the original block sequence.
    #[error("replacement {replacement_index} range {range:?} is invalid for length {len}")]
    InvalidRange {
        /// The replacement's position in the submitted list.
        replacement_index: usize,
        /// The invalid original-coordinate range.
        range: Range<usize>,
        /// The original block count.
        len: usize,
    },
    /// The resulting block sequence would contain the ID more than once.
    #[error("duplicate block id: {0:?}")]
    DuplicateBlockId(BlockId),
    /// An existing ID was reused for different content or metadata.
    #[error("block identity reused with different content or metadata: {0:?}")]
    BlockIdentityMismatch(BlockId),
}

/// An edit failure together with every submitted block of edit material.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("context edit failed: {reason}")]
pub struct EditFailure {
    /// The validation failure.
    #[source]
    pub reason: EditError,
    /// Every submitted replacement, including replacements invalidated by overlap.
    pub replacements: Vec<Replacement>,
    /// Blocks submitted for insertion after all replacements.
    pub appended: Vec<ContextBlock>,
}

/// An exclusive, uncommitted edit builder for a turn.
#[derive(Debug)]
pub struct ContextEdit<'a> {
    context: &'a mut TurnContext,
    replacements: Vec<Replacement>,
    appended: Vec<ContextBlock>,
}

impl<'a> ContextEdit<'a> {
    /// Returns the original block sequence while this edit is being prepared.
    pub fn blocks(&self) -> &[ContextBlock] {
        self.context.blocks()
    }

    /// Adds a replacement expressed against the original block sequence.
    pub fn replace(
        mut self,
        range: Range<usize>,
        with: impl IntoIterator<Item = ContextBlock>,
    ) -> Self {
        self.replacements.push(Replacement {
            range,
            with: with.into_iter().collect(),
        });
        self
    }

    /// Adds blocks to the end, after all replacement insertions.
    pub fn append(mut self, blocks: impl IntoIterator<Item = ContextBlock>) -> Self {
        self.appended.extend(blocks);
        self
    }

    /// Validates and applies the complete edit atomically.
    pub fn commit(self) -> Result<(), EditFailure> {
        self.context.apply(self.replacements, self.appended)
    }
}

impl TurnContext {
    /// Applies replacements and tail appends as one atomic edit.
    ///
    /// Every range is validated against the original block sequence before overlap
    /// resolution. Later overlapping operations invalidate earlier operations in
    /// full, and invalidated operations are not reconsidered. Ranges are half-open:
    /// an insertion at `p` conflicts with `[start, end)` when `start <= p < end`,
    /// while an insertion at `end` may coexist. Two insertions at the same point
    /// conflict. Even an empty replacement with no inserted blocks participates in
    /// overlap resolution.
    ///
    /// A sealed turn rejects every edit, including an empty one; an empty edit on
    /// an open turn succeeds. Effective replacement blocks and appended blocks are
    /// checked against every original block, including originals removed by this
    /// edit: an ID may be reused only when its content and metadata are unchanged.
    /// The final sequence must have unique IDs. General edits permit partial tool
    /// material; execution-specific append methods enforce pairing. On failure,
    /// the context is unchanged and all input material is returned in
    /// [`EditFailure`].
    pub fn apply(
        &mut self,
        mut replacements: Vec<Replacement>,
        appended: Vec<ContextBlock>,
    ) -> Result<(), EditFailure> {
        let effective = match self.validate_edit(&replacements, &appended) {
            Ok(effective) => effective,
            Err(reason) => {
                return Err(EditFailure {
                    reason,
                    replacements,
                    appended,
                });
            }
        };

        if replacements.is_empty() {
            self.blocks.extend(appended);
            return Ok(());
        }

        let removed = effective
            .iter()
            .map(|index| replacements[*index].range.end - replacements[*index].range.start)
            .sum::<usize>();
        let capacity = self.blocks.len() - removed
            + appended.len()
            + effective
                .iter()
                .map(|index| replacements[*index].with.len())
                .sum::<usize>();
        let mut output = Vec::with_capacity(capacity);
        let original = std::mem::take(&mut self.blocks);
        let mut source = original.into_iter();
        let mut source_index = 0;
        for index in effective {
            let replacement = &mut replacements[index];
            while source_index < replacement.range.start {
                output.push(source.next().expect("range validated"));
                source_index += 1;
            }
            while source_index < replacement.range.end {
                source.next().expect("range validated");
                source_index += 1;
            }
            output.extend(std::mem::take(&mut replacement.with));
        }
        output.extend(source);
        output.extend(appended);
        self.blocks = output;
        Ok(())
    }

    /// Starts an exclusive edit builder. Dropping it leaves this turn unchanged.
    pub fn edit(&mut self) -> ContextEdit<'_> {
        ContextEdit {
            context: self,
            replacements: Vec::new(),
            appended: Vec::new(),
        }
    }

    fn validate_edit(
        &self,
        replacements: &[Replacement],
        appended: &[ContextBlock],
    ) -> Result<Vec<usize>, EditError> {
        if self.lifecycle == TurnLifecycle::Sealed {
            return Err(EditError::SealedTurn);
        }

        let len = self.blocks.len();
        for (replacement_index, replacement) in replacements.iter().enumerate() {
            if replacement.range.start > replacement.range.end || replacement.range.end > len {
                return Err(EditError::InvalidRange {
                    replacement_index,
                    range: replacement.range.clone(),
                    len,
                });
            }
        }

        let mut effective: Vec<usize> = Vec::with_capacity(replacements.len());
        for (index, replacement) in replacements.iter().enumerate() {
            effective.retain(|&previous| {
                !(ranges_conflict(&replacement.range, &replacements[previous].range))
            });
            effective.push(index);
        }
        effective.sort_by_key(|index| replacements[*index].range.start);

        let originals = self
            .blocks
            .iter()
            .map(|block| (block.id(), block))
            .collect::<HashMap<_, _>>();
        for &index in &effective {
            for block in &replacements[index].with {
                validate_identity(&originals, block)?;
            }
        }
        for block in appended {
            validate_identity(&originals, block)?;
        }

        let mut ids = HashSet::with_capacity(len + appended.len());
        let mut source_index = 0;
        for &index in &effective {
            let range = &replacements[index].range;
            while source_index < range.start {
                let id = self.blocks[source_index].id();
                if !ids.insert(id) {
                    return Err(EditError::DuplicateBlockId(id));
                }
                source_index += 1;
            }
            for block in &replacements[index].with {
                if !ids.insert(block.id()) {
                    return Err(EditError::DuplicateBlockId(block.id()));
                }
            }
            source_index = range.end;
        }
        while source_index < len {
            let id = self.blocks[source_index].id();
            if !ids.insert(id) {
                return Err(EditError::DuplicateBlockId(id));
            }
            source_index += 1;
        }
        for block in appended {
            if !ids.insert(block.id()) {
                return Err(EditError::DuplicateBlockId(block.id()));
            }
        }
        Ok(effective)
    }
}

fn validate_identity(
    originals: &HashMap<BlockId, &ContextBlock>,
    incoming: &ContextBlock,
) -> Result<(), EditError> {
    if let Some(existing) = originals.get(&incoming.id())
        && *existing != incoming
    {
        return Err(EditError::BlockIdentityMismatch(incoming.id()));
    }
    Ok(())
}

fn ranges_conflict(left: &Range<usize>, right: &Range<usize>) -> bool {
    match (left.is_empty(), right.is_empty()) {
        (true, true) => left.start == right.start,
        (true, false) => right.start <= left.start && left.start < right.end,
        (false, true) => left.start <= right.start && right.start < left.end,
        (false, false) => left.start < right.end && right.start < left.end,
    }
}
