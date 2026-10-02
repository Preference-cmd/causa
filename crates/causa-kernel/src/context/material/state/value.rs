//! Controlled context material and independent request frames.

use crate::context::block::ContextBlock;
use serde::Serialize;

use super::EditFailure;

/// The ordered material selected for one model request.
///
/// This owned value is independent of the context that produced it. Invocation
/// identity belongs to the model request rather than this material selection.
#[derive(Debug, Clone)]
pub struct ContextFrame {
    /// Blocks presented to the model, in request order.
    pub blocks: Vec<ContextBlock>,
}

/// Ordered material with atomic, identity-preserving edits.
///
/// A context has no execution identity or terminal state. Generic edits and
/// imports allow local tool material without imposing execution pairing rules.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Context {
    pub(super) blocks: Vec<ContextBlock>,
}

impl Context {
    /// Creates empty material without allocating an execution identity.
    pub fn new() -> Self {
        Self::default()
    }

    /// Imports blocks, validating local identity uniqueness and returning all
    /// submitted material on failure.
    pub fn from_blocks(blocks: Vec<ContextBlock>) -> Result<Self, EditFailure> {
        let mut context = Self::new();
        context.apply(Vec::new(), blocks)?;
        Ok(context)
    }

    /// Borrows current material in its committed order.
    pub fn blocks(&self) -> &[ContextBlock] {
        &self.blocks
    }

    /// Transfers ownership of the current blocks without copying them.
    pub fn into_blocks(self) -> Vec<ContextBlock> {
        self.blocks
    }

    /// Copies current material into an independent request frame.
    pub fn frame(&self) -> ContextFrame {
        ContextFrame {
            blocks: self.blocks.clone(),
        }
    }
}
