//! Ordinary policy processors supplied as convenient runtime components.

use std::collections::HashSet;

use async_trait::async_trait;
use causa_kernel::{
    ProcessorContext, ProcessorError, TextPayload, ToolBatch, ToolBatchProcessor, ToolCallId,
    ToolOutput, ToolResultPayload, ToolResultStatus,
};

use crate::new_block_id;

/// Reject repeated calls with the same complete content identity.
///
/// The first declaration remains eligible for execution. Later duplicates
/// are completed as rejections while their declaration identities remain fixed.
#[derive(Debug, Default)]
pub struct DeduplicateProcessor;

#[async_trait]
impl ToolBatchProcessor for DeduplicateProcessor {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let duplicates = {
            let mut identities = HashSet::new();
            let mut duplicates = Vec::new();
            for entry in batch.results().iter().chain(batch.calls()) {
                let input = &entry.call().input;
                if !identities.insert(ToolCallId::new(&input.tool_name, &input.arguments)) {
                    duplicates.push(entry.call().call_block_id);
                }
            }
            duplicates
        };
        for call_id in duplicates {
            let Some(offset) = batch
                .calls()
                .iter()
                .position(|entry| entry.call().call_block_id == call_id)
            else {
                continue;
            };
            let index = batch.completed_len() + offset;
            batch
                .resolve_at(
                    index,
                    new_block_id(),
                    rejected_result(call_id, "duplicate tool call"),
                )
                .map_err(|error| ProcessorError::Failed(error.to_string()))?;
        }
        Ok(())
    }
}

/// Reject every still-pending declaration with a caller-supplied reason.
#[derive(Debug, Clone)]
pub struct RejectAllProcessor {
    reason: String,
}

impl RejectAllProcessor {
    /// Reject all pending calls with `reason` in their recorded output.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl ToolBatchProcessor for RejectAllProcessor {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        while !batch.calls().is_empty() {
            let index = batch.completed_len();
            let call_id = batch.calls()[0].call().call_block_id;
            batch
                .resolve_at(
                    index,
                    new_block_id(),
                    rejected_result(call_id, &self.reason),
                )
                .map_err(|error| ProcessorError::Failed(error.to_string()))?;
        }
        Ok(())
    }
}

fn rejected_result(call_block_id: causa_kernel::BlockId, reason: &str) -> ToolResultPayload {
    ToolResultPayload {
        call_block_id,
        status: ToolResultStatus::Rejected,
        output: ToolOutput::new(serde_json::json!({"error": reason})),
        media: Vec::new(),
        notes: vec![TextPayload::new(reason)],
    }
}
