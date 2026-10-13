//! Ordered before/dispatch/after processing and bound-name enforcement.

use super::{ToolProcessingError, ToolProcessorPhase, catalog::BoundTools, collection::resolve};
use crate::processors::process_phase;
use causa_kernel::{
    BatchError, ProcessorContext, ToolBatch, ToolOutput, ToolResultPayload, ToolResultStatus,
};

impl BoundTools<'_> {
    /// Processes a new, entirely pending batch using only this binding's fixed
    /// targets. The caller retains the batch on every return path.
    ///
    /// On controlled interruption, accepted results remain, started calls with
    /// no accepted result become unknown, and unstarted calls remain pending.
    /// A stop first seen after the last after-processor finished is not an
    /// interruption of this call: the batch is complete and `Ok` is returned.
    /// Ordinary failed, rejected, or unknown results do not make this method
    /// fail. It does not commit material or cancel the caller's parent token.
    // ToolProcessingError preserves the rejected payload without boxing.
    pub async fn process(self, batch: &mut ToolBatch) -> Result<(), ToolProcessingError> {
        batch
            .validate()
            .map_err(ToolProcessingError::InvalidBatch)?;
        if batch.completed_len() != 0 {
            return Err(ToolProcessingError::NotFreshBatch {
                completed: batch.completed_len(),
            });
        }
        let declaration_order = batch.declaration_ids();
        self.control.check().map_err(ToolProcessingError::Control)?;
        let context = ProcessorContext::new(
            &self.invocation_id.turn_id,
            self.invocation_id.round_id,
            &declaration_order,
            &self.control,
        );
        self.reject_unbound(batch)?;
        process_phase(
            &self.executor.options.before,
            batch,
            &context,
            ToolProcessorPhase::Before,
        )
        .await?;
        self.control.check().map_err(ToolProcessingError::Control)?;
        self.reject_unbound(batch)?;
        self.dispatch(batch).await?;
        // The after phase checks for a stop before each of its processors.
        // With none configured, a batch whose tools all returned is finished.
        process_phase(
            &self.executor.options.after,
            batch,
            &context,
            ToolProcessorPhase::After,
        )
        .await?;
        // A stop seen from here on does not fail the call: every tool has
        // returned and every processor has run, so the batch is reported as
        // processed and the caller decides what the stop means.
        batch
            .validate()
            .map_err(ToolProcessingError::InvalidBatch)?;
        if batch.completed_len() != declaration_order.len() {
            return Err(ToolProcessingError::InvalidBatch(BatchError::Incomplete {
                completed: batch.completed_len(),
                total: declaration_order.len(),
            }));
        }
        Ok(())
    }

    // Preserve the confirmed error API carrying an unboxed rejected payload.
    fn reject_unbound(&self, batch: &mut ToolBatch) -> Result<(), ToolProcessingError> {
        let rejected: Vec<_> = batch
            .calls()
            .iter()
            .filter(|entry| !self.targets.contains_key(&entry.call().input.tool_name))
            .map(|entry| {
                (
                    entry.call().call_block_id,
                    entry.call().input.tool_name.clone(),
                )
            })
            .collect();
        for (call_block_id, name) in rejected {
            resolve(
                batch,
                call_block_id,
                ToolResultPayload::new(
                    call_block_id,
                    ToolResultStatus::Rejected,
                    ToolOutput::new(serde_json::json!({"error": format!("unknown tool: {name}")})),
                ),
            )?;
        }
        Ok(())
    }
}
