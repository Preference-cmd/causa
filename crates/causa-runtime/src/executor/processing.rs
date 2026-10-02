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
    /// Ordinary failed, rejected, or unknown results do not make this method
    /// fail. It does not commit material or cancel the caller's parent token.
    // ToolProcessingError preserves the rejected payload without boxing.
    #[allow(clippy::result_large_err)]
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
        let context = ProcessorContext {
            turn_id: &self.invocation_id.turn_id,
            round_id: self.invocation_id.round_id,
            declaration_order: &declaration_order,
            control: &self.control,
        };
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
        self.control.check().map_err(ToolProcessingError::Control)?;
        process_phase(
            &self.executor.options.after,
            batch,
            &context,
            ToolProcessorPhase::After,
        )
        .await?;
        self.control.check().map_err(ToolProcessingError::Control)?;
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
    #[allow(clippy::result_large_err)]
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
                ToolResultPayload {
                    call_block_id,
                    status: ToolResultStatus::Rejected,
                    output: ToolOutput::new(
                        serde_json::json!({"error": format!("unknown tool: {name}")}),
                    ),
                    media: Vec::new(),
                    notes: Vec::new(),
                },
            )?;
        }
        Ok(())
    }
}
