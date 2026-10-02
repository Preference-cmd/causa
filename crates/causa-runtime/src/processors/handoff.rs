use crate::executor::dispatch::wait_for_stop;
use crate::executor::{ToolProcessingError, ToolProcessorPhase};
use causa_kernel::{
    BlockId, ProcessorContext, ToolBatch, ToolBatchProcessor, ToolCallPayload, ToolResultStatus,
};
use std::collections::HashMap;
use std::sync::Arc;

// ToolProcessingError preserves the rejected payload without boxing.
#[allow(clippy::result_large_err)]
pub(crate) async fn process_phase(
    processors: &[Arc<dyn ToolBatchProcessor>],
    batch: &mut ToolBatch,
    context: &ProcessorContext<'_>,
    phase: ToolProcessorPhase,
) -> Result<(), ToolProcessingError> {
    batch
        .validate()
        .map_err(ToolProcessingError::InvalidBatch)?;
    let mut previous = Snapshot::capture(batch);
    for (index, processor) in processors.iter().enumerate() {
        context
            .control
            .check()
            .map_err(ToolProcessingError::Control)?;
        let result = tokio::select! {
            biased;
            cause = wait_for_stop(context.control) => return Err(ToolProcessingError::Control(cause)),
            result = processor.process(batch, context) => result,
        };
        result.map_err(|error| ToolProcessingError::Processor {
            phase,
            index,
            error,
        })?;
        batch
            .validate()
            .map_err(|error| ToolProcessingError::InvalidProcessorBatch {
                phase,
                index,
                error,
            })?;
        let current = Snapshot::capture(batch);
        previous
            .check_transition(&current, phase)
            .map_err(|message| ToolProcessingError::Handoff {
                phase,
                index,
                message,
            })?;
        context
            .control
            .check()
            .map_err(ToolProcessingError::Control)?;
        previous = current;
    }
    Ok(())
}

struct EntrySnapshot {
    completed: Option<(BlockId, ToolResultStatus)>,
    input: ToolCallPayload,
    notes: Vec<causa_kernel::TextPayload>,
}

struct Snapshot {
    entries: HashMap<BlockId, EntrySnapshot>,
    completed_len: usize,
}

impl Snapshot {
    fn capture(batch: &ToolBatch) -> Self {
        let entries = batch
            .results()
            .iter()
            .chain(batch.calls())
            .map(|entry| {
                let completed = entry
                    .result()
                    .map(|(id, result)| (*id, result.status.clone()));
                let notes = entry
                    .result()
                    .map(|(_, result)| result.notes.clone())
                    .unwrap_or_else(|| entry.call().result_notes.clone());
                (
                    entry.call().call_block_id,
                    EntrySnapshot {
                        completed,
                        input: entry.call().input.clone(),
                        notes,
                    },
                )
            })
            .collect();
        Self {
            entries,
            completed_len: batch.completed_len(),
        }
    }

    fn check_transition(&self, next: &Self, phase: ToolProcessorPhase) -> Result<(), String> {
        if self.entries.len() != next.entries.len()
            || self.entries.keys().any(|id| !next.entries.contains_key(id))
        {
            return Err("processor changed the batch's fixed declaration membership".into());
        }
        if next.completed_len < self.completed_len {
            return Err("processor reopened a completed call".into());
        }
        for (id, before) in &self.entries {
            let after = &next.entries[id];
            match (&before.completed, &after.completed) {
                (Some(old), Some(new)) if old == new => {}
                (Some(_), _) => {
                    return Err(format!(
                        "processor changed result identity or status for {id:?}"
                    ));
                }
                (None, Some(_)) if phase == ToolProcessorPhase::Before => {}
                (None, Some(_)) => {
                    return Err("post-processor completed a call after dispatch".into());
                }
                (None, None) => {}
            }
            if (before.completed.is_some() || phase == ToolProcessorPhase::After)
                && before.input != after.input
            {
                return Err(format!("processor changed completed input for {id:?}"));
            }
            if !after.notes.starts_with(&before.notes) {
                return Err(format!("processor removed required notes for {id:?}"));
            }
        }
        Ok(())
    }
}
