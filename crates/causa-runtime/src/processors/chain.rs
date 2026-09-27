//! Ordered pre- and post-processing for fixed-membership tool batches.
//!
//! The chain borrows one batch for each processor and validates the batch
//! after every handoff. Processors own policy such as approval, rewriting,
//! deduplication, rejection, ordering, and output retention.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use causa_kernel::{
    BatchError, ProcessorContext, ProcessorError, ToolBatch, ToolBatchProcessor, ToolResultStatus,
};

/// Why a configured tool processor chain stopped.
#[derive(Debug, thiserror::Error)]
pub enum ToolProcessingError {
    /// A user processor could not complete its work.
    #[error(transparent)]
    Processor(#[from] ProcessorError),
    /// A processor violated the batch's fixed membership or immutable result
    /// boundary.
    #[error("tool batch processor violated its handoff contract: {0}")]
    Handoff(String),
}

/// Ordered processors that run before and after the executor's fixed batch
/// dispatch stage.
#[derive(Clone, Default)]
pub struct ToolProcessingChain {
    before: Vec<Arc<dyn ToolBatchProcessor>>,
    after: Vec<Arc<dyn ToolBatchProcessor>>,
}

impl ToolProcessingChain {
    /// Start building an ordered pre-/post-processing chain.
    pub fn builder() -> ToolProcessingBuilder {
        ToolProcessingBuilder::default()
    }

    /// Run the processors before fixed executor dispatch.
    pub(crate) async fn process_before(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ToolProcessingError> {
        self.run(&self.before, batch, ctx, Boundary::Before).await
    }

    /// Run the processors after every call has a result.
    pub(crate) async fn process_after(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ToolProcessingError> {
        if batch.completed_len() != batch.declaration_ids().len() {
            return Err(ToolProcessingError::Handoff(
                "post-processing requires every call to have a result".into(),
            ));
        }
        self.run(&self.after, batch, ctx, Boundary::After).await
    }

    async fn run(
        &self,
        processors: &[Arc<dyn ToolBatchProcessor>],
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
        boundary: Boundary,
    ) -> Result<(), ToolProcessingError> {
        validate(batch)?;
        let mut previous = Snapshot::capture(batch)?;
        for processor in processors {
            ctx.control.check().map_err(|error| {
                ToolProcessingError::Handoff(format!("processing control stopped: {error}"))
            })?;
            processor.process(batch, ctx).await?;
            ctx.control.check().map_err(|error| {
                ToolProcessingError::Handoff(format!("processing control stopped: {error}"))
            })?;
            validate(batch)?;
            let current = Snapshot::capture(batch)?;
            previous.check_transition(&current, boundary)?;
            previous = current;
        }
        Ok(())
    }

    /// Number of pre-processors.
    pub fn before_len(&self) -> usize {
        self.before.len()
    }

    /// Number of post-processors.
    pub fn after_len(&self) -> usize {
        self.after.len()
    }
}

impl std::fmt::Debug for ToolProcessingChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolProcessingChain")
            .field("before", &self.before.len())
            .field("after", &self.after.len())
            .finish()
    }
}

/// Builder for the fixed pre → executor → post processor chain.
#[derive(Default)]
pub struct ToolProcessingBuilder {
    before: Vec<Arc<dyn ToolBatchProcessor>>,
    after: Vec<Arc<dyn ToolBatchProcessor>>,
}

impl ToolProcessingBuilder {
    /// Append one processor before executor dispatch.
    pub fn before(mut self, processor: Arc<dyn ToolBatchProcessor>) -> Self {
        self.before.push(processor);
        self
    }

    /// Append one processor after executor dispatch.
    pub fn after(mut self, processor: Arc<dyn ToolBatchProcessor>) -> Self {
        self.after.push(processor);
        self
    }

    /// Finish assembling the chain. Processor order is insertion order.
    pub fn build(self) -> ToolProcessingChain {
        ToolProcessingChain {
            before: self.before,
            after: self.after,
        }
    }
}

#[derive(Clone, Copy)]
enum Boundary {
    Before,
    After,
}

struct EntrySnapshot {
    completed: Option<(causa_kernel::BlockId, ToolResultStatus)>,
    input: Option<causa_kernel::ToolCallPayload>,
    notes: Vec<causa_kernel::TextPayload>,
}

struct Snapshot {
    entries: HashMap<causa_kernel::BlockId, EntrySnapshot>,
    completed_len: usize,
}

impl Snapshot {
    fn capture(batch: &ToolBatch) -> Result<Self, ToolProcessingError> {
        let mut entries = HashMap::new();
        for entry in batch.results().iter().chain(batch.calls()) {
            let call_id = entry.call().call_block_id;
            let completed = entry
                .result()
                .map(|(id, result)| (*id, result.status.clone()));
            let input = Some(entry.call().input.clone());
            let notes = entry
                .result()
                .map(|(_, result)| result.notes.clone())
                .unwrap_or_else(|| entry.call().result_notes.clone());
            if entries
                .insert(
                    call_id,
                    EntrySnapshot {
                        completed,
                        input,
                        notes,
                    },
                )
                .is_some()
            {
                return Err(ToolProcessingError::Handoff(format!(
                    "duplicate declaration identity {call_id:?}"
                )));
            }
        }
        Ok(Self {
            entries,
            completed_len: batch.completed_len(),
        })
    }

    fn check_transition(&self, next: &Self, boundary: Boundary) -> Result<(), ToolProcessingError> {
        if self.entries.len() != next.entries.len()
            || self.entries.keys().any(|id| !next.entries.contains_key(id))
        {
            return Err(ToolProcessingError::Handoff(
                "processor changed the batch's fixed declaration membership".into(),
            ));
        }
        if next.completed_len < self.completed_len {
            return Err(ToolProcessingError::Handoff(
                "processor reopened a completed call".into(),
            ));
        }
        for (id, before) in &self.entries {
            let after = &next.entries[id];
            match (&before.completed, &after.completed) {
                (Some(old), Some(new)) if old == new => {}
                (Some(_), _) => {
                    return Err(ToolProcessingError::Handoff(format!(
                        "processor changed the result identity or status for {id:?}"
                    )));
                }
                (None, Some(_)) if matches!(boundary, Boundary::Before) => {}
                (None, Some(_)) => {
                    return Err(ToolProcessingError::Handoff(
                        "post-processor completed a call after dispatch".into(),
                    ));
                }
                (None, None) => {}
            }
            if (before.completed.is_some() || matches!(boundary, Boundary::After))
                && before.input != after.input
            {
                return Err(ToolProcessingError::Handoff(format!(
                    "post-processor changed executed input for {id:?}"
                )));
            }
            if !after.notes.starts_with(&before.notes) {
                return Err(ToolProcessingError::Handoff(format!(
                    "processor removed required notes for {id:?}"
                )));
            }
        }
        Ok(())
    }
}

fn validate(batch: &ToolBatch) -> Result<(), ToolProcessingError> {
    batch
        .validate()
        .map_err(|error: BatchError| ToolProcessingError::Handoff(error.to_string()))?;
    let ids = batch.declaration_ids();
    let unique: HashSet<_> = ids.iter().collect();
    if unique.len() != ids.len() {
        return Err(ToolProcessingError::Handoff(
            "batch contains duplicate declaration identities".into(),
        ));
    }
    Ok(())
}

/// A no-op processor, useful as an explicit pass-through stage in examples.
#[derive(Debug, Default)]
pub struct PassThroughProcessor;

#[async_trait]
impl ToolBatchProcessor for PassThroughProcessor {
    async fn process(
        &self,
        _batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        Ok(())
    }
}
