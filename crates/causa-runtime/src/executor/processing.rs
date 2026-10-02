use super::catalog::BoundTools;
use super::dispatch::{execute, wait_for_stop};
use super::error::{ToolProcessingError, ToolProcessorPhase};
use crate::new_block_id;
use crate::processors::process_phase;
use causa_kernel::{
    BatchError, BlockId, ProcessorContext, ToolBatch, ToolOutput, ToolResultPayload,
    ToolResultStatus,
};
use futures_util::StreamExt;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

impl BoundTools<'_> {
    /// Processes a new, entirely pending batch using only this binding's fixed
    /// targets. The caller retains the batch on every return path.
    ///
    /// On controlled interruption, accepted results remain, started calls with
    /// no accepted result become unknown, and unstarted calls remain pending.
    /// Ordinary failed, rejected, or unknown results do not make this method
    /// fail. It does not commit material or cancel the caller's parent token.
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

    async fn dispatch(&self, batch: &mut ToolBatch) -> Result<(), ToolProcessingError> {
        let started = Arc::new(Mutex::new(HashSet::new()));
        let calls: Vec<_> = batch
            .calls()
            .iter()
            .map(|entry| entry.call().clone())
            .collect();
        let tasks = calls.into_iter().map(|call| {
            let started = started.clone();
            let control = self.control.clone();
            let timeout = self.executor.options.call_timeout;
            let store = self.executor.options.artifact_store.as_deref();
            let tool = self.targets[&call.input.tool_name].clone();
            async move {
                // One poll can enter several siblings. Recheck before recording
                // a start so synchronous cancellation leaves later calls pending.
                control.check().map_err(ToolProcessingError::Control)?;
                let call_block_id = call.call_block_id;
                started
                    .lock()
                    .expect("started calls lock")
                    .insert(call_block_id);
                let call_control =
                    timeout.map_or(control.clone(), |timeout| control.with_timeout(timeout));
                let result = execute(tool, call, call_control, store).await;
                Ok((call_block_id, result))
            }
        });
        let mut tasks = futures_util::stream::FuturesUnordered::from_iter(tasks);
        let result = loop {
            if tasks.is_empty() {
                break Ok(());
            }
            let next = tokio::select! {
                biased;
                cause = wait_for_stop(&self.control) => break Err(ToolProcessingError::Control(cause)),
                result = tasks.next() => result,
            };
            match next {
                Some(Ok((call_block_id, result))) => {
                    if let Err(error) = resolve(batch, call_block_id, result) {
                        break Err(error);
                    }
                }
                Some(Err(error)) => break Err(error),
                None => break Ok(()),
            }
        };
        // Dropping only our own call futures never signals the caller token.
        drop(tasks);
        if result.is_err() {
            let started = started.lock().expect("started calls lock").clone();
            for call_block_id in started {
                if batch
                    .calls()
                    .iter()
                    .any(|entry| entry.call().call_block_id == call_block_id)
                {
                    // The batch was valid before dispatch and is modified only
                    // by checked completion, so settlement cannot fail.
                    resolve(batch, call_block_id, ToolResultPayload {
                        call_block_id,
                        status: ToolResultStatus::UnknownOutcome,
                        output: ToolOutput::new(serde_json::json!({
                            "error": "tool was started but no acceptable result was observed before the batch stopped"
                        })),
                        media: Vec::new(), notes: Vec::new(),
                    }).expect("valid pending declaration during settlement");
                }
            }
        }
        result
    }
}

// Preserve the actual rejected input in the confirmed public error shape.
#[allow(clippy::result_large_err)]
fn resolve(
    batch: &mut ToolBatch,
    call_block_id: BlockId,
    result: ToolResultPayload,
) -> Result<(), ToolProcessingError> {
    let pending = batch
        .calls()
        .iter()
        .position(|entry| entry.call().call_block_id == call_block_id);
    let Some(pending) = pending else {
        return Err(ToolProcessingError::ResultRejected {
            call_block_id,
            error: BatchError::InvalidPartition(format!(
                "declaration {call_block_id:?} is not pending"
            )),
            result,
        });
    };
    let index = batch.completed_len() + pending;
    loop {
        match batch.resolve_at(index, new_block_id(), result.clone()) {
            Ok(()) => return Ok(()),
            Err(BatchError::DuplicateResultBlockId(_)) => continue,
            Err(error) => {
                return Err(ToolProcessingError::ResultRejected {
                    call_block_id,
                    error,
                    result,
                });
            }
        }
    }
}
