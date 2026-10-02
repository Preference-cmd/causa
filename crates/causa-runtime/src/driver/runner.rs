use super::model::{control_cause, invoke, notify, stopped, wait_for_stop};
use super::{TurnInterruption, TurnOutcome, TurnResult};
use crate::{
    RunControl, RunEvent, ToolCatalogError, ToolExecutor, ToolProcessingError, TurnRunOptions,
    new_block_id,
};
use causa_kernel::{
    BlockContent, BlockMeta, Context, ContextBlock, InvocationId, ModelGateway, ModelRequest,
    ModelStopReason, RoundId, RoundInfo, ToolBatch, ToolCallContext, ToolResultStatus, TurnId,
    validate_tool_result_append,
};
use std::sync::Arc;

/// Drives logical model rounds over caller-owned materials.
///
/// Only fresh declarations from each accepted model output are executed.
/// Preparation and observation are supplied per run; tool policy belongs to
/// the executor. Controlled stops return materials, including the complete
/// current batch when its results have not committed.
pub struct TurnRunner {
    gateway: Arc<dyn ModelGateway>,
    executor: Arc<ToolExecutor>,
}
impl TurnRunner {
    /// Assemble the two execution dependencies.
    pub fn new(gateway: Arc<dyn ModelGateway>, executor: Arc<ToolExecutor>) -> Self {
        Self { gateway, executor }
    }

    /// Run using complete logical gateway calls.
    pub async fn run(
        &self,
        turn_id: TurnId,
        context: Context,
        options: TurnRunOptions,
        control: RunControl,
    ) -> TurnOutcome {
        self.execute(turn_id, context, options, control, false)
            .await
    }
    /// Run using streaming gateway calls, accepting only complete `Done` output.
    pub async fn run_streaming(
        &self,
        turn_id: TurnId,
        context: Context,
        options: TurnRunOptions,
        control: RunControl,
    ) -> TurnOutcome {
        self.execute(turn_id, context, options, control, true).await
    }
    #[tracing::instrument(name = "agent.turn", skip_all, fields(turn_id = %turn_id.0, model = %options.model.0))]
    async fn execute(
        &self,
        turn_id: TurnId,
        mut context: Context,
        options: TurnRunOptions,
        control: RunControl,
        streaming: bool,
    ) -> TurnOutcome {
        let (result, uncommitted_tool_batch) = self
            .run_loop(&turn_id, &mut context, &options, &control, streaming)
            .await;
        TurnOutcome {
            turn_id,
            context,
            result,
            uncommitted_tool_batch,
        }
    }
    async fn run_loop(
        &self,
        turn_id: &TurnId,
        context: &mut Context,
        options: &TurnRunOptions,
        control: &RunControl,
        streaming: bool,
    ) -> (TurnResult, Option<ToolBatch>) {
        let mut round = 0;
        let mut declared_before = 0u32;
        loop {
            if let Some(cause) = stopped(control, None) {
                return interrupted(cause, None);
            }
            if round >= options.limits.max_model_rounds {
                return interrupted(
                    TurnInterruption::MaxModelRounds {
                        limit: options.limits.max_model_rounds,
                    },
                    None,
                );
            }
            let invocation_id = InvocationId {
                turn_id: turn_id.clone(),
                round_id: RoundId(round),
            };
            let bound = match self
                .executor
                .bind(invocation_id.clone(), control.call_control())
                .await
            {
                Ok(bound) => bound,
                Err(ToolCatalogError::Control(error)) => {
                    return interrupted(control_cause(error, Some(invocation_id)), None);
                }
                Err(error) => {
                    return interrupted(
                        TurnInterruption::ToolCatalogFailed {
                            invocation_id,
                            error,
                        },
                        None,
                    );
                }
            };
            if let Some(cause) = stopped(control, Some(&invocation_id)) {
                return interrupted(cause, None);
            }
            let frame = if let Some(preparer) = &options.preparer {
                let round_info = RoundInfo {
                    invocation_id: &invocation_id,
                    model: &options.model,
                    tool_surface: bound.surface(),
                    generation: &options.generation,
                };
                let call_control = control.call_control();
                let prepared = tokio::select! {
                    biased;
                    error = wait_for_stop(control) => return interrupted(control_cause(error, Some(invocation_id.clone())), None),
                    result = preparer.prepare(context, &round_info, &call_control) => result,
                };
                if let Some(cause) = stopped(control, Some(&invocation_id)) {
                    return interrupted(cause, None);
                }
                match prepared {
                    Ok(frame) => frame,
                    Err(error) => {
                        return interrupted(
                            TurnInterruption::PrepareFailed {
                                invocation_id,
                                error,
                            },
                            None,
                        );
                    }
                }
            } else {
                context.frame()
            };
            let request = ModelRequest {
                invocation_id: invocation_id.clone(),
                frame,
                model: options.model.clone(),
                tool_surface: bound.surface().clone(),
                generation: options.generation.clone(),
                cache: options.cache,
            };
            notify(options, RunEvent::ModelRequestReady { request: &request });
            if let Some(cause) = stopped(control, Some(&invocation_id)) {
                return interrupted(cause, None);
            }
            let output =
                match invoke(self.gateway.as_ref(), &request, options, control, streaming).await {
                    Ok(output) => output,
                    Err(cause) => return interrupted(cause, None),
                };
            // These terminal reasons are selected before invoking the observer.
            match output.stop_reason {
                ModelStopReason::MaxTokens | ModelStopReason::Refusal => {
                    notify(
                        options,
                        RunEvent::ModelOutput {
                            invocation_id: &invocation_id,
                            output: &output,
                        },
                    );
                    let cause = if output.stop_reason == ModelStopReason::MaxTokens {
                        TurnInterruption::ModelMaxTokens {
                            invocation_id,
                            output,
                        }
                    } else {
                        TurnInterruption::ModelRefusal {
                            invocation_id,
                            output,
                        }
                    };
                    return interrupted(cause, None);
                }
                _ => {}
            }
            notify(
                options,
                RunEvent::ModelOutput {
                    invocation_id: &invocation_id,
                    output: &output,
                },
            );
            if let Some(cause) = stopped(control, Some(&invocation_id)) {
                return interrupted(cause, None);
            }
            let block_ids = (0..output.response.block_count())
                .map(|_| new_block_id())
                .collect::<Vec<_>>();
            let blocks = match output.response.to_blocks(output.stop_reason, &block_ids) {
                Ok(blocks) => blocks,
                Err(error) => {
                    return interrupted(
                        TurnInterruption::ModelConversionFailed {
                            invocation_id,
                            error,
                            output,
                            block_ids,
                        },
                        None,
                    );
                }
            };
            let calls = blocks
                .iter()
                .filter_map(|block| match block.content() {
                    BlockContent::ToolCall(payload) => {
                        Some(ToolCallContext::from_declaration(block.id(), payload))
                    }
                    _ => None,
                })
                .collect();
            let mut batch = match ToolBatch::new(calls) {
                Ok(batch) => batch,
                Err(error) => {
                    return interrupted(
                        TurnInterruption::ModelBatchFailed {
                            invocation_id,
                            error,
                            output,
                            blocks,
                        },
                        None,
                    );
                }
            };
            let start = context.blocks().len();
            if let Err(error) = context.apply(vec![], blocks) {
                return interrupted(
                    TurnInterruption::ModelCommitFailed {
                        invocation_id,
                        error,
                        output,
                    },
                    None,
                );
            }
            if output.stop_reason == ModelStopReason::EndTurn {
                notify(
                    options,
                    RunEvent::BlocksCommitted {
                        invocation_id: &invocation_id,
                        blocks: &context.blocks()[start..],
                    },
                );
                return (
                    TurnResult::Completed {
                        final_output: output,
                    },
                    None,
                );
            }
            let declared_this_round = batch.calls().len();
            let total = u32::try_from(declared_this_round)
                .ok()
                .and_then(|n| declared_before.checked_add(n));
            if total.is_none_or(|total| total > options.limits.max_tool_calls) {
                let cause = TurnInterruption::MaxToolCalls {
                    invocation_id: invocation_id.clone(),
                    limit: options.limits.max_tool_calls,
                    declared_before,
                    declared_this_round,
                };
                notify(
                    options,
                    RunEvent::BlocksCommitted {
                        invocation_id: &invocation_id,
                        blocks: &context.blocks()[start..],
                    },
                );
                return interrupted(cause, Some(batch));
            }
            declared_before = total.expect("checked tool declaration total");
            notify(
                options,
                RunEvent::BlocksCommitted {
                    invocation_id: &invocation_id,
                    blocks: &context.blocks()[start..],
                },
            );
            if let Some(cause) = stopped(control, Some(&invocation_id)) {
                return interrupted(cause, Some(batch));
            }
            notify(
                options,
                RunEvent::ToolBatchReady {
                    invocation_id: &invocation_id,
                    batch: &batch,
                },
            );
            if let Some(cause) = stopped(control, Some(&invocation_id)) {
                return interrupted(cause, Some(batch));
            }
            // The module must finish controlled collection. Never drop this future
            // in an outer cancellation select: it owns the started-call records.
            let processed = bound.process(&mut batch).await;
            notify(
                options,
                RunEvent::ToolBatchReturned {
                    invocation_id: &invocation_id,
                    batch: &batch,
                    result: processed.as_ref().map(|_| ()),
                },
            );
            if let Err(error) = processed {
                let cause = match error {
                    ToolProcessingError::Control(error) => {
                        control_cause(error, Some(invocation_id))
                    }
                    error => TurnInterruption::ToolProcessingFailed {
                        invocation_id,
                        error,
                    },
                };
                return interrupted(cause, Some(batch));
            }
            if let Some(cause) = stopped(control, Some(&invocation_id)) {
                return interrupted(cause, Some(batch));
            }
            let results = batch
                .results()
                .iter()
                .filter_map(|entry| entry.result().map(|(id, payload)| (*id, payload.clone())))
                .collect::<Vec<_>>();
            if let Err(error) = validate_tool_result_append(context.blocks(), &results) {
                return interrupted(
                    TurnInterruption::ToolResultValidationFailed {
                        invocation_id,
                        error,
                    },
                    Some(batch),
                );
            }
            let unknown = results.iter().find_map(|(_, result)| {
                (result.status == ToolResultStatus::UnknownOutcome).then_some(result.call_block_id)
            });
            let blocks = results
                .into_iter()
                .map(|(id, result)| {
                    ContextBlock::new(id, BlockContent::ToolResult(result), BlockMeta::default())
                })
                .collect();
            let start = context.blocks().len();
            if let Err(error) = context.apply(vec![], blocks) {
                return interrupted(
                    TurnInterruption::ToolCommitFailed {
                        invocation_id,
                        error,
                    },
                    Some(batch),
                );
            }
            // The successful commit consumes the logical U even when an observer
            // requests cancellation. Unknown is already a fixed terminal cause.
            if let Some(call_block_id) = unknown {
                let cause = TurnInterruption::UnknownToolOutcome {
                    invocation_id: invocation_id.clone(),
                    call_block_id,
                };
                notify(
                    options,
                    RunEvent::BlocksCommitted {
                        invocation_id: &invocation_id,
                        blocks: &context.blocks()[start..],
                    },
                );
                return interrupted(cause, None);
            }
            notify(
                options,
                RunEvent::BlocksCommitted {
                    invocation_id: &invocation_id,
                    blocks: &context.blocks()[start..],
                },
            );
            round += 1;
        }
    }
}
fn interrupted(
    cause: TurnInterruption,
    batch: Option<ToolBatch>,
) -> (TurnResult, Option<ToolBatch>) {
    (TurnResult::Interrupted { cause }, batch)
}
