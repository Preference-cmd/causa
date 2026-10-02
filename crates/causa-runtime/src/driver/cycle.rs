//! Round ordering, observation boundaries and terminal-result selection.

use super::model::{control_cause, invoke, notify, stopped};
use super::{TurnInterruption, TurnResult, TurnRunner, commit, request};
use crate::{RunControl, RunEvent, ToolCatalogError, ToolProcessingError, TurnRunOptions};
use causa_kernel::{Context, InvocationId, ModelStopReason, RoundId, ToolBatch, TurnId};

impl TurnRunner {
    pub(super) async fn run_loop(
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
            let request =
                match request::prepare(context, &invocation_id, bound.surface(), options, control)
                    .await
                {
                    Ok(request) => request,
                    Err(cause) => return interrupted(cause, None),
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
            let commit::ModelCommit {
                output,
                mut batch,
                start,
            } = match commit::model(context, &invocation_id, output) {
                Ok(committed) => committed,
                Err(cause) => return interrupted(cause, None),
            };
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
            let commit::ToolCommit { start, unknown } =
                match commit::tools(context, &invocation_id, &batch) {
                    Ok(committed) => committed,
                    Err(cause) => return interrupted(cause, Some(batch)),
                };
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
