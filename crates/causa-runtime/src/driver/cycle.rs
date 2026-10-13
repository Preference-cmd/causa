//! Round ordering, observation boundaries and terminal-result selection.

use super::model::{control_cause, invoke, notify, stopped};
use super::{TurnInterruption, TurnResult, TurnRunner, commit, request};
use crate::{RunEvent, ToolCatalogError, ToolProcessingError, TurnRunOptions};
use causa_kernel::{
    CallControl, Context, InvocationId, ModelStopReason, RoundId, ToolBatch, TurnId,
};

impl TurnRunner {
    pub(super) async fn run_loop(
        &self,
        turn_id: &TurnId,
        context: &mut Context,
        options: &TurnRunOptions,
        control: &CallControl,
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
                .bind(invocation_id.clone(), control.clone())
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
            notify(
                options,
                RunEvent::ModelOutput {
                    invocation_id: &invocation_id,
                    output: &output,
                },
            );
            // Only EndTurn and ToolUse are committed and continued. Every
            // other reason returns the complete uncommitted output, and is
            // selected before the stop check so a stop raised by the observer
            // cannot replace it. No wildcard arm is needed.
            match output.stop_reason {
                ModelStopReason::EndTurn | ModelStopReason::ToolUse => {}
                ModelStopReason::MaxTokens => {
                    return interrupted(
                        TurnInterruption::ModelMaxTokens {
                            invocation_id,
                            output,
                        },
                        None,
                    );
                }
                ModelStopReason::Refusal => {
                    return interrupted(
                        TurnInterruption::ModelRefusal {
                            invocation_id,
                            output,
                        },
                        None,
                    );
                }
                ModelStopReason::ContextWindowExceeded => {
                    return interrupted(
                        TurnInterruption::ModelContextWindowExceeded {
                            invocation_id,
                            output,
                        },
                        None,
                    );
                }
                ModelStopReason::Unrecognized => {
                    return interrupted(
                        TurnInterruption::ModelStopUnrecognized {
                            invocation_id,
                            output,
                        },
                        None,
                    );
                }
            }
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
            notify(
                options,
                RunEvent::BlocksCommitted {
                    invocation_id: &invocation_id,
                    blocks: &context.blocks()[start..],
                },
            );
            if output.stop_reason == ModelStopReason::EndTurn {
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
                return interrupted(
                    TurnInterruption::MaxToolCalls {
                        invocation_id,
                        limit: options.limits.max_tool_calls,
                        declared_before,
                        declared_this_round,
                    },
                    Some(batch),
                );
            }
            declared_before = total.expect("checked tool declaration total");
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
            // A fully processed batch is committed before any stop is
            // honored: the tools ran and the host's processors ran, so the
            // results are facts and there is nothing left for the host to
            // decide about them.
            let commit::ToolCommit { start, unknown } =
                match commit::tools(context, &invocation_id, &batch) {
                    Ok(committed) => committed,
                    Err(cause) => return interrupted(cause, Some(batch)),
                };
            notify(
                options,
                RunEvent::BlocksCommitted {
                    invocation_id: &invocation_id,
                    blocks: &context.blocks()[start..],
                },
            );
            // The successful commit consumes the uncommitted tool batch.
            // Unknown is a fixed terminal cause and is selected before a stop.
            if let Some(call_block_id) = unknown {
                return interrupted(
                    TurnInterruption::UnknownToolOutcome {
                        invocation_id,
                        call_block_id,
                    },
                    None,
                );
            }
            if let Some(cause) = stopped(control, Some(&invocation_id)) {
                return interrupted(cause, None);
            }
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
