//! The reference driver — retry scheduling, tool batch dispatch, artifact
//! spill, control plumbing, and trace construction behind the staged
//! perimeter. One runner during the transition; the canonical kernel never
//! references this module.
use super::config::TurnRunOptions;
use super::executor::ToolExecutor;
use crate::block::ToolCallPayload;
use crate::control::RunControl;
use crate::gateway::ModelGateway;
use crate::gateway::ModelRequest;
use crate::ids::{AttemptNumber, BlockId, InvocationId, RoundId};
use crate::model::{ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelStopReason};
use crate::tool_data::{
    ArtifactRef, ToolCallId, ToolExecutionOutcome, ToolOutput, ToolResultPayload, ToolResultStatus,
    Truncation, UnknownOutcomePolicy,
};
use crate::turn::{FrameScope, TurnContext};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn millis_since(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct DedupKey {
    tool_name: String,
    arguments: serde_json::Value,
}
impl DedupKey {
    fn new(tool_name: String, arguments: serde_json::Value) -> Self {
        Self {
            tool_name,
            arguments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "detail")]
pub enum TurnInterruption {
    ExplicitCancellation,
    TurnDeadlineExceeded,
    /// 终态模型调用失败的统一承载：可重试耗尽与非可重试（Permanent /
    /// InvalidRequest / UnknownOutcome）都落到这里，由 `last_kind` 区分；
    /// parent 取消导致的 `Cancelled` 映射为 `ExplicitCancellation`。
    RetryExhausted {
        last_kind: ModelInvokeErrorKind,
        last_error: String,
    },
    InvalidModelOutput {
        reason: String,
    },
    MaxModelRounds {
        limit: u32,
    },
    MaxToolCalls {
        limit: u32,
    },
    UnsafeUnknownOutcome {
        call_id: ToolCallId,
    },
    CompactionFailed {
        reason: String,
    },
    RunnerInvariantViolation {
        reason: String,
    },
    ModelMaxTokens,
    ModelRefusal,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttemptTrace {
    pub attempt: AttemptNumber,
    /// `None` = 成功 attempt；`Some(kind)` = 失败 attempt 的归类。
    pub kind: Option<ModelInvokeErrorKind>,
    pub is_retryable: bool,
    pub duration_ms: u64,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputSummary {
    pub stop_reason: crate::model::ModelStopReason,
    pub usage: Option<crate::model::ModelUsage>,
    pub tool_call_count: usize,
    pub assistant_text_bytes: usize,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallTrace {
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub position: usize,
    pub status: ToolResultStatus,
    pub truncation: Truncation,
    pub artifact: Option<ArtifactRef>,
    pub duration_ms: u64,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolBatchTrace {
    pub calls: Vec<ToolCallTrace>,
    /// 实际完成顺序（executor 返回即记录），非提交顺序。
    pub completion_order: Vec<ToolCallId>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelRoundTrace {
    pub round_id: RoundId,
    pub invocation_id: InvocationId,
    /// frame 物化时的 `source_version`（apply 之前的版本）。
    pub frame_version: crate::ids::ContextVersion,
    pub attempts: Vec<AttemptTrace>,
    pub output_summary: Option<OutputSummary>,
    pub applied_block_ids: Vec<BlockId>,
    pub tool_batch: Option<ToolBatchTrace>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurnTrace {
    pub rounds: Vec<ModelRoundTrace>,
    pub tool_calls_total: usize,
    pub total_duration_ms: u64,
}
impl TurnTrace {
    pub fn new() -> Self {
        Self {
            rounds: vec![],
            tool_calls_total: 0,
            total_duration_ms: 0,
        }
    }
}
impl Default for TurnTrace {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum TurnResult {
    Completed { final_output: ModelOutput },
    Interrupted { cause: TurnInterruption },
}

#[derive(Debug)]
pub struct TurnOutcome {
    pub context: TurnContext,
    pub result: TurnResult,
    pub trace: TurnTrace,
}

pub struct TurnRunner {
    gateway: Arc<dyn ModelGateway>,
    executor: Arc<ToolExecutor>,
}
impl TurnRunner {
    pub fn new(gateway: Arc<dyn ModelGateway>, executor: Arc<ToolExecutor>) -> Self {
        Self { gateway, executor }
    }
    pub async fn run(
        &self,
        mut context: TurnContext,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> TurnOutcome {
        let start = Instant::now();
        let mut round: u32 = 0;
        let mut tool_calls_total: usize = 0;
        let mut trace = TurnTrace::new();

        // Any terminal outcome seals the returned context and fills trace totals.
        fn finish(
            mut ctx: TurnContext,
            result: TurnResult,
            mut trace: TurnTrace,
            tool_calls_total: usize,
            start: Instant,
        ) -> TurnOutcome {
            ctx.seal();
            trace.tool_calls_total = tool_calls_total;
            trace.total_duration_ms = millis_since(start);
            TurnOutcome {
                context: ctx,
                result,
                trace,
            }
        }

        loop {
            // boundary checks
            if ctrl.should_stop() {
                let cause = if ctrl.is_cancelled() {
                    TurnInterruption::ExplicitCancellation
                } else {
                    TurnInterruption::TurnDeadlineExceeded
                };
                return finish(
                    context,
                    TurnResult::Interrupted { cause },
                    trace,
                    tool_calls_total,
                    start,
                );
            }
            if round >= options.policy.limits.max_model_rounds {
                return finish(
                    context,
                    TurnResult::Interrupted {
                        cause: TurnInterruption::MaxModelRounds {
                            limit: options.policy.limits.max_model_rounds,
                        },
                    },
                    trace,
                    tool_calls_total,
                    start,
                );
            }
            // materialize frame — canonical trigger evaluation consumes the
            // assembled frame policy (staged ownership, port-typed input)
            let frame = match context.frame(RoundId(round), &options.frame).await {
                Ok(f) => f,
                Err(e) => {
                    return finish(
                        context,
                        TurnResult::Interrupted {
                            cause: TurnInterruption::CompactionFailed {
                                reason: e.to_string(),
                            },
                        },
                        trace,
                        tool_calls_total,
                        start,
                    );
                }
            };
            let frame_version = match &frame.scope {
                FrameScope::Turn { source_version, .. } => *source_version,
            };
            let invocation = InvocationId {
                turn_id: context.turn_id(),
                round_id: RoundId(round),
            };
            // bounded logical retry: same InvocationId / ContextFrame, attempt+1
            let mut attempt: u32 = 1;
            let mut attempts: Vec<AttemptTrace> = Vec::new();
            let output: Result<ModelOutput, ModelInvokeError> = loop {
                let attempt_started = Instant::now();
                let attempt_ctrl = ctrl.for_attempt(options.policy.attempt_timeout);
                let req = ModelRequest {
                    invocation_id: invocation.clone(),
                    attempt: AttemptNumber(attempt),
                    frame: frame.clone(),
                    model: options.invocation.model.clone(),
                    tool_surface: options.invocation.tool_surface.clone(),
                    generation: options.invocation.generation.clone(),
                };
                match self.gateway.invoke(&req, &attempt_ctrl).await {
                    Ok(out) => {
                        attempts.push(AttemptTrace {
                            attempt: AttemptNumber(attempt),
                            kind: None,
                            is_retryable: false,
                            duration_ms: millis_since(attempt_started),
                        });
                        break Ok(out);
                    }
                    Err(e) => {
                        let retryable = options.policy.retry.allows(&e.kind);
                        attempts.push(AttemptTrace {
                            attempt: AttemptNumber(attempt),
                            kind: Some(e.kind.clone()),
                            is_retryable: retryable,
                            duration_ms: millis_since(attempt_started),
                        });
                        if retryable && attempt <= options.policy.retry.max_retries {
                            attempt += 1;
                            continue;
                        }
                        break Err(e);
                    }
                }
            };
            let output = match output {
                Ok(o) => o,
                Err(e) => {
                    let cause = if matches!(e.kind, ModelInvokeErrorKind::Cancelled)
                        && ctrl.is_cancelled()
                    {
                        TurnInterruption::ExplicitCancellation
                    } else {
                        TurnInterruption::RetryExhausted {
                            last_kind: e.kind.clone(),
                            last_error: e.message.clone(),
                        }
                    };
                    trace.rounds.push(ModelRoundTrace {
                        round_id: RoundId(round),
                        invocation_id: invocation,
                        frame_version,
                        attempts,
                        output_summary: None,
                        applied_block_ids: vec![],
                        tool_batch: None,
                    });
                    return finish(
                        context,
                        TurnResult::Interrupted { cause },
                        trace,
                        tool_calls_total,
                        start,
                    );
                }
            };
            let output_summary = Some(OutputSummary {
                stop_reason: output.stop_reason,
                usage: output.usage.clone(),
                tool_call_count: output.assistant.tool_calls.len(),
                assistant_text_bytes: output.assistant.text.0.len(),
            });
            // MaxTokens / Refusal never persist blocks (§5.6); they carry their
            // own dedicated interruption causes, so dispatch before apply.
            // This is driver policy: the canonical apply_model_output would
            // happily record them as facts.
            if matches!(
                output.stop_reason,
                ModelStopReason::MaxTokens | ModelStopReason::Refusal
            ) {
                let cause = if matches!(output.stop_reason, ModelStopReason::MaxTokens) {
                    TurnInterruption::ModelMaxTokens
                } else {
                    TurnInterruption::ModelRefusal
                };
                trace.rounds.push(ModelRoundTrace {
                    round_id: RoundId(round),
                    invocation_id: invocation,
                    frame_version,
                    attempts,
                    output_summary,
                    applied_block_ids: vec![],
                    tool_batch: None,
                });
                return finish(
                    context,
                    TurnResult::Interrupted { cause },
                    trace,
                    tool_calls_total,
                    start,
                );
            }
            let applied = match context.apply_model_output(invocation.clone(), output.clone()) {
                Ok(a) => a,
                Err(e) => {
                    trace.rounds.push(ModelRoundTrace {
                        round_id: RoundId(round),
                        invocation_id: invocation,
                        frame_version,
                        attempts,
                        output_summary,
                        applied_block_ids: vec![],
                        tool_batch: None,
                    });
                    return finish(
                        context,
                        TurnResult::Interrupted {
                            cause: TurnInterruption::InvalidModelOutput {
                                reason: e.to_string(),
                            },
                        },
                        trace,
                        tool_calls_total,
                        start,
                    );
                }
            };
            // ToolCallId is generated exactly once — inside apply_model_output.
            // The runner reuses the persisted tool.call payloads instead of
            // regenerating ids, so positions always match the context.
            let call_payloads: Vec<ToolCallPayload> = applied
                .block_ids
                .iter()
                .filter_map(|bid| {
                    context
                        .blocks()
                        .iter()
                        .find(|b| &b.id == bid)
                        .and_then(|b| match &b.payload {
                            crate::block::BlockPayload::ToolCall(p) => Some(p.clone()),
                            _ => None,
                        })
                })
                .collect();
            trace.rounds.push(ModelRoundTrace {
                round_id: RoundId(round),
                invocation_id: invocation.clone(),
                frame_version,
                attempts,
                output_summary,
                applied_block_ids: applied.block_ids.clone(),
                tool_batch: None,
            });
            match output.stop_reason {
                ModelStopReason::EndTurn => {
                    // apply_model_output guarantees EndTurn has empty tool_calls
                    return finish(
                        context,
                        TurnResult::Completed {
                            final_output: output,
                        },
                        trace,
                        tool_calls_total,
                        start,
                    );
                }
                ModelStopReason::ToolUse => {
                    tool_calls_total += call_payloads.len();
                    if tool_calls_total as u32 > options.policy.limits.max_tool_calls {
                        return finish(
                            context,
                            TurnResult::Interrupted {
                                cause: TurnInterruption::MaxToolCalls {
                                    limit: options.policy.limits.max_tool_calls,
                                },
                            },
                            trace,
                            tool_calls_total,
                            start,
                        );
                    }
                    // same-batch dedup by (tool_name + arguments); original
                    // order and positions preserved
                    let mut seen: HashMap<DedupKey, ()> = HashMap::new();
                    let mut to_exec: Vec<ToolCallPayload> = Vec::new();
                    let mut rejected: Vec<ToolExecutionOutcome> = Vec::new();
                    for payload in &call_payloads {
                        let key =
                            DedupKey::new(payload.tool_name.clone(), payload.arguments.clone());
                        if seen.insert(key, ()).is_some() {
                            rejected.push(ToolExecutionOutcome::new(ToolResultPayload {
                                call_id: payload.call_id.clone(),
                                status: ToolResultStatus::Rejected,
                                output: ToolOutput::new(
                                    serde_json::json!({"error": "duplicate tool call"}),
                                ),
                            }));
                        } else {
                            to_exec.push(payload.clone());
                        }
                    }
                    // parallel dispatch; record real completion order + duration
                    let completion_log: Arc<Mutex<Vec<(ToolCallId, u64)>>> =
                        Arc::new(Mutex::new(Vec::new()));
                    let futs = to_exec.into_iter().map(|payload| {
                        let cc = ctrl
                            .for_attempt(options.policy.attempt_timeout)
                            .for_call(options.execution.call_timeout);
                        let store = options.execution.artifact_store.clone();
                        let tc = options.execution.token_counter.clone();
                        let limits = options.execution.tool_output_limits.clone();
                        let exec = self.executor.clone();
                        let log = completion_log.clone();
                        async move {
                            let t0 = Instant::now();
                            let out = exec
                                .execute_with_limits(payload, cc, store, tc, limits)
                                .await;
                            log.lock()
                                .unwrap()
                                .push((out.result.call_id.clone(), millis_since(t0)));
                            out
                        }
                    });
                    let mut results = futures_util::future::join_all(futs).await;
                    results.extend(rejected);
                    results.sort_by_key(|r| r.result.call_id.position());
                    let (completion_order, call_durations): (
                        Vec<ToolCallId>,
                        HashMap<ToolCallId, u64>,
                    ) = {
                        let log = completion_log.lock().unwrap();
                        (
                            log.iter().map(|(id, _)| id.clone()).collect(),
                            log.iter().cloned().collect(),
                        )
                    };
                    let tool_names: HashMap<ToolCallId, String> = call_payloads
                        .iter()
                        .map(|p| (p.call_id.clone(), p.tool_name.clone()))
                        .collect();
                    // attach batch trace before committing so even a
                    // RunnerInvariantViolation keeps the observations
                    if let Some(rt) = trace.rounds.last_mut() {
                        rt.tool_batch = Some(ToolBatchTrace {
                            calls: results
                                .iter()
                                .map(|r| ToolCallTrace {
                                    call_id: r.result.call_id.clone(),
                                    tool_name: tool_names
                                        .get(&r.result.call_id)
                                        .cloned()
                                        .unwrap_or_default(),
                                    position: r.result.call_id.position(),
                                    status: r.result.status.clone(),
                                    truncation: r.result.output.truncation,
                                    artifact: r.result.output.artifact.clone(),
                                    duration_ms: call_durations
                                        .get(&r.result.call_id)
                                        .copied()
                                        .unwrap_or(0),
                                })
                                .collect(),
                            completion_order,
                        });
                    }
                    if let Err(e) = context.append_tool_results(
                        &results.iter().map(|o| o.result.clone()).collect::<Vec<_>>(),
                    ) {
                        return finish(
                            context,
                            TurnResult::Interrupted {
                                cause: TurnInterruption::RunnerInvariantViolation {
                                    reason: e.to_string(),
                                },
                            },
                            trace,
                            tool_calls_total,
                            start,
                        );
                    }
                    // UnknownOutcome policy: Stop interrupts, Continue proceeds;
                    // parent should_stop is checked at the next loop top.
                    if let Some(uu) = results.iter().find(|r| {
                        r.result.status == ToolResultStatus::UnknownOutcome
                            && r.policy == UnknownOutcomePolicy::Stop
                    }) {
                        return finish(
                            context,
                            TurnResult::Interrupted {
                                cause: TurnInterruption::UnsafeUnknownOutcome {
                                    call_id: uu.result.call_id.clone(),
                                },
                            },
                            trace,
                            tool_calls_total,
                            start,
                        );
                    }
                    round += 1;
                }
                ModelStopReason::MaxTokens | ModelStopReason::Refusal => {
                    unreachable!("handled before apply_model_output")
                }
            }
        }
    }
}
