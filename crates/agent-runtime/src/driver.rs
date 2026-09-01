//! The reference driver — retry scheduling, tool batch dispatch, artifact
//! spill, control plumbing, and trace construction. Graduated from the
//! kernel's staged `internal/` perimeter by Slice 12: the kernel is facts
//! and contracts only; the canonical consumer of those contracts lives
//! here, one layer up.
use crate::config::TurnRunOptions;
use crate::control::RunControl;
use crate::executor::ToolExecutor;
use reimagine_context_kernel::AttemptNumber;
use reimagine_context_kernel::FramePolicy;
use reimagine_context_kernel::ModelGateway;
use reimagine_context_kernel::ModelRequest;
use reimagine_context_kernel::ModelStopReason;
use reimagine_context_kernel::ToolCallPayload;
use reimagine_context_kernel::{ArtifactRef, ToolCallId, ToolResultStatus, Truncation};
use reimagine_context_kernel::{BlockId, ConversationId, FrameScope, InvocationId, RoundId};
use reimagine_context_kernel::{ConversationError, ConversationState, SealedResult, merged_frame};
use reimagine_context_kernel::{ModelInvokeError, ModelInvokeErrorKind, ModelOutput};
use reimagine_context_kernel::{ToolExecutionOutcome, UnknownOutcomePolicy};
use reimagine_context_kernel::{TurnContext, TurnSnapshot};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::hook::{HookCtx, PassthroughHook, ToolUseHook};
use futures_util::StreamExt;

fn millis_since(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

// Tool-use filtering lives behind `crate::hook::ToolUseHook`.
// `TurnRunner` defaults to `PassthroughHook` (no opinion); concrete
// filter policies live in `crate::filter` or the host layer.

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
    pub stop_reason: ModelStopReason,
    pub usage: Option<reimagine_context_kernel::ModelUsage>,
    pub tool_call_count: usize,
    pub response_text_bytes: usize,
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
    pub frame_version: reimagine_context_kernel::ContextVersion,
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TurnResult {
    Completed { final_output: ModelOutput },
    Interrupted { cause: TurnInterruption },
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnOutcome {
    /// The sealed active turn at handoff. Serialized through the
    /// `turn_context_as_snapshot` adapter (see `reimagine_context_kernel::turn_context_as_snapshot`):
    /// the in-memory `TurnContext` is the mutable fact machine, but
    /// once sealed its snapshot projection is the canonical wire shape.
    /// On reload we rebuild a sealed `TurnContext` via
    /// `from_validated_blocks` + `seal()`.
    #[serde(with = "reimagine_context_kernel::turn_context_as_snapshot")]
    pub context: TurnContext,
    pub result: TurnResult,
    pub trace: TurnTrace,
}

// Wire-contract note (Slice 5A, 2026-09-02 review): `TurnResult`,
// `TurnOutcome`, `ConversationOutcome` and the `TurnTrace` family are
// embedded in `agent_runtime::event::ContextEvent` and delivered over
// IPC. Since Slice 12 their Rust item paths live in `agent-runtime`
// (graduated from the kernel's staged perimeter) and may move between
// layers without notice — but their serde shapes are a load-bearing
// external contract and must not change without a breaking migration of
// the event wire format. `tests/serialization.rs` pins the shapes.

/// The conversation entry's counterpart to [`TurnOutcome`]: consume/return —
/// the state comes back with the active turn sealed inside and its outcome
/// stamped; the host then calls `commit` (Completed) or `abort_turn`
/// (Interrupted).
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ConversationOutcome {
    pub state: ConversationState,
    pub result: TurnResult,
    pub trace: TurnTrace,
}

/// The frame source — the single fork point between the two runner entries.
#[derive(Clone, Copy)]
enum FrameSource<'a> {
    /// Policy-shaped materialization over the active turn (Turn scope).
    Turn(&'a FramePolicy),
    /// Lossless merged view (Conversation scope) — policy-inert in Slice 2;
    /// conversation-level budget/compaction is Slice 5.
    Conversation {
        conversation_id: &'a ConversationId,
        history: &'a [TurnSnapshot],
    },
}

pub struct TurnRunner {
    gateway: Arc<dyn ModelGateway>,
    executor: Arc<ToolExecutor>,
    /// Seam for tool-use filtering. `TurnRunner::new()` defaults to
    /// `PassthroughHook` (no filter applied — the driver carries no
    /// opinion). Custom hooks (e.g. `FilterChain`, which implements
    /// `ToolUseHook`) plug in via `TurnRunner::with_hook`.
    hook: Arc<dyn ToolUseHook>,
}
impl TurnRunner {
    pub fn new(gateway: Arc<dyn ModelGateway>, executor: Arc<ToolExecutor>) -> Self {
        Self {
            gateway,
            executor,
            hook: Arc::new(PassthroughHook),
        }
    }

    pub fn with_hook(
        gateway: Arc<dyn ModelGateway>,
        executor: Arc<ToolExecutor>,
        hook: Arc<dyn ToolUseHook>,
    ) -> Self {
        Self {
            gateway,
            executor,
            hook,
        }
    }
    /// Slice 1 entry, unchanged in shape: frames materialize from the active
    /// turn alone (Turn scope, policy-shaped).
    pub async fn run(
        &self,
        mut context: TurnContext,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> TurnOutcome {
        let start = Instant::now();
        let (result, mut trace, tool_calls_total) = self
            .drive(
                &mut context,
                FrameSource::Turn(&options.frame),
                &options,
                &ctrl,
            )
            .await;
        // Every drive exit is terminal; the entry owns all terminal
        // bookkeeping (totals, duration, sealing).
        trace.tool_calls_total = tool_calls_total;
        trace.total_duration_ms = millis_since(start);
        context.seal();
        TurnOutcome {
            context,
            result,
            trace,
        }
    }

    /// Slice 2 entry: frames materialize as the lossless merged view over
    /// committed history plus the active turn (Conversation scope — the
    /// `options.frame` policy is deliberately inert here; conversation-level
    /// budget/compaction is Slice 5). Consume/return: the state comes back
    /// with the active turn sealed and outcome-stamped; the host then calls
    /// `commit` (Completed) or `abort_turn` (Interrupted).
    pub async fn run_in_conversation(
        &self,
        mut state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> Result<ConversationOutcome, ConversationError> {
        // Entry gates — caller bugs fail fast, before the state machine.
        let active_id = match state.active_turn() {
            Some(t) => t.turn_id(),
            None => return Err(ConversationError::NoActiveTurn),
        };
        if state.active_turn().expect("checked above").is_sealed() {
            return Err(ConversationError::TurnAlreadySealed);
        }
        // Field-split borrow: read conversation id and history while driving
        // the active turn mutably; stamping happens after the loop through
        // the public `seal_turn`, so no second &mut seam is exposed.
        let (conversation_id, history, active) = state.runner_parts();
        let active = active.expect("NoActiveTurn checked above");
        let start = Instant::now();
        let (result, mut trace, tool_calls_total) = self
            .drive(
                active,
                FrameSource::Conversation {
                    conversation_id,
                    history,
                },
                &options,
                &ctrl,
            )
            .await;
        trace.tool_calls_total = tool_calls_total;
        trace.total_duration_ms = millis_since(start);
        let stamp = match &result {
            TurnResult::Completed { .. } => SealedResult::Completed,
            TurnResult::Interrupted { .. } => SealedResult::Interrupted,
        };
        state
            .seal_turn(active_id, stamp)
            .expect("active turn still present");
        Ok(ConversationOutcome {
            state,
            result,
            trace,
        })
    }

    /// The shared state machine — both entries run this loop; the frame
    /// source is the only fork. Every exit is terminal; the entries own
    /// sealing.
    async fn drive(
        &self,
        active: &mut TurnContext,
        frames: FrameSource<'_>,
        options: &TurnRunOptions,
        ctrl: &RunControl,
    ) -> (TurnResult, TurnTrace, usize) {
        let mut round: u32 = 0;
        let mut tool_calls_total: usize = 0;
        let mut trace = TurnTrace::new();

        loop {
            // boundary checks
            if ctrl.should_stop() {
                let cause = if ctrl.is_cancelled() {
                    TurnInterruption::ExplicitCancellation
                } else {
                    TurnInterruption::TurnDeadlineExceeded
                };
                return (TurnResult::Interrupted { cause }, trace, tool_calls_total);
            }
            if round >= options.policy.limits.max_model_rounds {
                return (
                    TurnResult::Interrupted {
                        cause: TurnInterruption::MaxModelRounds {
                            limit: options.policy.limits.max_model_rounds,
                        },
                    },
                    trace,
                    tool_calls_total,
                );
            }
            // materialize the round's frame — the single fork point between
            // the entries; the fact machine only ever offers the lossless
            // projection, the policy orchestrates anything beyond it
            let frame = match frames {
                FrameSource::Turn(policy) => {
                    match policy.materialize(active, RoundId(round)).await {
                        Ok(f) => f,
                        Err(e) => {
                            return (
                                TurnResult::Interrupted {
                                    cause: TurnInterruption::CompactionFailed {
                                        reason: e.to_string(),
                                    },
                                },
                                trace,
                                tool_calls_total,
                            );
                        }
                    }
                }
                FrameSource::Conversation {
                    conversation_id,
                    history,
                } => merged_frame(conversation_id, history, active, RoundId(round)),
            };
            let frame_version = match &frame.scope {
                FrameScope::Turn { source_version, .. }
                | FrameScope::Conversation { source_version, .. } => *source_version,
            };
            let invocation = InvocationId {
                turn_id: active.turn_id(),
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
                    return (TurnResult::Interrupted { cause }, trace, tool_calls_total);
                }
            };
            let output_summary = Some(OutputSummary {
                stop_reason: output.stop_reason,
                usage: output.usage.clone(),
                tool_call_count: output.response.tool_calls.len(),
                response_text_bytes: output.response.text.0.len(),
            });
            // Every round lands in the trace exactly once, right after the
            // output is summarized — including the paths that never apply a
            // block. Later stages fill their fields through `last_mut`.
            trace.rounds.push(ModelRoundTrace {
                round_id: RoundId(round),
                invocation_id: invocation.clone(),
                frame_version,
                attempts,
                output_summary,
                applied_block_ids: vec![],
                tool_batch: None,
            });
            // MaxTokens / Refusal never persist blocks (§5.6); they carry their
            // own dedicated interruption causes, so dispatch before apply.
            // This is driver policy: the canonical append_model_output would
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
                return (TurnResult::Interrupted { cause }, trace, tool_calls_total);
            }
            let applied = match active.append_model_output(
                invocation.clone(),
                &output.response,
                output.stop_reason,
            ) {
                Ok(a) => a,
                Err(e) => {
                    return (
                        TurnResult::Interrupted {
                            cause: TurnInterruption::InvalidModelOutput {
                                reason: e.to_string(),
                            },
                        },
                        trace,
                        tool_calls_total,
                    );
                }
            };
            // The receipt carries the prepared tool calls in model draft
            // order — the same payloads the kernel committed, with ids
            // generated exactly once. No re-reading of blocks.
            let call_payloads: Vec<ToolCallPayload> = applied.tool_calls;
            trace
                .rounds
                .last_mut()
                .expect("round trace just pushed")
                .applied_block_ids = applied.block_ids.clone();
            match output.stop_reason {
                ModelStopReason::EndTurn => {
                    // append_model_output guarantees EndTurn has empty tool_calls
                    return (
                        TurnResult::Completed {
                            final_output: output,
                        },
                        trace,
                        tool_calls_total,
                    );
                }
                ModelStopReason::ToolUse => {
                    tool_calls_total += call_payloads.len();
                    if tool_calls_total as u32 > options.policy.limits.max_tool_calls {
                        return (
                            TurnResult::Interrupted {
                                cause: TurnInterruption::MaxToolCalls {
                                    limit: options.policy.limits.max_tool_calls,
                                },
                            },
                            trace,
                            tool_calls_total,
                        );
                    }
                    // ToolUse hook: the tool-use filter seam. `TurnRunner::new()`
                    // defaults to `PassthroughHook` (no filter applied — opt in
                    // via `with_hook`). `FilterChain` plugs in via `with_hook`,
                    // implementing `ToolUseHook` directly.
                    let (to_exec, rejected): (Vec<ToolCallPayload>, Vec<ToolExecutionOutcome>) = {
                        let call_control = ctrl
                            .for_attempt(options.policy.attempt_timeout)
                            .for_call(options.execution.call_timeout);
                        let conversation_id = match &frames {
                            FrameSource::Conversation {
                                conversation_id, ..
                            } => Some(*conversation_id),
                            FrameSource::Turn(_) => None,
                        };
                        let hook_ctx = HookCtx {
                            turn_id: &active.turn_id(),
                            conversation_id,
                            round_id: RoundId(round),
                            control: &call_control,
                        };
                        let outcome = self.hook.apply(call_payloads.clone(), &hook_ctx).await;
                        (outcome.to_execute, outcome.rejected)
                    };
                    // parallel dispatch; completion order comes from the
                    // stream (each future is yielded as it finishes), so no
                    // shared log is needed
                    let futs = to_exec.into_iter().map(|payload| {
                        let cc = ctrl
                            .for_attempt(options.policy.attempt_timeout)
                            .for_call(options.execution.call_timeout);
                        let store = options.execution.artifact_store.clone();
                        let tc = options.execution.token_counter.clone();
                        let limits = options.execution.tool_output_limits.clone();
                        let exec = self.executor.clone();
                        async move {
                            let t0 = Instant::now();
                            let out = exec
                                .execute_with_limits(payload, cc, store, tc, limits)
                                .await;
                            (out, millis_since(t0))
                        }
                    });
                    let mut stream = futures_util::stream::FuturesUnordered::from_iter(futs);
                    let mut results: Vec<ToolExecutionOutcome> = Vec::with_capacity(stream.len());
                    let mut completion_order: Vec<ToolCallId> = Vec::with_capacity(stream.len());
                    let mut call_durations: HashMap<ToolCallId, u64> = HashMap::new();
                    while let Some((out, duration_ms)) = stream.next().await {
                        completion_order.push(out.result.call_id.clone());
                        call_durations.insert(out.result.call_id.clone(), duration_ms);
                        results.push(out);
                    }
                    results.extend(rejected);
                    // Canonical order = model draft order, taken from the
                    // receipt's position — not from ToolCallId encoding. The
                    // kernel re-derives the same order from call block
                    // sequences when committing.
                    let order_index: HashMap<ToolCallId, usize> = call_payloads
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (p.call_id.clone(), i))
                        .collect();
                    results.sort_by_key(|r| order_index.get(&r.result.call_id).copied());
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
                                    position: order_index
                                        .get(&r.result.call_id)
                                        .copied()
                                        .unwrap_or_default(),
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
                    if let Err(e) = active
                        .append_tool_results(results.iter().map(|o| o.result.clone()).collect())
                    {
                        return (
                            TurnResult::Interrupted {
                                cause: TurnInterruption::RunnerInvariantViolation {
                                    reason: e.to_string(),
                                },
                            },
                            trace,
                            tool_calls_total,
                        );
                    }
                    // UnknownOutcome policy: Stop interrupts, Continue proceeds;
                    // parent should_stop is checked at the next loop top.
                    if let Some(uu) = results.iter().find(|r| {
                        r.result.status == ToolResultStatus::UnknownOutcome
                            && r.policy == UnknownOutcomePolicy::Stop
                    }) {
                        return (
                            TurnResult::Interrupted {
                                cause: TurnInterruption::UnsafeUnknownOutcome {
                                    call_id: uu.result.call_id.clone(),
                                },
                            },
                            trace,
                            tool_calls_total,
                        );
                    }
                    round += 1;
                }
                ModelStopReason::MaxTokens | ModelStopReason::Refusal => {
                    unreachable!("handled before append_model_output")
                }
            }
        }
    }
}
