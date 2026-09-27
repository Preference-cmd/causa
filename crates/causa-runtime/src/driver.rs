//! The execution stack here — retry scheduling, tool batch processing,
//! control plumbing, and trace construction. The canonical consumer
//! of the kernel's contracts lives here, one layer up.
use crate::budget::FramePolicy;
use crate::config::TurnRunOptions;
use crate::config::UnknownOutcomePolicy;
use crate::control::RunControl;
use crate::conversation::{ConversationError, ConversationState, SealedResult};
use crate::executor::ToolExecutor;
use crate::ids::new_block_id;
use causa_kernel::AttemptNumber;
use causa_kernel::ModelGateway;
use causa_kernel::ModelRequest;
use causa_kernel::ModelStopReason;
use causa_kernel::ToolCallPayload;
use causa_kernel::merged_frame;
use causa_kernel::{ArtifactRef, BatchError, ToolResultStatus, Truncation};
use causa_kernel::{AttemptControl, ModelUsage, StreamDelta};
use causa_kernel::{BlockId, ConversationId, FrameScope, InvocationId, RoundId};
use causa_kernel::{ModelInvokeError, ModelInvokeErrorKind, ModelOutput};
use causa_kernel::{TurnContext, TurnSnapshot};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tracing::Instrument;

use crate::processors::ToolProcessingChain;
use causa_kernel::{
    ProcessorContext, ToolBatch, ToolCallContext, ToolOutput, ToolResultPayload,
    model_output_block_count,
};
use futures_util::StreamExt;

fn millis_since(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

async fn wait_for_stop(ctrl: &RunControl) -> TurnInterruption {
    tokio::select! {
        biased;
        _ = ctrl.cancellation_token().cancelled() => TurnInterruption::ExplicitCancellation,
        _ = async {
            match ctrl.deadline() {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        } => TurnInterruption::TurnDeadlineExceeded,
    }
}

fn stop_cause(ctrl: &RunControl) -> TurnInterruption {
    if ctrl.is_cancelled() {
        TurnInterruption::ExplicitCancellation
    } else {
        TurnInterruption::TurnDeadlineExceeded
    }
}

fn resolve_batch_result(
    batch: &mut ToolBatch,
    call_block_id: BlockId,
    result: ToolResultPayload,
) -> Result<(), String> {
    let pending_offset = batch
        .calls()
        .iter()
        .position(|entry| entry.call().call_block_id == call_block_id)
        .ok_or_else(|| format!("declaration {call_block_id:?} is not pending"))?;
    let index = batch.completed_len() + pending_offset;
    loop {
        match batch.resolve_at(index, new_block_id(), result.clone()) {
            Ok(()) => return Ok(()),
            Err(BatchError::DuplicateResultBlockId(_)) => continue,
            Err(error) => return Err(error.to_string()),
        }
    }
}

fn settle_started_calls(
    batch: &mut ToolBatch,
    started: &std::sync::Mutex<HashMap<BlockId, String>>,
) {
    let started_ids: Vec<BlockId> = started
        .lock()
        .expect("started calls lock")
        .keys()
        .copied()
        .collect();
    for call_block_id in started_ids {
        if !batch
            .calls()
            .iter()
            .any(|entry| entry.call().call_block_id == call_block_id)
        {
            continue;
        }
        let result = ToolResultPayload {
            call_block_id,
            status: ToolResultStatus::UnknownOutcome,
            output: ToolOutput::new(serde_json::json!({
                "error": "tool was started but no result was observed before the batch stopped"
            })),
            media: Vec::new(),
            notes: Vec::new(),
        };
        if let Err(error) = resolve_batch_result(batch, call_block_id, result) {
            tracing::error!(call_block_id = ?call_block_id, %error, "failed to settle started tool call");
        }
    }
}

fn batch_trace(
    batch: &ToolBatch,
    declaration_order: &[BlockId],
    started: &std::sync::Mutex<HashMap<BlockId, String>>,
    durations: &HashMap<BlockId, u64>,
    completion_order: &[BlockId],
) -> ToolBatchTrace {
    let started = started.lock().expect("started calls lock");
    let mut calls: Vec<ToolCallTrace> = batch
        .results()
        .iter()
        .chain(batch.calls())
        .filter_map(|entry| {
            let call_block_id = entry.call().call_block_id;
            let (result_id, result) = entry.result()?;
            let _ = result_id;
            Some(ToolCallTrace {
                call_block_id,
                tool_name: started
                    .get(&call_block_id)
                    .cloned()
                    .unwrap_or_else(|| entry.call().input.tool_name.clone()),
                position: declaration_order
                    .iter()
                    .position(|id| *id == call_block_id)
                    .unwrap_or_default(),
                status: result.status.clone(),
                truncation: result.output.truncation,
                artifact: result.output.artifact.clone(),
                duration_ms: durations.get(&call_block_id).copied().unwrap_or(0),
            })
        })
        .collect();
    calls.sort_by_key(|call: &ToolCallTrace| call.position);
    ToolBatchTrace {
        calls,
        completion_order: completion_order.to_vec(),
    }
}

fn record_batch_trace(
    trace: &mut TurnTrace,
    round: u32,
    batch: &ToolBatch,
    declaration_order: &[BlockId],
    started: &std::sync::Mutex<HashMap<BlockId, String>>,
    durations: &HashMap<BlockId, u64>,
    completion_order: &[BlockId],
) {
    if let Some(round_trace) = trace
        .rounds
        .last_mut()
        .filter(|round_trace| round_trace.round_id == RoundId(round))
    {
        round_trace.tool_batch = Some(batch_trace(
            batch,
            declaration_order,
            started,
            durations,
            completion_order,
        ));
    }
}

/// Why a turn terminated in `TurnResult::Interrupted`. The serde shape
/// (`kind` / `detail`) is part of the event wire contract; see the
/// wire-contract note on [`TurnOutcome`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "detail")]
pub enum TurnInterruption {
    /// The host cancelled the turn through the shared `RunControl` token
    /// (including cancellation observed mid-call, mid-stream, or during
    /// retry backoff).
    ExplicitCancellation,
    /// The turn deadline carried by `RunControl` passed (checked at every
    /// loop top; retry backoff caps its wait at the remaining deadline and
    /// re-checks before each attempt, and a retry loop that gives up after
    /// the deadline has passed ends here rather than with the last
    /// attempt's error kind).
    TurnDeadlineExceeded,
    /// Uniform carrier for terminal model-call failures: retry exhaustion
    /// and non-retryable kinds (Permanent / InvalidRequest / UnknownOutcome)
    /// all land here, discriminated by `last_kind`; a parent-caused
    /// `Cancelled` maps to `ExplicitCancellation` instead.
    RetryExhausted {
        /// Error kind of the final failed attempt — retry exhaustion and
        /// non-retryable kinds alike.
        last_kind: ModelInvokeErrorKind,
        /// Error message of the final failed attempt.
        last_error: String,
    },
    /// The kernel refused the model response —
    /// `TurnContext::append_model_output` rejected it.
    InvalidModelOutput {
        /// The kernel's rejection message.
        reason: String,
    },
    /// The loop reached `TurnLimits::max_model_rounds` before the model
    /// ended the turn.
    MaxModelRounds {
        /// The configured limit that was reached.
        limit: u32,
    },
    /// The turn emitted more than `TurnLimits::max_tool_calls` tool calls
    /// (counted when the model emits the declarations).
    MaxToolCalls {
        /// The configured limit that was exceeded.
        limit: u32,
    },
    /// A tool call ended `UnknownOutcome` while its trusted declaration
    /// demands `Stop` — the call's result is unknowable, so continuing
    /// is unsafe.
    UnsafeUnknownOutcome {
        /// The offending call.
        call_block_id: BlockId,
    },
    /// Turn-scope frame materialization failed — `FramePolicy::materialize`
    /// returned an error.
    CompactionFailed {
        /// The materialization error message.
        reason: String,
    },
    /// A driver invariant broke (e.g. the fact machine refused an append)
    /// — a caller bug, not a model or tool failure.
    RunnerInvariantViolation {
        /// The invariant failure message.
        reason: String,
    },
    /// A configured processor failed to complete its stage.
    ProcessorFailed {
        /// The processor failure detail.
        reason: String,
    },
    /// The model stopped on the token ceiling; the driver dispatches this
    /// before applying, so no blocks persist.
    ModelMaxTokens,
    /// The model refused; the driver dispatches this before applying, so
    /// no blocks persist.
    ModelRefusal,
}

/// One model attempt inside a round — an entry of the retry ledger.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AttemptTrace {
    /// 1-based attempt number within the round's invocation.
    pub attempt: AttemptNumber,
    /// `None` = successful attempt; `Some(kind)` = the failure classification.
    pub kind: Option<ModelInvokeErrorKind>,
    /// Whether the failure was classified retryable (it may still not have
    /// been retried once `RetryPolicy::max_retries` was spent).
    pub is_retryable: bool,
    /// Wall-clock duration of the attempt, in milliseconds.
    pub duration_ms: u64,
}
/// Summary of the model output that closed a round, recorded before any
/// tool dispatch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputSummary {
    /// The model's stop reason (`EndTurn`, `ToolUse`, `MaxTokens`, `Refusal`).
    pub stop_reason: ModelStopReason,
    /// Token usage reported by the gateway, when the provider supplied it.
    pub usage: Option<causa_kernel::ModelUsage>,
    /// Number of tool calls in the model's response payload.
    pub tool_call_count: usize,
    /// Byte length of the response text.
    pub response_text_bytes: usize,
}
/// One dispatched tool call, as observed by the executor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCallTrace {
    /// Id pairing this entry with the committed tool-call block.
    pub call_block_id: BlockId,
    /// Tool name from the draft payload (resolved by `call_id`).
    pub tool_name: String,
    /// The call's index in the model-emitted draft order.
    pub position: usize,
    /// Final result status of the call.
    pub status: ToolResultStatus,
    /// Output truncation marker (`Truncation::None` unless truncated).
    pub truncation: Truncation,
    /// Reference to the full output when an output-budget processor spilled
    /// it to an `ArtifactStore`.
    pub artifact: Option<ArtifactRef>,
    /// Executor-measured wall-clock duration in milliseconds (`0` for
    /// host-precomputed outcomes).
    pub duration_ms: u64,
}
/// One dispatched tool batch, as observed by the executor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolBatchTrace {
    /// Observed per-call outcomes keyed by declaration ID, in declaration
    /// order. Pending calls that never started have no outcome entry.
    pub calls: Vec<ToolCallTrace>,
    /// Actual completion order (recorded as the executor returns), not submission order.
    pub completion_order: Vec<BlockId>,
}
/// One model round: framing, attempts, output, applied blocks, and its
/// tool batch (when any).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelRoundTrace {
    /// 0-based round index within the turn.
    pub round_id: RoundId,
    /// Turn + round identity every attempt of the round shares (a retry
    /// bumps only the attempt number).
    pub invocation_id: InvocationId,
    /// The `source_version` at frame materialization (the version before apply).
    pub frame_version: causa_kernel::ContextVersion,
    /// One entry per model attempt, in order.
    pub attempts: Vec<AttemptTrace>,
    /// `None` when no output was produced (every attempt failed).
    pub output_summary: Option<OutputSummary>,
    /// Ids of the blocks the model door committed this round (optional
    /// text first, then one tool call per draft).
    pub applied_block_ids: Vec<BlockId>,
    /// The dispatched batch, when the round dispatched tools.
    pub tool_batch: Option<ToolBatchTrace>,
}
/// The turn's trace: one record per model round plus totals.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurnTrace {
    /// Rounds in execution order — including rounds whose invocation
    /// never produced output.
    pub rounds: Vec<ModelRoundTrace>,
    /// Total tool calls emitted by the model, counted once at emission.
    pub tool_calls_total: usize,
    /// Wall-clock duration of this run in milliseconds.
    pub total_duration_ms: u64,
}
impl TurnTrace {
    /// An empty trace — no rounds, zero totals; the fresh-drive starting
    /// point.
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

/// How a turn ended: completed or interrupted. The
/// serde shapes are a load-bearing wire contract (embedded in
/// `ContextEvent`); `tests/serialization.rs` pins them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TurnResult {
    /// The model ended the turn (`EndTurn`) — no pending work remains.
    Completed {
        /// The final model output.
        final_output: ModelOutput,
    },
    /// The turn stopped early; `cause` discriminates why.
    Interrupted {
        /// Why the turn was interrupted.
        cause: TurnInterruption,
    },
}

/// Result of a bare-turn entry: context, terminal result, trace, and any
/// uncommitted tool batch returned when processing stopped mid-batch.
pub struct TurnOutcome {
    /// The sealed active turn at handoff.
    pub context: TurnContext,
    /// Terminal result and stop reason.
    pub result: TurnResult,
    /// Execution trace.
    pub trace: TurnTrace,
    /// The current uncommitted batch when a processor, cancellation, deadline,
    /// or invalid tool outcome stopped the chain. The caller owns this batch.
    pub uncommitted_tool_batch: Option<ToolBatch>,
}
impl std::fmt::Debug for TurnOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnOutcome")
            .field("context", &self.context)
            .field("result", &self.result)
            .field("trace", &self.trace)
            .field(
                "has_uncommitted_tool_batch",
                &self.uncommitted_tool_batch.is_some(),
            )
            .finish()
    }
}

// Wire-contract note: `TurnResult`, `TurnOutcome`, `ConversationOutcome`
// and the `TurnTrace` family are embedded in `causa_runtime::event::ContextEvent`
// and delivered over IPC. Their Rust item paths live in this crate and may
// move between layers without notice — but their serde shapes are a
// load-bearing external contract and must not change without a breaking
// migration of the event wire format. `tests/serialization.rs` pins the
// shapes.

/// The conversation entry's consume/return result and any uncommitted tool
/// batch returned when processing stopped mid-batch.
pub struct ConversationOutcome {
    /// State back from the run, with the active turn sealed and outcome-stamped.
    pub state: ConversationState,
    /// The active turn's terminal result.
    pub result: TurnResult,
    /// Observations from the execution.
    pub trace: TurnTrace,
    /// Any current uncommitted batch returned to the caller.
    pub uncommitted_tool_batch: Option<ToolBatch>,
}
impl std::fmt::Debug for ConversationOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationOutcome")
            .field("result", &self.result)
            .field("trace", &self.trace)
            .field(
                "has_uncommitted_tool_batch",
                &self.uncommitted_tool_batch.is_some(),
            )
            .finish()
    }
}

/// The frame source — the single fork point between the two runner entries.
#[derive(Clone, Copy)]
enum FrameSource<'a> {
    /// Policy-shaped materialization over the active turn (Turn scope).
    Turn(&'a FramePolicy),
    /// Lossless merged view (Conversation scope) — the frame policy is
    /// deliberately inert here.
    Conversation {
        conversation_id: &'a ConversationId,
        history: &'a [TurnSnapshot],
    },
}

/// The model-phase fork point: a batch `invoke`, or a delta `stream` whose
/// items are forwarded to the interaction seam. Everything downstream of
/// the assembled `ModelOutput` is shared state machine.
#[derive(Clone, Copy)]
pub(crate) enum ModelPhase {
    Batch,
    Stream,
}

/// The execution stack over kernel facts, the model gateway, tool executor,
/// run control, and ordered batch processors.
pub struct TurnRunner {
    gateway: Arc<dyn ModelGateway>,
    executor: Arc<ToolExecutor>,
    processors: ToolProcessingChain,
}
impl TurnRunner {
    /// Create a runner with no configured tool processors.
    pub fn new(gateway: Arc<dyn ModelGateway>, executor: Arc<ToolExecutor>) -> Self {
        Self {
            gateway,
            executor,
            processors: ToolProcessingChain::default(),
        }
    }

    /// Create a runner with an ordered pre-/post-processing chain.
    pub fn with_tool_processors(
        gateway: Arc<dyn ModelGateway>,
        executor: Arc<ToolExecutor>,
        processors: ToolProcessingChain,
    ) -> Self {
        Self {
            gateway,
            executor,
            processors,
        }
    }
    /// Frames materialize from the active turn alone (Turn scope,
    /// policy-shaped).
    pub async fn run(
        &self,
        context: TurnContext,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> TurnOutcome {
        let span = tracing::info_span!(
            "agent.turn",
            turn_id = %context.turn_id().0,
            scope = "turn"
        );
        self.run_inner(context, options, ctrl, ModelPhase::Batch)
            .instrument(span)
            .await
    }

    async fn run_inner(
        &self,
        mut context: TurnContext,
        options: TurnRunOptions,
        ctrl: RunControl,
        phase: ModelPhase,
    ) -> TurnOutcome {
        let start = Instant::now();
        let (result, mut trace, tool_calls_total, uncommitted_tool_batch) = self
            .drive(
                &mut context,
                FrameSource::Turn(&options.frame),
                &options,
                &ctrl,
                phase,
            )
            .await;
        // Every drive exit is terminal; the entry seals committed facts.
        trace.tool_calls_total = tool_calls_total;
        trace.total_duration_ms = millis_since(start);
        context.seal();
        TurnOutcome {
            context,
            result,
            trace,
            uncommitted_tool_batch,
        }
    }

    /// The streaming twin of [`TurnRunner::run`]. The model
    /// phase consumes provider deltas — each one forwarded to
    /// `options.interaction.on_delta` — instead of a single batch result;
    /// retry bookkeeping, tool dispatch, traces, and sealing are the same
    /// shared state machine. A retried attempt re-streams the same frame;
    /// deltas already observed are advisory history, and the host decides
    /// how to present the partial-then-reset flow.
    pub async fn run_streaming(
        &self,
        context: TurnContext,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> TurnOutcome {
        let span = tracing::info_span!(
            "agent.turn",
            turn_id = %context.turn_id().0,
            scope = "turn.streaming"
        );
        self.run_inner(context, options, ctrl, ModelPhase::Stream)
            .instrument(span)
            .await
    }

    /// Frames materialize as the lossless merged view over committed history
    /// plus the active turn (Conversation scope — the `options.frame` policy
    /// is deliberately inert here). Consume/return: the state comes back
    /// with the active turn sealed and outcome-stamped; the host then calls
    /// `commit` (Completed) or `abort_turn` (Interrupted).
    pub async fn run_in_conversation(
        &self,
        state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> Result<ConversationOutcome, ConversationError> {
        self.drive_conversation(state, options, ctrl, ModelPhase::Batch)
            .await
    }

    /// The streaming twin of [`TurnRunner::run_in_conversation`]
    /// — same consume/return contract and entry gates, delta-driven model
    /// phase. `options.frame` stays deliberately inert here.
    pub async fn run_in_conversation_streaming(
        &self,
        state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> Result<ConversationOutcome, ConversationError> {
        self.drive_conversation(state, options, ctrl, ModelPhase::Stream)
            .await
    }

    /// The conversation entry body — entry gates, merged-frame source,
    /// terminal bookkeeping, and outcome-stamped sealing shared by both
    /// conversation entry points.
    async fn drive_conversation(
        &self,
        state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
        phase: ModelPhase,
    ) -> Result<ConversationOutcome, ConversationError> {
        // One `agent.turn` span per entry, ids and scope only — never
        // message content. The span is entered per poll via `Instrument`,
        // so the future stays `Send`.
        let scope = if matches!(phase, ModelPhase::Stream) {
            "conversation.streaming"
        } else {
            "conversation"
        };
        let turn_id = state
            .active_turn()
            .map(|t| t.turn_id().0)
            .unwrap_or_default();
        let span = tracing::info_span!(
            "agent.turn",
            turn_id = %turn_id,
            conversation_id = %state.conversation_id().0,
            scope = scope,
        );
        self.drive_conversation_inner(state, options, ctrl, phase)
            .instrument(span)
            .await
    }

    /// The conversation entry body — entry gates, merged-frame source,
    /// terminal bookkeeping, and outcome-stamped sealing shared by both
    /// conversation entry points.
    async fn drive_conversation_inner(
        &self,
        mut state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
        phase: ModelPhase,
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
        let (result, mut trace, tool_calls_total, uncommitted_tool_batch) = self
            .drive_from(
                active,
                FrameSource::Conversation {
                    conversation_id,
                    history: &history,
                },
                &options,
                &ctrl,
                phase,
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
            uncommitted_tool_batch,
        })
    }

    /// One streaming model attempt: forward every delta to the interaction
    /// seam, then return the `Done` output (falling back to a stream-phase
    /// `Usage` delta when `Done` carries none). An `Error` delta maps to the
    /// same error shape the `invoke` path produces, so the shared retry
    /// loop and trace machinery apply unchanged; a stream that ends
    /// without `Done` is `UnknownOutcome`.
    async fn stream_attempt(
        &self,
        req: &ModelRequest,
        attempt_ctrl: &AttemptControl,
        options: &TurnRunOptions,
        round_id: RoundId,
    ) -> Result<ModelOutput, ModelInvokeError> {
        use futures_util::StreamExt;
        let mut stream = self.gateway.stream(req, attempt_ctrl).await?;
        let mut usage_seen: Option<ModelUsage> = None;
        loop {
            let item = tokio::select! {
                biased;
                _ = attempt_ctrl.cancellation_token().cancelled() => {
                    return Err(ModelInvokeError::new(
                        ModelInvokeErrorKind::Cancelled,
                        "cancelled during stream",
                    ));
                }
                next = stream.next() => match next {
                    Some(item) => item,
                    None => {
                        return Err(ModelInvokeError::new(
                            ModelInvokeErrorKind::UnknownOutcome,
                            "stream ended without Done",
                    ));
                    }
                },
            };
            options.interaction.on_delta(round_id, &item).await;
            match item {
                StreamDelta::Done { final_output, .. } => {
                    let mut out = final_output;
                    if out.usage.is_none() {
                        out.usage = usage_seen;
                    }
                    return Ok(out);
                }
                StreamDelta::Usage(u) => usage_seen = Some(u),
                StreamDelta::Error { kind, message } => {
                    return Err(ModelInvokeError::new(kind, message));
                }
                StreamDelta::TextDelta { .. }
                | StreamDelta::ReasoningDelta { .. }
                | StreamDelta::ToolCallDelta { .. } => {}
            }
        }
    }

    /// The shared state machine — every entry runs this loop; the frame
    /// source and the model phase are the only forks. Every exit is
    /// terminal; the entries own
    /// sealing/stamping.
    async fn drive(
        &self,
        active: &mut TurnContext,
        frames: FrameSource<'_>,
        options: &TurnRunOptions,
        ctrl: &RunControl,
        phase: ModelPhase,
    ) -> (TurnResult, TurnTrace, usize, Option<ToolBatch>) {
        self.drive_from(active, frames, options, ctrl, phase).await
    }

    async fn drive_from(
        &self,
        active: &mut TurnContext,
        frames: FrameSource<'_>,
        options: &TurnRunOptions,
        ctrl: &RunControl,
        phase: ModelPhase,
    ) -> (TurnResult, TurnTrace, usize, Option<ToolBatch>) {
        let mut round: u32 = 0;
        let mut tool_calls_total: usize = 0;
        let mut trace = TurnTrace::new();
        // The first model phase of this drive advertises the host-declared
        // surface (the `TurnInvocation` baseline); every later round
        // boundary re-snapshots the runner's executor, so catalog changes
        // — including ones this turn's own tool calls triggered — reach the
        // next request. Retries within one round reuse that round's
        // snapshot.
        let first_model_round = round;

        loop {
            for text in options.interaction.pending_inputs().await {
                if let Err(e) = active.append_input(new_block_id(), text, "user.steering") {
                    return (
                        TurnResult::Interrupted {
                            cause: TurnInterruption::RunnerInvariantViolation {
                                reason: e.to_string(),
                            },
                        },
                        trace,
                        tool_calls_total,
                        None,
                    );
                }
            }
            // boundary checks
            if ctrl.should_stop() {
                let cause = if ctrl.is_cancelled() {
                    TurnInterruption::ExplicitCancellation
                } else {
                    TurnInterruption::TurnDeadlineExceeded
                };
                return (
                    TurnResult::Interrupted { cause },
                    trace,
                    tool_calls_total,
                    None,
                );
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
                    None,
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
                                None,
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
            // Per-round tool surface (see `first_model_round` above): the
            // host baseline for the first model phase, a fresh executor
            // snapshot at every later round boundary. Retries inside the
            // attempt loop below reuse the same snapshot, so advertising
            // and dispatch routing stay one decision per round.
            let round_surface = if round == first_model_round {
                options.invocation.tool_surface.clone()
            } else {
                self.executor.tool_surface().await
            };
            // The round span covers the bounded retry loop; spans are
            // entered per poll via `Instrument`, so the future stays `Send`.
            // Fields carry ids and names only — never frame content or
            // arguments.
            let round_span = tracing::debug_span!(
                "agent.round",
                turn_id = %invocation.turn_id.0,
                round_id = round
            );
            let (attempts, output) = async {
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
                        tool_surface: round_surface.clone(),
                        generation: options.invocation.generation.clone(),
                        cache: options.invocation.cache,
                    };
                    let attempt_span = tracing::debug_span!(
                        "agent.attempt",
                        attempt = attempt,
                        model = %options.invocation.model.0
                    );
                    let result = match phase {
                        ModelPhase::Batch => {
                            self.gateway
                                .invoke(&req, &attempt_ctrl)
                                .instrument(attempt_span)
                                .await
                        }
                        ModelPhase::Stream => {
                            self.stream_attempt(&req, &attempt_ctrl, options, RoundId(round))
                                .instrument(attempt_span)
                                .await
                        }
                    };
                    match result {
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
                                // Cancellation- and deadline-aware backoff:
                                // sleep between attempts, racing the shared
                                // token so a cancel lands immediately instead
                                // of after the wait, and capping the wait at
                                // the remaining turn deadline so the driver
                                // never holds the turn hostage in its own
                                // backoff.
                                let delay = options.policy.retry.backoff_delay(attempt + 1);
                                let capped = ctrl
                                    .remaining_turn_time()
                                    .map_or(delay, |remaining| delay.min(remaining));
                                if !capped.is_zero() {
                                    tokio::select! {
                                        biased;
                                        _ = attempt_ctrl.cancellation_token().cancelled() => {
                                            break Err(ModelInvokeError::new(
                                                ModelInvokeErrorKind::Cancelled,
                                                "cancelled during retry backoff",
                                            ));
                                        }
                                        _ = tokio::time::sleep(capped) => {}
                                    }
                                }
                                // Stop guard before spending another attempt:
                                // a cancel or a deadline that landed during the
                                // backoff (or the previous attempt) ends the
                                // loop here. The outer mapping attributes the
                                // turn's end to `TurnDeadlineExceeded` whenever
                                // the deadline has passed, so the terminal
                                // reason matches the round-boundary check.
                                if ctrl.should_stop() {
                                    if ctrl.is_cancelled() {
                                        break Err(ModelInvokeError::new(
                                            ModelInvokeErrorKind::Cancelled,
                                            "cancelled during retry backoff",
                                        ));
                                    }
                                    break Err(ModelInvokeError::new(
                                        ModelInvokeErrorKind::TimedOut,
                                        "turn deadline passed during retry backoff",
                                    ));
                                }
                                attempt += 1;
                                continue;
                            }
                            break Err(e);
                        }
                    }
                };
                (attempts, output)
            }
            .instrument(round_span)
            .await;
            let output = match output {
                Ok(o) => o,
                Err(e) => {
                    let cause = if matches!(e.kind, ModelInvokeErrorKind::Cancelled)
                        && ctrl.is_cancelled()
                    {
                        TurnInterruption::ExplicitCancellation
                    } else if ctrl.should_stop() && !ctrl.is_cancelled() {
                        // The turn deadline passed while the retry loop was
                        // still working — mid-attempt or during backoff — so
                        // the turn ends with the same verdict the
                        // round-boundary check would produce, not with the
                        // last attempt's error kind.
                        TurnInterruption::TurnDeadlineExceeded
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
                    return (
                        TurnResult::Interrupted { cause },
                        trace,
                        tool_calls_total,
                        None,
                    );
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
            // MaxTokens / Refusal never persist blocks; they carry their
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
                return (
                    TurnResult::Interrupted { cause },
                    trace,
                    tool_calls_total,
                    None,
                );
            }
            let block_ids = (0..model_output_block_count(&output.response))
                .map(|_| new_block_id())
                .collect();
            let applied = match active.append_model_output(
                invocation.clone(),
                &output.response,
                output.stop_reason,
                block_ids,
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
                        None,
                    );
                }
            };
            // The receipt carries the prepared tool calls in model draft
            // order — the same payloads the kernel committed, with ids
            // generated exactly once. No re-reading of blocks.
            let declarations = applied.tool_calls;
            let call_payloads: Vec<ToolCallPayload> = declarations
                .iter()
                .map(|(_, payload)| payload.clone())
                .collect();
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
                        None,
                    );
                }
                ModelStopReason::ToolUse => {
                    tool_calls_total += call_payloads.len();
                    let contexts = declarations
                        .iter()
                        .map(|(id, payload)| ToolCallContext::from_declaration(*id, payload))
                        .collect();
                    let batch = match ToolBatch::new(contexts) {
                        Ok(batch) => batch,
                        Err(error) => {
                            return (
                                TurnResult::Interrupted {
                                    cause: TurnInterruption::RunnerInvariantViolation {
                                        reason: error.to_string(),
                                    },
                                },
                                trace,
                                tool_calls_total,
                                None,
                            );
                        }
                    };
                    if tool_calls_total as u32 > options.policy.limits.max_tool_calls {
                        return (
                            TurnResult::Interrupted {
                                cause: TurnInterruption::MaxToolCalls {
                                    limit: options.policy.limits.max_tool_calls,
                                },
                            },
                            trace,
                            tool_calls_total,
                            Some(batch),
                        );
                    }
                    let declaration_order = batch.declaration_ids();
                    let turn_id = active.turn_id();
                    let conversation_id = match &frames {
                        FrameSource::Conversation {
                            conversation_id, ..
                        } => Some(*conversation_id),
                        FrameSource::Turn(_) => None,
                    };
                    let control = ctrl.for_attempt(None).for_call(None);
                    let processor_context = ProcessorContext {
                        conversation_id,
                        turn_id: &turn_id,
                        round_id: RoundId(round),
                        declaration_order: &declaration_order,
                        control: &control,
                    };
                    match self
                        .run_batch(active, batch, &processor_context, options, ctrl, &mut trace)
                        .await
                    {
                        Ok(()) => {}
                        Err((cause, batch)) => {
                            return (
                                TurnResult::Interrupted { cause },
                                trace,
                                tool_calls_total,
                                batch,
                            );
                        }
                    }
                    round += 1;
                }
                ModelStopReason::MaxTokens | ModelStopReason::Refusal => {
                    unreachable!("handled before append_model_output")
                }
            }
        }
    }

    /// Run the processor → executor → processor sequence over one batch.
    async fn run_batch(
        &self,
        active: &mut TurnContext,
        mut batch: ToolBatch,
        ctx: &ProcessorContext<'_>,
        options: &TurnRunOptions,
        ctrl: &RunControl,
        trace: &mut TurnTrace,
    ) -> Result<(), (TurnInterruption, Option<ToolBatch>)> {
        let round = ctx.round_id.0;
        let started = Arc::new(std::sync::Mutex::new(HashMap::<BlockId, String>::new()));
        let durations = HashMap::<BlockId, u64>::new();
        let completion_order = Vec::<BlockId>::new();
        let pre = tokio::select! {
            biased;
            cause = wait_for_stop(ctrl) => Err(cause),
            result = self.processors.process_before(&mut batch, ctx) => Ok(result),
        };
        match pre {
            Err(cause) => {
                record_batch_trace(
                    trace,
                    round,
                    &batch,
                    ctx.declaration_order,
                    &started,
                    &durations,
                    &completion_order,
                );
                return Err((cause, Some(batch)));
            }
            Ok(Err(error)) => {
                record_batch_trace(
                    trace,
                    round,
                    &batch,
                    ctx.declaration_order,
                    &started,
                    &durations,
                    &completion_order,
                );
                if ctrl.should_stop() {
                    return Err((stop_cause(ctrl), Some(batch)));
                }
                return Err((
                    TurnInterruption::ProcessorFailed {
                        reason: error.to_string(),
                    },
                    Some(batch),
                ));
            }
            Ok(Ok(())) => {}
        }
        if ctrl.should_stop() {
            record_batch_trace(
                trace,
                round,
                &batch,
                ctx.declaration_order,
                &started,
                &durations,
                &completion_order,
            );
            return Err((stop_cause(ctrl), Some(batch)));
        }

        let mut durations = durations;
        let mut completion_order = completion_order;
        let executor = self.executor.clone();
        let store = options.execution.artifact_store.clone();
        let control = ctrl.clone();
        let call_timeout = options.execution.call_timeout;
        let calls: Vec<(BlockId, ToolCallContext)> = batch
            .calls()
            .iter()
            .map(|entry| (entry.call().call_block_id, entry.call().clone()))
            .collect();
        let tasks = calls.into_iter().map(|(call_block_id, call)| {
            let started = started.clone();
            let executor = executor.clone();
            let control = control.clone();
            let store = store.clone();
            async move {
                // FuturesUnordered can poll several newly-created futures
                // in one `next()` call. A sibling may cancel the turn during
                // that same poll, so every task must check the shared control
                // before claiming that it started.
                if control.should_stop() {
                    return (call_block_id, None, None);
                }
                started
                    .lock()
                    .expect("started calls lock")
                    .insert(call_block_id, call.input.tool_name.clone());
                // Start the call-scoped timeout only when this future is
                // polled for dispatch; time spent in model/catalog/pre-stage
                // work does not consume a tool's execution budget.
                let call_control = control.for_attempt(None).for_call(call_timeout);
                let begin = Instant::now();
                let outcome = executor.execute(call, call_control, store).await;
                (call_block_id, Some(outcome), Some(millis_since(begin)))
            }
        });
        let mut stream = futures_util::stream::FuturesUnordered::from_iter(tasks);
        let mut stopped = None;
        while !stream.is_empty() {
            let next = tokio::select! {
                biased;
                cause = wait_for_stop(ctrl) => {
                    stopped = Some(cause);
                    None
                }
                next = stream.next() => next,
            };
            let Some((call_block_id, outcome, duration_ms)) = next else {
                break;
            };
            let (Some(outcome), Some(duration_ms)) = (outcome, duration_ms) else {
                stopped = Some(stop_cause(ctrl));
                break;
            };
            match outcome {
                Ok(result) => {
                    completion_order.push(call_block_id);
                    durations.insert(call_block_id, duration_ms);
                    if result.call_block_id != call_block_id {
                        stopped = Some(TurnInterruption::RunnerInvariantViolation {
                            reason: format!(
                                "tool result targets {:?}, expected {:?}",
                                result.call_block_id, call_block_id
                            ),
                        });
                        break;
                    }
                    if let Err(error) = resolve_batch_result(&mut batch, call_block_id, result) {
                        stopped =
                            Some(TurnInterruption::RunnerInvariantViolation { reason: error });
                        break;
                    }
                }
                Err(error) => {
                    stopped = Some(TurnInterruption::RunnerInvariantViolation {
                        reason: error.to_string(),
                    });
                    break;
                }
            }
        }
        if stopped.is_some() {
            // Preserve the selected cause, then signal every in-flight tool
            // through the shared run token. This internal signal must not
            // replace a deadline or runner-invariant cause with cancellation.
            ctrl.cancellation_token().cancel();
        }
        drop(stream);
        if let Some(cause) = stopped {
            settle_started_calls(&mut batch, &started);
            record_batch_trace(
                trace,
                round,
                &batch,
                ctx.declaration_order,
                &started,
                &durations,
                &completion_order,
            );
            return Err((cause, Some(batch)));
        }

        if batch.completed_len() != batch.declaration_ids().len() {
            settle_started_calls(&mut batch, &started);
            record_batch_trace(
                trace,
                round,
                &batch,
                ctx.declaration_order,
                &started,
                &durations,
                &completion_order,
            );
            return Err((
                TurnInterruption::RunnerInvariantViolation {
                    reason: "fixed executor stage left a call unresolved".into(),
                },
                Some(batch),
            ));
        }
        let post = tokio::select! {
            biased;
            cause = wait_for_stop(ctrl) => Err(cause),
            result = self.processors.process_after(&mut batch, ctx) => Ok(result),
        };
        match post {
            Err(cause) => {
                record_batch_trace(
                    trace,
                    round,
                    &batch,
                    ctx.declaration_order,
                    &started,
                    &durations,
                    &completion_order,
                );
                return Err((cause, Some(batch)));
            }
            Ok(Err(error)) => {
                record_batch_trace(
                    trace,
                    round,
                    &batch,
                    ctx.declaration_order,
                    &started,
                    &durations,
                    &completion_order,
                );
                if ctrl.should_stop() {
                    return Err((stop_cause(ctrl), Some(batch)));
                }
                return Err((
                    TurnInterruption::ProcessorFailed {
                        reason: error.to_string(),
                    },
                    Some(batch),
                ));
            }
            Ok(Ok(())) => {}
        }
        if ctrl.should_stop() {
            record_batch_trace(
                trace,
                round,
                &batch,
                ctx.declaration_order,
                &started,
                &durations,
                &completion_order,
            );
            return Err((stop_cause(ctrl), Some(batch)));
        }

        let tool_names: HashMap<BlockId, String> = batch
            .results()
            .iter()
            .map(|entry| {
                let id = entry.call().call_block_id;
                let name = started
                    .lock()
                    .expect("started calls lock")
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| entry.call().input.tool_name.clone());
                (id, name)
            })
            .collect();
        let entries = batch.results();
        let unsafe_unknown = entries.iter().find_map(|entry| {
            let (_, result) = entry.result()?;
            (result.status == ToolResultStatus::UnknownOutcome
                && options.policy.unknown_outcome.resolve(
                    tool_names
                        .get(&entry.call().call_block_id)
                        .map(String::as_str)
                        .unwrap_or(""),
                ) == UnknownOutcomePolicy::Stop)
                .then_some(entry.call().call_block_id)
        });
        record_batch_trace(
            trace,
            round,
            &batch,
            ctx.declaration_order,
            &started,
            &durations,
            &completion_order,
        );

        let results = entries
            .iter()
            .filter_map(|entry| {
                let (result_id, result) = entry.result()?;
                Some((*result_id, result.clone()))
            })
            .collect();
        if let Err(error) = active.append_tool_results(results) {
            return Err((
                TurnInterruption::RunnerInvariantViolation {
                    reason: error.to_string(),
                },
                Some(batch),
            ));
        }
        if ctrl.should_stop() {
            return Err((stop_cause(ctrl), None));
        }
        if let Some(call_block_id) = unsafe_unknown {
            return Err((
                TurnInterruption::UnsafeUnknownOutcome { call_block_id },
                None,
            ));
        }
        Ok(())
    }
}
