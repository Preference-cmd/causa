//! The reference driver — retry scheduling, tool batch dispatch, artifact
//! spill, control plumbing, and trace construction. Graduated from the
//! kernel's staged `internal/` perimeter by Slice 12: the kernel is facts
//! and contracts only; the canonical consumer of those contracts lives
//! here, one layer up.
use crate::budget::FramePolicy;
use crate::config::TurnRunOptions;
use crate::control::RunControl;
use crate::conversation::{ConversationError, ConversationState, SealedResult};
use crate::executor::ToolExecutor;
use crate::interaction::BatchDecision;
use causa_kernel::AttemptNumber;
use causa_kernel::ModelGateway;
use causa_kernel::ModelRequest;
use causa_kernel::ModelStopReason;
use causa_kernel::TextPayload;
use causa_kernel::ToolCallPayload;
use causa_kernel::{ArtifactRef, ToolCallId, ToolResultStatus, Truncation};
use causa_kernel::{AttemptControl, ModelUsage, StreamDelta};
use causa_kernel::{BlockContent, merged_frame};
use causa_kernel::{BlockId, ConversationId, FrameScope, InvocationId, RoundId};
use causa_kernel::{ModelInvokeError, ModelInvokeErrorKind, ModelOutput};
use causa_kernel::{ToolExecutionOutcome, UnknownOutcomePolicy};
use causa_kernel::{TurnContext, TurnSnapshot};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::Instrument;

use crate::hook::{HookCtx, HookOutcome, PassthroughHook, ToolUseHook};
use futures_util::StreamExt;

fn millis_since(t: Instant) -> u64 {
    t.elapsed().as_millis() as u64
}

/// The turn's committed-but-unanswered tool calls, in block order — the
/// model-emitted draft order results pair into. Shared by the resume
/// validation and the approval-resume prologue.
pub(crate) fn unanswered_tool_calls(active: &TurnContext) -> Vec<ToolCallPayload> {
    let mut pending: Vec<ToolCallPayload> = Vec::new();
    for b in active.blocks() {
        match &b.content {
            BlockContent::ToolCall(c) => pending.push(c.clone()),
            BlockContent::ToolResult(r) => pending.retain(|p| p.call_id != r.call_id),
            _ => {}
        }
    }
    pending
}

// Tool-use filtering lives behind `crate::hook::ToolUseHook`.
// `TurnRunner` defaults to `PassthroughHook` (no opinion); concrete
// filter policies live in `crate::filter` or the host layer.

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
    /// loop top).
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
    /// (counted at dispatch time, including a batch paused pending
    /// approval).
    MaxToolCalls {
        /// The configured limit that was exceeded.
        limit: u32,
    },
    /// A tool call ended `UnknownOutcome` while its trusted declaration
    /// demands `Stop` — the call's result is unknowable, so continuing
    /// is unsafe.
    UnsafeUnknownOutcome {
        /// The offending call.
        call_id: ToolCallId,
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
    /// The model stopped on the token ceiling; the driver dispatches this
    /// before applying, so no blocks persist (§5.6).
    ModelMaxTokens,
    /// The model refused; the driver dispatches this before applying, so
    /// no blocks persist (§5.6).
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
    pub call_id: ToolCallId,
    /// Tool name from the draft payload (resolved by `call_id`).
    pub tool_name: String,
    /// The call's index in the model-emitted draft order.
    pub position: usize,
    /// Final result status of the call.
    pub status: ToolResultStatus,
    /// Output truncation marker (`Truncation::None` unless truncated).
    pub truncation: Truncation,
    /// Reference to the spilled full output, when truncation used an
    /// `ArtifactStore`.
    pub artifact: Option<ArtifactRef>,
    /// Executor-measured wall-clock duration in milliseconds (`0` for
    /// host-precomputed outcomes).
    pub duration_ms: u64,
}
/// One dispatched tool batch, as observed by the executor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolBatchTrace {
    /// Per-call traces in canonical (model draft) order — executor
    /// results, hook rejections, and host-precomputed outcomes alike.
    pub calls: Vec<ToolCallTrace>,
    /// Actual completion order (recorded as the executor returns), not submission order.
    pub completion_order: Vec<ToolCallId>,
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
    /// Total tool calls emitted, counted at dispatch time (a paused batch
    /// counts once, at emission — the resume does not re-count).
    pub tool_calls_total: usize,
    /// Wall-clock duration of this run in milliseconds (a resumed
    /// continuation measures only the continuation).
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

/// How a turn ended: completed, interrupted, or paused mid-turn. The
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
    /// Resumable suspension (Slice 7) — not a terminal state: the turn's
    /// facts stay open in the outcome's `context` / `state` (the single
    /// fact source — Slice 6.5 removed the duplicated snapshot from this
    /// variant), the prepared batch is neither executed nor rejected, and
    /// `resume_turn` / `TurnRunner::resume` continue the same turn from
    /// the [`Continuation`].
    Paused {
        /// The single continuation: pause position, control counts,
        /// prepared work, and queued inputs.
        continuation: Continuation,
    },
}

/// The hook-prepared work an approval pause checkpoints (Slice 6.5,
/// Decision 7): the hook has already filtered / rejected / rewritten the
/// model-emitted batch, and those decisions cannot be re-derived on
/// resume — they are saved, not re-run. The original payloads remain
/// derivable from the committed tool-call blocks (fact identity); the
/// independent rewrite and rejections live here as serializable data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreparedApproval {
    /// Calls the hook admitted — arguments possibly rewritten — in model
    /// draft order. These are what still await the host's decision at
    /// resume; the resume decision must cover them exactly once.
    pub awaiting: Vec<ToolCallPayload>,
    /// Calls the hook rejected. The stored outcomes commit verbatim at
    /// resume (with their original call ids) and can never be re-decided —
    /// a resume decision touching them is rejected before anything
    /// executes.
    pub rejected: Vec<ToolExecutionOutcome>,
}

/// Where a turn paused. The reference driver currently emits only the
/// approval position; the steering position exists so hosts that suspend
/// before a model round produce a continuation the same resume machinery
/// consumes (Decision 6.2: the pause point and the round together define
/// the next step — never free-floating `Option`s).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum PausePoint {
    /// The approval gate paused behind a hook-prepared batch.
    AwaitingApproval {
        /// The hook's prepared work awaiting the host's decision.
        prepared: PreparedApproval,
        /// Advisory decision budget the host granted itself, as *remaining*
        /// time so the variant stays serde-friendly (a host anchors
        /// `Instant::now() + d`). Carried for information only — the resume
        /// never derives a deadline from it.
        deadline: Option<Duration>,
    },
    /// Paused before a model round so the host can steer; no batch awaits
    /// (a steering continuation over a turn with unanswered tool calls is
    /// rejected at resume — the batch could never be answered).
    PausedForSteering,
}

/// The single continuation of a paused turn (Slice 6.5): everything
/// resuming needs beyond the paused outcome's fact state, the runner, the
/// new options, and the new control. Rounds and quotas are read from here
/// — never from the trace, which is observational and may be trimmed
/// freely without changing where a resume continues or what it may spend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Continuation {
    /// Where and why the turn paused.
    pub pause_point: PausePoint,
    /// The round the continuation sits at: for an approval pause, the
    /// paused round whose batch executes first at resume (the loop then
    /// continues at round+1); for a steering pause, the first round whose
    /// model phase has not run yet.
    pub round: u32,
    /// Tool calls already counted against the turn's quota at emission
    /// time (a paused batch was counted once, at emission — the resume
    /// never re-counts it). The authoritative control count across
    /// resumes.
    pub accounted_tool_calls: usize,
    /// Inputs received but not yet committed. A resume commits them first
    /// (`user.steering` label), before the resume request's own inject —
    /// "queued → injected → pulled" order; identical texts are independent
    /// inputs and are never deduped.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub queued_inputs: Vec<TextPayload>,
}

/// A bare-turn entry's result: the turn context (sealed unless the turn
/// paused), the outcome, and the trace.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnOutcome {
    /// The sealed active turn at handoff. Serialized through the
    /// `turn_context_as_snapshot` adapter (see `causa_kernel::turn_context_as_snapshot`):
    /// the in-memory `TurnContext` is the mutable fact machine, but
    /// once sealed its snapshot projection is the canonical wire shape.
    /// On reload we rebuild a sealed `TurnContext` via
    /// `from_validated_blocks` + `seal()`.
    #[serde(with = "causa_kernel::turn_context_as_snapshot")]
    pub context: TurnContext,
    /// Terminal result or pause — see [`TurnResult`].
    pub result: TurnResult,
    /// What happened during the run — see [`TurnTrace`].
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
    /// The state back from the run, its active turn sealed and
    /// outcome-stamped; the host then calls `commit` (Completed),
    /// `abort_turn` (Interrupted), or `resume_turn` (Paused).
    pub state: ConversationState,
    /// The active turn's outcome.
    pub result: TurnResult,
    /// The active turn's trace (a resumed continuation appends to the
    /// paused phase's trace).
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

/// The model-phase fork point: a batch `invoke`, or a delta `stream` whose
/// items are forwarded to the interaction seam. Everything downstream of
/// the assembled `ModelOutput` is shared state machine.
#[derive(Clone, Copy)]
pub(crate) enum ModelPhase {
    Batch,
    Stream,
}

/// Continuation payload for a resumed turn (Slice 7, reworked by Slice
/// 6.5): the validated continuation plus the new decision, consumed once
/// by `drive_from`'s prologue. Built by the public resume entries after
/// validation; rounds, counts, prepared work, and queued inputs all come
/// from the [`Continuation`] — the trace rides along as observation only.
pub(crate) struct ResumeState {
    /// The paused turn's continuation (pause position, round, counts,
    /// prepared work, queued inputs).
    pub(crate) continuation: Continuation,
    /// The resume request's new decision for the prepared batch. `Some`
    /// iff the pause point is `AwaitingApproval` (enforced by validation).
    pub(crate) decision: Option<HookOutcome>,
    /// The resume request's new inputs, appended before the next model
    /// round — after the continuation's queued inputs.
    pub(crate) inject: Vec<TextPayload>,
    /// The paused phase's trace — rounds append to it, totals re-derive
    /// from the continuation's count. A trimmed trace changes nothing.
    pub(crate) trace: TurnTrace,
}

/// What a batch dispatch feeds the executor: live payloads, or
/// pre-computed outcomes (the host's `BatchDecision::Reject` copy).
enum BatchWork {
    Execute(Vec<ToolCallPayload>),
    Precomputed(Vec<ToolExecutionOutcome>),
}

/// The reference driver over the kernel's ports: frame materialization,
/// bounded model retry, tool batch dispatch through the hook and
/// interaction seams, run control plumbing, and trace construction.
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
    /// A runner with the default [`PassthroughHook`] — no tool-use
    /// filtering (the driver carries no opinion; opt in via
    /// [`TurnRunner::with_hook`]).
    pub fn new(gateway: Arc<dyn ModelGateway>, executor: Arc<ToolExecutor>) -> Self {
        Self {
            gateway,
            executor,
            hook: Arc::new(PassthroughHook),
        }
    }

    /// A runner with a custom [`ToolUseHook`] applied between model
    /// output and tool dispatch (e.g. a
    /// [`FilterChain`](crate::filter::FilterChain)).
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
        let (result, mut trace, tool_calls_total) = self
            .drive(
                &mut context,
                FrameSource::Turn(&options.frame),
                &options,
                &ctrl,
                phase,
            )
            .await;
        // Every drive exit is terminal or paused; the entry owns all
        // bookkeeping (totals, duration, sealing — withheld on pause).
        trace.tool_calls_total = tool_calls_total;
        trace.total_duration_ms = millis_since(start);
        if !matches!(result, TurnResult::Paused { .. }) {
            context.seal();
        }
        TurnOutcome {
            context,
            result,
            trace,
        }
    }

    /// Slice 6 entry: the streaming twin of [`TurnRunner::run`]. The model
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

    /// Slice 2 entry: frames materialize as the lossless merged view over
    /// committed history plus the active turn (Conversation scope — the
    /// `options.frame` policy is deliberately inert here; conversation-level
    /// budget/compaction is Slice 5). Consume/return: the state comes back
    /// with the active turn sealed and outcome-stamped; the host then calls
    /// `commit` (Completed) or `abort_turn` (Interrupted).
    pub async fn run_in_conversation(
        &self,
        state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> Result<ConversationOutcome, ConversationError> {
        self.drive_conversation(state, options, ctrl, ModelPhase::Batch, None)
            .await
    }

    /// Slice 6 entry: the streaming twin of [`TurnRunner::run_in_conversation`]
    /// — same consume/return contract and entry gates, delta-driven model
    /// phase. `options.frame` stays deliberately inert here.
    pub async fn run_in_conversation_streaming(
        &self,
        state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> Result<ConversationOutcome, ConversationError> {
        self.drive_conversation(state, options, ctrl, ModelPhase::Stream, None)
            .await
    }

    /// Slice 7, reworked by Slice 6.5: continue a paused conversation
    /// turn. The continuation half of `run_in_conversation` — same
    /// consume/return contract; the free function
    /// `crate::resume::resume_turn` is the public face: it validates the
    /// complete paused outcome (stamp, open turn, continuation-vs-facts,
    /// decision coverage) and builds the [`ResumeState`] from the
    /// outcome's [`Continuation`].
    pub(crate) async fn resume_conversation(
        &self,
        state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
        resume: ResumeState,
    ) -> Result<ConversationOutcome, ConversationError> {
        // Resumes continue in batch phase: the facts are complete, and a
        // streaming continuation is future work (see the Slice 7 note).
        self.drive_conversation(state, options, ctrl, ModelPhase::Batch, Some(resume))
            .await
    }

    /// Slice 7, reworked by Slice 6.5: continue a paused bare turn — the
    /// continuation half of [`TurnRunner::run`]/[`TurnRunner::run_streaming`].
    /// Consumes the **complete paused outcome** plus the new
    /// [`ResumeRequest`](crate::resume::ResumeRequest) (decision + inject);
    /// the host no longer assembles reason/trace by hand. Validation runs
    /// before anything executes or any fact changes — a rejected request
    /// returns the untouched paused material in the
    /// [`ResumeRejection`](crate::resume::ResumeRejection). Rounds, quota
    /// counts, prepared hook work, and queued inputs all come from the
    /// outcome's [`Continuation`]: a trimmed or empty trace changes
    /// nothing, and lower limits stop the turn before external execution.
    pub async fn resume(
        &self,
        outcome: TurnOutcome,
        request: crate::resume::ResumeRequest,
        options: TurnRunOptions,
        ctrl: RunControl,
    ) -> Result<TurnOutcome, crate::resume::ResumeRejection<TurnOutcome>> {
        if let Err(reason) =
            crate::resume::validate_resume(&outcome.result, &outcome.context, &request)
        {
            return Err(crate::resume::ResumeRejection { reason, outcome });
        }
        let TurnResult::Paused { continuation } = outcome.result else {
            unreachable!("validated above");
        };
        let TurnOutcome {
            mut context, trace, ..
        } = outcome;
        let start = Instant::now();
        let (result, mut trace, tool_calls_total) = self
            .drive_from(
                &mut context,
                FrameSource::Turn(&options.frame),
                &options,
                &ctrl,
                ModelPhase::Batch,
                Some(ResumeState {
                    continuation,
                    decision: request.decision,
                    inject: request.inject,
                    trace,
                }),
            )
            .await;
        trace.tool_calls_total = tool_calls_total;
        trace.total_duration_ms = millis_since(start);
        if !matches!(result, TurnResult::Paused { .. }) {
            context.seal();
        }
        Ok(TurnOutcome {
            context,
            result,
            trace,
        })
    }

    /// The conversation entry body — entry gates, merged-frame source,
    /// terminal bookkeeping, and outcome-stamped sealing shared by both
    /// conversation entries and the resume path.
    async fn drive_conversation(
        &self,
        state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
        phase: ModelPhase,
        resume: Option<ResumeState>,
    ) -> Result<ConversationOutcome, ConversationError> {
        // Observability baseline (Slice 6.6): one `agent.turn` span per
        // entry, ids and scope only — never message content. The span is
        // entered per poll via `Instrument`, so the future stays `Send`.
        let scope = if resume.is_some() {
            "conversation.resume"
        } else if matches!(phase, ModelPhase::Stream) {
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
        self.drive_conversation_inner(state, options, ctrl, phase, resume)
            .instrument(span)
            .await
    }

    /// The conversation entry body — entry gates, merged-frame source,
    /// terminal bookkeeping, and outcome-stamped sealing shared by both
    /// conversation entries and the resume path.
    async fn drive_conversation_inner(
        &self,
        mut state: ConversationState,
        options: TurnRunOptions,
        ctrl: RunControl,
        phase: ModelPhase,
        resume: Option<ResumeState>,
    ) -> Result<ConversationOutcome, ConversationError> {
        // Entry gates — caller bugs fail fast, before the state machine.
        let active_id = match state.active_turn() {
            Some(t) => t.turn_id(),
            None => return Err(ConversationError::NoActiveTurn),
        };
        if state.active_turn().expect("checked above").is_sealed() {
            return Err(ConversationError::TurnAlreadySealed);
        }
        // A paused turn occupies the slot; fresh-driving it would run the
        // same turn twice. Resume (resume_turn) or abort instead.
        if resume.is_none() && state.sealed_result() == Some(SealedResult::Paused) {
            return Err(ConversationError::TurnAlreadyActive);
        }
        // Field-split borrow: read conversation id and history while driving
        // the active turn mutably; stamping happens after the loop through
        // the public `seal_turn`, so no second &mut seam is exposed.
        let (conversation_id, history, active) = state.runner_parts();
        let active = active.expect("NoActiveTurn checked above");
        let start = Instant::now();
        let (result, mut trace, tool_calls_total) = self
            .drive_from(
                active,
                FrameSource::Conversation {
                    conversation_id,
                    history: &history,
                },
                &options,
                &ctrl,
                phase,
                resume,
            )
            .await;
        trace.tool_calls_total = tool_calls_total;
        trace.total_duration_ms = millis_since(start);
        let stamp = match &result {
            TurnResult::Completed { .. } => SealedResult::Completed,
            TurnResult::Interrupted { .. } => SealedResult::Interrupted,
            TurnResult::Paused { .. } => SealedResult::Paused,
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
    /// source, the model phase, and the resume prologue are the only
    /// forks. Every exit is terminal or paused; the entries own
    /// sealing/stamping.
    async fn drive(
        &self,
        active: &mut TurnContext,
        frames: FrameSource<'_>,
        options: &TurnRunOptions,
        ctrl: &RunControl,
        phase: ModelPhase,
    ) -> (TurnResult, TurnTrace, usize) {
        self.drive_from(active, frames, options, ctrl, phase, None)
            .await
    }

    async fn drive_from(
        &self,
        active: &mut TurnContext,
        frames: FrameSource<'_>,
        options: &TurnRunOptions,
        ctrl: &RunControl,
        phase: ModelPhase,
        resume: Option<ResumeState>,
    ) -> (TurnResult, TurnTrace, usize) {
        let mut round: u32;
        let mut tool_calls_total: usize;
        let mut trace: TurnTrace;
        let mut pending_inject: Vec<TextPayload>;
        match resume {
            None => {
                round = 0;
                tool_calls_total = 0;
                trace = TurnTrace::new();
                pending_inject = Vec::new();
            }
            Some(r) => {
                let ResumeState {
                    continuation,
                    decision,
                    inject,
                    trace: resumed_trace,
                } = r;
                let Continuation {
                    pause_point,
                    round: paused_round,
                    accounted_tool_calls,
                    queued_inputs,
                } = continuation;
                round = paused_round;
                tool_calls_total = accounted_tool_calls;
                trace = resumed_trace;
                // Queued inputs land before the request's inject ("queued
                // → injected → pulled"); both commit at the loop top with
                // the `user.steering` label.
                pending_inject = queued_inputs;
                pending_inject.extend(inject);
                // A resumed turn whose control is already spent exits
                // before touching facts.
                if ctrl.should_stop() {
                    let cause = if ctrl.is_cancelled() {
                        TurnInterruption::ExplicitCancellation
                    } else {
                        TurnInterruption::TurnDeadlineExceeded
                    };
                    return (TurnResult::Interrupted { cause }, trace, tool_calls_total);
                }
                // Exhausted limits stop the turn BEFORE external
                // execution: a lowered `max_tool_calls` refuses the
                // withheld batch itself, a lowered `max_model_rounds`
                // refuses the paused round's batch (its results would
                // never reach a model call). Steering pauses re-enter the
                // loop instead, where the boundary checks apply before any
                // model call.
                if tool_calls_total as u64 > options.policy.limits.max_tool_calls as u64 {
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
                if matches!(pause_point, PausePoint::AwaitingApproval { .. })
                    && round >= options.policy.limits.max_model_rounds
                {
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
                // Approval-resume prologue: execute the paused round's
                // batch. The saved hook rejections stay rejected; the new
                // decision covers the remaining awaiting calls exactly
                // once (validated at the entry, re-checked by `run_batch`).
                // The batch was already counted into `accounted_tool_calls`
                // at emission, so the prologue never re-counts it.
                if let Some(decision) = decision {
                    let PausePoint::AwaitingApproval {
                        prepared,
                        deadline: _,
                    } = pause_point
                    else {
                        unreachable!("entry validation pairs a decision with an approval pause");
                    };
                    // The paused batch is exactly the committed-but-
                    // unanswered tool calls of this turn, in block order —
                    // the model-emitted draft order the results pair into.
                    let draft = unanswered_tool_calls(active);
                    let mut rejected = prepared.rejected;
                    rejected.extend(decision.rejected);
                    if let Err(cause) = self
                        .run_batch(
                            active,
                            &draft,
                            BatchWork::Execute(decision.to_execute),
                            rejected,
                            options,
                            ctrl,
                            round,
                            &mut trace,
                        )
                        .await
                    {
                        return (TurnResult::Interrupted { cause }, trace, tool_calls_total);
                    }
                    round += 1;
                }
            }
        }

        // The first model phase of this drive advertises the host-declared
        // surface (the `TurnInvocation` baseline); every later round
        // boundary re-snapshots the runner's executor, so catalog changes
        // — including ones this turn's own tool calls triggered — reach the
        // next request (Slice 10's per-round refresh, consumed here).
        // Retries within one round reuse that round's snapshot.
        let first_model_round = round;

        loop {
            // Steering (Slice 7): resume-time injections first, then the
            // round-boundary pull. Both append with the `user.steering`
            // source label; the next model round sees them.
            if !pending_inject.is_empty() {
                for text in std::mem::take(&mut pending_inject) {
                    if let Err(e) = active.append_input(text, "user.steering") {
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
                }
            }
            for text in options.interaction.pending_inputs().await {
                if let Err(e) = active.append_input(text, "user.steering") {
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
            }
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
            // Observability baseline (Slice 6.6): the round span covers the
            // bounded retry loop; spans are entered per poll via
            // `Instrument`, so the future stays `Send`. Fields carry ids
            // and names only — never frame content or arguments.
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
                                // Cancellation-aware backoff: sleep between
                                // attempts, racing the shared token so a cancel
                                // lands immediately instead of after the wait.
                                let delay = options.policy.retry.backoff_delay(attempt + 1);
                                if !delay.is_zero() {
                                    tokio::select! {
                                        biased;
                                        _ = attempt_ctrl.cancellation_token().cancelled() => {
                                            break Err(ModelInvokeError::new(
                                                ModelInvokeErrorKind::Cancelled,
                                                "cancelled during retry backoff",
                                            ));
                                        }
                                        _ = tokio::time::sleep(delay) => {}
                                    }
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
                    // Emitted calls count at dispatch time — including the
                    // batch still pending behind a pause (the resume
                    // therefore must not re-count it).
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
                    let (hook_to_exec, hook_rejected): (
                        Vec<ToolCallPayload>,
                        Vec<ToolExecutionOutcome>,
                    ) = {
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
                    // Slice 7: the second gate — the host's batch decision.
                    // Default `Proceed`; `Pause` suspends the turn with the
                    // model-emitted batch as `pending_calls`, before any
                    // execution or rejection lands in the facts.
                    let decision = options.interaction.decide_batch(&hook_to_exec).await;
                    match decision {
                        BatchDecision::Pause { deadline } => {
                            // The checkpoint saves the hook's prepared work
                            // (admitted-with-rewrites + rejections) so the
                            // resume never re-runs the hook and never
                            // re-admits a rejected call. The outcome's
                            // context is the single fact source — no
                            // duplicated snapshot rides on the variant.
                            return (
                                TurnResult::Paused {
                                    continuation: Continuation {
                                        pause_point: PausePoint::AwaitingApproval {
                                            prepared: PreparedApproval {
                                                awaiting: hook_to_exec,
                                                rejected: hook_rejected,
                                            },
                                            deadline,
                                        },
                                        round,
                                        accounted_tool_calls: tool_calls_total,
                                        queued_inputs: Vec::new(),
                                    },
                                },
                                trace,
                                tool_calls_total,
                            );
                        }
                        BatchDecision::Proceed => {
                            if let Err(cause) = self
                                .run_batch(
                                    active,
                                    &call_payloads,
                                    BatchWork::Execute(hook_to_exec),
                                    hook_rejected,
                                    options,
                                    ctrl,
                                    round,
                                    &mut trace,
                                )
                                .await
                            {
                                return (
                                    TurnResult::Interrupted { cause },
                                    trace,
                                    tool_calls_total,
                                );
                            }
                        }
                        BatchDecision::Rewrite(rewritten) => {
                            if let Err(cause) = self
                                .run_batch(
                                    active,
                                    &call_payloads,
                                    BatchWork::Execute(rewritten),
                                    hook_rejected,
                                    options,
                                    ctrl,
                                    round,
                                    &mut trace,
                                )
                                .await
                            {
                                return (
                                    TurnResult::Interrupted { cause },
                                    trace,
                                    tool_calls_total,
                                );
                            }
                        }
                        BatchDecision::Reject { results } => {
                            if let Err(cause) = self
                                .run_batch(
                                    active,
                                    &call_payloads,
                                    BatchWork::Precomputed(results),
                                    hook_rejected,
                                    options,
                                    ctrl,
                                    round,
                                    &mut trace,
                                )
                                .await
                            {
                                return (
                                    TurnResult::Interrupted { cause },
                                    trace,
                                    tool_calls_total,
                                );
                            }
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

    /// Execute a tool batch and commit its results — the shared core of
    /// the normal ToolUse dispatch and the approval-resume prologue.
    // Private shared core; every parameter is a distinct axis (facts,
    // order, work, policy, control, round, trace) — grouping would only
    // obscure it.
    #[allow(clippy::too_many_arguments)]
    /// `draft_order` is the model-emitted payload order; the canonical
    /// result order follows it (host rewrites and pre-computed rejects
    /// still pair by `call_id`). The batch trace attaches to the round's
    /// trace entry before committing, so even a
    /// `RunnerInvariantViolation` keeps the observations.
    async fn run_batch(
        &self,
        active: &mut TurnContext,
        draft_order: &[ToolCallPayload],
        work: BatchWork,
        mut rejected: Vec<ToolExecutionOutcome>,
        options: &TurnRunOptions,
        ctrl: &RunControl,
        round: u32,
        trace: &mut TurnTrace,
    ) -> Result<(), TurnInterruption> {
        // Batch completeness is runner policy (the kernel's tool door
        // intentionally accepts partial commits): every model-emitted call
        // must be covered exactly once by the work + rejected decision —
        // no silent drops, no duplicates, no foreign ids. Validated BEFORE
        // dispatch so a broken decision cannot spend side effects on calls
        // the facts can never pair.
        let expected: HashSet<ToolCallId> = draft_order.iter().map(|p| p.call_id.clone()).collect();
        let mut covered: HashSet<ToolCallId> = HashSet::new();
        let work_ids: Vec<&ToolCallId> = match &work {
            BatchWork::Execute(payloads) => payloads.iter().map(|p| &p.call_id).collect(),
            BatchWork::Precomputed(outcomes) => {
                outcomes.iter().map(|o| &o.result.call_id).collect()
            }
        };
        for call_id in work_ids
            .into_iter()
            .chain(rejected.iter().map(|o| &o.result.call_id))
        {
            if !expected.contains(call_id) {
                return Err(TurnInterruption::RunnerInvariantViolation {
                    reason: format!(
                        "batch decision covers call {:?} which the model did not emit",
                        call_id.0
                    ),
                });
            }
            if !covered.insert(call_id.clone()) {
                return Err(TurnInterruption::RunnerInvariantViolation {
                    reason: format!("batch decision covers call {:?} more than once", call_id.0),
                });
            }
        }
        if covered.len() != expected.len() {
            let missing: Vec<String> = expected
                .difference(&covered)
                .map(|id| id.0.clone())
                .collect();
            return Err(TurnInterruption::RunnerInvariantViolation {
                reason: format!(
                    "batch decision does not cover every emitted call: missing {missing:?}"
                ),
            });
        }
        // parallel dispatch; completion order comes from the
        // stream (each future is yielded as it finishes), so no
        // shared log is needed
        let (mut results, completion_order, call_durations): (
            Vec<ToolExecutionOutcome>,
            Vec<ToolCallId>,
            HashMap<ToolCallId, u64>,
        ) = match work {
            BatchWork::Execute(to_exec) => {
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
                let mut results = Vec::with_capacity(stream.len());
                let mut completion_order = Vec::with_capacity(stream.len());
                let mut call_durations: HashMap<ToolCallId, u64> = HashMap::new();
                while let Some((out, duration_ms)) = stream.next().await {
                    completion_order.push(out.result.call_id.clone());
                    call_durations.insert(out.result.call_id.clone(), duration_ms);
                    results.push(out);
                }
                (results, completion_order, call_durations)
            }
            BatchWork::Precomputed(precomputed) => (
                precomputed,
                Vec::new(), // host-supplied: no executor completion order
                HashMap::new(),
            ),
        };
        results.append(&mut rejected);
        // Post-execution identity: every returned outcome must answer one
        // of the batch's own calls — a tool fabricating a foreign id fails
        // loudly here instead of as a kernel pairing error after the fact.
        if let Some(foreign) = results
            .iter()
            .find(|r| !expected.contains(&r.result.call_id))
        {
            return Err(TurnInterruption::RunnerInvariantViolation {
                reason: format!(
                    "tool outcome answers foreign call {:?}",
                    foreign.result.call_id.0
                ),
            });
        }
        // Canonical order = model draft order, taken from the
        // receipt's position — not from ToolCallId encoding. The
        // kernel re-derives the same order from call block
        // sequences when committing.
        let order_index: HashMap<ToolCallId, usize> = draft_order
            .iter()
            .enumerate()
            .map(|(i, p)| (p.call_id.clone(), i))
            .collect();
        results.sort_by_key(|r| order_index.get(&r.result.call_id).copied());
        let tool_names: HashMap<ToolCallId, String> = draft_order
            .iter()
            .map(|p| (p.call_id.clone(), p.tool_name.clone()))
            .collect();
        // attach batch trace before committing so even a
        // RunnerInvariantViolation keeps the observations
        if let Some(rt) = trace
            .rounds
            .last_mut()
            .filter(|rt| rt.round_id == RoundId(round))
        {
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
                        duration_ms: call_durations.get(&r.result.call_id).copied().unwrap_or(0),
                    })
                    .collect(),
                completion_order,
            });
        }
        if let Err(e) =
            active.append_tool_results(results.iter().map(|o| o.result.clone()).collect())
        {
            return Err(TurnInterruption::RunnerInvariantViolation {
                reason: e.to_string(),
            });
        }
        // UnknownOutcome policy: Stop interrupts, Continue proceeds;
        // parent should_stop is checked at the next loop top.
        if let Some(uu) = results.iter().find(|r| {
            r.result.status == ToolResultStatus::UnknownOutcome
                && r.policy == UnknownOutcomePolicy::Stop
        }) {
            return Err(TurnInterruption::UnsafeUnknownOutcome {
                call_id: uu.result.call_id.clone(),
            });
        }
        Ok(())
    }
}
