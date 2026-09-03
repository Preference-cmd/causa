//! `resume_turn` — the continuation half of a paused turn (Slice 7).
//!
//! A pause suspends a turn between the tool-use gate and executor
//! dispatch (or, for steering, before the next model round); resuming
//! re-enters the same drive state machine with the withheld decision.
//! The decision vocabulary is deliberately NOT new: approve / reject /
//! rewrite are exactly the three constructions of the existing
//! `HookOutcome` (Slice 12 Decision 4), so `decide_batch`'s pause and
//! `resume_turn`'s release are two halves of one gate.

use crate::config::TurnRunOptions;
use crate::control::RunControl;
use crate::driver::{ConversationOutcome, PausedReason, ResumeState, TurnRunner, TurnTrace};
use crate::hook::HookOutcome;
use causa_kernel::{ConversationError, ConversationState, SealedResult, TextPayload};

/// The resume payload (Slice 7): everything the continuation needs beyond
/// the runner, the paused state, and the run options. One struct so the
/// entry signatures stay stable as the payload grows.
pub struct ResumeRequest {
    /// The `PausedReason` the pause carried (from
    /// `TurnResult::Paused.reason`).
    pub pending: PausedReason,
    /// The paused phase's trace; resumed rounds append to it and totals
    /// are not reset — pause and resume are two phases of one turn.
    pub trace: TurnTrace,
    /// The withheld approval decision: `HookOutcome` passthrough =
    /// approve, all-rejected = reject, edited `to_execute` = rewrite. It
    /// must cover `pending_calls` (its ids pair with the committed
    /// tool-call blocks; a mismatch surfaces later as a
    /// `RunnerInvariantViolation`).
    pub withheld: HookOutcome,
    /// Steering inputs appended before the next model round (empty for
    /// pure approvals).
    pub inject: Vec<TextPayload>,
}

/// Resume a paused conversation turn.
///
/// - `state` must carry an **open** active turn stamped
///   [`SealedResult::Paused`] — exactly what the driver left behind when
///   the turn paused.
/// - `request` bundles the [`PausedReason`], the paused phase's trace
///   (rounds append, totals persist), the withheld `HookOutcome`
///   decision, and the steering injection — see [`ResumeRequest`].
///
/// Resumes continue in batch phase — the facts are complete, and a
/// streaming continuation is future work (the `TurnInteraction` still
/// receives batch decisions and steering pulls).
pub async fn resume_turn(
    runner: &TurnRunner,
    state: ConversationState,
    request: ResumeRequest,
    options: TurnRunOptions,
    ctrl: RunControl,
) -> Result<ConversationOutcome, ConversationError> {
    let turn_id = match state.active_turn() {
        Some(t) => t.turn_id(),
        None => return Err(ConversationError::NoActiveTurn),
    };
    if state.active_turn().expect("checked above").is_sealed() {
        return Err(ConversationError::TurnAlreadySealed);
    }
    if state.sealed_result() != Some(SealedResult::Paused) {
        return Err(ConversationError::NotPaused(turn_id));
    }
    let ResumeRequest {
        pending,
        trace,
        withheld,
        inject,
    } = request;
    let tool_calls_total = trace.tool_calls_total;
    let resume = match pending {
        PausedReason::AwaitingApproval {
            pending_calls,
            deadline: _,
        } => {
            // The withheld batch belongs to the last traced round — the
            // one the model had just finished when the gate paused it.
            let batch_round = trace.rounds.last().map(|r| r.round_id.0).ok_or_else(|| {
                ConversationError::InvalidSequence(
                    "approval resume requires the paused turn's trace".into(),
                )
            })?;
            ResumeState {
                continue_round: batch_round,
                trace,
                tool_calls_total,
                batch: Some((withheld, pending_calls)),
                inject,
            }
        }
        PausedReason::PausedForSteering {
            pending_round_id,
            queued_inputs: _,
        } => ResumeState {
            continue_round: pending_round_id.0,
            trace,
            tool_calls_total,
            batch: None,
            inject,
        },
    };
    runner
        .resume_conversation(state, options, ctrl, resume)
        .await
}
