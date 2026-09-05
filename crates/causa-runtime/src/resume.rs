//! `resume_turn` — the continuation half of a paused turn (Slice 7,
//! reworked by Slice 6.5): both resume entries consume the **complete
//! paused outcome** plus a request that carries only what is new.
//!
//! A pause suspends a turn between the tool-use gate and executor
//! dispatch (or, for steering, before the next model round). Everything
//! about *where* the turn paused — round, quota counts, the hook's
//! prepared work, queued inputs — is checkpointed in the outcome's
//! [`Continuation`]; the trace is observational and may be trimmed
//! freely. The decision vocabulary is deliberately not new: approve /
//! reject / rewrite are the three constructions of the existing
//! `HookOutcome` (Slice 12 Decision 4), so `decide_batch`'s pause and
//! `resume_turn`'s release are two halves of one gate.
//!
//! Validation runs before anything executes: a rejected request returns
//! the untouched paused material in [`ResumeRejection`] — no model call,
//! no tool execution, no fact appended or sealed. Input errors are never
//! turned into terminal outcomes of the original run.

use std::collections::HashSet;

use causa_kernel::{TextPayload, ToolCallId, TurnContext};

use crate::config::TurnRunOptions;
use crate::control::RunControl;
use crate::conversation::SealedResult;
use crate::driver::{
    Continuation, ConversationOutcome, PausePoint, ResumeState, TurnResult, TurnRunner,
    unanswered_tool_calls,
};
use crate::hook::HookOutcome;

/// The resume payload (Slice 7, reworked by Slice 6.5): only what is NEW
/// at resume time. The host passes the complete paused outcome — the
/// single checkpoint — plus this request.
pub struct ResumeRequest {
    /// The new decision for the calls still awaiting approval: approve =
    /// `HookOutcome::passthrough(awaiting)`, reject = all-rejected,
    /// rewrite = edited `to_execute`. It must cover the continuation's
    /// prepared `awaiting` calls exactly once — hook rejections saved in
    /// the continuation can never be re-decided. `None` only for steering
    /// resumes; a decision on a steering pause is rejected.
    pub decision: Option<HookOutcome>,
    /// New inputs appended before the next model round, after the
    /// continuation's queued inputs (empty for pure approvals). Identical
    /// texts are independent inputs and are never deduped.
    pub inject: Vec<TextPayload>,
}

/// A rejected resume request: the reason plus the complete paused
/// material, returned untouched for correction. Validation happens before
/// any side effect, so the outcome is exactly what the caller handed in.
#[derive(Debug)]
pub struct ResumeRejection<T> {
    /// Why the resume request was rejected.
    pub reason: String,
    /// The complete paused material, exactly as received.
    pub outcome: T,
}

impl<T> ResumeRejection<T> {
    /// The untouched paused material — correct the request and retry.
    pub fn into_outcome(self) -> T {
        self.outcome
    }
}

impl<T> std::fmt::Display for ResumeRejection<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "resume rejected: {}", self.reason)
    }
}

impl<T: std::fmt::Debug> std::error::Error for ResumeRejection<T> {}

/// Resume a paused conversation turn from its complete paused outcome.
///
/// - `outcome.result` must be [`TurnResult::Paused`], the state's active
///   turn open and stamped [`SealedResult::Paused`] — exactly what the
///   driver left behind when the turn paused.
/// - `request` carries the new decision (approval pauses) and the new
///   steering injection; see [`ResumeRequest`].
///
/// Counters continue from the continuation (`round`,
/// `accounted_tool_calls`) — never re-derived from the trace. Lower
/// limits stop the turn before external execution: a lowered
/// `max_tool_calls` refuses the withheld batch itself, a lowered
/// `max_model_rounds` refuses the paused round's batch.
///
/// Resumes continue in batch phase — the facts are complete, and a
/// streaming continuation is future work (the `TurnInteraction` still
/// receives batch decisions and steering pulls).
pub async fn resume_turn(
    runner: &TurnRunner,
    outcome: ConversationOutcome,
    request: ResumeRequest,
    options: TurnRunOptions,
    ctrl: RunControl,
) -> Result<ConversationOutcome, ResumeRejection<ConversationOutcome>> {
    let TurnResult::Paused { .. } = &outcome.result else {
        return Err(ResumeRejection {
            reason: "resume requires a paused outcome".into(),
            outcome,
        });
    };
    // State-shape checks (conversation-only; the bare-turn entry checks
    // the outcome's context instead). Every rejection returns the
    // untouched material.
    let Some(active) = outcome.state.active_turn() else {
        return Err(ResumeRejection {
            reason: "paused outcome has no active turn".into(),
            outcome,
        });
    };
    if active.is_sealed() {
        return Err(ResumeRejection {
            reason: "the paused active turn must be open".into(),
            outcome,
        });
    }
    if outcome.state.sealed_result() != Some(SealedResult::Paused) {
        return Err(ResumeRejection {
            reason: "the conversation state is not stamped Paused".into(),
            outcome,
        });
    }
    if let Err(reason) = validate_resume(
        &outcome.result,
        outcome.state.active_turn().expect("checked above"),
        &request,
    ) {
        return Err(ResumeRejection { reason, outcome });
    }
    let TurnResult::Paused { continuation } = outcome.result else {
        unreachable!("validated above");
    };
    let ConversationOutcome { state, trace, .. } = outcome;
    let resumed = runner
        .resume_conversation(
            state,
            options,
            ctrl,
            ResumeState {
                continuation,
                decision: request.decision,
                inject: request.inject,
                trace,
            },
        )
        .await
        // The entry gates (no active turn / sealed / fresh-run-on-paused)
        // are all covered by the validation above, so the runner cannot
        // reject here.
        .expect("resume validation covered the runner entry gates");
    Ok(resumed)
}

/// Entry-level validation shared by both resume entries: the outcome must
/// be paused, the fact record open, and the continuation consistent with
/// the request and the facts. Returns a clone of the continuation for the
/// driver's `ResumeState`.
pub(crate) fn validate_resume(
    result: &TurnResult,
    facts: &TurnContext,
    request: &ResumeRequest,
) -> Result<Continuation, String> {
    let TurnResult::Paused { continuation } = result else {
        return Err("resume requires a paused outcome".into());
    };
    if facts.is_sealed() {
        return Err("the paused turn must be open to resume".into());
    }
    validate_continuation(continuation, request, facts)?;
    Ok(continuation.clone())
}

/// Continuation-vs-request-vs-facts validation: the pause point pairs
/// with the decision vocabulary, the decision covers the prepared batch
/// exactly once (no foreign ids, no duplicates, no omissions), and the
/// continuation's prepared work is exactly the turn's unanswered tool
/// calls.
pub(crate) fn validate_continuation(
    continuation: &Continuation,
    request: &ResumeRequest,
    facts: &TurnContext,
) -> Result<(), String> {
    let unanswered = unanswered_tool_calls(facts);
    match &continuation.pause_point {
        PausePoint::AwaitingApproval { prepared, .. } => {
            let decision = request
                .decision
                .as_ref()
                .ok_or_else(|| "an approval pause requires a decision (HookOutcome)".to_string())?;
            // The decision covers the awaiting calls exactly once.
            let awaiting: HashSet<&ToolCallId> =
                prepared.awaiting.iter().map(|p| &p.call_id).collect();
            let mut covered: HashSet<&ToolCallId> = HashSet::new();
            for id in decision
                .to_execute
                .iter()
                .map(|p| &p.call_id)
                .chain(decision.rejected.iter().map(|o| &o.result.call_id))
            {
                if !awaiting.contains(id) {
                    return Err(format!(
                        "decision covers call {:?} which is not awaiting approval",
                        id.0
                    ));
                }
                if !covered.insert(id) {
                    return Err(format!("decision covers call {:?} more than once", id.0));
                }
            }
            if covered.len() != awaiting.len() {
                let missing: Vec<String> = awaiting
                    .difference(&covered)
                    .map(|id| id.0.clone())
                    .collect();
                return Err(format!(
                    "decision does not cover every awaiting call: missing {missing:?}"
                ));
            }
            // The checkpoint's prepared work must be exactly the turn's
            // unanswered calls: awaiting plus saved hook rejections.
            let mut checkpoint: HashSet<&ToolCallId> = awaiting;
            checkpoint.extend(prepared.rejected.iter().map(|o| &o.result.call_id));
            let fact_ids: HashSet<&ToolCallId> = unanswered.iter().map(|p| &p.call_id).collect();
            if checkpoint != fact_ids {
                return Err("continuation does not match the turn's unanswered tool calls".into());
            }
            Ok(())
        }
        PausePoint::PausedForSteering => {
            if request.decision.is_some() {
                return Err("a steering pause takes no decision".into());
            }
            if !unanswered.is_empty() {
                return Err("steering continuation would leave unanswered tool calls".into());
            }
            Ok(())
        }
    }
}
