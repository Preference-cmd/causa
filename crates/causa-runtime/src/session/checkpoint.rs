//! The versioned save envelope for one material-free session and the
//! comparable description of the configuration it was produced under.
//!
//! The envelope is a plain data interface: [`Session`] produces it, a host
//! persists it (harness / examples — the runtime ships no store), and
//! [`SessionCheckpoint::restore`] consumes a re-assembled configuration plus
//! the loaded envelope. Phase C covers only the sessions Phase A/B support —
//! no seed, no projected materials, no per-work tool binding. When later
//! phases extend the envelope, they must declare version compatibility or a
//! migration rule; this format never silently guesses at unknown sections.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use causa_kernel::{CacheDirective, ContentPart, ConversationId, GenerationOptions, ToolSurface};

use crate::budget::WindowBudget;
use crate::config::{RetryPolicy, ToolOutputLimits, TurnLimits, UnknownOutcomeConfig};
use crate::conversation::ConversationState;
use crate::driver::ConversationOutcome;
use crate::resume::ResumeRequest;

use super::types::{CancelReceipt, FinishedKind, SessionConfig, WorkReceipt, WorkRef, WorkState};

/// The envelope format version written by [`Session::checkpoint`].
pub const SESSION_CHECKPOINT_VERSION: u32 = 1;

/// The complete save envelope of one session — everything restore needs
/// beyond freshly assembled capabilities, at an explicit format version.
///
/// One phase payload only: an `Idle` envelope carries the
/// [`ConversationState`]; a `Paused` envelope carries the **complete**
/// [`ConversationOutcome`], whose state is the only saved active context.
/// The envelope is never a second, drifting copy of a live session's state —
/// it is a value the host persists and later hands back.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct SessionCheckpoint {
    /// The envelope format version — [`SESSION_CHECKPOINT_VERSION`] for
    /// envelopes this runtime writes. Restore rejects any version it does
    /// not implement instead of guessing at unknown shapes.
    pub version: u32,
    /// The conversation the envelope belongs to. Restore registers it as-is;
    /// collision with a still-held conversation id is the harness's check
    /// (the session has no global instance table).
    pub conversation_id: ConversationId,
    /// The saved conversation phase — the Idle state or the complete paused
    /// outcome. Exactly one is present, and the paused variant carries its
    /// state inside the outcome.
    pub phase: CheckpointPhase,
    /// Every retained work: refs, revisions, terminal results, faults, and
    /// original deadlines. Terminal results stay observable after restore;
    /// a restored work is never a new work.
    pub works: Vec<SavedWork>,
    /// The session's TurnId allocation progress at save time, so restore
    /// never hands out an identity a previous owner already used.
    pub next_turn: u64,
    /// The accepted submit requests, replayable by key after restore.
    pub submit_keys: Vec<SavedSubmitKey>,
    /// The accepted resume requests, replayable by key after restore.
    pub resume_keys: Vec<SavedResumeKey>,
    /// The accepted cancel requests, replayable by key after restore.
    pub cancel_keys: Vec<SavedCancelKey>,
    /// The comparable description of the configuration the session ran
    /// under; restore validates it against the freshly assembled options.
    pub config: SessionConfigDescription,
    /// Whether the owner had closed the session at save time. Restoring a
    /// closed envelope yields a closed session: accepted keys still replay,
    /// new work is refused — the closed state is preserved, not upgraded.
    pub closed: bool,
}

/// The saved conversation phase: the idle state, or the complete paused
/// outcome plus the work ref it belongs to.
///
/// The `Paused` arm is deliberately the complete outcome (the same shape the
/// in-memory slot holds), so the enum is as large as that outcome. Boxing it
/// would add indirection for a value there is exactly one of, matching the
/// accepted trade-off on the session's `Slot`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
#[non_exhaustive]
#[allow(clippy::large_enum_variant)]
pub enum CheckpointPhase {
    /// The session was idle: the state (with its committed history) is the
    /// whole conversation payload; its active slot is empty.
    Idle {
        /// The idle conversation state.
        state: ConversationState,
    },
    /// The session held a paused work: the complete outcome (its own state,
    /// the paused result, and the continuation) is the whole payload. No
    /// second state copy is saved beside it.
    Paused {
        /// The complete paused outcome, including its state and continuation.
        outcome: ConversationOutcome,
        /// The paused work the outcome belongs to.
        work: WorkRef,
    },
}

/// One retained work's saved view: everything `observe` reported plus the
/// absolute deadline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedWork {
    /// The work's identity (conversation + assigned turn id).
    pub work: WorkRef,
    /// The work's revision at save time; restore continues the sequence
    /// from here, so a caller's pre-save revision stays meaningful.
    pub revision: u64,
    /// The work's state at save time — `Paused` only in the paused phase
    /// payload's work; `Accepted` / `Running` cannot be checkpointed.
    pub state: WorkState,
    /// The terminal result when the work is `Finished`.
    pub finished: Option<FinishedKind>,
    /// The fault reason when the work is `Faulted`.
    pub fault: Option<String>,
    /// The work's deadline as an absolute UTC expiry. Monotonic deadlines
    /// have no wall-clock meaning across processes, so the remaining time is
    /// re-derived at restore — never re-granted in full.
    pub deadline_utc: Option<SystemTime>,
}

/// One accepted submit under its request key, saved for replay.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedSubmitKey {
    /// The caller's idempotency key.
    pub key: String,
    /// The parts the request was accepted for; a same-key retry with
    /// different parts is a conflict after restore, exactly as before it.
    pub parts: Vec<ContentPart>,
    /// The original acceptance receipt.
    pub receipt: WorkReceipt,
}

/// One accepted resume under its request key, saved for replay.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedResumeKey {
    /// The caller's idempotency key.
    pub key: String,
    /// The work the resume targeted.
    pub work: WorkRef,
    /// The paused revision the resume named.
    pub revision: u64,
    /// The request it was accepted for (decision + injected inputs).
    pub request: ResumeRequest,
}

/// One accepted cancel under its request key, saved for replay.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedCancelKey {
    /// The caller's idempotency key.
    pub key: String,
    /// The work the cancel targeted.
    pub work: WorkRef,
    /// The original cancel receipt.
    pub receipt: CancelReceipt,
}

/// The comparable description of one assembled configuration: the plain-data
/// projection of [`TurnRunOptions`] plus the session coordination config.
///
/// Restore re-derives the description from the newly assembled options and
/// rejects a mismatch — the same model ref, tool surface, and execution
/// options must be in force, so a checkpoint cannot silently resume under a
/// swapped model or narrowed surface. Ports the description cannot compare
/// by value (the interaction seam, the gateway, the executor, the concrete
/// compaction / counter implementations behind the presence flags) stay the
/// harness's assembly responsibility: this description pins *what was
/// configured*, not the identity of behavior objects behind it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct SessionConfigDescription {
    /// The model the runs invoke (`ModelRef` as its raw string).
    pub model: String,
    /// The tool surface advertised on the first model phase.
    pub tool_surface: ToolSurface,
    /// Sampling parameters sent with every attempt.
    pub generation: GenerationOptions,
    /// Prompt-cache instruction.
    pub cache: CacheDirective,
    /// Retry schedule for failed model attempts.
    pub retry: RetryPolicy,
    /// Per-turn loop bounds.
    pub limits: TurnLimits,
    /// Per-model-attempt budget.
    pub attempt_timeout: Option<Duration>,
    /// Unknown-outcome continuation configuration.
    pub unknown_outcome: UnknownOutcomeConfig,
    /// Fallback tool-output truncation limit.
    pub tool_output_limits: ToolOutputLimits,
    /// Per-tool-name truncation-limit overrides.
    pub tool_output_limits_overrides: HashMap<String, ToolOutputLimits>,
    /// Per-tool-call deadline.
    pub call_timeout: Option<Duration>,
    /// Frame-compaction trigger thresholds.
    pub window_budget: WindowBudget,
    /// Whether a compaction implementation is configured.
    pub compaction_configured: bool,
    /// Whether a frame token counter is configured.
    pub frame_counter_configured: bool,
    /// Whether a tool-output artifact store is configured.
    pub artifact_store_configured: bool,
    /// Whether a truncation-estimation counter is configured.
    pub truncation_counter_configured: bool,
    /// The session coordination config (retention capacity, work deadline).
    pub session: SessionConfig,
}

impl SessionConfigDescription {
    /// Describe an assembled configuration — the projection restore compares
    /// against, and the description [`Session::checkpoint`] records.
    pub fn of(options: &crate::config::TurnRunOptions, config: &SessionConfig) -> Self {
        Self {
            model: options.invocation.model.0.clone(),
            tool_surface: options.invocation.tool_surface.clone(),
            generation: options.invocation.generation.clone(),
            cache: options.invocation.cache,
            retry: options.policy.retry.clone(),
            limits: options.policy.limits.clone(),
            attempt_timeout: options.policy.attempt_timeout,
            unknown_outcome: options.policy.unknown_outcome.clone(),
            tool_output_limits: options.execution.tool_output_limits,
            tool_output_limits_overrides: options.execution.tool_output_limits_overrides.clone(),
            call_timeout: options.execution.call_timeout,
            window_budget: options.frame.window_budget,
            compaction_configured: options.frame.compaction.is_some(),
            frame_counter_configured: options.frame.token_counter.is_some(),
            artifact_store_configured: options.execution.artifact_store.is_some(),
            truncation_counter_configured: options.execution.token_counter.is_some(),
            session: config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TurnRunOptions;
    use crate::conversation::SealedResult;
    use crate::driver::{Continuation, PausePoint, TurnResult, TurnTrace};
    use causa_kernel::{TextPayload, TurnId};

    #[test]
    fn envelope_roundtrips_through_json_without_shape_drift() {
        let conversation_id = ConversationId("conv-c1".into());
        let turn_id = TurnId::new("conv-c1-work-0");
        let checkpoint = SessionCheckpoint {
            version: SESSION_CHECKPOINT_VERSION,
            conversation_id: conversation_id.clone(),
            phase: CheckpointPhase::Paused {
                outcome: paused_outcome(&conversation_id, &turn_id),
                work: WorkRef {
                    conversation_id: conversation_id.clone(),
                    turn_id: turn_id.clone(),
                },
            },
            works: vec![SavedWork {
                work: WorkRef {
                    conversation_id: conversation_id.clone(),
                    turn_id,
                },
                revision: 1,
                state: WorkState::Paused,
                finished: None,
                fault: None,
                deadline_utc: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(42)),
            }],
            next_turn: 1,
            submit_keys: vec![SavedSubmitKey {
                key: "s1".into(),
                parts: vec![ContentPart::Text(TextPayload::new("hello"))],
                receipt: WorkReceipt {
                    work: WorkRef {
                        conversation_id: conversation_id.clone(),
                        turn_id: TurnId::new("conv-c1-work-0"),
                    },
                    accepted_revision: 0,
                },
            }],
            resume_keys: vec![],
            cancel_keys: vec![],
            config: SessionConfigDescription::of(
                &TurnRunOptions::default(),
                &SessionConfig::default(),
            ),
            closed: false,
        };

        let json = serde_json::to_value(&checkpoint).expect("envelope serializes");
        let back: SessionCheckpoint =
            serde_json::from_value(json.clone()).expect("envelope deserializes");
        assert_eq!(serde_json::to_value(&back).unwrap(), json);
        assert_eq!(back.version, SESSION_CHECKPOINT_VERSION);
        assert_eq!(back.next_turn, 1);
        assert!(!back.closed);
        let CheckpointPhase::Paused { work, .. } = &back.phase else {
            panic!("the paused phase survives the roundtrip");
        };
        assert_eq!(work.conversation_id, conversation_id);
    }

    /// A minimal paused outcome: one open turn, sealed and stamped `Paused`,
    /// with a steering continuation and no saved batches.
    fn paused_outcome(conversation_id: &ConversationId, turn_id: &TurnId) -> ConversationOutcome {
        let mut state = ConversationState::new(conversation_id.clone());
        state.begin_turn(turn_id.clone()).expect("the turn admits");
        state
            .active_turn_mut()
            .expect("begin_turn admitted an active turn")
            .append_input(TextPayload::new("hi"), "user")
            .expect("the input appends");
        state
            .seal_turn(turn_id.clone(), SealedResult::Paused)
            .expect("the turn seals");
        ConversationOutcome {
            state,
            result: TurnResult::Paused {
                continuation: Continuation {
                    pause_point: PausePoint::PausedForSteering,
                    round: 0,
                    accounted_tool_calls: 0,
                    queued_inputs: Vec::new(),
                },
            },
            trace: TurnTrace::new(),
        }
    }

    #[test]
    fn the_config_description_pins_the_options_it_was_derived_from() {
        let mut options = TurnRunOptions::default();
        options.invocation.model = causa_kernel::ModelRef::new("m1");
        options.policy.limits.max_model_rounds = 3;
        let config = SessionConfig {
            retained_work_capacity: 8,
            work_deadline: Some(Duration::from_secs(5)),
        };

        let described = SessionConfigDescription::of(&options, &config);
        assert_eq!(described.model, "m1");
        assert_eq!(described.limits.max_model_rounds, 3);
        assert_eq!(described.session, config);
        assert_eq!(described, SessionConfigDescription::of(&options, &config));

        let mut swapped = options.clone();
        swapped.invocation.model = causa_kernel::ModelRef::new("m2");
        assert_ne!(
            described,
            SessionConfigDescription::of(&swapped, &config),
            "a swapped model is visible to the comparison"
        );
    }
}
