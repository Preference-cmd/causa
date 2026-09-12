//! Private execution core for `crate::session` — the work registry, the
//! single execution slot, the state-change epoch, and the worker task.
//!
//! Nothing here is public: the module root owns the public vocabulary and
//! this module owns the coordination. One owner holds one execution slot —
//! whose `Running` arm has already handed the [`ConversationState`] to the
//! runner loop — plus one published work view per accepted work.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use futures_util::FutureExt;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use causa_kernel::{ContentPart, ConversationId, TurnId, TurnSnapshot};

use crate::config::TurnRunOptions;
use crate::control::RunControl;
use crate::conversation::ConversationState;
use crate::driver::{Continuation, ConversationOutcome, TurnInterruption, TurnResult, TurnRunner};
use crate::resume::ResumeRequest;

use super::checkpoint::{
    SESSION_CHECKPOINT_VERSION, SavedCancelKey, SavedResumeKey, SavedSubmitKey, SavedWork,
    SessionCheckpoint, SessionConfigDescription,
};
use super::{
    CancelOutcome, CancelReceipt, FinishedKind, SessionConfig, SessionError, SubmitRequest,
    WaitEnd, WaitOutcome, WorkObservation, WorkReceipt, WorkRef, WorkState,
};

/// The published view of one retained work.
struct WorkEntry {
    revision: u64,
    state: WorkState,
    finished: Option<FinishedKind>,
    fault: Option<String>,
    /// The per-work deadline measured from acceptance; it survives a pause, so
    /// a resume continues under the same absolute bound rather than a fresh
    /// one.
    deadline: Option<Instant>,
}

/// The single execution slot — the one place the writable conversation lives
/// (or is suspended).
///
/// The `Paused` arm is deliberately the complete `ConversationOutcome` (its
/// state, result, and trace), which makes the enum as large as that outcome.
/// Boxing it would add indirection for a value there is exactly one of, so
/// the size difference is accepted rather than papered over.
#[allow(clippy::large_enum_variant)]
enum Slot {
    /// The idle conversation; a `submit` may begin a turn in it.
    Idle(ConversationState),
    /// A work is running: the state has moved into the worker task, which is
    /// the only place it is writable. The token is the session's stop signal.
    Running {
        work: WorkRef,
        token: CancellationToken,
    },
    /// A paused work's **complete** outcome, including its state and
    /// continuation. Never a second writable `ConversationState`.
    Paused(ConversationOutcome),
    /// The worker exited abnormally; no complete outcome exists.
    Faulted { reason: String },
}

impl Slot {
    /// The admission error for a non-idle slot, or `None` when a `submit`
    /// may begin a turn. One active work per conversation: no queue, no
    /// steering, no implicit approval.
    fn admission_error(&self, id: &ConversationId) -> Option<SessionError> {
        match self {
            Slot::Idle(_) => None,
            Slot::Running { work, .. } => Some(SessionError::Busy {
                active: work.clone(),
            }),
            Slot::Paused(outcome) => Some(SessionError::Busy {
                active: WorkRef {
                    conversation_id: id.clone(),
                    turn_id: paused_turn_id(outcome),
                },
            }),
            Slot::Faulted { reason } => Some(SessionError::Faulted {
                reason: reason.clone(),
            }),
        }
    }

    /// Borrow the paused outcome for `work`, or explain why it cannot
    /// resume. A `Running` slot is `Busy` (another work owns the
    /// conversation); anything else holding no resumable material is
    /// `NotPaused`.
    fn paused_for(&self, work: &WorkRef) -> Result<&ConversationOutcome, SessionError> {
        match self {
            Slot::Paused(outcome) if paused_turn_id(outcome) == work.turn_id => Ok(outcome),
            Slot::Running { work: active, .. } => Err(SessionError::Busy {
                active: active.clone(),
            }),
            _ => Err(SessionError::NotPaused(work.clone())),
        }
    }

    /// Move the paused outcome for `work` out of the slot, leaving it
    /// `Running` under the caller's new token. The single match validates
    /// and extracts together, so no second resume can interleave between
    /// the check and the handoff.
    fn take_paused_for(
        &mut self,
        work: &WorkRef,
        token: CancellationToken,
    ) -> Result<ConversationOutcome, SessionError> {
        match self {
            Slot::Paused(outcome) if paused_turn_id(outcome) == work.turn_id => {
                let previous = std::mem::replace(
                    self,
                    Slot::Running {
                        work: work.clone(),
                        token,
                    },
                );
                let Slot::Paused(outcome) = previous else {
                    unreachable!("the paused arm was matched above");
                };
                Ok(outcome)
            }
            Slot::Running { work: active, .. } => Err(SessionError::Busy {
                active: active.clone(),
            }),
            _ => Err(SessionError::NotPaused(work.clone())),
        }
    }

    /// Terminate the paused `work` in place with no new external call:
    /// abort its active turn and leave the slot `Idle` with the aborted
    /// state. Returns the aborted facts plus the retained continuation.
    /// A rejection leaves the slot untouched.
    fn stop_paused(
        &mut self,
        work: &WorkRef,
    ) -> Result<(TurnSnapshot, Continuation), SessionError> {
        match self {
            Slot::Paused(outcome) if paused_turn_id(outcome) == work.turn_id => {}
            Slot::Running { work: active, .. } => {
                return Err(SessionError::Busy {
                    active: active.clone(),
                });
            }
            _ => return Err(SessionError::NotPaused(work.clone())),
        }
        // The guard above established the arm; the placeholder below is the
        // only safe-Rust way to move the aborted state out from behind
        // `&mut` (the transient is never observable — the real state is
        // assigned before the lock is released).
        let previous = std::mem::replace(
            self,
            Slot::Idle(ConversationState::new(work.conversation_id.clone())),
        );
        let Slot::Paused(outcome) = previous else {
            unreachable!("the paused arm was matched above");
        };
        let ConversationOutcome {
            mut state, result, ..
        } = outcome;
        let TurnResult::Paused { continuation } = result else {
            unreachable!("a paused slot holds a paused result");
        };
        let facts = state
            .abort_turn(work.turn_id.clone())
            .expect("the paused active turn aborts")
            .snapshot();
        *self = Slot::Idle(state);
        Ok((facts, continuation))
    }
}

/// One accepted `submit` under its request key: the receipt plus the parts it
/// was accepted for, so a same-key retry can be replayed or reported as
/// `Conflict`.
struct SubmitKey {
    receipt: WorkReceipt,
    parts: Vec<ContentPart>,
}

/// One accepted `resume` under its request key: the receipt plus the work,
/// the paused revision it named, and the request it was accepted for — a
/// same-key repeat with a different argument set is `Conflict`, not a second
/// execution.
struct ResumeKey {
    receipt: WorkReceipt,
    work: WorkRef,
    revision: u64,
    request: ResumeRequest,
}

/// One accepted `cancel` under its request key: the receipt plus the work it
/// targeted.
struct CancelKey {
    receipt: CancelReceipt,
    work: WorkRef,
}

/// The lock-protected registry. Request keys dedup per operation — each
/// operation keeps its own table, so one `request_key` string may be reused
/// across operations without colliding, and every lookup is precisely typed
/// with no operation tag to re-assert.
struct Inner {
    slot: Slot,
    works: HashMap<WorkRef, WorkEntry>,
    submit_keys: HashMap<String, SubmitKey>,
    resume_keys: HashMap<String, ResumeKey>,
    cancel_keys: HashMap<String, CancelKey>,
    next_turn: u64,
    closed: bool,
}

/// The shared coordination core. Handles hold `Arc<SessionCore>`; the worker
/// holds one too (never a `JoinHandle`), so no strong reference cycle forms and
/// the worker's terminal publish still lands after the owner is dropped.
pub(super) struct SessionCore {
    id: ConversationId,
    runner: Arc<TurnRunner>,
    options: TurnRunOptions,
    config: SessionConfig,
    inner: Mutex<Inner>,
    /// Monotonic state-change epoch. Waiters subscribe first, then observe,
    /// then block on `changed()` — the subscribe-before-observe order is what
    /// makes the wait lost-wakeup safe.
    epoch: watch::Sender<u64>,
}

impl SessionCore {
    /// Build the core around an already-validated idle state.
    pub(super) fn new(
        state: ConversationState,
        runner: Arc<TurnRunner>,
        options: TurnRunOptions,
        config: SessionConfig,
    ) -> Arc<Self> {
        let id = state.conversation_id().clone();
        let (epoch, _initial_receiver) = watch::channel(0_u64);
        Arc::new(Self {
            id,
            runner,
            options,
            config,
            inner: Mutex::new(Inner {
                slot: Slot::Idle(state),
                works: HashMap::new(),
                submit_keys: HashMap::new(),
                resume_keys: HashMap::new(),
                cancel_keys: HashMap::new(),
                next_turn: 0,
                closed: false,
            }),
            epoch,
        })
    }

    /// The conversation this core owns.
    pub(super) fn conversation_id(&self) -> ConversationId {
        self.id.clone()
    }

    /// Export the versioned save envelope — the complete idle state or the
    /// complete paused outcome, every retained work, the request-key replay
    /// tables, the allocation progress, and the configuration description.
    ///
    /// Only a quiescent slot exports: an accepted / running work (a cancel
    /// in progress included) is `Busy` and a faulted session refuses — no
    /// complete outcome exists to save. A closed session still exports; the
    /// envelope records `closed` so restoring it preserves the closed
    /// acceptance state.
    pub(super) fn checkpoint(&self) -> Result<SessionCheckpoint, SessionError> {
        let inner = self.lock();
        let phase = match &inner.slot {
            Slot::Idle(state) => super::checkpoint::CheckpointPhase::Idle {
                state: state.clone(),
            },
            Slot::Paused(outcome) => super::checkpoint::CheckpointPhase::Paused {
                outcome: outcome.clone(),
                work: WorkRef {
                    conversation_id: self.id.clone(),
                    turn_id: paused_turn_id(outcome),
                },
            },
            Slot::Running { work, .. } => {
                return Err(SessionError::Busy {
                    active: work.clone(),
                });
            }
            Slot::Faulted { reason } => {
                return Err(SessionError::Faulted {
                    reason: reason.clone(),
                });
            }
        };
        // Deterministic record order: two exports of the same quiescent
        // state list their records identically. The envelopes still differ
        // whenever any retained work carries a deadline — each export
        // re-anchors that work's absolute expiry to the export moment — so
        // byte equality only holds for deadline-free states.
        let mut works: Vec<SavedWork> = inner
            .works
            .iter()
            .map(|(work, entry)| SavedWork {
                work: work.clone(),
                revision: entry.revision,
                state: entry.state,
                finished: entry.finished.clone(),
                fault: entry.fault.clone(),
                deadline_utc: entry.deadline.map(absolute_expiry),
            })
            .collect();
        works.sort_by(|a, b| a.work.turn_id.0.cmp(&b.work.turn_id.0));
        let mut submit_keys: Vec<SavedSubmitKey> = inner
            .submit_keys
            .iter()
            .map(|(key, record)| SavedSubmitKey {
                key: key.clone(),
                parts: record.parts.clone(),
                receipt: record.receipt.clone(),
            })
            .collect();
        submit_keys.sort_by(|a, b| a.key.cmp(&b.key));
        let mut resume_keys: Vec<SavedResumeKey> = inner
            .resume_keys
            .iter()
            .map(|(key, record)| SavedResumeKey {
                key: key.clone(),
                work: record.work.clone(),
                revision: record.revision,
                request: record.request.clone(),
            })
            .collect();
        resume_keys.sort_by(|a, b| a.key.cmp(&b.key));
        let mut cancel_keys: Vec<SavedCancelKey> = inner
            .cancel_keys
            .iter()
            .map(|(key, record)| SavedCancelKey {
                key: key.clone(),
                work: record.work.clone(),
                receipt: record.receipt.clone(),
            })
            .collect();
        cancel_keys.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(SessionCheckpoint {
            version: SESSION_CHECKPOINT_VERSION,
            conversation_id: self.id.clone(),
            phase,
            works,
            next_turn: inner.next_turn,
            submit_keys,
            resume_keys,
            cancel_keys,
            config: SessionConfigDescription::of(&self.options, &self.config),
            closed: inner.closed,
        })
    }

    /// Rebuild the core from a validated checkpoint's idle state: the saved
    /// works, key tables, allocation progress, and closed state register
    /// as-is — the caller (restore) validated material consistency first.
    pub(super) fn restore_idle(
        state: ConversationState,
        registry: super::checkpoint::SavedRegistry,
        runner: Arc<TurnRunner>,
        options: TurnRunOptions,
        config: SessionConfig,
    ) -> Arc<Self> {
        let id = state.conversation_id().clone();
        Self::assemble(id, Slot::Idle(state), registry, runner, options, config)
    }

    /// Rebuild the core around a validated checkpoint's paused outcome. The
    /// outcome keeps its open active turn and continuation; the registered
    /// work stays `Paused` and advances only on an explicit resume.
    pub(super) fn restore_paused(
        outcome: ConversationOutcome,
        registry: super::checkpoint::SavedRegistry,
        runner: Arc<TurnRunner>,
        options: TurnRunOptions,
        config: SessionConfig,
    ) -> Arc<Self> {
        let id = outcome.state.conversation_id().clone();
        Self::assemble(id, Slot::Paused(outcome), registry, runner, options, config)
    }

    /// The shared restore assembly: convert the envelope's records into the
    /// live registry. Deadlines come back as the remaining time derived from
    /// the saved absolute expiry — never the original duration.
    fn assemble(
        id: ConversationId,
        slot: Slot,
        registry: super::checkpoint::SavedRegistry,
        runner: Arc<TurnRunner>,
        options: TurnRunOptions,
        config: SessionConfig,
    ) -> Arc<Self> {
        let super::checkpoint::SavedRegistry {
            works,
            submit_keys,
            resume_keys,
            cancel_keys,
            next_turn,
            closed,
        } = registry;
        let works = works
            .into_iter()
            .map(|saved| {
                let entry = WorkEntry {
                    revision: saved.revision,
                    state: saved.state,
                    finished: saved.finished,
                    fault: saved.fault,
                    deadline: saved.deadline_utc.map(remaining_deadline),
                };
                (saved.work, entry)
            })
            .collect();
        let submit_keys = submit_keys
            .into_iter()
            .map(|saved| {
                (
                    saved.key,
                    SubmitKey {
                        receipt: saved.receipt,
                        parts: saved.parts,
                    },
                )
            })
            .collect();
        let resume_keys = resume_keys
            .into_iter()
            .map(|saved| {
                (
                    saved.key,
                    ResumeKey {
                        // The resume receipt is exactly (work, accepted
                        // revision) — the saved pair reconstructs it without
                        // a separate envelope field.
                        receipt: WorkReceipt {
                            work: saved.work.clone(),
                            accepted_revision: saved.revision,
                        },
                        work: saved.work,
                        revision: saved.revision,
                        request: saved.request,
                    },
                )
            })
            .collect();
        let cancel_keys = cancel_keys
            .into_iter()
            .map(|saved| {
                (
                    saved.key,
                    CancelKey {
                        receipt: saved.receipt,
                        work: saved.work,
                    },
                )
            })
            .collect();
        let (epoch, _initial_receiver) = watch::channel(0_u64);
        Arc::new(Self {
            id,
            runner,
            options,
            config,
            inner: Mutex::new(Inner {
                slot,
                works,
                submit_keys,
                resume_keys,
                cancel_keys,
                next_turn,
                closed,
            }),
            epoch,
        })
    }

    /// Stop accepting work and send the active work's stop signal.
    pub(super) fn close(&self) {
        let mut inner = self.lock();
        inner.closed = true;
        if let Slot::Running { token, .. } = &inner.slot {
            token.cancel();
        }
        drop(inner);
        self.bump_epoch();
    }

    /// Accept one work, or reject it without side effects.
    pub(super) fn submit(
        self: &Arc<Self>,
        request: SubmitRequest,
    ) -> Result<WorkReceipt, SessionError> {
        let mut inner = self.lock();
        // Local dedup is checked before the busy and closed guards: a retry of
        // a lost receipt must resolve to the original receipt even while the
        // work runs and after the session closed. A key that was never accepted
        // (because a submit was rejected) is absent here, so fixing a rejected
        // submit keeps the key free.
        if let Some(record) = inner.submit_keys.get(&request.request_key) {
            return if record.parts == request.parts {
                Ok(record.receipt.clone())
            } else {
                Err(SessionError::Conflict)
            };
        }
        if inner.closed {
            return Err(SessionError::Closed);
        }
        if let Some(error) = inner.slot.admission_error(&self.id) {
            return Err(error);
        }
        if inner.works.len() >= self.config.retained_work_capacity {
            return Err(SessionError::CapacityExceeded);
        }
        if request.parts.is_empty() {
            return Err(SessionError::InvalidInput(
                "submit requires at least one content part".into(),
            ));
        }

        let (turn_id, next_turn) = self.fresh_turn_id(&inner);
        let Slot::Idle(state) = &mut inner.slot else {
            unreachable!("idle slot checked above");
        };
        if let Err(error) = state.begin_turn(turn_id.clone()) {
            return Err(SessionError::InvalidInput(format!(
                "begin_turn rejected the new work: {error}"
            )));
        }
        if let Err(error) = state
            .active_turn_mut()
            .expect("begin_turn just admitted an active turn")
            .append_parts(request.parts.clone(), "user")
        {
            // A rejected append must not leave an empty active turn behind.
            let _ = state.abort_turn(turn_id.clone());
            return Err(SessionError::InvalidInput(format!(
                "submit parts rejected: {error}"
            )));
        }
        inner.next_turn = next_turn;

        let token = CancellationToken::new();
        let deadline = self.config.work_deadline.map(|d| Instant::now() + d);
        let ctrl = RunControl::new(token.clone(), deadline);
        let work = WorkRef {
            conversation_id: self.id.clone(),
            turn_id: turn_id.clone(),
        };
        let receipt = WorkReceipt {
            work: work.clone(),
            accepted_revision: 0,
        };
        inner.works.insert(
            work.clone(),
            WorkEntry {
                revision: 0,
                state: WorkState::Accepted,
                finished: None,
                fault: None,
                deadline,
            },
        );
        inner.submit_keys.insert(
            request.request_key,
            SubmitKey {
                receipt: receipt.clone(),
                parts: request.parts,
            },
        );

        // Exactly one writable ConversationState: move it into the worker.
        let Slot::Idle(state) = std::mem::replace(
            &mut inner.slot,
            Slot::Running {
                work: work.clone(),
                token,
            },
        ) else {
            unreachable!("the slot was admitted idle above");
        };
        drop(inner);
        self.bump_epoch();

        let options = self.options.clone();
        let core = Arc::clone(self);
        let marking = work.clone();
        let drive_core = Arc::clone(&core);
        let drive = async move {
            drive_core.mark_running(&marking);
            drive_core
                .runner
                .run_in_conversation(state, options, ctrl)
                .await
                .map_err(|error| format!("runner rejected the work: {error}"))
        };
        tokio::spawn(supervise(core, work, "worker", drive));
        Ok(receipt)
    }

    /// Continue one paused work, or reject the request without side effects.
    ///
    /// Synchronous and atomic under the registry lock — no await point between
    /// validating the paused material and switching the work back to
    /// `Running`, mirroring `submit`'s accept-once guarantee. Validation runs
    /// against the paused outcome before the slot is touched, so a rejected
    /// request leaves the material exactly as it was.
    pub(super) fn resume(
        self: &Arc<Self>,
        work: &WorkRef,
        expected_revision: u64,
        request_key: String,
        request: ResumeRequest,
    ) -> Result<WorkReceipt, SessionError> {
        let mut inner = self.lock();
        // Local dedup is checked before the state and closed guards: a retry of
        // a lost receipt must resolve to the original receipt even after the
        // session closed. A key accepted for a different work, revision, or
        // request is a conflict, not a second execution.
        if let Some(record) = inner.resume_keys.get(&request_key) {
            return if &record.work == work
                && record.revision == expected_revision
                && record.request == request
            {
                Ok(record.receipt.clone())
            } else {
                Err(SessionError::Conflict)
            };
        }
        if inner.closed {
            return Err(SessionError::Closed);
        }
        if work.conversation_id != self.id {
            return Err(SessionError::NotFound(work.clone()));
        }
        let Some(entry) = inner.works.get(work) else {
            return Err(SessionError::NotFound(work.clone()));
        };
        if matches!(entry.state, WorkState::Finished | WorkState::Faulted) {
            return Err(SessionError::NotPaused(work.clone()));
        }
        // Copy the scalars out before the slot is mutated: the deadline is the
        // one stored at acceptance, so a resume continues under the same
        // absolute bound rather than a fresh one.
        let paused_revision = entry.revision;
        let deadline = entry.deadline;
        // Validate against the paused outcome before anything moves: a
        // rejected request executes nothing and leaves the material untouched.
        {
            let outcome = inner.slot.paused_for(work)?;
            if expected_revision != paused_revision {
                return Err(SessionError::StaleRevision {
                    work: work.clone(),
                    expected: expected_revision,
                    actual: paused_revision,
                });
            }
            let active = outcome
                .state
                .active_turn()
                .expect("a paused outcome keeps its active turn open");
            if let Err(reason) = crate::resume::validate_resume(&outcome.result, active, &request) {
                return Err(SessionError::InvalidResume(reason));
            }
        }

        // Accept: the outcome moves out of the slot and the work returns to
        // `Running` under the same lock that validated it, so no second
        // resume can interleave.
        let token = CancellationToken::new();
        let outcome = inner.slot.take_paused_for(work, token.clone())?;
        let ctrl = RunControl::new(token, deadline);
        let receipt = WorkReceipt {
            work: work.clone(),
            accepted_revision: paused_revision,
        };
        update(&mut inner, work, |entry| entry.state = WorkState::Running);
        inner.resume_keys.insert(
            request_key,
            ResumeKey {
                receipt: receipt.clone(),
                work: work.clone(),
                revision: expected_revision,
                request: request.clone(),
            },
        );
        drop(inner);
        self.bump_epoch();

        let core = Arc::clone(self);
        let options = self.options.clone();
        let drive_core = Arc::clone(&core);
        let drive = async move {
            crate::resume::resume_turn(drive_core.runner.as_ref(), outcome, request, options, ctrl)
                .await
                .map_err(|rejection| {
                    format!("resume rejected after acceptance: {}", rejection.reason)
                })
        };
        tokio::spawn(supervise(core, work.clone(), "resume worker", drive));
        Ok(receipt)
    }

    /// Cancel one work: fire its own stop token while it runs, or terminate a
    /// paused work in place.
    ///
    /// Synchronous and atomic under the registry lock. A terminal work is never
    /// rewritten — the cancel reports `AlreadyTerminal` and leaves its result
    /// intact.
    pub(super) fn cancel(
        self: &Arc<Self>,
        work: &WorkRef,
        request_key: String,
    ) -> Result<CancelReceipt, SessionError> {
        let mut inner = self.lock();
        // Local dedup is checked before the closed guard: a retry of a lost
        // receipt resolves to the original receipt without signalling or
        // terminating anything again, even after the session closed.
        if let Some(record) = inner.cancel_keys.get(&request_key) {
            return if &record.work == work {
                Ok(record.receipt.clone())
            } else {
                Err(SessionError::Conflict)
            };
        }
        if inner.closed {
            return Err(SessionError::Closed);
        }
        if work.conversation_id != self.id || !inner.works.contains_key(work) {
            return Err(SessionError::NotFound(work.clone()));
        }
        // A terminal work keeps its result: the cancel only records the key.
        if matches!(
            inner.works.get(work).expect("checked above").state,
            WorkState::Finished | WorkState::Faulted
        ) {
            let receipt = CancelReceipt {
                work: work.clone(),
                outcome: CancelOutcome::AlreadyTerminal,
            };
            inner.cancel_keys.insert(
                request_key,
                CancelKey {
                    receipt: receipt.clone(),
                    work: work.clone(),
                },
            );
            return Ok(receipt);
        }
        // Still executing (accepted-and-not-yet-started, or running): its
        // own control token is enough — the runner ends and publishes. No
        // state changes yet, so no epoch bump: waiters re-observe on the
        // terminal publish.
        if let Slot::Running {
            work: active,
            token,
        } = &inner.slot
        {
            if active == work {
                token.cancel();
                let receipt = CancelReceipt {
                    work: work.clone(),
                    outcome: CancelOutcome::Signalled,
                };
                inner.cancel_keys.insert(
                    request_key,
                    CancelKey {
                        receipt: receipt.clone(),
                        work: work.clone(),
                    },
                );
                return Ok(receipt);
            }
            return Err(SessionError::Busy {
                active: active.clone(),
            });
        }

        // Paused: terminate in place with no new external call.
        let (facts, continuation) = inner.slot.stop_paused(work)?;
        let receipt = CancelReceipt {
            work: work.clone(),
            outcome: CancelOutcome::Stopped,
        };
        update(&mut inner, work, |entry| {
            entry.state = WorkState::Finished;
            entry.finished = Some(FinishedKind::Interrupted {
                cause: TurnInterruption::ExplicitCancellation,
                facts,
                continuation: Some(continuation),
            });
        });
        inner.cancel_keys.insert(
            request_key,
            CancelKey {
                receipt: receipt.clone(),
                work: work.clone(),
            },
        );
        drop(inner);
        self.bump_epoch();
        Ok(receipt)
    }

    /// Wait until no work is running.
    ///
    /// Subscribe-before-check, the same order as `wait`, so a change between
    /// the observation and the block is never lost. The core is kept alive by
    /// the caller; a dropped epoch sender ends the wait rather than hanging.
    pub(super) async fn quiesce(&self) {
        let mut rx = self.epoch.subscribe();
        loop {
            let running = matches!(&self.lock().slot, Slot::Running { .. });
            if !running {
                return;
            }
            if rx.changed().await.is_err() {
                return;
            }
        }
    }

    /// Read one work's published view.
    pub(super) fn observe(&self, work: &WorkRef) -> Result<WorkObservation, SessionError> {
        let inner = self.lock();
        observation(&inner, &self.id, work)
    }

    /// Wait finitely for a returnable state.
    pub(super) async fn wait(
        &self,
        work: &WorkRef,
        timeout: Duration,
    ) -> Result<WaitOutcome, SessionError> {
        let deadline = Instant::now() + timeout;
        // Subscribe before the first observe: any change after this point is
        // seen either by the observe or by `changed()`, never lost.
        let mut rx = self.epoch.subscribe();
        loop {
            let obs = self.observe(work)?;
            if matches!(
                obs.state,
                WorkState::Paused | WorkState::Finished | WorkState::Faulted
            ) {
                return Ok(WaitOutcome {
                    observation: obs,
                    end: WaitEnd::ReachedState,
                });
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(WaitOutcome {
                    observation: obs,
                    end: WaitEnd::TimedOut,
                });
            }
            tokio::select! {
                // The core (and therefore the sender) is kept alive by the
                // handle performing this wait, so `changed()` cannot fail
                // here; any wakeup simply re-observes.
                _ = rx.changed() => {}
                _ = tokio::time::sleep(deadline - now) => {
                    // Re-observe so the timeout snapshot is consistent.
                }
            }
        }
    }

    /// The next unused `TurnId`. Starts from `next_turn`, skips ids already in
    /// committed history and ids already handed to a retained work, and never
    /// walks the counter back — so no identity is ever reused, even after a
    /// work is interrupted or its slot cleared.
    fn fresh_turn_id(&self, inner: &Inner) -> (TurnId, u64) {
        let mut n = inner.next_turn;
        loop {
            let candidate = TurnId::new(format!("{}-work-{n}", self.id.0));
            n += 1;
            let in_history = match &inner.slot {
                Slot::Idle(state) => state
                    .history()
                    .iter()
                    .any(|entry| entry.snapshot.turn_id == candidate),
                _ => false,
            };
            let in_works = inner.works.keys().any(|work| work.turn_id == candidate);
            if !in_history && !in_works {
                return (candidate, n);
            }
        }
    }

    /// Lock the registry, recovering from a poisoned mutex: a worker panic is
    /// published as `Faulted`, not allowed to brick the session.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Publish a state change: every waiter re-checks.
    fn bump_epoch(&self) {
        self.epoch
            .send_modify(|version| *version = version.wrapping_add(1));
    }

    /// `Accepted` → `Running`, published before the runner is polled.
    fn mark_running(&self, work: &WorkRef) {
        let mut inner = self.lock();
        update(&mut inner, work, |entry| entry.state = WorkState::Running);
        drop(inner);
        self.bump_epoch();
    }

    /// Publish a fault: the slot becomes unusable and the work becomes
    /// terminal, so a fault can never look like a permanent `Running`.
    fn mark_faulted(&self, work: &WorkRef, reason: String) {
        let mut inner = self.lock();
        if !matches!(&inner.slot, Slot::Running { work: active, .. } if active == work) {
            return;
        }
        inner.slot = Slot::Faulted {
            reason: reason.clone(),
        };
        update(&mut inner, work, |entry| {
            entry.state = WorkState::Faulted;
            entry.fault = Some(reason);
        });
        drop(inner);
        self.bump_epoch();
    }

    /// Publish the runner's outcome: `Completed` commits into history,
    /// `Interrupted` carries the real aborted facts (a snapshot) plus the
    /// cause publicly and stays out of history, `Paused` retains the complete
    /// outcome.
    ///
    /// The `Completed` / `Interrupted` commit/abort runs before the lock is
    /// taken — it works on the owned state the worker handed back, so the
    /// registry is never held while facts validate. The `Paused` arm is the
    /// exception: its verdict depends on the work's own stop token, so the
    /// token check and the install share one lock hold — a cancel that already
    /// fired wins over the pause, and a cancel that has not fired yet will see
    /// `Paused` and stop it in place, so the pause can never slip past a
    /// cancel.
    fn publish_outcome(&self, work: &WorkRef, outcome: ConversationOutcome) {
        let ConversationOutcome {
            mut state,
            result,
            trace,
        } = outcome;
        if matches!(result, TurnResult::Paused { .. }) {
            let TurnResult::Paused { continuation } = result else {
                unreachable!("matched above");
            };
            let mut inner = self.lock();
            let cancelled = matches!(
                &inner.slot,
                Slot::Running { work: active, token } if active == work && token.is_cancelled()
            );
            let (slot, kind, finished, fault) = if cancelled {
                match state.abort_turn(work.turn_id.clone()) {
                    Ok(facts) => (
                        Slot::Idle(state),
                        WorkState::Finished,
                        Some(FinishedKind::Interrupted {
                            cause: TurnInterruption::ExplicitCancellation,
                            facts: facts.snapshot(),
                            continuation: Some(continuation),
                        }),
                        None,
                    ),
                    Err(error) => {
                        let reason =
                            format!("abort_turn rejected the cancelled paused turn: {error}");
                        (
                            Slot::Faulted {
                                reason: reason.clone(),
                            },
                            WorkState::Faulted,
                            None,
                            Some(reason),
                        )
                    }
                }
            } else {
                (
                    Slot::Paused(ConversationOutcome {
                        state,
                        result: TurnResult::Paused { continuation },
                        trace,
                    }),
                    WorkState::Paused,
                    None,
                    None,
                )
            };
            inner.slot = slot;
            update(&mut inner, work, |entry| {
                entry.state = kind;
                entry.finished = finished;
                entry.fault = fault;
            });
            drop(inner);
            self.bump_epoch();
            return;
        }
        let (slot, kind, finished, fault) = match result {
            TurnResult::Completed { final_output } => match state.commit(work.turn_id.clone()) {
                Ok(_) => (
                    Slot::Idle(state),
                    WorkState::Finished,
                    Some(FinishedKind::Completed { final_output }),
                    None,
                ),
                Err(error) => {
                    let reason = format!("commit rejected the completed turn: {error}");
                    (
                        Slot::Faulted {
                            reason: reason.clone(),
                        },
                        WorkState::Faulted,
                        None,
                        Some(reason),
                    )
                }
            },
            TurnResult::Interrupted { cause } => match state.abort_turn(work.turn_id.clone()) {
                Ok(facts) => (
                    Slot::Idle(state),
                    WorkState::Finished,
                    Some(FinishedKind::Interrupted {
                        cause,
                        facts: facts.snapshot(),
                        continuation: None,
                    }),
                    None,
                ),
                Err(error) => {
                    let reason = format!("abort_turn rejected the interrupted turn: {error}");
                    (
                        Slot::Faulted {
                            reason: reason.clone(),
                        },
                        WorkState::Faulted,
                        None,
                        Some(reason),
                    )
                }
            },
            TurnResult::Paused { .. } => unreachable!("the paused arm returned above"),
        };
        let mut inner = self.lock();
        inner.slot = slot;
        update(&mut inner, work, |entry| {
            entry.state = kind;
            entry.finished = finished;
            entry.fault = fault;
        });
        drop(inner);
        self.bump_epoch();
    }
}

/// The turn id of a paused slot's open active turn. `drive_conversation`
/// stamps `Paused` only after sealing the (deliberately still open) active
/// turn, so the arm always has one.
fn paused_turn_id(outcome: &ConversationOutcome) -> TurnId {
    outcome
        .state
        .active_turn()
        .expect("a paused outcome keeps its active turn open")
        .turn_id()
}

/// The absolute UTC expiry of a monotonic deadline, measured against a
/// single wall-clock anchor taken now. Monotonic deadlines have no
/// cross-process meaning, so the envelope records the expiry and restore
/// re-derives the remaining time from it — never re-granting the original
/// duration. Clock consistency across the save/restore boundary is the
/// harness's responsibility.
fn absolute_expiry(deadline: Instant) -> SystemTime {
    let anchor_instant = Instant::now();
    let anchor_system = SystemTime::now();
    let expiry = if deadline >= anchor_instant {
        anchor_system.checked_add(deadline - anchor_instant)
    } else {
        anchor_system.checked_sub(anchor_instant - deadline)
    };
    expiry.expect("a work deadline is within SystemTime's range")
}

/// Rebuild the monotonic deadline from a saved absolute UTC expiry: the
/// remaining wall-clock time from now. An expiry already in the past (or a
/// clock that moved backwards past it) yields `now` — an elapsed deadline
/// the driver stops on before its first dispatch.
fn remaining_deadline(expiry: SystemTime) -> Instant {
    let remaining = expiry
        .duration_since(SystemTime::now())
        .unwrap_or(Duration::ZERO);
    Instant::now() + remaining
}

/// Look up one work's view; unknown or foreign refs are `NotFound`.
fn observation(
    inner: &Inner,
    id: &ConversationId,
    work: &WorkRef,
) -> Result<WorkObservation, SessionError> {
    if work.conversation_id != *id {
        return Err(SessionError::NotFound(work.clone()));
    }
    let entry = inner
        .works
        .get(work)
        .ok_or_else(|| SessionError::NotFound(work.clone()))?;
    Ok(WorkObservation {
        work: work.clone(),
        revision: entry.revision,
        state: entry.state,
        paused: match &inner.slot {
            Slot::Paused(outcome) if paused_turn_id(outcome) == work.turn_id => {
                let TurnResult::Paused { continuation } = &outcome.result else {
                    unreachable!("a paused slot holds a paused result");
                };
                Some(continuation.pause_point.clone())
            }
            _ => None,
        },
        finished: entry.finished.clone(),
        fault: entry.fault.clone(),
    })
}

/// Apply one published mutation and bump the work's revision.
fn update(inner: &mut Inner, work: &WorkRef, mutate: impl FnOnce(&mut WorkEntry)) {
    if let Some(entry) = inner.works.get_mut(work) {
        mutate(entry);
        entry.revision += 1;
    }
}

/// Drive one work to its terminal publish: run `drive` to a
/// [`ConversationOutcome`], then publish it. A driver rejection after
/// acceptance is an invariant break (the request was validated under the
/// lock), so it faults with the caller's message rather than dropping the
/// material; a panic is caught and faulted the same way.
///
/// The guard is built before the future is spawned, so even a runtime that
/// drops the task before its first poll publishes a fault. It holds no
/// `JoinHandle` and creates no reference cycle.
fn supervise(
    core: Arc<SessionCore>,
    work: WorkRef,
    panic_label: &'static str,
    drive: impl std::future::Future<Output = Result<ConversationOutcome, String>>,
) -> impl std::future::Future<Output = ()> {
    let mut guard = WorkerGuard {
        core,
        work,
        armed: true,
    };
    async move {
        match AssertUnwindSafe(drive).catch_unwind().await {
            Ok(Ok(outcome)) => guard.core.publish_outcome(&guard.work, outcome),
            Ok(Err(reason)) => guard.core.mark_faulted(&guard.work, reason),
            Err(payload) => guard.core.mark_faulted(
                &guard.work,
                format!("{panic_label} panicked: {}", panic_message(&*payload)),
            ),
        }
        guard.disarm();
    }
}

/// Publishes task cancellation even when the executor never polls it again.
/// No complete outcome is available once the task loses its owned state.
struct WorkerGuard {
    core: Arc<SessionCore>,
    work: WorkRef,
    armed: bool,
}

impl WorkerGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        if self.armed {
            self.core.mark_faulted(
                &self.work,
                "worker task dropped before publishing its outcome".into(),
            );
        }
    }
}

/// Best-effort text of a caught panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
