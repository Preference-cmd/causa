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
use std::time::{Duration, Instant};

use futures_util::FutureExt;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use causa_kernel::{ContentPart, ConversationId, TurnId};

use crate::config::TurnRunOptions;
use crate::control::RunControl;
use crate::conversation::ConversationState;
use crate::driver::{ConversationOutcome, TurnResult, TurnRunner};

use super::{
    FinishedKind, SessionConfig, SessionError, SubmitRequest, WaitEnd, WaitOutcome,
    WorkObservation, WorkReceipt, WorkRef, WorkState,
};

/// The request-table operation a key belongs to. `submit` is the only
/// operation today; a later `resume` / `cancel` extends this without reusing
/// the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Op {
    Submit,
}

/// One accepted request-key record: the original receipt plus the arguments it
/// was accepted for, so a repeat with different arguments is `Conflict`.
struct Accepted {
    receipt: WorkReceipt,
    parts: Vec<ContentPart>,
}

/// The published view of one retained work.
struct WorkEntry {
    revision: u64,
    state: WorkState,
    finished: Option<FinishedKind>,
    fault: Option<String>,
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

/// The lock-protected registry.
struct Inner {
    slot: Slot,
    works: HashMap<WorkRef, WorkEntry>,
    by_key: HashMap<(Op, String), Accepted>,
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
                by_key: HashMap::new(),
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
        if inner.closed {
            return Err(SessionError::Closed);
        }
        // Local dedup is checked before the busy guard: a retry of a lost
        // receipt must resolve to the original receipt even while the work
        // runs. A key that was never accepted (because a submit was rejected)
        // is absent here, so fixing a rejected submit keeps the key free.
        if let Some(accepted) = inner.by_key.get(&(Op::Submit, request.request_key.clone())) {
            return if accepted.parts == request.parts {
                Ok(accepted.receipt.clone())
            } else {
                Err(SessionError::Conflict)
            };
        }
        // One active work per conversation: no queue, no steering, no
        // implicit approval.
        match &inner.slot {
            Slot::Idle(_) => {}
            Slot::Running { work, .. } => {
                return Err(SessionError::Busy {
                    active: work.clone(),
                });
            }
            Slot::Paused(outcome) => {
                return Err(SessionError::Busy {
                    active: WorkRef {
                        conversation_id: self.id.clone(),
                        turn_id: paused_turn_id(outcome),
                    },
                });
            }
            Slot::Faulted { reason } => {
                return Err(SessionError::Faulted {
                    reason: reason.clone(),
                });
            }
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
            },
        );
        inner.by_key.insert(
            (Op::Submit, request.request_key),
            Accepted {
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
            unreachable!("idle slot checked above");
        };
        drop(inner);
        self.bump_epoch();

        let options = self.options.clone();
        tokio::spawn(worker(Arc::clone(self), work, state, options, ctrl));
        Ok(receipt)
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
    fn publish_outcome(&self, work: &WorkRef, outcome: ConversationOutcome) {
        let mut inner = self.lock();
        let ConversationOutcome {
            mut state,
            result,
            trace,
        } = outcome;
        let (slot, kind, finished, fault) = match result {
            TurnResult::Paused { continuation } => (
                Slot::Paused(ConversationOutcome {
                    state,
                    result: TurnResult::Paused { continuation },
                    trace,
                }),
                WorkState::Paused,
                None,
                None,
            ),
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
        };
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

/// The worker task body: one accepted work, driven by the reference runner.
///
/// Holds `Arc<SessionCore>` (never a `JoinHandle`) so the session can forget
/// the task and no strong reference cycle forms.
async fn worker(
    core: Arc<SessionCore>,
    work: WorkRef,
    state: ConversationState,
    options: TurnRunOptions,
    ctrl: RunControl,
) {
    core.mark_running(&work);
    let outcome = AssertUnwindSafe(core.runner.run_in_conversation(state, options, ctrl))
        .catch_unwind()
        .await;
    match outcome {
        Ok(Ok(outcome)) => core.publish_outcome(&work, outcome),
        Ok(Err(error)) => core.mark_faulted(&work, format!("runner rejected the work: {error}")),
        Err(payload) => core.mark_faulted(
            &work,
            format!("worker panicked: {}", panic_message(&*payload)),
        ),
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
