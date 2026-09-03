//! ConversationState — the conversation-level fact aggregate: ordered
//! completed turns, the active turn host, and the eligibility stamp.
//!
//! This is not a second fact source: block-level facts live only in
//! `TurnContext` / `TurnSnapshot`; the aggregate owns deterministic order
//! (`TurnSequence`, assigned exactly once by `commit`), the active turn, and
//! the `ConversationVersion` that ticks at every controlled transition
//! (`begin_turn` / `commit` / `abort_turn`).

use serde::{Deserialize, Serialize};

use crate::context::ids::{
    ConversationId, ConversationVersion, FrameId, FrameScope, RoundId, TurnId, TurnSequence,
};
use crate::context::turn::{ContextFrame, ModelContext, TurnContext, TurnSnapshot};

/// Kernel-side eligibility stamp recorded when the driver finalizes the
/// active turn. Marker only — the rich cause stays with the caller via the
/// runner's `TurnResult` (`TurnInterruption` is driver vocabulary and must not enter the facts layer).
///
/// `Paused` (Slice 7) stamps a turn that is *not* sealed: the active
/// `TurnContext` stays open so `resume_turn` can continue it. `commit`
/// rejects a Paused stamp (`TurnPaused`) — only `abort_turn` (host gives
/// up) or a resumed completion/commit may close the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SealedResult {
    /// The turn finished successfully — the only stamp `commit` accepts.
    Completed,
    /// The turn was cut short. Seals the turn, but `commit` rejects it —
    /// only `abort_turn` closes the slot.
    Interrupted,
    /// The turn is suspended but *not* sealed (Slice 7): the active
    /// `TurnContext` stays open for `resume_turn`; `commit` rejects it
    /// with `TurnPaused`.
    Paused,
}

/// Committed history: the turn snapshots in ascending `TurnSequence` order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderedTurns(Vec<TurnSnapshot>);
impl OrderedTurns {
    /// An empty history.
    pub fn empty() -> Self {
        Self(Vec::new())
    }
    /// Read-only view of the snapshots in `TurnSequence` order.
    pub fn ordered(&self) -> &[TurnSnapshot] {
        &self.0
    }
}

/// Rejections of the conversation-level controlled operations. Pure state
/// guards — every variant is deterministic from the aggregate's facts.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConversationError {
    /// An open (unsealed) active turn exists; seal-then-commit or abort it first.
    #[error("a turn is already active in this conversation")]
    TurnAlreadyActive,
    /// The active turn is already sealed; commit or abort it before
    /// starting a new one.
    #[error("active turn is already sealed; abort it before starting a new one")]
    TurnAlreadySealed,
    /// The operation needs an active turn (e.g. `frame`), but the slot is empty.
    #[error("no active turn in this conversation")]
    NoActiveTurn,
    /// The given turn id is not the active turn — including a repeated
    /// commit after the slot was cleared.
    #[error("unknown turn: {0:?}")]
    UnknownTurn(TurnId),
    /// `begin_turn` with an id already present in committed history.
    #[error("duplicate turn id: {0:?}")]
    DuplicateTurnId(TurnId),
    /// `commit` on a turn that is not sealed-and-`Completed`.
    #[error("turn not completed, cannot commit: {0:?}")]
    TurnNotCompleted(TurnId),
    /// `commit` on a paused turn; resume it or abort it instead.
    #[error("turn is paused, cannot commit until resumed: {0:?}")]
    TurnPaused(TurnId),
    /// The turn is not paused (no `Paused` stamp), so it cannot be resumed.
    #[error("turn is not paused, cannot resume: {0:?}")]
    NotPaused(TurnId),
    /// Replay validation failed in `from_snapshots`; carries the reason
    /// (non-monotonic `TurnSequence` or a block-level violation).
    #[error("invalid conversation state: {0}")]
    InvalidSequence(String),
}

/// Conversation-level controlled operations:
///
/// - `begin_turn` admits a fresh active turn (rejects concurrent active and
///   turn-id collisions with committed history);
/// - `seal_turn` is the only stamping path — for `Completed`/`Interrupted`
///   it seals the active `TurnContext` and records the outcome in one step
///   (invariant `sealed_result ∈ {Completed, Interrupted} ⇒ active.is_sealed()`);
///   for `Paused` (Slice 7) it records the stamp while the active
///   `TurnContext` stays open, so a later `resume_turn` can continue it
///   (invariant `sealed_result == Paused ⇒ active` is open);
/// - `commit` is the exactly-once transition into history: it alone assigns
///   the `TurnSequence` (struct-update on the turn's snapshot), rejects
///   anything not sealed-and-`Completed` (`Paused` gets the dedicated
///   `TurnPaused` rejection), and clears the active slot — a repeated
///   commit therefore lands on `UnknownTurn` (rejection, not idempotence);
/// - `abort_turn` discards the active turn in any state (open, paused,
///   sealed); history is untouched either way, and an aborted turn's id
///   may be reused.
#[derive(Serialize, Deserialize)]
pub struct ConversationState {
    conversation_id: ConversationId,
    completed_turns: OrderedTurns,
    /// The sealed active turn at handoff. Serialized through the
    /// `option_turn_context_as_snapshot` adapter (see
    /// `crate::context::turn`): the in-memory `TurnContext` is the
    /// mutable fact machine; once sealed its snapshot projection is the
    /// canonical wire shape. On reload we rebuild a sealed
    /// `TurnContext` via `from_validated_blocks` + `seal()`.
    #[serde(with = "crate::context::turn::option_turn_context_as_snapshot")]
    active_turn: Option<TurnContext>,
    sealed_result: Option<SealedResult>,
    version: ConversationVersion,
}

impl std::fmt::Debug for ConversationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationState")
            .field("conversation_id", &self.conversation_id)
            .field("version", &self.version)
            .field("snapshot_count", &self.completed_turns.0.len())
            .field("next_turn_sequence", &self.next_turn_sequence())
            .field("active_turn", &self.active_turn)
            .field("sealed_result", &self.sealed_result)
            .finish()
    }
}

impl ConversationState {
    /// A fresh, empty conversation: no history, no active turn, version zero.
    pub fn new(conversation_id: ConversationId) -> Self {
        Self {
            conversation_id,
            completed_turns: OrderedTurns::empty(),
            active_turn: None,
            sealed_result: None,
            version: ConversationVersion(0),
        }
    }

    /// The next `TurnSequence` the kernel will assign on `commit`. Derived
    /// from `completed_turns.last() + 1` so the source of truth is history;
    /// there is no in-memory field to drift. `TurnSequence(0)` if empty.
    pub fn next_turn_sequence(&self) -> TurnSequence {
        self.completed_turns
            .0
            .last()
            .map(|s| TurnSequence(s.turn_sequence.0 + 1))
            .unwrap_or(TurnSequence(0))
    }

    /// Admit a fresh active turn. Rejects while any active exists (sealed
    /// ones must be committed or aborted first) and when the id collides
    /// with committed history — an aborted turn's id is reusable.
    pub fn begin_turn(&mut self, turn_id: TurnId) -> Result<&mut TurnContext, ConversationError> {
        self.assert_stamp_invariant();
        if let Some(active) = &self.active_turn {
            return Err(if active.is_sealed() {
                ConversationError::TurnAlreadySealed
            } else {
                ConversationError::TurnAlreadyActive
            });
        }
        if self.completed_turns.0.iter().any(|s| s.turn_id == turn_id) {
            return Err(ConversationError::DuplicateTurnId(turn_id));
        }
        self.active_turn = Some(TurnContext::new(turn_id));
        self.version = self.version.next();
        Ok(self.active_turn.as_mut().expect("just inserted"))
    }

    /// The only stamping path. `Completed`/`Interrupted` seal the active
    /// `TurnContext` and record the outcome atomically; `Paused` records
    /// the stamp while the turn stays open (Slice 7 — the driver-owned
    /// counterpart of `TurnContext::seal`, deliberately withheld for
    /// resumable pauses).
    pub fn seal_turn(
        &mut self,
        turn_id: TurnId,
        result: SealedResult,
    ) -> Result<(), ConversationError> {
        self.assert_stamp_invariant();
        match self.active_turn.as_mut() {
            Some(active) if active.turn_id() == turn_id => {
                if result != SealedResult::Paused {
                    active.seal();
                }
                self.sealed_result = Some(result);
                Ok(())
            }
            _ => Err(ConversationError::UnknownTurn(turn_id)),
        }
    }

    /// Read-only view of the active turn.
    pub fn active_turn(&self) -> Option<&TurnContext> {
        self.active_turn.as_ref()
    }

    /// The eligibility stamp, if any (`Some(Paused)` marks a resumable
    /// turn).
    pub fn sealed_result(&self) -> Option<SealedResult> {
        self.sealed_result
    }

    /// Mutable view of the active turn — the host's door access for
    /// `append_input` before running.
    pub fn active_turn_mut(&mut self) -> Option<&mut TurnContext> {
        self.active_turn.as_mut()
    }

    /// Exactly-once commit: requires the turn id to match the active slot,
    /// the turn to be sealed, and the stamp to read `Completed`. Assigns the
    /// next `TurnSequence` and appends the turn's snapshot to history.
    /// Rejection (not idempotence): after the first commit the slot is
    /// empty, so a repeated commit returns `UnknownTurn`.
    pub fn commit(&mut self, turn_id: TurnId) -> Result<TurnSnapshot, ConversationError> {
        self.assert_stamp_invariant();
        match self.active_turn.as_ref() {
            Some(active) if active.turn_id() == turn_id => {}
            _ => return Err(ConversationError::UnknownTurn(turn_id)),
        }
        if self.sealed_result == Some(SealedResult::Paused) {
            // Facts of a paused turn are still open; resume it (or abort)
            // instead of committing.
            return Err(ConversationError::TurnPaused(turn_id));
        }
        let active = self.active_turn.as_ref().expect("matched above");
        if !active.is_sealed() || self.sealed_result != Some(SealedResult::Completed) {
            return Err(ConversationError::TurnNotCompleted(turn_id));
        }
        let active = self.active_turn.take().expect("matched above");
        self.sealed_result = None;
        let turn_sequence = self.next_turn_sequence();
        let snapshot = TurnSnapshot {
            turn_sequence,
            ..active.snapshot()
        };
        self.completed_turns.0.push(snapshot.clone());
        self.version = self.version.next();
        Ok(snapshot)
    }

    /// Discard the active turn in any state (open, sealed-completed,
    /// sealed-interrupted). History is untouched; the returned
    /// `TurnContext` is for caller inspection only (no reopen/unseal). An
    /// aborted turn's id may be reused by a later `begin_turn`.
    pub fn abort_turn(&mut self, turn_id: TurnId) -> Result<TurnContext, ConversationError> {
        self.assert_stamp_invariant();
        match self.active_turn.as_ref() {
            Some(active) if active.turn_id() == turn_id => {}
            _ => return Err(ConversationError::UnknownTurn(turn_id)),
        }
        let active = self.active_turn.take().expect("matched above");
        self.sealed_result = None;
        self.version = self.version.next();
        Ok(active)
    }

    /// Lossless merged view: committed history (TurnSequence ascending,
    /// blocks in BlockSequence order) followed by the active turn's blocks,
    /// under the Conversation scope identity. Sync and policy-free by
    /// design — budget, selection and compaction over the merged view are
    /// Slice 5 and will orchestrate through the policy layer, never mutate
    /// facts. The only failure source is a missing active turn.
    pub fn frame(&self, round_id: RoundId) -> Result<ContextFrame, ConversationError> {
        let active = match &self.active_turn {
            Some(active) => active,
            None => return Err(ConversationError::NoActiveTurn),
        };
        Ok(merged_frame(
            &self.conversation_id,
            self.completed_turns.ordered(),
            active,
            round_id,
        ))
    }

    /// Borrow-split for a conversation driver's consume/return flow: the
    /// conversation id and committed history are read while the active
    /// turn is driven mutably. Public since Slice 12 — the canonical
    /// driver lives outside the kernel (agent-runtime), and this is the
    /// exact seam any external conversation driver needs. Stamping still
    /// goes through the public `seal_turn` afterwards, so no second `&mut`
    /// seam exists.
    pub fn runner_parts(&mut self) -> (&ConversationId, &[TurnSnapshot], Option<&mut TurnContext>) {
        (
            &self.conversation_id,
            self.completed_turns.ordered(),
            self.active_turn.as_mut(),
        )
    }

    /// Validated replay path: rebuild a conversation from committed
    /// snapshots. The active slot starts empty (live paths never enter
    /// here); `ConversationVersion` resets to zero (replay is a fresh load —
    /// cross-persistence version semantics are Slice 5). Validation closed
    /// set: `turn_sequence` strictly increasing and distinct (gaps allowed —
    /// future trimming territory), and every snapshot's blocks pass the same
    /// `TurnContext::from_validated_blocks` checks; any violation maps to
    /// `ConversationError::InvalidSequence` so callers never touch
    /// `ContextError`. `source_version` is accepted as a recorded fact — it
    /// counts fact commits and is not derivable from the blocks.
    pub fn from_snapshots(
        conversation_id: ConversationId,
        snapshots: Vec<TurnSnapshot>,
    ) -> Result<Self, ConversationError> {
        let mut last_seq = TurnSequence(0);
        for snapshot in &snapshots {
            if snapshot.turn_sequence < last_seq {
                return Err(ConversationError::InvalidSequence(format!(
                    "turn_sequence not strictly increasing: {:?}",
                    snapshot.turn_sequence
                )));
            }
            TurnContext::validate_blocks(&snapshot.turn_id, snapshot.blocks.as_slice())
                .map_err(|e| ConversationError::InvalidSequence(e.to_string()))?;
            last_seq = TurnSequence(snapshot.turn_sequence.0 + 1);
        }
        Ok(Self {
            conversation_id,
            completed_turns: OrderedTurns(snapshots),
            active_turn: None,
            sealed_result: None,
            version: ConversationVersion(0),
        })
    }

    /// Committed history in ascending `TurnSequence` order.
    pub fn completed_turns(&self) -> &[TurnSnapshot] {
        self.completed_turns.ordered()
    }

    /// Number of committed turns in history.
    pub fn snapshot_count(&self) -> usize {
        self.completed_turns.0.len()
    }

    /// The `ConversationVersion`, ticked at every controlled transition
    /// (`begin_turn` / `commit` / `abort_turn`); zero for a fresh or
    /// replayed conversation.
    pub fn version(&self) -> ConversationVersion {
        self.version
    }

    /// The conversation's identity.
    pub fn conversation_id(&self) -> &ConversationId {
        &self.conversation_id
    }

    /// Canonical-path invariant: a `Completed`/`Interrupted` stamp exists
    /// only while the active turn is sealed; a `Paused` stamp exists only
    /// while it is open (Slice 7). Enforced by construction (`seal_turn`
    /// seals exactly when it does not stamp `Paused`; every other path
    /// only clears). Commit does not rely on this — it checks both facts
    /// defensively.
    fn assert_stamp_invariant(&self) {
        debug_assert!(match self.sealed_result {
            None => true,
            Some(SealedResult::Paused) => {
                self.active_turn.as_ref().is_some_and(|t| !t.is_sealed())
            }
            Some(_) => self.active_turn.as_ref().is_some_and(|t| t.is_sealed()),
        });
    }
}

/// Shared lossless merged materialization over committed history plus the
/// active turn — the single semantics both `ConversationState::frame()` and
/// a conversation driver's merged-view entry use. Public since Slice 12:
/// the canonical driver lives outside the kernel and materializes frames
/// through this projection. No reordering, no dedup, no trimming; nothing
/// is written back.
pub fn merged_frame(
    conversation_id: &ConversationId,
    history: &[TurnSnapshot],
    active: &TurnContext,
    round_id: RoundId,
) -> ContextFrame {
    let mut blocks = Vec::new();
    for snapshot in history {
        blocks.extend(snapshot.blocks.as_slice().iter().cloned());
    }
    blocks.extend(active.blocks().iter().cloned());
    let scope = FrameScope::Conversation {
        conversation_id: conversation_id.clone(),
        active_turn_id: active.turn_id(),
        source_version: active.version(),
    };
    ContextFrame {
        frame_id: FrameId::from_scope(&scope, round_id),
        scope,
        round_id,
        model_context: ModelContext { blocks },
    }
}
