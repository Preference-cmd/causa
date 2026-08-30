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
/// runner's `TurnResult`（`TurnInterruption` 是 staged 词汇，不得进入事实层）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealedResult {
    Completed,
    Interrupted,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderedTurns(Vec<TurnSnapshot>);
impl OrderedTurns {
    pub fn empty() -> Self {
        Self(Vec::new())
    }
    pub fn ordered(&self) -> &[TurnSnapshot] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConversationError {
    #[error("a turn is already active in this conversation")]
    TurnAlreadyActive,
    #[error("active turn is already sealed; abort it before starting a new one")]
    TurnAlreadySealed,
    #[error("no active turn in this conversation")]
    NoActiveTurn,
    #[error("unknown turn: {0:?}")]
    UnknownTurn(TurnId),
    #[error("duplicate turn id: {0:?}")]
    DuplicateTurnId(TurnId),
    #[error("turn not completed, cannot commit: {0:?}")]
    TurnNotCompleted(TurnId),
    #[error("invalid conversation state: {0}")]
    InvalidSequence(String),
}

/// Conversation-level controlled operations:
///
/// - `begin_turn` admits a fresh active turn (rejects concurrent active and
///   turn-id collisions with committed history);
/// - `seal_turn` is the only stamping path — it seals the active
///   `TurnContext` and records the outcome in one step, so the invariant
///   `sealed_result.is_some() ⇒ active.is_sealed()` holds by construction;
/// - `commit` is the exactly-once transition into history: it alone assigns
///   the `TurnSequence` (struct-update on the turn's snapshot), rejects
///   anything not sealed-and-`Completed`, and clears the active slot — a
///   repeated commit therefore lands on `UnknownTurn` (rejection, not
///   idempotence);
/// - `abort_turn` discards the active turn in any state (open,
///   sealed-completed, sealed-interrupted); history is untouched either way,
///   and an aborted turn's id may be reused.
pub struct ConversationState {
    conversation_id: ConversationId,
    completed_turns: OrderedTurns,
    active_turn: Option<TurnContext>,
    sealed_result: Option<SealedResult>,
    next_turn_sequence: TurnSequence,
    version: ConversationVersion,
}

impl std::fmt::Debug for ConversationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationState")
            .field("conversation_id", &self.conversation_id)
            .field("version", &self.version)
            .field("snapshot_count", &self.completed_turns.0.len())
            .field("next_turn_sequence", &self.next_turn_sequence)
            .field("active_turn", &self.active_turn)
            .field("sealed_result", &self.sealed_result)
            .finish()
    }
}

impl ConversationState {
    pub fn new(conversation_id: ConversationId) -> Self {
        Self {
            conversation_id,
            completed_turns: OrderedTurns::empty(),
            active_turn: None,
            sealed_result: None,
            next_turn_sequence: TurnSequence(0),
            version: ConversationVersion(0),
        }
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

    /// The only stamping path: seal the active `TurnContext` and record its
    /// outcome atomically. Driver-owned, like `TurnContext::seal` itself.
    pub fn seal_turn(
        &mut self,
        turn_id: TurnId,
        result: SealedResult,
    ) -> Result<(), ConversationError> {
        self.assert_stamp_invariant();
        match self.active_turn.as_mut() {
            Some(active) if active.turn_id() == turn_id => {
                active.seal();
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
        let active = self.active_turn.as_ref().expect("matched above");
        if !active.is_sealed() || self.sealed_result != Some(SealedResult::Completed) {
            return Err(ConversationError::TurnNotCompleted(turn_id));
        }
        let active = self.active_turn.take().expect("matched above");
        self.sealed_result = None;
        let turn_sequence = self.next_turn_sequence;
        self.next_turn_sequence = TurnSequence(turn_sequence.0 + 1);
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

    /// Disjoint borrow-split for the staged runner's consume/return flow:
    /// the conversation id and committed history are read while the active
    /// turn is driven mutably. Crate-internal only — the runner stamps via
    /// the public `seal_turn` afterwards, so no second &mut seam reaches
    /// external callers.
    pub(crate) fn runner_parts(
        &mut self,
    ) -> (&ConversationId, &[TurnSnapshot], Option<&mut TurnContext>) {
        (
            &self.conversation_id,
            self.completed_turns.ordered(),
            self.active_turn.as_mut(),
        )
    }

    pub fn completed_turns(&self) -> &[TurnSnapshot] {
        self.completed_turns.ordered()
    }

    pub fn snapshot_count(&self) -> usize {
        self.completed_turns.0.len()
    }

    pub fn version(&self) -> ConversationVersion {
        self.version
    }

    pub fn conversation_id(&self) -> &ConversationId {
        &self.conversation_id
    }

    /// Canonical-path invariant: a stamp exists only while the active turn
    /// is sealed. Enforced by construction (`seal_turn` seals first; every
    /// other path only clears). Commit does not rely on this — it checks
    /// both facts defensively.
    fn assert_stamp_invariant(&self) {
        debug_assert!(
            self.sealed_result.is_none()
                || self.active_turn.as_ref().is_some_and(|t| t.is_sealed())
        );
    }
}

/// Shared lossless merged materialization over committed history plus the
/// active turn — the single semantics both `ConversationState::frame()` and
/// the staged runner's conversation entry use. No reordering, no dedup, no
/// trimming; nothing is written back.
pub(crate) fn merged_frame(
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
