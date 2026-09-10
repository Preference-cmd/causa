//! The session aggregate — `ConversationState`, its ordering vocabulary,
//! and the `ConversationStore` archive port. Migrated here from the kernel
//! in Slice 6.5 (Context / harness separation, first batch): the single
//! active slot, completed-only history admission, and commit-time sequence
//! assignment are *reference-harness* decisions, not fact-layer invariants,
//! so they live with the reference driver. The kernel keeps the facts
//! ([`causa_kernel::TurnContext`] / [`causa_kernel::TurnSnapshot`]), the
//! validated recovery entries, and the shared [`causa_kernel::merged_frame`]
//! projection; it never depends back on this crate.
//!
//! ## Snapshot vs. session entry (Slice 6.5)
//!
//! A [`causa_kernel::TurnSnapshot`] describes one record only — identity,
//! blocks, fact version, write lifecycle. Session ordering lives in
//! [`HistoryEntry`] (`sequence` + `snapshot`), assigned exactly once by
//! [`ConversationState::commit`]. Old wire payloads that embedded
//! `turn_sequence` inside the snapshot are migrated by extracting the
//! sequence into the entry (the persistence example carries the reference
//! recipe); a bare snapshot alone promises fact recovery, never a
//! resumable execution checkpoint.

use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize};

use causa_kernel::{
    ContextError, ContextFrame, ConversationId, RoundId, TurnContext, TurnSnapshot, merged_frame,
};

/// Position of a committed turn within a session's history, assigned
/// exactly once by [`ConversationState::commit`]. Migrated from the kernel
/// with the session aggregate (Slice 6.5): the ordering *rule* — completed
/// turns only, dense sequence at commit — is reference-harness policy, not
/// a fact-layer invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TurnSequence(pub u64);

/// Counts controlled transitions of a session (`begin_turn` / `commit` /
/// `abort_turn`); the aggregate-level analogue of `ContextVersion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationVersion(pub u64);
impl ConversationVersion {
    /// Returns the successor version.
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

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

/// One committed session entry: the snapshot plus the session order the
/// record was admitted at. Slice 6.5 moved `turn_sequence` out of the
/// snapshot itself — the record describes only its own facts; the session
/// owns the order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// The `TurnSequence` `commit` assigned to this record.
    pub sequence: TurnSequence,
    /// The turn's committed facts.
    pub snapshot: TurnSnapshot,
}

impl HistoryEntry {
    /// Convenience view of the wrapped record.
    pub fn snapshot(&self) -> &TurnSnapshot {
        &self.snapshot
    }
}

/// Rejections of the session-level controlled operations. Pure state
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
    UnknownTurn(causa_kernel::TurnId),
    /// `begin_turn` with an id already present in committed history.
    #[error("duplicate turn id: {0:?}")]
    DuplicateTurnId(causa_kernel::TurnId),
    /// `commit` on a turn that is not sealed-and-`Completed`.
    #[error("turn not completed, cannot commit: {0:?}")]
    TurnNotCompleted(causa_kernel::TurnId),
    /// `commit` on a paused turn; resume it or abort it instead.
    #[error("turn is paused, cannot commit until resumed: {0:?}")]
    TurnPaused(causa_kernel::TurnId),
    /// The turn is not paused (no `Paused` stamp), so it cannot be resumed.
    #[error("turn is not paused, cannot resume: {0:?}")]
    NotPaused(causa_kernel::TurnId),
    /// Replay validation failed in `from_history`; carries the reason
    /// (non-monotonic `TurnSequence`, a duplicate turn id, or a
    /// block-level violation).
    #[error("invalid conversation state: {0}")]
    InvalidSequence(String),
}

/// Session-level controlled operations over the kernel's facts:
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
///   the `TurnSequence` (as a [`HistoryEntry`]), rejects anything not
///   sealed-and-`Completed` (`Paused` gets the dedicated `TurnPaused`
///   rejection), and clears the active slot — a repeated commit therefore
///   lands on `UnknownTurn` (rejection, not idempotence);
/// - `abort_turn` discards the active turn in any state (open, paused,
///   sealed); history is untouched either way, and an aborted turn's id
///   may be reused.
///
/// The single active slot, completed-only admission, and commit-time
/// ordering are this crate's reference-harness defaults — a custom harness
/// composes the kernel facts differently without touching them.
#[derive(Serialize)]
pub struct ConversationState {
    conversation_id: ConversationId,
    /// Committed history in `TurnSequence` order — entries, not bare
    /// snapshots, since Slice 6.5 (the order is session vocabulary).
    history: Vec<HistoryEntry>,
    /// The sealed active turn at handoff. Serialized through the
    /// `option_turn_context_as_snapshot` adapter (see
    /// `causa_kernel`): the in-memory `TurnContext` is the
    /// mutable fact machine; once sealed its snapshot projection is the
    /// canonical wire shape. On reload we rebuild a sealed
    /// `TurnContext` via `from_validated_blocks` + `seal()`.
    #[serde(with = "causa_kernel::option_turn_context_as_snapshot")]
    active_turn: Option<TurnContext>,
    sealed_result: Option<SealedResult>,
    version: ConversationVersion,
}

/// Wire-shaped field carrier for [`ConversationState`]: the exact derived
/// shape the `Serialize` derive emits, kept in one place so the manual
/// `Deserialize` impl below cannot drift from it. The wire shape itself is
/// unchanged — this only adds load-time validation.
#[derive(Deserialize)]
struct ConversationStateFields {
    conversation_id: ConversationId,
    history: Vec<HistoryEntry>,
    #[serde(with = "causa_kernel::option_turn_context_as_snapshot")]
    active_turn: Option<TurnContext>,
    sealed_result: Option<SealedResult>,
    version: ConversationVersion,
}

impl<'de> Deserialize<'de> for ConversationState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let fields = ConversationStateFields::deserialize(deserializer)?;
        // `begin_turn` never admits an active turn whose id is already in
        // committed history, so a payload claiming otherwise never came
        // from this aggregate.
        if let Some(active) = &fields.active_turn
            && fields
                .history
                .iter()
                .any(|entry| entry.snapshot.turn_id == active.turn_id())
        {
            return Err(serde::de::Error::custom(format!(
                "active turn {:?} duplicates a committed history id",
                active.turn_id()
            )));
        }
        // Same closed set as `from_history`: a corrupt or hand-edited
        // payload is rejected with the strength the in-memory paths
        // enforce by construction.
        validate_history(&fields.history).map_err(serde::de::Error::custom)?;
        Ok(Self {
            conversation_id: fields.conversation_id,
            history: fields.history,
            active_turn: fields.active_turn,
            sealed_result: fields.sealed_result,
            version: fields.version,
        })
    }
}

/// The validated-replay closed set shared by
/// [`ConversationState::from_history`] and the aggregate's `Deserialize`
/// impl: `TurnSequence` strictly increasing (gaps allowed — future
/// trimming territory), turn ids distinct (duplicate records must not
/// silently collapse into one history), and every snapshot's blocks pass
/// the kernel's `TurnContext::validate_blocks`; any violation maps to
/// `ConversationError::InvalidSequence`.
fn validate_history(entries: &[HistoryEntry]) -> Result<(), ConversationError> {
    let mut last_seq = TurnSequence(0);
    let mut seen_ids = std::collections::HashSet::new();
    for entry in entries {
        if entry.sequence < last_seq {
            return Err(ConversationError::InvalidSequence(format!(
                "turn_sequence not strictly increasing: {:?}",
                entry.sequence
            )));
        }
        if !seen_ids.insert(entry.snapshot.turn_id.clone()) {
            return Err(ConversationError::InvalidSequence(format!(
                "duplicate turn id in history: {:?}",
                entry.snapshot.turn_id
            )));
        }
        TurnContext::validate_blocks(&entry.snapshot.turn_id, entry.snapshot.blocks.as_slice())
            .map_err(|e: ContextError| ConversationError::InvalidSequence(e.to_string()))?;
        last_seq = TurnSequence(entry.sequence.0 + 1);
    }
    Ok(())
}

impl std::fmt::Debug for ConversationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationState")
            .field("conversation_id", &self.conversation_id)
            .field("version", &self.version)
            .field("history_len", &self.history.len())
            .field("next_turn_sequence", &self.next_turn_sequence())
            .field("active_turn", &self.active_turn)
            .field("sealed_result", &self.sealed_result)
            .finish()
    }
}

impl ConversationState {
    /// A fresh, empty session: no history, no active turn, version zero.
    pub fn new(conversation_id: ConversationId) -> Self {
        Self {
            conversation_id,
            history: Vec::new(),
            active_turn: None,
            sealed_result: None,
            version: ConversationVersion(0),
        }
    }

    /// The next `TurnSequence` the session will assign on `commit`. Derived
    /// from `history.last() + 1` so the source of truth is history; there
    /// is no in-memory field to drift. `TurnSequence(0)` if empty.
    pub fn next_turn_sequence(&self) -> TurnSequence {
        self.history
            .last()
            .map(|e| TurnSequence(e.sequence.0 + 1))
            .unwrap_or(TurnSequence(0))
    }

    /// Admit a fresh active turn. Rejects while any active exists (sealed
    /// ones must be committed or aborted first) and when the id collides
    /// with committed history — an aborted turn's id is reusable.
    pub fn begin_turn(
        &mut self,
        turn_id: causa_kernel::TurnId,
    ) -> Result<&mut TurnContext, ConversationError> {
        self.assert_stamp_invariant();
        if let Some(active) = &self.active_turn {
            return Err(if active.is_sealed() {
                ConversationError::TurnAlreadySealed
            } else {
                ConversationError::TurnAlreadyActive
            });
        }
        if self.history.iter().any(|e| e.snapshot.turn_id == turn_id) {
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
        turn_id: causa_kernel::TurnId,
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
    /// the turn to be sealed, and the stamp to read `Completed`. Assigns
    /// the next `TurnSequence`, appends the [`HistoryEntry`] to history,
    /// and clears the active slot. Rejection (not idempotence): after the
    /// first commit the slot is empty, so a repeated commit returns
    /// `UnknownTurn`.
    pub fn commit(
        &mut self,
        turn_id: causa_kernel::TurnId,
    ) -> Result<HistoryEntry, ConversationError> {
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
        let sequence = self.next_turn_sequence();
        let entry = HistoryEntry {
            sequence,
            snapshot: active.snapshot(),
        };
        self.history.push(entry.clone());
        self.version = self.version.next();
        Ok(entry)
    }

    /// Discard the active turn in any state (open, paused, sealed-completed,
    /// sealed-interrupted). History is untouched; the returned
    /// `TurnContext` is for caller inspection only (no reopen/unseal). An
    /// aborted turn's id may be reused by a later `begin_turn`.
    pub fn abort_turn(
        &mut self,
        turn_id: causa_kernel::TurnId,
    ) -> Result<TurnContext, ConversationError> {
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

    /// Lossless merged view: committed history (sequence ascending,
    /// blocks in BlockSequence order) followed by the active turn's blocks,
    /// under the Conversation scope identity. Sync and policy-free by
    /// design — budget, selection and compaction over the merged view are
    /// Slice 5 territory and orchestrate through the policy layer, never
    /// mutate facts. The only failure source is a missing active turn.
    pub fn frame(&self, round_id: RoundId) -> Result<ContextFrame, ConversationError> {
        let active = match &self.active_turn {
            Some(active) => active,
            None => return Err(ConversationError::NoActiveTurn),
        };
        let history: Vec<TurnSnapshot> = self.history.iter().map(|e| e.snapshot.clone()).collect();
        Ok(merged_frame(
            &self.conversation_id,
            &history,
            active,
            round_id,
        ))
    }

    /// Borrow-split for a conversation driver's consume/return flow: the
    /// conversation id and committed history are read while the active
    /// turn is driven mutably. Public since Slice 12 — this is the exact
    /// seam any external conversation driver needs. Stamping still goes
    /// through the public `seal_turn` afterwards, so no second `&mut`
    /// seam exists.
    pub fn runner_parts(
        &mut self,
    ) -> (&ConversationId, Vec<TurnSnapshot>, Option<&mut TurnContext>) {
        (
            &self.conversation_id,
            self.history.iter().map(|e| e.snapshot.clone()).collect(),
            self.active_turn.as_mut(),
        )
    }

    /// Validated replay path: rebuild a session from committed history
    /// entries. The active slot starts empty (live paths never enter
    /// here); `ConversationVersion` resets to zero (replay is a fresh load —
    /// cross-persistence version semantics are Slice 5). Validation runs
    /// the shared `validate_history` closed set — any violation maps to
    /// `ConversationError::InvalidSequence` so callers never touch
    /// `ContextError`. `source_version` is accepted as a recorded fact — it
    /// counts fact commits and is not derivable from the blocks.
    pub fn from_history(
        conversation_id: ConversationId,
        entries: Vec<HistoryEntry>,
    ) -> Result<Self, ConversationError> {
        validate_history(&entries)?;
        Ok(Self {
            conversation_id,
            history: entries,
            active_turn: None,
            sealed_result: None,
            version: ConversationVersion(0),
        })
    }

    /// Committed history in ascending `TurnSequence` order.
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Number of committed entries in history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// The `ConversationVersion`, ticked at every controlled transition
    /// (`begin_turn` / `commit` / `abort_turn`); zero for a fresh or
    /// replayed session.
    pub fn version(&self) -> ConversationVersion {
        self.version
    }

    /// The session's identity.
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

/// Persist one session's committed history as [`HistoryEntry`] records —
/// the archive convenience port of the reference harness. Migrated from the
/// kernel in Slice 6.5 with the session aggregate: it is a session-archive
/// contract (completed-turn history), not a cross-harness capability, and
/// it is **not** wired into `commit` — the host's harness calls
/// `save_entry` after `ConversationState::commit`, keeping persistence
/// policy (batch writes, compression, fsync cadence, retry strategy)
/// host-owned.
///
/// Full paused outcomes are NOT stored through this port: the host saves
/// them in its own checkpoint document (fact state + continuation can live
/// atomically in one file).
///
/// Implementations are expected to be `Send + Sync` so they can sit
/// behind an `Arc` in the host's wiring.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Persist one committed history entry. Caller is the host's harness,
    /// invoked inside or after `ConversationState::commit`. Idempotent on
    /// `(conversation_id, sequence)` — repeated writes of the same entry
    /// are valid (host may retry).
    ///
    /// Entries do not carry their `ConversationId` (the id is implicit in
    /// the host's commit flow), so the trait takes it explicitly.
    async fn save_entry(
        &self,
        conversation_id: &ConversationId,
        entry: &HistoryEntry,
    ) -> Result<(), ConversationStoreError>;

    /// Load committed history for `conversation_id`. Returned entries are
    /// in their own `TurnSequence` ascending order; the session validates
    /// strict monotonicity through `ConversationState::from_history`.
    /// Returns `ConversationStoreError::NotFound` when the conversation
    /// has never been written.
    async fn load_entries(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<HistoryEntry>, ConversationStoreError>;
}

/// Session-archive error surface. Distinct from
/// `ArtifactStore::StoreError` (artifact persistence). Implementations
/// map their native failures onto these variants so the host has a
/// single error type to handle.
#[derive(Debug, thiserror::Error)]
pub enum ConversationStoreError {
    /// No entry has ever been written for this conversation; carries the
    /// conversation id.
    #[error("conversation not found: {0}")]
    NotFound(String),
    /// The underlying storage failed; carries the human-readable cause.
    #[error("io error: {0}")]
    Io(String),
    /// An entry could not be (de)serialized; carries the human-readable
    /// cause.
    #[error("serialization error: {0}")]
    Serialization(String),
    /// Stored data is present but unusable; carries the human-readable cause.
    #[error("corrupted data: {0}")]
    Corrupted(String),
}
