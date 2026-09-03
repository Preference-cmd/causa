use serde::{Deserialize, Serialize};

/// Unique identity of a turn. Opaque string; scopes every block, sequence,
/// and version belonging to that turn, and is checked on every model-door
/// invocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnId(pub String);
impl TurnId {
    /// Wraps an arbitrary string as a turn id.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Ordinal of a model round within a turn (0-based). Enters the frame-identity
/// and tool-call-id hash preimages, so the same logical call in different
/// rounds never collides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoundId(pub u32);

/// Identity of one model invocation: which turn, and which round inside it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InvocationId {
    /// The turn the invocation belongs to.
    pub turn_id: TurnId,
    /// The round within the turn.
    pub round_id: RoundId,
}

/// Monotonic per-turn block ordinal, assigned once when the fact machine
/// commits a block; dense and gap-free (validated on replay).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockSequence(pub u64);

/// Counts canonical fact commits of a turn; bumps exactly once per non-empty
/// commit and pins the frame identity for a round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextVersion(pub u64);
impl ContextVersion {
    /// Returns the successor version.
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

/// Unique identity of a fact block: the owning turn plus the block's dense
/// position within that turn.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId {
    /// The turn the block belongs to.
    pub turn_id: TurnId,
    /// The block's position in the turn's monotonic sequence.
    pub sequence: BlockSequence,
}

/// Frame identity/provenance tag. `Turn` scopes a single-turn projection;
/// `Conversation` scopes the lossless merged view (history + active turn).
/// Adding a scope variant is additive; the scope never changes block-level
/// operation rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameScope {
    /// Single-turn projection: only one turn's committed facts.
    Turn {
        /// The turn being projected.
        turn_id: TurnId,
        /// The turn's `ContextVersion` the frame was built from; part of the
        /// frame-identity preimage.
        source_version: ContextVersion,
    },
    /// Lossless merged view: conversation history plus the active turn.
    Conversation {
        /// The conversation being projected.
        conversation_id: ConversationId,
        /// The turn currently active in the conversation.
        active_turn_id: TurnId,
        /// The active turn's `ContextVersion`. History snapshots are
        /// immutable and identified by `TurnSequence`; within one round the
        /// (conversation_id, active_turn_id, source_version) triple is
        /// constant, so it pins the frame input.
        source_version: ContextVersion,
    },
}

/// Deterministic frame identity: a blake3 digest of the scope and round,
/// truncated to 16 hex chars. Equal `(scope, round)` inputs always yield the
/// same id; see `from_scope` for the preimage formats.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrameId(pub String);
impl FrameId {
    /// Scope-driven deterministic derivation — the canonical entry. The Turn
    /// branch replicates the historical preimage byte-for-byte
    /// (`turn|version|round`, colon-separated) so every pre-Slice-2 frame id
    /// is unchanged.
    pub fn from_scope(scope: &FrameScope, round_id: RoundId) -> Self {
        let input = match scope {
            FrameScope::Turn {
                turn_id,
                source_version,
            } => format!("{}:{}:{}", turn_id.0, source_version.0, round_id.0),
            FrameScope::Conversation {
                conversation_id,
                active_turn_id,
                source_version,
            } => format!(
                "conversation|{}|{}|{}|{}",
                conversation_id.0, active_turn_id.0, source_version.0, round_id.0
            ),
        };
        let hex = blake3::hash(input.as_bytes()).to_hex();
        Self(hex[..16].to_string())
    }

    /// Turn-scope thin wrapper — the historical entry, kept for Slice 1
    /// call sites and pinned equal to `from_scope(Turn)` by test.
    pub fn deterministic(
        turn_id: &TurnId,
        source_version: ContextVersion,
        round_id: RoundId,
    ) -> Self {
        Self::from_scope(
            &FrameScope::Turn {
                turn_id: turn_id.clone(),
                source_version,
            },
            round_id,
        )
    }
}

/// Unique identity of a conversation aggregate. Opaque string; scopes the
/// conversation's turns, sequences, and versions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationId(pub String);

/// Position of a committed turn within a conversation's history, assigned
/// exactly once by the conversation's commit transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TurnSequence(pub u64);

/// Counts controlled transitions of a conversation (`begin_turn` / `commit`
/// / `abort_turn`); the aggregate-level analogue of `ContextVersion`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationVersion(pub u64);
impl ConversationVersion {
    /// Returns the successor version.
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}
