use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identity of a turn. Opaque string checked on every model-door
/// invocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnId(pub String);
impl TurnId {
    /// Wraps an arbitrary string as a turn id.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Ordinal of a model round within a turn (0-based).
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

/// Counts canonical fact commits of a turn; bumps once per non-empty commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextVersion(pub u64);
impl ContextVersion {
    /// Returns the successor version.
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

/// UUID identity of one fact block, independent of its current position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BlockId(pub Uuid);
impl BlockId {
    /// Wraps a UUID supplied by the caller.
    pub const fn new(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

/// Projection provenance tag. `Turn` scopes a single-turn projection;
/// `Conversation` scopes the lossless merged view (history + active turn).
/// Adding a scope variant is additive; the scope never changes block-level
/// operation rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameScope {
    /// Single-turn projection: only one turn's committed facts.
    Turn {
        /// The turn being projected.
        turn_id: TurnId,
        /// The turn's `ContextVersion` the frame was built from.
        source_version: ContextVersion,
    },
    /// Lossless merged view: conversation history plus the active turn.
    Conversation {
        /// The conversation being projected.
        conversation_id: ConversationId,
        /// The turn currently active in the conversation.
        active_turn_id: TurnId,
        /// The active turn's `ContextVersion`. History snapshots are
        /// immutable; their session ordering lives in the runtime's
        /// `HistoryEntry`. Within one round the
        /// (conversation_id, active_turn_id, source_version) triple records
        /// the source coordinates. The projected block content is carried
        /// separately by `ContextFrame::model_context`.
        source_version: ContextVersion,
    },
}

/// Unique identity of a conversation aggregate. Opaque string.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationId(pub String);
