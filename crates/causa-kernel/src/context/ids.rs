use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Identity of an execution, independent of its context material.
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
