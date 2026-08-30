use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TurnId(pub String);
impl TurnId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RoundId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InvocationId {
    pub turn_id: TurnId,
    pub round_id: RoundId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockSequence(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextVersion(pub u64);
impl ContextVersion {
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId {
    pub turn_id: TurnId,
    pub sequence: BlockSequence,
}

/// Frame identity/provenance tag. `Turn` scopes a single-turn projection;
/// `Conversation` scopes the lossless merged view (history + active turn).
/// Adding a scope variant is additive; the scope never changes block-level
/// operation rules.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameScope {
    Turn {
        turn_id: TurnId,
        source_version: ContextVersion,
    },
    Conversation {
        conversation_id: ConversationId,
        active_turn_id: TurnId,
        /// The active turn's `ContextVersion`. History snapshots are
        /// immutable and identified by `TurnSequence`; within one round the
        /// (conversation_id, active_turn_id, source_version) triple is
        /// constant, so it pins the frame input.
        source_version: ContextVersion,
    },
}

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TurnSequence(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConversationVersion(pub u64);
impl ConversationVersion {
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}
