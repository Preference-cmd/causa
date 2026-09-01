//! Conversation persistence port — Slice 5A Phase B.
//!
//! `ConversationStore` is the kernel's narrow trait for persisting
//! completed-turn snapshots. It is **not** wired into the kernel's
//! commit path: `ConversationState::commit` produces the snapshot, the
//! host's harness then calls `save_snapshot`. This separation keeps
//! persistence policy (batch writes, compression, fsync cadence, retry
//! strategy) outside the kernel.
//!
//! ## Why no `FsConversationStore` here?
//!
//! The kernel ships no storage implementation. Concrete stores are host
//! concerns; an `FsConversationStore` reference lives in
//! `reimagine-agent-runtime/examples/fs_conversation_store.rs`.
//!
//! ## Why only `TurnSnapshot`?
//!
//! `ConversationState` is the kernel's live mutable state; its in-memory
//! active turn is not a deserializable artifact. `TurnSnapshot` is the
//! canonical history projection (Slice 2). The kernel validates snapshot
//! sequence monotonicity through `ConversationState::from_snapshots`;
//! the store is just a typed key/value surface.
//!
//! ## `StoreError`
//!
//! Distinct from `ArtifactStore::StoreError` — different domain error
//! face (artifact persistence vs. conversation persistence). Both can
//! coexist in the same host without ambiguity.

use async_trait::async_trait;

use crate::context::ids::ConversationId;
use crate::context::turn::TurnSnapshot;

/// Persist a single completed-turn snapshot; load committed history for
/// a conversation. The trait is invoked by the host's harness — never
/// by the kernel itself.
///
/// Implementations are expected to be `Send + Sync` so they can sit
/// behind an `Arc` in the host's wiring.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    /// Persist a single completed-turn snapshot. Caller is the host's
    /// harness, invoked inside or after `ConversationState::commit`.
    /// Idempotent on `(conversation_id, turn_sequence)` — repeated
    /// writes of the same snapshot are valid (host may retry).
    async fn save_snapshot(&self, snapshot: &TurnSnapshot) -> Result<(), ConversationStoreError>;

    /// Load committed history for `conversation_id`. Returned snapshots
    /// are in the snapshot's own `TurnSequence` ascending order; the
    /// kernel validates strict monotonicity through
    /// `ConversationState::from_snapshots`. Returns
    /// `ConversationStoreError::NotFound` when the conversation has
    /// never been written.
    async fn load_snapshots(
        &self,
        conversation_id: &ConversationId,
    ) -> Result<Vec<TurnSnapshot>, ConversationStoreError>;
}

/// Conversation-persistence error surface. Distinct from
/// `ArtifactStore::StoreError` (artifact persistence). Implementations
/// map their native failures onto these variants so the host has a
/// single error type to handle.
#[derive(Debug, thiserror::Error)]
pub enum ConversationStoreError {
    #[error("conversation not found: {0}")]
    NotFound(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("corrupted data: {0}")]
    Corrupted(String),
}
