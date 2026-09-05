//! The context model — the external rule interface of the kernel.
//!
//! Everything in this subtree is the validated vocabulary of a conversation
//! context: the block content shapes, the turn state machine and its deterministic
//! projections, model/tool value shapes, and identifiers. These types are
//! transparent data plus controlled transitions; they never call out to
//! behavior. Ports live in `crate::ports`; the canonical consumer of both
//! lives in `causa-runtime`. Session aggregation (single active slot,
//! completed-only history, commit-time ordering) is runtime vocabulary since
//! Slice 6.5 — `ConversationState` and the store port live there now; the
//! kernel keeps the facts (`TurnContext` / `TurnSnapshot`) and the shared
//! `merged_frame` projection.

pub mod block;
pub mod ids;
pub mod model;
pub mod tool_data;
pub mod turn;
