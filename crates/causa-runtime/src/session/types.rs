//! Public values for session work identity, observations, requests, and errors.

use std::sync::Arc;
use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, ModelOutput, ToolBatch, TurnContext, TurnId};

use crate::driver::TurnInterruption;

/// Identity of one accepted work: its conversation and assigned turn.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkRef {
    /// The conversation the work belongs to.
    pub conversation_id: ConversationId,
    /// The turn id assigned at acceptance; never reused by later work.
    pub turn_id: TurnId,
}

/// The session's management view of one work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkState {
    /// Accepted and not yet advancing the runner.
    Accepted,
    /// The runner is advancing this work.
    Running,
    /// Terminal: the work completed or was interrupted.
    Finished,
    /// Terminal: the worker exited abnormally.
    Faulted,
}

/// The terminal result retained for one finished work.
#[derive(Debug, Clone)]
pub enum FinishedKind {
    /// The model ended the turn and the turn was committed into history.
    Completed {
        /// The turn's final model output.
        final_output: ModelOutput,
    },
    /// The turn stopped early and its real facts stayed out of completed history.
    Interrupted {
        /// Why the turn was interrupted.
        cause: TurnInterruption,
        /// Owned facts from the aborted turn.
        facts: TurnContext,
        /// Uncommitted calls and results returned with the interrupted outcome.
        /// The shared pointer lets observations remain cloneable without
        /// cloning or serializing the batch itself.
        uncommitted_tool_batch: Option<Arc<ToolBatch>>,
    },
}

/// What a cancel request did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelOutcome {
    /// The work was running and its control token was fired.
    Signalled,
    /// The work was already terminal; its result was left unchanged.
    AlreadyTerminal,
}

/// Receipt for an accepted cancel request.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CancelReceipt {
    /// The work the request targeted.
    pub work: WorkRef,
    /// What the cancel did.
    pub outcome: CancelOutcome,
}

/// One submission: a caller-scoped request key and the new task's content.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SubmitRequest {
    /// Repeating this key with the same parts replays the original receipt;
    /// using it with different parts returns [`SessionError::Conflict`].
    pub request_key: String,
    /// The new task's content; must not be empty.
    pub parts: Vec<ContentPart>,
}

/// Receipt for an accepted submit.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkReceipt {
    /// The accepted work.
    pub work: WorkRef,
    /// Revision at acceptance; zero for new work.
    pub accepted_revision: u64,
}

/// One consistent snapshot of a work's published state.
#[derive(Debug, Clone)]
pub struct WorkObservation {
    /// The observed work.
    pub work: WorkRef,
    /// Revision increments on each observable state change.
    pub revision: u64,
    /// Current state.
    pub state: WorkState,
    /// The terminal result when finished.
    pub finished: Option<FinishedKind>,
    /// Retained fault reason when faulted.
    pub fault: Option<String>,
}

/// Why a finite wait returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitEnd {
    /// The work reached a terminal state.
    ReachedState,
    /// The timeout elapsed first.
    TimedOut,
}

/// Result of a finite wait: one snapshot and its completion reason.
#[derive(Debug, Clone)]
pub struct WaitOutcome {
    /// The snapshot taken when the wait ended.
    pub observation: WorkObservation,
    /// Whether the wait reached a terminal state or timed out.
    pub end: WaitEnd,
}

/// Session coordination configuration.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    /// Maximum number of retained works.
    pub retained_work_capacity: usize,
    /// Per-work deadline measured from acceptance.
    pub work_deadline: Option<Duration>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            retained_work_capacity: 256,
            work_deadline: None,
        }
    }
}

/// Rejections of session operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SessionError {
    /// The work does not belong to this session or is unknown.
    #[error("no such work in this session: {0:?}")]
    NotFound(WorkRef),
    /// Another work currently owns the session.
    #[error("session is busy with an active work: {active:?}")]
    Busy {
        /// Active work that must finish before new work can be submitted.
        active: WorkRef,
    },
    /// A request key was reused with different arguments.
    #[error("request key reused with different arguments")]
    Conflict,
    /// The retained-work capacity is exhausted.
    #[error("session retained-work capacity is exhausted")]
    CapacityExceeded,
    /// The submitted input is unusable.
    #[error("invalid submit input: {0}")]
    InvalidInput(String),
    /// The owner has been dropped or the session was closed.
    #[error("session is closed and accepts no new work")]
    Closed,
    /// The worker faulted; the session refuses more work.
    #[error("session worker faulted: {reason}")]
    Faulted {
        /// Retained fault reason.
        reason: String,
    },
}
