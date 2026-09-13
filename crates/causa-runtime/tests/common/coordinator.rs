//! Reference multi-session coordinator — the Slice 8 Phase D harness.
//!
//! Test-side only: compiled into the acceptance targets, never exported by
//! `causa-runtime`. It holds several [`Session`] owners keyed by conversation
//! and adds exactly the cross-session responsibilities the single session
//! deliberately leaves out (proposal §13.5):
//!
//! 1. **create** — assemble a session from the harness's resolved profile,
//!    admit its initial task, and only then return a receipt. Same-key retries
//!    replay the original receipt; the same key with different arguments is a
//!    conflict. This create table is separate from the session's own
//!    submit/resume/cancel tables.
//! 2. **lookup / routing** — find a session handle by `ConversationId`, or
//!    route a `WorkRef` to its session. Routing never duplicates the session's
//!    work table: a ref for a known conversation but an unknown turn is
//!    rejected by the session itself.
//! 3. **trusted depth** — a child is created from a parent `WorkRef` the
//!    coordinator already registered; the depth is derived here and never
//!    supplied by the caller.
//! 4. **finite run capacity** — a slot is occupied while a registered work is
//!    `Accepted`/`Running` and freed when it pauses or finishes. Admission,
//!    and the capacity re-application on `resume`, are checked before
//!    anything is accepted.
//! 5. **collective shutdown** — close every held session and settle it.
//!
//! Everything runs under one `std::sync::Mutex` with no `.await` inside it:
//! `SessionHandle::submit` / `observe` / `resume` / `cancel` are synchronous,
//! so check-and-admit is atomic without ever holding the coordination lock
//! across a model call, a tool call, or a child work. Lock order is
//! coordinator-table → session-core, one-way; a session never calls back here.
//!
//! Contract notes:
//!
//! - `create` enforces the harness's `ConversationId` uniqueness (proposal
//!   §13.4): a second create for a registered conversation is refused with
//!   [`CoordinatorError::ConversationExists`] instead of replacing the live
//!   session. A next work for an existing session goes through its handle.
//! - The assembly function runs synchronously **inside** the admission lock,
//!   so it must not perform external reads or call back into the coordinator.
//!   Phase E's material preparation must pre-resolve what it needs before
//!   `create`, or move assembly out of the lock first: proposal §7 forbids
//!   holding the global management lock across assembly that reaches outside
//!   the coordinator.
//! - Capacity counts works admitted by `create` or a new coordinator `resume`.
//!   Resuming a later work updates the counted reference for that session;
//!   replaying an old receipt leaves the counted reference unchanged.
//!   Direct `SessionHandle`
//!   use — including handles handed out by [`Coordinator::handle`] /
//!   [`Coordinator::route`] — bypasses it by design (proposal §7: a caller
//!   that does not use the coordinator needs no capacity gate). A host that
//!   wants every running admission counted must use `create` / coordinator
//!   `resume`; submitting a new work through a handle still bypasses the gate.
//! - After collective shutdown the coordinator refuses `resume` with
//!   [`CoordinatorError::Closed`], same-key replays included, so nothing new
//!   can be accepted inside the wind-down window. The session's own replay
//!   table stays reachable through a `SessionHandle`; the cross-session
//!   `create` replay (the D2 contract) is still served.
//!
//! Out of scope by phase boundary: persisting the create table (F5),
//! model-facing tool wrappers and call-id pairing (F-tool), per-work materials
//! (E), and context editing (Q). Holding sessions here does not publish a
//! `SessionManager`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use causa_kernel::{ContentPart, ConversationId, ModelGateway, Tool};
use causa_runtime::{
    ConversationState, ResumeRequest, Session, SessionBuildRejection, SessionCheckpoint,
    SessionConfig, SessionError, SessionHandle, SubmitRequest, TurnRunOptions, TurnRunner,
    WaitOutcome, WorkObservation, WorkReceipt, WorkRef, WorkState,
};

use super::{PausingInteraction, runner_with};

/// The coordinator's finite limits. Both axes are explicit: no hidden
/// unlimited default (proposal §9).
#[derive(Debug, Clone, Copy)]
pub struct CoordinatorConfig {
    /// Works that may be `Accepted`/`Running` across every held session.
    /// `0` is a legal finite value meaning no work may start.
    pub max_parallel: usize,
    /// Trusted creation depth ceiling; a root is depth `0`.
    pub max_depth: u32,
}

/// Where a create sits in the trusted creation relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreateOrigin {
    /// A root session, depth `0`.
    Root,
    /// A child of a work the coordinator already admitted. Only the reference
    /// is given; the depth is resolved from the coordinator's own table.
    Child {
        /// The creating work — the coordinator's trusted parent anchor.
        parent: WorkRef,
    },
}

/// One typed create request. The whole value is the conflict identity of the
/// create: a same-key retry replays only when every field matches.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateRequest {
    /// Caller-scoped idempotency key for the cross-session create table.
    pub request_key: String,
    /// The conversation the harness names for the new session.
    pub conversation_id: ConversationId,
    /// The initial task's content parts.
    pub parts: Vec<ContentPart>,
    /// Root or trusted child.
    pub origin: CreateOrigin,
    /// Harness-resolved capability label; opaque here (it selects a profile
    /// in `profile_factory` and participates in conflict comparison).
    pub selection: String,
}

impl CreateRequest {
    /// A root create.
    pub fn root(
        conversation_id: &str,
        request_key: &str,
        parts: Vec<ContentPart>,
        selection: &str,
    ) -> Self {
        Self {
            request_key: request_key.into(),
            conversation_id: ConversationId(conversation_id.into()),
            parts,
            origin: CreateOrigin::Root,
            selection: selection.into(),
        }
    }

    /// A child create whose depth the coordinator derives from `parent`.
    pub fn child(
        conversation_id: &str,
        request_key: &str,
        parts: Vec<ContentPart>,
        selection: &str,
        parent: &WorkRef,
    ) -> Self {
        Self {
            request_key: request_key.into(),
            conversation_id: ConversationId(conversation_id.into()),
            parts,
            origin: CreateOrigin::Child {
                parent: parent.clone(),
            },
            selection: selection.into(),
        }
    }
}

/// The four by-value inputs of `Session::new`, as assembled by the harness.
pub struct SessionParts {
    /// The idle conversation state; its identity must match the request.
    pub state: ConversationState,
    /// The assembled runner (real gateway, executor, hook).
    pub runner: Arc<TurnRunner>,
    /// The assembled run options (policy, interaction, capabilities).
    pub options: TurnRunOptions,
    /// The session's own coordination configuration.
    pub config: SessionConfig,
}

/// Everything the coordinator itself can refuse.
///
/// Request refusals happen before anything is accepted and never consume the
/// create key; `Session` carries the session's own local rejection through.
#[derive(Debug)]
pub enum CoordinatorError {
    /// The create key was reused with different arguments.
    Conflict,
    /// The coordinator was shut down and this is not a same-key replay.
    Closed,
    /// A session is already registered for the conversation. A create never
    /// replaces a live session (proposal §13.4: the harness guarantees one
    /// owner per `ConversationId`); a next work goes through its handle.
    ConversationExists(ConversationId),
    /// The finite run capacity is exhausted (checked before assembly).
    CapacityExceeded,
    /// The trusted depth ceiling would be exceeded.
    DepthExceeded {
        /// The derived depth of the refused create.
        depth: u32,
        /// The configured ceiling.
        max: u32,
    },
    /// The parent reference is not a work this coordinator admitted.
    UnknownParent(WorkRef),
    /// No session is registered for the conversation.
    UnknownConversation(ConversationId),
    /// The assembled state's identity differs from the request's.
    ConversationMismatch {
        /// The identity the request named.
        expected: ConversationId,
        /// The identity the assembled state carries.
        actual: ConversationId,
    },
    /// `Session::new` refused the assembled parts; all inputs come back.
    Build(Box<SessionBuildRejection>),
    /// The session rejected the operation locally.
    Session(SessionError),
}

impl PartialEq for CoordinatorError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Conflict, Self::Conflict)
            | (Self::Closed, Self::Closed)
            | (Self::CapacityExceeded, Self::CapacityExceeded) => true,
            (
                Self::DepthExceeded { depth: a, max: b },
                Self::DepthExceeded { depth: c, max: d },
            ) => a == c && b == d,
            (Self::UnknownParent(a), Self::UnknownParent(b)) => a == b,
            (Self::UnknownConversation(a), Self::UnknownConversation(b)) => a == b,
            (
                Self::ConversationMismatch {
                    expected: a,
                    actual: b,
                },
                Self::ConversationMismatch {
                    expected: c,
                    actual: d,
                },
            ) => a == c && b == d,
            (Self::ConversationExists(a), Self::ConversationExists(b)) => a == b,
            (Self::Build(a), Self::Build(b)) => a.error == b.error,
            (Self::Session(a), Self::Session(b)) => a == b,
            // Every remaining pairing is unequal. Naming each current variant
            // keeps the match exhaustive: a new variant fails to compile until
            // it states its own equality, instead of silently comparing false.
            (Self::Conflict | Self::Closed | Self::CapacityExceeded, _)
            | (Self::ConversationExists(_), _)
            | (Self::DepthExceeded { .. }, _)
            | (Self::UnknownParent(_), _)
            | (Self::UnknownConversation(_), _)
            | (Self::ConversationMismatch { .. }, _)
            | (Self::Build(_), _)
            | (Self::Session(_), _) => false,
        }
    }
}

/// One accepted create: the create table entry, and nothing the session
/// already owns. No work state, result, history, or execution slot is copied.
struct Entry {
    request: CreateRequest,
    depth: u32,
    /// The work this create admitted; the trusted parent anchor for children.
    work: WorkRef,
    /// The latest work admitted here, observed for capacity without copying
    /// the session's execution state. A receipt replay never changes it.
    capacity_work: WorkRef,
    /// The original receipt, replayed for a same-key retry.
    receipt: WorkReceipt,
    /// The lifecycle owner, kept so collective shutdown can settle it.
    session: Arc<Session>,
    /// The shared operation entry.
    handle: SessionHandle,
}

#[derive(Default)]
struct Table {
    closed: bool,
    by_conversation: HashMap<ConversationId, Entry>,
    create_index: HashMap<String, ConversationId>,
}

/// The reference multi-session coordinator; see the module docs.
pub struct Coordinator {
    config: CoordinatorConfig,
    factory: Box<dyn Fn(&CreateRequest) -> SessionParts + Send + Sync>,
    table: Mutex<Table>,
}

impl Coordinator {
    /// Build a coordinator over the harness's assembly function.
    ///
    /// The factory runs synchronously inside the admission lock and must not
    /// call back into this coordinator.
    pub fn new(
        config: CoordinatorConfig,
        factory: impl Fn(&CreateRequest) -> SessionParts + Send + Sync + 'static,
    ) -> Self {
        Self {
            config,
            factory: Box::new(factory),
            table: Mutex::new(Table::default()),
        }
    }

    /// Create one session and admit its initial task.
    ///
    /// Synchronous: dedup, the depth and capacity checks, assembly, and the
    /// initial `submit` all happen under the coordination lock with no await
    /// point, so concurrent same-key creates admit exactly one work.
    ///
    /// A conversation is owned by at most one session: a create for a
    /// registered `ConversationId` is refused with
    /// [`CoordinatorError::ConversationExists`] instead of replacing the live
    /// owner.
    ///
    /// # Errors
    ///
    /// See [`CoordinatorError`]; every refusal leaves no session, no work, and
    /// no consumed create key.
    pub fn create(&self, request: CreateRequest) -> Result<WorkReceipt, CoordinatorError> {
        let mut table = self
            .table
            .lock()
            .expect("the coordinator table is never poisoned");

        // Replay first: a same-key retry keeps working after shutdown.
        if let Some(id) = table.create_index.get(&request.request_key).cloned() {
            let entry = table
                .by_conversation
                .get(&id)
                .expect("the create index always points at a registered session");
            return if entry.request == request {
                Ok(entry.receipt.clone())
            } else {
                Err(CoordinatorError::Conflict)
            };
        }

        if table.closed {
            return Err(CoordinatorError::Closed);
        }
        if table.by_conversation.contains_key(&request.conversation_id) {
            return Err(CoordinatorError::ConversationExists(
                request.conversation_id.clone(),
            ));
        }
        let depth = match &request.origin {
            CreateOrigin::Root => 0,
            CreateOrigin::Child { parent } => {
                let parent_entry = table
                    .by_conversation
                    .get(&parent.conversation_id)
                    .ok_or_else(|| CoordinatorError::UnknownParent(parent.clone()))?;
                if parent_entry.work != *parent {
                    return Err(CoordinatorError::UnknownParent(parent.clone()));
                }
                parent_entry.depth + 1
            }
        };
        if depth > self.config.max_depth {
            return Err(CoordinatorError::DepthExceeded {
                depth,
                max: self.config.max_depth,
            });
        }
        if Self::in_use(&table) >= self.config.max_parallel {
            return Err(CoordinatorError::CapacityExceeded);
        }

        let parts = (self.factory)(&request);
        if parts.state.conversation_id() != &request.conversation_id {
            return Err(CoordinatorError::ConversationMismatch {
                expected: request.conversation_id.clone(),
                actual: parts.state.conversation_id().clone(),
            });
        }
        let session = Session::new(parts.state, parts.runner, parts.options, parts.config)
            .map_err(|rejection| CoordinatorError::Build(Box::new(rejection)))?;
        let handle = session.handle();
        let receipt = handle
            .submit(SubmitRequest {
                request_key: request.request_key.clone(),
                parts: request.parts.clone(),
            })
            .map_err(CoordinatorError::Session)?;

        table
            .create_index
            .insert(request.request_key.clone(), request.conversation_id.clone());
        table.by_conversation.insert(
            request.conversation_id.clone(),
            Entry {
                depth,
                work: receipt.work.clone(),
                capacity_work: receipt.work.clone(),
                receipt: receipt.clone(),
                session: Arc::new(session),
                handle,
                request,
            },
        );
        Ok(receipt)
    }

    /// The operation entry for a registered conversation.
    ///
    /// # Errors
    ///
    /// [`CoordinatorError::UnknownConversation`] when nothing is registered.
    pub fn handle(&self, id: &ConversationId) -> Result<SessionHandle, CoordinatorError> {
        let table = self
            .table
            .lock()
            .expect("the coordinator table is never poisoned");
        table
            .by_conversation
            .get(id)
            .map(|entry| entry.handle.clone())
            .ok_or_else(|| CoordinatorError::UnknownConversation(id.clone()))
    }

    /// Route a work reference to its session's handle.
    ///
    /// Routing is by conversation only; whether the turn belongs to that
    /// session is the session's own check ([`SessionError::NotFound`]).
    ///
    /// # Errors
    ///
    /// [`CoordinatorError::UnknownConversation`] when nothing is registered.
    pub fn route(&self, work: &WorkRef) -> Result<SessionHandle, CoordinatorError> {
        self.handle(&work.conversation_id)
    }

    /// Read one work's published state through its session.
    ///
    /// # Errors
    ///
    /// [`CoordinatorError::UnknownConversation`] or the session's
    /// [`SessionError::NotFound`].
    pub fn observe(&self, work: &WorkRef) -> Result<WorkObservation, CoordinatorError> {
        self.route(work)?
            .observe(work)
            .map_err(CoordinatorError::Session)
    }

    /// Wait finitely for one work to reach `Paused` / `Finished` / `Faulted`.
    ///
    /// The coordination lock is not held across the wait: the routed handle is
    /// an owned clone.
    ///
    /// # Errors
    ///
    /// [`CoordinatorError::UnknownConversation`] or the session's
    /// [`SessionError::NotFound`].
    pub async fn wait(
        &self,
        work: &WorkRef,
        timeout: Duration,
    ) -> Result<WaitOutcome, CoordinatorError> {
        self.route(work)?
            .wait(work, timeout)
            .await
            .map_err(CoordinatorError::Session)
    }

    /// Continue a paused work, replaying accepted receipts before applying
    /// run capacity to new admissions.
    ///
    /// A capacity refusal returns before `SessionHandle::resume` is called,
    /// so the work stays paused and the resume key is not consumed. Once the
    /// coordinator has been shut down, resume is refused with
    /// [`CoordinatorError::Closed`] — a same-key replay included — so no new
    /// acceptance can slip into the wind-down window; the session's own replay
    /// stays reachable through a `SessionHandle`.
    ///
    /// # Errors
    ///
    /// [`CoordinatorError::CapacityExceeded`], or the session's rejection.
    pub fn resume(
        &self,
        work: &WorkRef,
        expected_revision: u64,
        request_key: &str,
        request: ResumeRequest,
    ) -> Result<WorkReceipt, CoordinatorError> {
        let mut table = self
            .table
            .lock()
            .expect("the coordinator table is never poisoned");
        let entry = table
            .by_conversation
            .get(&work.conversation_id)
            .ok_or_else(|| CoordinatorError::UnknownConversation(work.conversation_id.clone()))?;
        if table.closed {
            return Err(CoordinatorError::Closed);
        }
        if let Some(receipt) = entry
            .handle
            .resume_receipt(work, expected_revision, request_key, &request)
            .map_err(CoordinatorError::Session)?
        {
            return Ok(receipt);
        }
        match entry.handle.observe(work) {
            Ok(observation) if observation.state == WorkState::Paused => {
                if Self::in_use(&table) >= self.config.max_parallel {
                    return Err(CoordinatorError::CapacityExceeded);
                }
            }
            Ok(_) => {}
            Err(error) => return Err(CoordinatorError::Session(error)),
        }
        let receipt = entry
            .handle
            .resume(work, expected_revision, request_key.to_string(), request)
            .map_err(CoordinatorError::Session)?;
        table
            .by_conversation
            .get_mut(&work.conversation_id)
            .expect("the held table lock preserves the entry")
            .capacity_work = work.clone();
        Ok(receipt)
    }

    /// Export one held session's save envelope (observation helper for tests).
    ///
    /// # Errors
    ///
    /// [`CoordinatorError::UnknownConversation`] or the session's rejection.
    pub fn checkpoint(&self, id: &ConversationId) -> Result<SessionCheckpoint, CoordinatorError> {
        let session = {
            let table = self
                .table
                .lock()
                .expect("the coordinator table is never poisoned");
            table
                .by_conversation
                .get(id)
                .map(|entry| Arc::clone(&entry.session))
                .ok_or_else(|| CoordinatorError::UnknownConversation(id.clone()))?
        };
        session.checkpoint().map_err(CoordinatorError::Session)
    }

    /// The trusted depth of a create's work, if the coordinator admitted it.
    pub fn depth_of(&self, work: &WorkRef) -> Option<u32> {
        let table = self
            .table
            .lock()
            .expect("the coordinator table is never poisoned");
        table
            .by_conversation
            .get(&work.conversation_id)
            .filter(|entry| entry.work == *work)
            .map(|entry| entry.depth)
    }

    /// How many sessions are registered.
    pub fn session_count(&self) -> usize {
        self.table
            .lock()
            .expect("the coordinator table is never poisoned")
            .by_conversation
            .len()
    }

    /// Close every held session and settle it.
    ///
    /// The closed flag and the owner list are taken under one lock, so a
    /// concurrent create either completes and is collected or observes
    /// `closed` — none escapes. The awaits happen after the lock is released.
    pub async fn shutdown(&self) {
        let sessions: Vec<Arc<Session>> = {
            let mut table = self
                .table
                .lock()
                .expect("the coordinator table is never poisoned");
            table.closed = true;
            table
                .by_conversation
                .values()
                .map(|entry| Arc::clone(&entry.session))
                .collect()
        };
        for session in sessions {
            session.shutdown().await;
        }
    }

    /// Slots occupied right now: registered works observed as `Accepted` or
    /// `Running`. A paused or terminal work frees its slot; an unreadable work
    /// is counted as occupied rather than assumed free.
    fn in_use(table: &Table) -> usize {
        table
            .by_conversation
            .values()
            .filter(|entry| match entry.handle.observe(&entry.capacity_work) {
                Ok(observation) => {
                    matches!(observation.state, WorkState::Accepted | WorkState::Running)
                }
                Err(_) => true,
            })
            .count()
    }
}

/// The harness-resolved capabilities behind one create `selection` label.
pub struct Profile {
    /// The model gateway this profile resolves to.
    pub gateway: Arc<dyn ModelGateway>,
    /// The tool capability this profile resolves to.
    pub tools: Vec<Arc<dyn Tool>>,
    /// Whether the approval gate pauses on the first tool-use batch.
    pub pause_on_tool_use: bool,
    /// The session's own coordination configuration.
    pub config: SessionConfig,
}

impl Profile {
    /// A profile that runs to completion with no tools.
    pub fn completing(gateway: Arc<dyn ModelGateway>) -> Self {
        Self {
            gateway,
            tools: Vec::new(),
            pause_on_tool_use: false,
            config: SessionConfig::default(),
        }
    }

    /// A profile whose first tool-use batch pauses for approval.
    pub fn pausing(gateway: Arc<dyn ModelGateway>, tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            gateway,
            tools,
            pause_on_tool_use: true,
            config: SessionConfig::default(),
        }
    }

    /// Replace the tool capability set.
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Replace the session coordination configuration.
    pub fn with_config(mut self, config: SessionConfig) -> Self {
        self.config = config;
        self
    }
}

/// Selection label → profile.
pub type Profiles = HashMap<&'static str, Profile>;

/// The reference assembly function: resolve a create's `selection` into the
/// real runner/options a [`Session`] is built from.
///
/// A selection with no wired profile panics on the caller's thread — the
/// harness must resolve every selection it accepts (proposal §13.4).
pub fn profile_factory(
    profiles: Profiles,
) -> impl Fn(&CreateRequest) -> SessionParts + Send + Sync + 'static {
    move |request: &CreateRequest| {
        let profile = profiles
            .get(request.selection.as_str())
            .unwrap_or_else(|| panic!("no profile wired for selection {:?}", request.selection));
        let options = if profile.pause_on_tool_use {
            TurnRunOptions {
                interaction: Arc::new(PausingInteraction),
                ..Default::default()
            }
        } else {
            TurnRunOptions::default()
        };
        SessionParts {
            state: ConversationState::new(request.conversation_id.clone()),
            runner: Arc::new(runner_with(profile.gateway.clone(), profile.tools.clone())),
            options,
            config: profile.config.clone(),
        }
    }
}
