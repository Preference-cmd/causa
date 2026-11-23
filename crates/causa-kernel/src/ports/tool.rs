//! Tool behavior — the `Tool` trait and the `ArtifactStore` port, plus the
//! execution vocabulary they share with drivers: the bound surface and call
//! context. Recorded facts (results, outputs, artifacts) live in
//! `crate::context::tool_data`; batch dispatch lives in `causa-runtime`'s
//! executor. A tool returns its recorded result and nothing else —
//! handling unknown outcomes and retaining output belong to execution
//! consumers, not tool declarations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::context::block::{TextPayload, ToolCallPayload};
use crate::context::ids::BlockId;
use crate::context::tool_data::{ArtifactKind, ArtifactRef, ToolResultPayload};
use crate::context::tools::ToolDefinition;
use crate::ports::control::CallControl;

/// The tool definitions bound for one invocation: what the executor can
/// dispatch this round. A driver records a difference from the context's
/// tool state as a [`crate::ToolChange`] block, and renderers read the tools
/// from the frame, never from this value.
///
/// Serializable for logs and tests; the serialized form is not part of the
/// stable set and may change in any release.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolSurface {
    /// Model-facing [`ToolDefinition`]s, in offer order.
    pub definitions: Vec<ToolDefinition>,
}

impl ToolSurface {
    /// A surface offering no tools.
    pub fn empty() -> Self {
        Self {
            definitions: Vec::new(),
        }
    }

    /// Builds a surface from the given definitions.
    pub fn from_definitions(definitions: Vec<ToolDefinition>) -> Self {
        Self { definitions }
    }
}

/// Non-persisted, bind-time typed metadata carried beside [`ToolDefinition`].
///
/// Hosts and tool sources store arbitrary typed values keyed by [`TypeId`].
/// Causa carries metadata from tool binding to [`crate::ProcessorContext`] and
/// never inspects, interprets, or serializes it: metadata never enters
/// [`crate::Context`], [`crate::ToolChange`], or model wire requests, and a
/// change in metadata alone records no [`crate::ToolChange`].
///
/// Cloning is O(1) per entry because each value is reference-counted. An empty
/// `ToolMetadata` allocates nothing.
#[derive(Clone, Default)]
pub struct ToolMetadata {
    entries: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl ToolMetadata {
    /// An empty metadata bag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `value` and returns `self` for chained construction. Replaces
    /// any previous value of type `T`.
    pub fn with<T: Any + Send + Sync>(mut self, value: T) -> Self {
        self.insert(value);
        self
    }

    /// Inserts `value` under `TypeId::of::<T>()`, returning any previous value
    /// of type `T`.
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) -> Option<Arc<T>> {
        self.entries
            .insert(TypeId::of::<T>(), Arc::new(value))
            .and_then(|prev| prev.downcast::<T>().ok())
    }

    /// Borrows the value stored for type `T`, or `None` if no value of that
    /// exact type is present.
    pub fn get<T: Any>(&self) -> Option<&T> {
        self.entries
            .get(&TypeId::of::<T>())
            .and_then(|arc| (&**arc as &dyn Any).downcast_ref::<T>())
    }

    /// Returns `true` when a value of exact type `T` is stored.
    pub fn contains<T: Any>(&self) -> bool {
        self.entries.contains_key(&TypeId::of::<T>())
    }

    /// Returns the number of stored metadata entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no metadata entries are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl fmt::Debug for ToolMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolMetadata")
            .field("len", &self.entries.len())
            .finish_non_exhaustive()
    }
}

/// One tool as its catalog describes it: the model-facing definition and
/// non-persisted [`ToolMetadata`].
///
/// A [`crate::DynamicToolSource`] lists these from
/// [`crate::DynamicToolSource::describe`], and tool processors see the bound
/// set through [`crate::ProcessorContext::tools`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolDescription {
    /// The model-facing definition.
    pub definition: ToolDefinition,
    /// Non-persisted metadata declared beside the definition.
    pub metadata: ToolMetadata,
}

impl ToolDescription {
    /// A description with empty metadata.
    pub fn new(definition: ToolDefinition) -> Self {
        Self {
            definition,
            metadata: ToolMetadata::default(),
        }
    }

    /// Replaces the tool metadata.
    pub fn with_metadata(mut self, metadata: ToolMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Inserts a single metadata value of type `T`, replacing any prior value
    /// of the same type.
    pub fn with_meta<T: Any + Send + Sync>(mut self, value: T) -> Self {
        self.metadata.insert(value);
        self
    }
}

/// Non-persisted, call-scoped typed attachments handed from before-processors
/// to tool execution.
///
/// Each entry is stored behind an [`Arc`] keyed by [`TypeId`], keeping
/// [`ToolCallContext::clone`] O(1) per entry during executor dispatch. An
/// empty `CallAttachments` allocates nothing. Attachments are never persisted
/// in [`crate::Context`], never included in [`ToolResultPayload`], and never
/// sent to model protocols; [`crate::ToolBatch::resolve_at`] clears an entry's
/// attachments as soon as the call resolves.
#[derive(Clone, Default)]
pub struct CallAttachments {
    entries: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl CallAttachments {
    /// An empty attachment bag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `value` and returns `self` for chained construction. Replaces
    /// any previous attachment of type `T`.
    pub fn with<T: Any + Send + Sync>(mut self, value: T) -> Self {
        self.insert(value);
        self
    }

    /// Inserts `value` under `TypeId::of::<T>()`, returning any previous
    /// attachment of type `T`.
    pub fn insert<T: Any + Send + Sync>(&mut self, value: T) -> Option<Arc<T>> {
        self.entries
            .insert(TypeId::of::<T>(), Arc::new(value))
            .and_then(|prev| prev.downcast::<T>().ok())
    }

    /// Borrows the attachment stored for type `T`, or `None` if no attachment
    /// of that exact type is present.
    pub fn get<T: Any>(&self) -> Option<&T> {
        self.entries
            .get(&TypeId::of::<T>())
            .and_then(|arc| (&**arc as &dyn Any).downcast_ref::<T>())
    }

    /// Returns `true` when an attachment of exact type `T` is stored.
    pub fn contains<T: Any>(&self) -> bool {
        self.entries.contains_key(&TypeId::of::<T>())
    }

    /// Removes and returns the attachment stored for type `T`, if present.
    pub fn remove<T: Any + Send + Sync>(&mut self) -> Option<Arc<T>> {
        self.entries
            .remove(&TypeId::of::<T>())
            .and_then(|prev| prev.downcast::<T>().ok())
    }

    /// Drops all stored attachments.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the number of stored attachments.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no attachments are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl fmt::Debug for CallAttachments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallAttachments")
            .field("len", &self.entries.len())
            .finish_non_exhaustive()
    }
}

/// Identity of one tool dispatch, handed to [`Tool::execute`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolCallContext {
    /// The declaration block whose result this dispatch produces.
    pub call_block_id: BlockId,
    /// The effective input after any pre-processing edits.
    pub input: ToolCallPayload,
    /// Notes added before execution. `ToolBatch::resolve_at` prepends these
    /// to the returned result's notes; tool implementations should return
    /// only notes they add and must not copy this field into the result.
    pub result_notes: Vec<TextPayload>,
    pub(crate) attachments: CallAttachments,
}
impl ToolCallContext {
    /// A call context with no result notes and no attachments.
    pub fn new(call_block_id: BlockId, input: ToolCallPayload) -> Self {
        Self {
            call_block_id,
            input,
            result_notes: Vec::new(),
            attachments: CallAttachments::default(),
        }
    }

    /// Builds execution material from a committed declaration.
    pub fn from_declaration(call_block_id: BlockId, declaration: &ToolCallPayload) -> Self {
        Self {
            call_block_id,
            input: declaration.clone(),
            result_notes: Vec::new(),
            attachments: CallAttachments::default(),
        }
    }

    /// Borrows the call-scoped attachment of exact type `T`, if present.
    pub fn attachment<T: Any>(&self) -> Option<&T> {
        self.attachments.get::<T>()
    }

    /// Borrows the full call-scoped attachment bag.
    pub fn attachments(&self) -> &CallAttachments {
        &self.attachments
    }

    /// Attaches a value of type `T`, replacing any prior attachment of the same
    /// type. Useful for unit-testing [`Tool::execute`] directly without a
    /// [`crate::ToolBatch`].
    pub fn with_attachment<T: Any + Send + Sync>(mut self, value: T) -> Self {
        self.attachments.insert(value);
        self
    }
}

/// Provenance a caller attaches to bytes handed to [`ArtifactStore::persist`],
/// so the store can label what it receives.
#[non_exhaustive]
pub struct ArtifactHint {
    /// The tool whose output produced the bytes.
    pub tool_name: String,
    /// The tool call the bytes belong to.
    pub call_block_id: BlockId,
    /// The [`ArtifactKind`] classification of the bytes.
    pub kind: ArtifactKind,
}
impl ArtifactHint {
    /// A hint for the artifact one call produced.
    pub fn new(tool_name: impl Into<String>, call_block_id: BlockId, kind: ArtifactKind) -> Self {
        Self {
            tool_name: tool_name.into(),
            call_block_id,
            kind,
        }
    }
}

/// Error surface of the [`ArtifactStore`] port.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StoreError {
    /// Persisting the artifact failed; carries the human-readable cause.
    #[error("persist failed: {0}")]
    Persist(String),
    /// Reading an artifact back failed; carries the human-readable cause.
    #[error("read failed: {0}")]
    Read(String),
}

/// Persistence port for tool artifacts — where oversized tool outputs are
/// spilled and read back. Callers persist raw bytes plus an [`ArtifactHint`]
/// and receive an [`ArtifactRef`] to embed in the result; concrete stores
/// are host concerns.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    /// Stores `data`, tagged with `hint`, and returns the [`ArtifactRef`]
    /// describing the persisted artifact.
    async fn persist(&self, data: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError>;
    /// Reads back the bytes stored under `id`; an optional byte range bounds
    /// the read.
    async fn read(
        &self,
        id: &str,
        range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError>;
}

/// The tool port: one callable tool the model can invoke. Implementations
/// live outside the kernel; the driver's executor dispatches them. A tool
/// returns only its recorded result. Handling unknown outcomes and retaining
/// output belong to the execution consumer, not tool declarations.
///
/// # Requirements on `execute`
///
/// An executor may poll every call of one batch on a single task, with
/// nothing spawned. An implementation therefore:
///
/// - must not block the thread. Blocking or CPU-heavy work runs on a
///   blocking thread (for example `tokio::task::spawn_blocking`) and the
///   tool awaits it;
/// - must watch its [`CallControl`] itself. A turn stop — cancellation or
///   the turn deadline — drops the call future: the executor stops polling,
///   a started call settles as
///   [`ToolResultStatus::UnknownOutcome`](crate::context::tool_data::ToolResultStatus::UnknownOutcome).
///   Once the executor observes the stop it polls the call no further, so a
///   value returned after that is not recorded. Returning promptly on
///   cancellation releases the resources the future still owns; it does not
///   make the result observed;
/// - must return, cleanup included, by its effective call deadline to
///   report
///   [`ToolResultStatus::TimedOut`](crate::context::tool_data::ToolResultStatus::TimedOut)
///   or
///   [`ToolResultStatus::Cancelled`](crate::context::tool_data::ToolResultStatus::Cancelled).
///   The executor enforces a call deadline with `timeout_at`, which polls the
///   call before its own timer and drops the future once the deadline has
///   expired. A tool that races the same deadline and returns without
///   awaiting anything further is therefore recorded. A tool that awaits
///   cleanup after it stops has to stop early enough for that cleanup to
///   finish before the deadline; otherwise the future is dropped mid-cleanup
///   and the call settles as `UnknownOutcome`.
///
/// Dropping the future runs on the executor's task, so [`Drop`] must not
/// block and must be idempotent. A future that starts external work — a
/// spawned task, a child process, a remote operation — cleans it up in
/// [`Drop`].
///
/// For a tool that meets these, a stop takes effect at its next yield point.
/// A tool that holds the thread also holds its sibling calls and the
/// executor's own stop and timeout checks, so nothing is guaranteed for it:
/// not cancellation, not the deadline, not a per-call timeout.
///
/// Calls of one batch are concurrent, not parallel, and their number is not
/// bounded. A tool that needs a limit shares a semaphore across its calls.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The model-facing [`ToolDefinition`] for this tool.
    fn definition(&self) -> ToolDefinition;
    /// Non-persisted metadata carried beside [`Tool::definition`]. An executor
    /// reads it once when binding the tool and exposes it to tool processors
    /// through [`crate::ProcessorContext::tool`] and
    /// [`crate::ProcessorContext::tool_metadata`]. The default is empty.
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::default()
    }
    /// Runs one call under the call's [`CallControl`], returning the
    /// recorded result. `store` is the host's [`ArtifactStore`] when one is
    /// configured: a tool that produces media or oversized output persists
    /// the bytes there and returns a reference; a tool that does not need
    /// storage ignores it.
    async fn execute(
        &self,
        ctx: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload;
}
