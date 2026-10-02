use causa_kernel::{
    BatchError, BlockId, ControlError, ProcessorError, SourceError, ToolResultPayload,
};

/// Failure to construct or change a tool registry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolRegistryError {
    /// Static tool definitions must have distinct names.
    #[error("duplicate static tool: {name}")]
    DuplicateTool {
        /// Conflicting advertised name.
        name: String,
    },
    /// A source with this identity is already registered.
    #[error("dynamic source already registered: {source_id}")]
    DuplicateSource {
        /// Conflicting source identity.
        source_id: String,
    },
    /// A requested source is not registered.
    #[error("no dynamic source registered: {source_id}")]
    UnknownSource {
        /// Requested source identity.
        source_id: String,
    },
}

/// Failure to bind a complete, unambiguous invocation catalog.
#[derive(Debug, thiserror::Error)]
pub enum ToolCatalogError {
    /// The parent invocation control stopped catalog acquisition.
    #[error(transparent)]
    Control(ControlError),
    /// A catalog could not be listed; its original error is retained.
    #[error("source {source_id} failed: {error}")]
    Source {
        /// Identity of the failed source registration.
        source_id: String,
        /// Original listing failure.
        error: SourceError,
    },
    /// A name occurred more than once in the assembled catalog.
    #[error("duplicate advertised tool: {name}")]
    DuplicateTool {
        /// Conflicting advertised name.
        name: String,
    },
    /// A static tool changed the name registered at construction.
    #[error("static tool name changed from {registered} to {actual}")]
    StaticNameChanged {
        /// Name registered at construction.
        registered: String,
        /// Name returned during this binding.
        actual: String,
    },
}

/// Position of a processor in the executor's fixed processing flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProcessorPhase {
    /// Before dispatch; pending inputs may be changed or completed.
    Before,
    /// After dispatch; completed output may be edited or reordered.
    After,
}

/// Why controlled processing returned the caller's current batch early.
// The confirmed API retains the actual rejected payload without indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, thiserror::Error)]
pub enum ToolProcessingError {
    /// The binding's parent cancellation or deadline stopped processing.
    #[error(transparent)]
    Control(ControlError),
    /// The input or internally produced batch is structurally invalid.
    #[error("invalid tool batch: {0}")]
    InvalidBatch(BatchError),
    /// This entry point only accepts new, entirely pending declarations.
    #[error("tool batch already contains {completed} completed calls")]
    NotFreshBatch {
        /// Number of completed calls at entry.
        completed: usize,
    },
    /// A processor returned its own error.
    #[error("{phase:?} processor {index} failed: {error}")]
    Processor {
        /// Processing phase.
        phase: ToolProcessorPhase,
        /// Zero-based index within that phase's configured processors.
        index: usize,
        /// Original processor error.
        error: ProcessorError,
    },
    /// A processor returned a structurally invalid batch.
    #[error("{phase:?} processor {index} returned an invalid batch: {error}")]
    InvalidProcessorBatch {
        /// Processing phase.
        phase: ToolProcessorPhase,
        /// Zero-based processor index.
        index: usize,
        /// Original batch validation error.
        error: BatchError,
    },
    /// A processor violated fixed membership or immutable completed facts.
    #[error("{phase:?} processor {index} violated its handoff: {message}")]
    Handoff {
        /// Processing phase.
        phase: ToolProcessorPhase,
        /// Zero-based processor index.
        index: usize,
        /// Specific violated handoff constraint.
        message: String,
    },
    /// An actual tool result could not be associated with its declaration.
    #[error("tool result rejected for declaration {call_block_id:?}: {error}")]
    ResultRejected {
        /// Declaration actually dispatched.
        call_block_id: BlockId,
        /// Original association or installation error.
        error: BatchError,
        /// Actual rejected payload, preserved for diagnosis.
        result: ToolResultPayload,
    },
}
