//! Optional synchronous observations; consumers own retained copies.

use crate::executor::ToolProcessingError;
use causa_kernel::{ContextBlock, InvocationId, ModelOutput, ModelRequest, StreamDelta, ToolBatch};

/// Borrowed execution observations. Callbacks must return promptly.
///
/// No event history is retained by the runner. `ModelDelta` excludes `Done`
/// and `Error`; a committed notification describes only the actual append.
#[derive(Debug)]
#[non_exhaustive]
pub enum RunEvent<'a> {
    /// A complete logical request, immediately before gateway entry.
    #[non_exhaustive]
    ModelRequestReady {
        /// The request to be sent.
        request: &'a ModelRequest,
    },
    /// A nonterminal streaming observation.
    #[non_exhaustive]
    ModelDelta {
        /// The logical invocation.
        invocation_id: &'a InvocationId,
        /// Text, reasoning, tool-call or usage observation.
        delta: &'a StreamDelta,
    },
    /// A complete model output, before material conversion.
    #[non_exhaustive]
    ModelOutput {
        /// The logical invocation.
        invocation_id: &'a InvocationId,
        /// The accepted complete output.
        output: &'a ModelOutput,
    },
    /// An atomic append succeeded, including an empty append.
    #[non_exhaustive]
    BlocksCommitted {
        /// The logical invocation.
        invocation_id: &'a InvocationId,
        /// Actual appended blocks borrowed from the context.
        blocks: &'a [ContextBlock],
    },
    /// Newly committed declarations are about to enter tool processing.
    #[non_exhaustive]
    ToolBatchReady {
        /// The logical invocation.
        invocation_id: &'a InvocationId,
        /// The complete uncommitted batch.
        batch: &'a ToolBatch,
    },
    /// Tool processing actually returned; results are not yet committed.
    #[non_exhaustive]
    ToolBatchReturned {
        /// The logical invocation.
        invocation_id: &'a InvocationId,
        /// The complete returned batch.
        batch: &'a ToolBatch,
        /// The module's return status, independent of subsequent commit.
        result: Result<(), &'a ToolProcessingError>,
    },
}
/// A synchronous callback borrowing each observation for the duration of the call.
pub type RunObserver = dyn for<'a> Fn(&RunEvent<'a>) + Send + Sync;
