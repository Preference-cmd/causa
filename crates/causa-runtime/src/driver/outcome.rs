use crate::executor::{ToolCatalogError, ToolProcessingError};
use causa_kernel::{
    BatchError, BlockId, Context, ContextBlock, EditFailure, InvocationId, ModelBlockError,
    ModelInvokeError, ModelOutput, PrepareError, ToolBatch, ToolResultError, TurnId,
};

/// Materials returned on completion or controlled interruption.
#[derive(Debug)]
pub struct TurnOutcome {
    /// Identity supplied by the caller.
    pub turn_id: TurnId,
    /// All successfully committed materials; still editable.
    pub context: Context,
    /// The selected terminal result.
    pub result: TurnResult,
    /// Complete current batch whose declarations, but not results, were committed.
    pub uncommitted_tool_batch: Option<ToolBatch>,
}
/// The selected execution result, without an implicit execution history.
#[derive(Debug)]
pub enum TurnResult {
    /// A final model output was committed.
    Completed {
        /// Last complete output, including usage and reasoning.
        final_output: ModelOutput,
    },
    /// Controlled termination with all committed materials returned.
    Interrupted {
        /// The precise stage failure or control reason.
        cause: TurnInterruption,
    },
}
/// Typed interruption causes retaining the relevant failure inputs.
#[derive(Debug)]
pub enum TurnInterruption {
    /// Caller cancellation stopped an unselected execution stage.
    Cancelled {
        /// Logical invocation, or none when stopping between invocations.
        invocation_id: Option<InvocationId>,
    },
    /// The inherited absolute execution deadline stopped a stage.
    DeadlineExceeded {
        /// Logical invocation, or none when stopping between invocations.
        invocation_id: Option<InvocationId>,
    },
    /// The logical invocation limit was reached before creating a new invocation.
    MaxModelRounds {
        /// Configured maximum for this execution.
        limit: u32,
    },
    /// New declarations committed, but their cumulative count exceeds the limit.
    MaxToolCalls {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Configured maximum for this execution.
        limit: u32,
        /// Number of declarations accepted in earlier invocations.
        declared_before: u32,
        /// Untruncated number of newly committed declarations.
        declared_this_round: usize,
    },
    /// Tool binding failed before preparation or model invocation.
    ToolCatalogFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: ToolCatalogError,
    },
    /// Preparation failed; any edits already committed by the preparer remain.
    PrepareFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: PrepareError,
    },
    /// The logical gateway call failed; its original classification is retained.
    ModelCallFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: ModelInvokeError,
    },
    /// Generation reached its token limit; the complete output was not committed.
    ModelMaxTokens {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// The complete uncommitted model output, including usage and reasoning.
        output: ModelOutput,
    },
    /// The model refused; the complete output was not committed.
    ModelRefusal {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// The complete uncommitted model output, including usage and reasoning.
        output: ModelOutput,
    },
    /// Pure conversion failed before commit, retaining output and supplied IDs.
    ModelConversionFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: ModelBlockError,
        /// The complete uncommitted model output, including usage and reasoning.
        output: ModelOutput,
        /// All IDs supplied to the failed pure conversion.
        block_ids: Vec<BlockId>,
    },
    /// Fresh batch construction failed before the converted blocks were committed.
    ModelBatchFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: BatchError,
        /// The complete uncommitted model output, including usage and reasoning.
        output: ModelOutput,
        /// All converted blocks rejected before model commit.
        blocks: Vec<ContextBlock>,
    },
    /// Atomic model append failed; the edit error retains every submitted block.
    ModelCommitFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: EditFailure,
        /// The complete uncommitted model output, including usage and reasoning.
        output: ModelOutput,
    },
    /// Tool processing failed; the complete current batch is returned in the outcome.
    ToolProcessingFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: ToolProcessingError,
    },
    /// Result scope validation failed; the complete batch remains uncommitted.
    ToolResultValidationFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: ToolResultError,
    },
    /// Atomic tool-result append failed; both the batch and edit inputs are retained.
    ToolCommitFailed {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Original typed failure, including any inputs owned by that error.
        error: EditFailure,
    },
    /// The full result batch committed, then execution stopped on an unknown result.
    UnknownToolOutcome {
        /// The logical invocation associated with this stage.
        invocation_id: InvocationId,
        /// Declaration of the first unknown result in final postprocessor order.
        call_block_id: BlockId,
    },
}
impl std::fmt::Display for TurnInterruption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
