//! Every struct-like variant of the library-built enums matches with `..`.
//!
//! The functions only need to compile: each pattern names all current fields and
//! ends in `..`, which is the form a caller can keep across added fields. The
//! doc tests in `src/lib.rs` hold the same patterns without `..`.
// Naming each field is the point: the doc tests use the same lists.
#![allow(dead_code, clippy::unneeded_wildcard_pattern)]

fn batcherror(value: &causa::kernel::BatchError) -> bool {
    matches!(
        value,
        causa::kernel::BatchError::InvalidIndex {
            index: _,
            completed: _,
            total: _,
            ..
        } | causa::kernel::BatchError::MismatchedCall {
            expected: _,
            actual: _,
            ..
        } | causa::kernel::BatchError::Incomplete {
            completed: _,
            total: _,
            ..
        }
    )
}

fn editerror(value: &causa::kernel::EditError) -> bool {
    matches!(
        value,
        causa::kernel::EditError::InvalidRange {
            replacement_index: _,
            range: _,
            len: _,
            ..
        }
    )
}

fn toolresulterror(value: &causa::kernel::ToolResultError) -> bool {
    matches!(
        value,
        causa::kernel::ToolResultError::MissingDeclaration {
            call_block_id: _,
            ..
        } | causa::kernel::ToolResultError::AlreadyCompleted {
            call_block_id: _,
            ..
        } | causa::kernel::ToolResultError::DuplicateResult {
            call_block_id: _,
            ..
        }
    )
}

fn modelblockerror(value: &causa::kernel::ModelBlockError) -> bool {
    matches!(
        value,
        causa::kernel::ModelBlockError::EndTurnWithToolCalls { count: _, .. }
            | causa::kernel::ModelBlockError::BlockIdCountMismatch {
                expected: _,
                actual: _,
                ..
            }
            | causa::kernel::ModelBlockError::EmptyToolName { tool_index: _, .. }
            | causa::kernel::ModelBlockError::ArgumentsNotObject { tool_index: _, .. }
    )
}

#[cfg(feature = "runtime")]
fn runevent(value: &causa::runtime::RunEvent<'_>) -> bool {
    matches!(
        value,
        causa::runtime::RunEvent::ModelRequestReady { request: _, .. }
            | causa::runtime::RunEvent::ModelDelta {
                invocation_id: _,
                delta: _,
                ..
            }
            | causa::runtime::RunEvent::ModelOutput {
                invocation_id: _,
                output: _,
                ..
            }
            | causa::runtime::RunEvent::BlocksCommitted {
                invocation_id: _,
                blocks: _,
                ..
            }
            | causa::runtime::RunEvent::ToolBatchReady {
                invocation_id: _,
                batch: _,
                ..
            }
            | causa::runtime::RunEvent::ToolBatchReturned {
                invocation_id: _,
                batch: _,
                result: _,
                ..
            }
    )
}

#[cfg(feature = "runtime")]
fn turninterruption(value: &causa::runtime::TurnInterruption) -> bool {
    matches!(
        value,
        causa::runtime::TurnInterruption::Cancelled {
            invocation_id: _,
            ..
        } | causa::runtime::TurnInterruption::DeadlineExceeded {
            invocation_id: _,
            ..
        } | causa::runtime::TurnInterruption::MaxModelRounds { limit: _, .. }
            | causa::runtime::TurnInterruption::MaxToolCalls {
                invocation_id: _,
                limit: _,
                declared_before: _,
                declared_this_round: _,
                ..
            }
            | causa::runtime::TurnInterruption::ToolCatalogFailed {
                invocation_id: _,
                error: _,
                ..
            }
            | causa::runtime::TurnInterruption::PrepareFailed {
                invocation_id: _,
                error: _,
                ..
            }
            | causa::runtime::TurnInterruption::ModelCallFailed {
                invocation_id: _,
                error: _,
                ..
            }
            | causa::runtime::TurnInterruption::ModelMaxTokens {
                invocation_id: _,
                output: _,
                ..
            }
            | causa::runtime::TurnInterruption::ModelRefusal {
                invocation_id: _,
                output: _,
                ..
            }
            | causa::runtime::TurnInterruption::ModelContextWindowExceeded {
                invocation_id: _,
                output: _,
                ..
            }
            | causa::runtime::TurnInterruption::ModelStopUnrecognized {
                invocation_id: _,
                output: _,
                ..
            }
            | causa::runtime::TurnInterruption::ModelConversionFailed {
                invocation_id: _,
                error: _,
                output: _,
                block_ids: _,
                ..
            }
            | causa::runtime::TurnInterruption::ModelBatchFailed {
                invocation_id: _,
                error: _,
                output: _,
                blocks: _,
                ..
            }
            | causa::runtime::TurnInterruption::ModelCommitFailed {
                invocation_id: _,
                error: _,
                output: _,
                ..
            }
            | causa::runtime::TurnInterruption::ToolProcessingFailed {
                invocation_id: _,
                error: _,
                ..
            }
            | causa::runtime::TurnInterruption::ToolResultValidationFailed {
                invocation_id: _,
                error: _,
                ..
            }
            | causa::runtime::TurnInterruption::ToolCommitFailed {
                invocation_id: _,
                error: _,
                ..
            }
            | causa::runtime::TurnInterruption::UnknownToolOutcome {
                invocation_id: _,
                call_block_id: _,
                ..
            }
    )
}

#[cfg(feature = "runtime")]
fn toolprocessingerror(value: &causa::runtime::ToolProcessingError) -> bool {
    matches!(
        value,
        causa::runtime::ToolProcessingError::NotFreshBatch { completed: _, .. }
            | causa::runtime::ToolProcessingError::Processor {
                phase: _,
                index: _,
                error: _,
                ..
            }
            | causa::runtime::ToolProcessingError::InvalidProcessorBatch {
                phase: _,
                index: _,
                error: _,
                ..
            }
            | causa::runtime::ToolProcessingError::Handoff {
                phase: _,
                index: _,
                message: _,
                ..
            }
            | causa::runtime::ToolProcessingError::ResultRejected {
                call_block_id: _,
                error: _,
                result: _,
                ..
            }
    )
}

#[cfg(feature = "runtime")]
fn toolcatalogerror(value: &causa::runtime::ToolCatalogError) -> bool {
    matches!(
        value,
        causa::runtime::ToolCatalogError::Source {
            source_id: _,
            error: _,
            ..
        } | causa::runtime::ToolCatalogError::DuplicateTool { name: _, .. }
            | causa::runtime::ToolCatalogError::StaticNameChanged {
                registered: _,
                actual: _,
                ..
            }
    )
}

#[cfg(feature = "runtime")]
fn toolregistryerror(value: &causa::runtime::ToolRegistryError) -> bool {
    matches!(
        value,
        causa::runtime::ToolRegistryError::DuplicateTool { name: _, .. }
            | causa::runtime::ToolRegistryError::DuplicateSource { source_id: _, .. }
            | causa::runtime::ToolRegistryError::UnknownSource { source_id: _, .. }
    )
}
