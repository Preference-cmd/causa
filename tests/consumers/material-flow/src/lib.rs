//! Public facade consumers for the material-flow contracts.
//!
//! This independent workspace depends on causa alone within the Causa family.
//!
//! The modules below hold no code. Each doc test is a pattern that names every
//! field of a struct-like variant and leaves out `..`; it must fail to compile,
//! so adding a field to one of those variants cannot break a caller. The same
//! patterns with `..` compile in `tests/variant_patterns.rs`.
#![deny(unsafe_code)]

/// Variants of the kernel's library-built enums.
pub mod kernel_variants {
    /// Patterns over `BatchError` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::BatchError) {
    ///     if let causa::kernel::BatchError::InvalidIndex { index: _, completed: _, total: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::BatchError) {
    ///     if let causa::kernel::BatchError::MismatchedCall { expected: _, actual: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::BatchError) {
    ///     if let causa::kernel::BatchError::Incomplete { completed: _, total: _ } = e {}
    /// }
    /// ```
    pub struct BatchErrorPatterns;

    /// Patterns over `EditError` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::EditError) {
    ///     if let causa::kernel::EditError::InvalidRange { replacement_index: _, range: _, len: _ } = e {}
    /// }
    /// ```
    pub struct EditErrorPatterns;

    /// Patterns over `ToolResultError` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::ToolResultError) {
    ///     if let causa::kernel::ToolResultError::MissingDeclaration { call_block_id: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::ToolResultError) {
    ///     if let causa::kernel::ToolResultError::AlreadyCompleted { call_block_id: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::ToolResultError) {
    ///     if let causa::kernel::ToolResultError::DuplicateResult { call_block_id: _ } = e {}
    /// }
    /// ```
    pub struct ToolResultErrorPatterns;

    /// Patterns over `ModelBlockError` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::ModelBlockError) {
    ///     if let causa::kernel::ModelBlockError::EndTurnWithToolCalls { count: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::ModelBlockError) {
    ///     if let causa::kernel::ModelBlockError::BlockIdCountMismatch { expected: _, actual: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::ModelBlockError) {
    ///     if let causa::kernel::ModelBlockError::EmptyToolName { tool_index: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::kernel::ModelBlockError) {
    ///     if let causa::kernel::ModelBlockError::ArgumentsNotObject { tool_index: _ } = e {}
    /// }
    /// ```
    pub struct ModelBlockErrorPatterns;
}

/// Variants of the runtime's library-built enums.
#[cfg(feature = "runtime")]
pub mod runtime_variants {
    /// Patterns over `RunEvent` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::RunEvent<'_>) {
    ///     if let causa::runtime::RunEvent::ModelRequestReady { request: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::RunEvent<'_>) {
    ///     if let causa::runtime::RunEvent::ModelDelta { invocation_id: _, delta: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::RunEvent<'_>) {
    ///     if let causa::runtime::RunEvent::ModelOutput { invocation_id: _, output: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::RunEvent<'_>) {
    ///     if let causa::runtime::RunEvent::BlocksCommitted { invocation_id: _, blocks: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::RunEvent<'_>) {
    ///     if let causa::runtime::RunEvent::ToolBatchReady { invocation_id: _, batch: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::RunEvent<'_>) {
    ///     if let causa::runtime::RunEvent::ToolBatchReturned { invocation_id: _, batch: _, result: _ } = e {}
    /// }
    /// ```
    pub struct RunEventPatterns;

    /// Patterns over `TurnInterruption` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::Cancelled { invocation_id: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::DeadlineExceeded { invocation_id: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::MaxModelRounds { limit: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::MaxToolCalls { invocation_id: _, limit: _, declared_before: _, declared_this_round: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ToolCatalogFailed { invocation_id: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::PrepareFailed { invocation_id: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ModelCallFailed { invocation_id: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ModelMaxTokens { invocation_id: _, output: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ModelRefusal { invocation_id: _, output: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ModelContextWindowExceeded { invocation_id: _, output: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ModelStopUnrecognized { invocation_id: _, output: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ModelConversionFailed { invocation_id: _, error: _, output: _, block_ids: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ModelBatchFailed { invocation_id: _, error: _, output: _, blocks: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ModelCommitFailed { invocation_id: _, error: _, output: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ToolProcessingFailed { invocation_id: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ToolResultValidationFailed { invocation_id: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::ToolCommitFailed { invocation_id: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::TurnInterruption) {
    ///     if let causa::runtime::TurnInterruption::UnknownToolOutcome { invocation_id: _, call_block_id: _ } = e {}
    /// }
    /// ```
    pub struct TurnInterruptionPatterns;

    /// Patterns over `ToolProcessingError` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolProcessingError) {
    ///     if let causa::runtime::ToolProcessingError::NotFreshBatch { completed: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolProcessingError) {
    ///     if let causa::runtime::ToolProcessingError::Processor { phase: _, index: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolProcessingError) {
    ///     if let causa::runtime::ToolProcessingError::InvalidProcessorBatch { phase: _, index: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolProcessingError) {
    ///     if let causa::runtime::ToolProcessingError::Handoff { phase: _, index: _, message: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolProcessingError) {
    ///     if let causa::runtime::ToolProcessingError::ResultRejected { call_block_id: _, error: _, result: _ } = e {}
    /// }
    /// ```
    pub struct ToolProcessingErrorPatterns;

    /// Patterns over `ToolCatalogError` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolCatalogError) {
    ///     if let causa::runtime::ToolCatalogError::Source { source_id: _, error: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolCatalogError) {
    ///     if let causa::runtime::ToolCatalogError::DuplicateTool { name: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolCatalogError) {
    ///     if let causa::runtime::ToolCatalogError::StaticNameChanged { registered: _, actual: _ } = e {}
    /// }
    /// ```
    pub struct ToolCatalogErrorPatterns;

    /// Patterns over `ToolRegistryError` that list every field and omit `..`.
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolRegistryError) {
    ///     if let causa::runtime::ToolRegistryError::DuplicateTool { name: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolRegistryError) {
    ///     if let causa::runtime::ToolRegistryError::DuplicateSource { source_id: _ } = e {}
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// fn f(e: causa::runtime::ToolRegistryError) {
    ///     if let causa::runtime::ToolRegistryError::UnknownSource { source_id: _ } = e {}
    /// }
    /// ```
    pub struct ToolRegistryErrorPatterns;
}
