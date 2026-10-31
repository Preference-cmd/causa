//! Composition layer — adapters bridging kernel ports into each other.
//!
//! [`ToolBridge`] adapts a `(DynamicToolSource, ToolDefinition)`
//! pair into a plain [`Tool`]. It is the executor's dynamic-dispatch
//! adapter (an invocation binding wraps a dynamic definition in a
//! bridge so it runs the same path as a local tool) and the
//! canonical home of the `ToolExecutionError` → status mapping. Hosts that
//! prefer snapshot semantics (describe once, register as static tools) reuse
//! the same bridge directly, passing each tool's hints to
//! [`ToolBridge::with_hints`]; for live catalogs prefer
//! `ToolExecutor::register_dynamic`.

use async_trait::async_trait;
use causa_kernel::{
    ArtifactStore, CallControl, DynamicToolSource, Tool, ToolCallContext, ToolDefinition,
    ToolExecutionError, ToolHints, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use serde_json::json;

/// A single dynamic-source tool exposed through the plain [`Tool`]
/// interface. The bridge routes `execute` to
/// [`DynamicToolSource::invoke`] (the source de-namespaces the
/// call; the host's artifact store is forwarded for media ingest) and
/// maps invoke errors onto structured results: a timeout or cancellation
/// is `UnknownOutcome` (the call may have run server-side), tagged with
/// `meta.extra.reason` = `"source_timed_out"` or `"source_cancelled"`; an
/// out-of-catalog name is `Rejected`, unavailability and protocol
/// failures are `Failed` with a model-readable message, and those mappings
/// carry no reason. The bridge returns
/// the recorded result only — what an `UnknownOutcome` result does next is
/// caller's responsibility (the runner commits the complete batch and
/// stops; a standalone bridge consumer chooses how to use its result).
///
/// Snapshot semantics: the [`ToolDefinition`] and [`ToolHints`] are fixed
/// at construction — listing changes after bridging are not picked up. For
/// live catalogs prefer `ToolExecutor::register_dynamic`.
pub struct ToolBridge {
    source: std::sync::Arc<dyn DynamicToolSource>,
    definition: ToolDefinition,
    hints: ToolHints,
}

impl ToolBridge {
    /// Bridge one tool definition of `source` into a static `Tool` that
    /// declares no hints.
    pub fn new(source: std::sync::Arc<dyn DynamicToolSource>, definition: ToolDefinition) -> Self {
        Self {
            source,
            definition,
            hints: ToolHints::default(),
        }
    }

    /// Sets the hints the bridged tool declares, usually the ones its
    /// source listed with the definition.
    pub fn with_hints(mut self, hints: ToolHints) -> Self {
        self.hints = hints;
        self
    }
}

#[async_trait]
impl Tool for ToolBridge {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn hints(&self) -> ToolHints {
        self.hints.clone()
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        match self.source.invoke(ctx, control, store).await {
            Ok(result) => result,
            Err(error) => {
                // The source observed no result. A timeout or cancellation
                // records why, aligning with the executor's reasons.
                let (status, reason) = match &error {
                    ToolExecutionError::TimedOut => {
                        (ToolResultStatus::UnknownOutcome, Some("source_timed_out"))
                    }
                    ToolExecutionError::Cancelled => {
                        (ToolResultStatus::UnknownOutcome, Some("source_cancelled"))
                    }
                    ToolExecutionError::UnknownTool(_) => (ToolResultStatus::Rejected, None),
                    ToolExecutionError::Unavailable(_) | ToolExecutionError::Protocol(_) => {
                        (ToolResultStatus::Failed, None)
                    }
                };
                let mut output = ToolOutput::new(json!({ "error": error.to_string() }));
                if let Some(reason) = reason {
                    output.meta.get_or_insert_default().extra = Some(json!({ "reason": reason }));
                }
                ToolResultPayload::new(ctx.call_block_id, status, output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causa_kernel::{CancellationToken, ToolCallPayload};

    /// What the source stub reports: an echoed success, a (clonable) error,
    /// or a terminal status and output of its own.
    enum StubReply {
        Echo,
        Error(ToolExecutionError),
        Reported(ToolResultStatus, ToolOutput),
    }

    struct StubSource(StubReply);

    #[async_trait]
    impl DynamicToolSource for StubSource {
        fn id(&self) -> &str {
            "stub"
        }
        async fn list(&self) -> Result<Vec<ToolDefinition>, causa_kernel::SourceError> {
            Ok(vec![ToolDefinition::new(
                "mcp_stub_echo",
                "stub",
                serde_json::json!({"type": "object"}),
            )])
        }
        async fn invoke(
            &self,
            call: &ToolCallContext,
            _control: &CallControl,
            _store: Option<&dyn causa_kernel::ArtifactStore>,
        ) -> Result<ToolResultPayload, ToolExecutionError> {
            match &self.0 {
                StubReply::Echo => Ok(ToolResultPayload::new(
                    call.call_block_id,
                    ToolResultStatus::Succeeded,
                    ToolOutput::new(call.input.arguments.clone()),
                )),
                StubReply::Error(e) => Err(e.clone()),
                StubReply::Reported(status, output) => Ok(ToolResultPayload::new(
                    call.call_block_id,
                    status.clone(),
                    output.clone(),
                )),
            }
        }
    }

    fn bridge(reply: StubReply) -> ToolBridge {
        ToolBridge::new(
            std::sync::Arc::new(StubSource(reply)),
            ToolDefinition::new(
                "mcp_stub_echo",
                "stub",
                serde_json::json!({"type": "object"}),
            ),
        )
    }

    fn call() -> ToolCallContext {
        ToolCallContext::new(
            crate::new_block_id(),
            ToolCallPayload::new("mcp_stub_echo", serde_json::json!({"k": "v"})),
        )
    }

    #[tokio::test]
    async fn success_passes_through() {
        let result = bridge(StubReply::Echo)
            .execute(&call(), &ctrl(), None)
            .await;
        assert_eq!(result.status, ToolResultStatus::Succeeded);
        assert_eq!(result.output.content, serde_json::json!({"k": "v"}));
    }

    #[tokio::test]
    async fn reported_status_and_output_pass_through() {
        let output = ToolOutput::new(serde_json::json!({ "partial": "first half" }));
        let call = call();
        let result = bridge(StubReply::Reported(
            ToolResultStatus::TimedOut,
            output.clone(),
        ))
        .execute(&call, &ctrl(), None)
        .await;
        assert_eq!(result.call_block_id, call.call_block_id);
        assert_eq!(result.status, ToolResultStatus::TimedOut);
        assert_eq!(result.output, output);
        assert!(
            result.output.meta.is_none(),
            "a source-reported result carries no bridge reason"
        );
    }

    #[tokio::test]
    async fn identical_dynamic_calls_keep_their_declaration_identity() {
        let first = call();
        let second = call();
        assert_ne!(first.call_block_id, second.call_block_id);
        let bridge = bridge(StubReply::Echo);
        let control = ctrl();
        let (a, b) = tokio::join!(
            bridge.execute(&first, &control, None),
            bridge.execute(&second, &control, None),
        );
        assert_eq!(a.call_block_id, first.call_block_id);
        assert_eq!(b.call_block_id, second.call_block_id);
        assert_eq!(a.output, b.output);
    }

    #[tokio::test]
    async fn error_mapping_is_canonical() {
        let cases: Vec<(ToolExecutionError, ToolResultStatus, Option<&str>)> = vec![
            (
                ToolExecutionError::TimedOut,
                ToolResultStatus::UnknownOutcome,
                Some("source_timed_out"),
            ),
            (
                ToolExecutionError::Cancelled,
                ToolResultStatus::UnknownOutcome,
                Some("source_cancelled"),
            ),
            (
                ToolExecutionError::UnknownTool("mcp_stub_nothere".into()),
                ToolResultStatus::Rejected,
                None,
            ),
            (
                ToolExecutionError::Unavailable("down".into()),
                ToolResultStatus::Failed,
                None,
            ),
            (
                ToolExecutionError::Protocol("bad frame".into()),
                ToolResultStatus::Failed,
                None,
            ),
        ];
        for (error, expected, expected_reason) in cases {
            let result = bridge(StubReply::Error(error.clone()))
                .execute(&call(), &ctrl(), None)
                .await;
            assert_eq!(result.status, expected, "mapping for {error}");
            let reason = result
                .output
                .meta
                .as_ref()
                .and_then(|meta| meta.extra.as_ref())
                .and_then(|extra| extra.get("reason"))
                .and_then(|reason| reason.as_str());
            assert_eq!(reason, expected_reason, "reason for {error}");
            assert!(
                result
                    .output
                    .content
                    .to_string()
                    .contains(&error.to_string()),
                "model-readable copy for {error}"
            );
        }
    }

    fn ctrl() -> CallControl {
        CallControl::new(CancellationToken::new(), None)
    }
}
