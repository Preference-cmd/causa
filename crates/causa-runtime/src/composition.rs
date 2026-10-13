//! Composition layer — adapters bridging kernel ports into each other.
//!
//! [`ToolBridge`] adapts a `(DynamicToolSource, ToolDefinition)`
//! pair into a plain [`Tool`]. It is the executor's dynamic-dispatch
//! adapter (an invocation binding wraps a dynamic definition in a
//! bridge so it runs the same path as a local tool) and the
//! canonical home of the `ToolExecutionError` → status mapping. Hosts that
//! prefer snapshot semantics (list once, register as static tools) reuse
//! the same bridge directly; for live catalogs prefer
//! `ToolExecutor::register_dynamic`.

use async_trait::async_trait;
use causa_kernel::{
    ArtifactStore, CallControl, DynamicToolSource, Tool, ToolCallContext, ToolDefinition,
    ToolExecutionError, ToolOutput, ToolResultPayload, ToolResultStatus,
};

/// A single dynamic-source tool exposed through the plain [`Tool`]
/// interface. The bridge routes `execute` to
/// [`DynamicToolSource::invoke`] (the source de-namespaces the
/// call; the host's artifact store is forwarded for media ingest) and
/// maps invoke errors onto structured results: a timeout or cancellation
/// is `UnknownOutcome` (the call may have run server-side), an
/// out-of-catalog name is `Rejected`, unavailability and protocol
/// failures are `Failed` with a model-readable message. The bridge returns
/// the recorded result only — what an `UnknownOutcome` result does next is
/// caller's responsibility (the runner commits the complete batch and
/// stops; a standalone bridge consumer chooses how to use its result).
///
/// Snapshot semantics: the [`ToolDefinition`] is fixed at construction —
/// listing changes after bridging are not picked up. For live catalogs
/// prefer `ToolExecutor::register_dynamic`.
pub struct ToolBridge {
    source: std::sync::Arc<dyn DynamicToolSource>,
    definition: ToolDefinition,
}

impl ToolBridge {
    /// Bridge one tool definition of `source` into a static `Tool`.
    pub fn new(source: std::sync::Arc<dyn DynamicToolSource>, definition: ToolDefinition) -> Self {
        Self { source, definition }
    }
}

#[async_trait]
impl Tool for ToolBridge {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        match self.source.invoke(ctx, control, store).await {
            Ok(result) => result,
            Err(e) => ToolResultPayload::new(
                ctx.call_block_id,
                match &e {
                    ToolExecutionError::TimedOut | ToolExecutionError::Cancelled => {
                        ToolResultStatus::UnknownOutcome
                    }
                    ToolExecutionError::UnknownTool(_) => ToolResultStatus::Rejected,
                    ToolExecutionError::Unavailable(_) | ToolExecutionError::Protocol(_) => {
                        ToolResultStatus::Failed
                    }
                },
                ToolOutput::new(serde_json::json!({ "error": e.to_string() })),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causa_kernel::{CancellationToken, ToolCallPayload};

    /// Source stub: `None` succeeds with the call's arguments as output;
    /// `Some` yields the (clonable) error.
    struct StubSource(Option<ToolExecutionError>);

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
                Some(e) => Err(e.clone()),
                None => Ok(ToolResultPayload::new(
                    call.call_block_id,
                    ToolResultStatus::Succeeded,
                    ToolOutput::new(call.input.arguments.clone()),
                )),
            }
        }
    }

    fn bridge(error: Option<ToolExecutionError>) -> ToolBridge {
        ToolBridge::new(
            std::sync::Arc::new(StubSource(error)),
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
        let result = bridge(None).execute(&call(), &ctrl(), None).await;
        assert_eq!(result.status, ToolResultStatus::Succeeded);
        assert_eq!(result.output.content, serde_json::json!({"k": "v"}));
    }

    #[tokio::test]
    async fn identical_dynamic_calls_keep_their_declaration_identity() {
        let first = call();
        let second = call();
        assert_ne!(first.call_block_id, second.call_block_id);
        let bridge = bridge(None);
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
        let cases: Vec<(ToolExecutionError, ToolResultStatus)> = vec![
            (
                ToolExecutionError::TimedOut,
                ToolResultStatus::UnknownOutcome,
            ),
            (
                ToolExecutionError::Cancelled,
                ToolResultStatus::UnknownOutcome,
            ),
            (
                ToolExecutionError::UnknownTool("mcp_stub_nothere".into()),
                ToolResultStatus::Rejected,
            ),
            (
                ToolExecutionError::Unavailable("down".into()),
                ToolResultStatus::Failed,
            ),
            (
                ToolExecutionError::Protocol("bad frame".into()),
                ToolResultStatus::Failed,
            ),
        ];
        for (error, expected) in cases {
            let result = bridge(Some(error.clone()))
                .execute(&call(), &ctrl(), None)
                .await;
            assert_eq!(result.status, expected, "mapping for {error}");
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
