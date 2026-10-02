//! Implementation of the composition component.

use async_trait::async_trait;
use causa_kernel::{
    ArtifactStore, CallControl, DynamicToolSource, Tool, ToolCallContext, ToolDefinition,
    ToolExecutionError, ToolOutput, ToolResultPayload, ToolResultStatus,
};

/// A single dynamic-source tool exposed through the plain [`Tool`]
/// interface. The bridge routes `execute` to
/// [`DynamicToolSource::invoke_with_store`] (the source de-namespaces the
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

    async fn execute(&self, ctx: &ToolCallContext, control: &CallControl) -> ToolResultPayload {
        self.execute_with_store(ctx, control, None).await
    }

    async fn execute_with_store(
        &self,
        ctx: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        match self.source.invoke_with_store(ctx, control, store).await {
            Ok(result) => result,
            Err(e) => ToolResultPayload {
                call_block_id: ctx.call_block_id,
                status: match &e {
                    ToolExecutionError::TimedOut | ToolExecutionError::Cancelled => {
                        ToolResultStatus::UnknownOutcome
                    }
                    ToolExecutionError::UnknownTool(_) => ToolResultStatus::Rejected,
                    ToolExecutionError::Unavailable(_) | ToolExecutionError::Protocol(_) => {
                        ToolResultStatus::Failed
                    }
                },
                output: ToolOutput::new(serde_json::json!({ "error": e.to_string() })),
                media: Vec::new(),
                notes: Vec::new(),
            },
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
            Ok(vec![ToolDefinition {
                name: "mcp_stub_echo".into(),
                description: "stub".into(),
                parameters: serde_json::json!({"type": "object"}),
            }])
        }
        async fn invoke(
            &self,
            call: &ToolCallContext,
            _control: &CallControl,
        ) -> Result<ToolResultPayload, ToolExecutionError> {
            match &self.0 {
                Some(e) => Err(e.clone()),
                None => Ok(ToolResultPayload {
                    call_block_id: call.call_block_id,
                    status: ToolResultStatus::Succeeded,
                    output: ToolOutput::new(call.input.arguments.clone()),
                    media: Vec::new(),
                    notes: Vec::new(),
                }),
            }
        }
    }

    fn bridge(error: Option<ToolExecutionError>) -> ToolBridge {
        ToolBridge::new(
            std::sync::Arc::new(StubSource(error)),
            ToolDefinition {
                name: "mcp_stub_echo".into(),
                description: "stub".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
        )
    }

    fn call() -> ToolCallContext {
        ToolCallContext {
            call_block_id: crate::new_block_id(),
            input: ToolCallPayload {
                tool_name: "mcp_stub_echo".into(),
                arguments: serde_json::json!({"k": "v"}),
            },
            result_notes: Vec::new(),
        }
    }

    #[tokio::test]
    async fn success_passes_through() {
        let result = bridge(None).execute(&call(), &ctrl()).await;
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
            bridge.execute(&first, &control),
            bridge.execute(&second, &control),
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
            let result = bridge(Some(error.clone())).execute(&call(), &ctrl()).await;
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
