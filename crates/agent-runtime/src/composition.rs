//! Composition layer — adapters bridging kernel ports into each other.
//!
//! Slice 10: [`ToolBridge`] adapts a `(DynamicToolSource, ToolDefinition)`
//! pair into a plain [`Tool`]. It is the executor's dynamic-dispatch
//! adapter (`ToolExecutor::execute_with_limits` wraps a dynamic call in a
//! bridge so it runs the same static path as a local tool) and the
//! canonical home of the `ToolExecutionError` → status mapping. Hosts that
//! prefer snapshot semantics (list once, register as static tools) reuse
//! the same bridge directly; for live catalogs prefer
//! `ToolExecutor::register_dynamic`.

use async_trait::async_trait;
use reimagine_context_kernel::ToolCallPayload;
use reimagine_context_kernel::{
    CallControl, DynamicToolSource, Tool, ToolCallContext, ToolDefinition, ToolExecutionError,
    ToolExecutionOutcome, ToolOutput, ToolResultPayload, ToolResultStatus,
};

/// A single dynamic-source tool exposed through the plain [`Tool`]
/// interface. The bridge routes `execute` to
/// [`DynamicToolSource::invoke`] (the source de-namespaces the call) and
/// maps invoke errors onto structured outcomes: a timeout or cancellation
/// is `UnknownOutcome` (the call may have run server-side), an
/// out-of-catalog name is `Rejected`, unavailability and protocol
/// failures are `Failed` with a model-readable message.
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

    async fn execute(&self, ctx: &ToolCallContext, control: &CallControl) -> ToolExecutionOutcome {
        let payload = ToolCallPayload {
            call_id: ctx.call_id.clone(),
            tool_name: ctx.tool_name.clone(),
            arguments: ctx.arguments.clone(),
        };
        match self.source.invoke(&payload, control).await {
            Ok(outcome) => outcome,
            Err(e) => ToolExecutionOutcome::new(ToolResultPayload {
                call_id: ctx.call_id.clone(),
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
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reimagine_context_kernel::{CancellationToken, ToolCallId};

    /// Source stub: `None` succeeds with the call's arguments as output;
    /// `Some` yields the (clonable) error.
    struct StubSource(Option<ToolExecutionError>);

    #[async_trait]
    impl DynamicToolSource for StubSource {
        fn id(&self) -> &str {
            "stub"
        }
        async fn list(&self) -> Result<Vec<ToolDefinition>, reimagine_context_kernel::SourceError> {
            Ok(vec![ToolDefinition {
                name: "mcp_stub_echo".into(),
                description: "stub".into(),
                parameters: serde_json::json!({"type": "object"}),
            }])
        }
        async fn invoke(
            &self,
            call: &ToolCallPayload,
            _control: &CallControl,
        ) -> Result<ToolExecutionOutcome, ToolExecutionError> {
            match &self.0 {
                Some(e) => Err(e.clone()),
                None => Ok(ToolExecutionOutcome::new(ToolResultPayload {
                    call_id: call.call_id.clone(),
                    status: ToolResultStatus::Succeeded,
                    output: ToolOutput::new(call.arguments.clone()),
                })),
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
            call_id: ToolCallId("call-1".into()),
            tool_name: "mcp_stub_echo".into(),
            arguments: serde_json::json!({"k": "v"}),
        }
    }

    #[tokio::test]
    async fn success_passes_through() {
        let outcome = bridge(None).execute(&call(), &ctrl()).await;
        assert_eq!(outcome.result.status, ToolResultStatus::Succeeded);
        assert_eq!(outcome.result.output.content, serde_json::json!({"k": "v"}));
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
            let outcome = bridge(Some(error.clone())).execute(&call(), &ctrl()).await;
            assert_eq!(outcome.result.status, expected, "mapping for {error}");
            assert!(
                outcome
                    .result
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
