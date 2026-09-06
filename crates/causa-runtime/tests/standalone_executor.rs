//! Standalone executor evidence (Slice 13 Phase B.3): tools execute
//! directly through `ToolExecutor::execute_with_limits` — no `TurnRunner`,
//! no `ConversationState`, no session — with the same result pairing,
//! error mapping, and limit semantics the reference driver gets.

use async_trait::async_trait;
use causa_kernel::{
    CallControl, DynamicToolSource, Tool, ToolCallContext, ToolCallId, ToolCallPayload,
    ToolDefinition, ToolExecutionError, ToolExecutionOutcome, ToolOutput, ToolOutputLimits,
    ToolResultPayload, ToolResultStatus,
};
use causa_runtime::{ToolBridge, ToolExecutor};
use serde_json::json;
use std::sync::Arc;

fn call(name: &str, args: serde_json::Value) -> ToolCallPayload {
    ToolCallPayload {
        call_id: ToolCallId("standalone:call".into()),
        tool_name: name.into(),
        arguments: args,
    }
}

fn ctrl() -> CallControl {
    CallControl::new(tokio_util::sync::CancellationToken::new(), None)
}

/// A plain local tool: echoes arguments back.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echo".into(),
            parameters: json!({"type": "object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, _control: &CallControl) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(ctx.arguments.clone()),
            media: Vec::new(),
        })
    }
}

/// Source stub for the bridge: echoes, or yields a caller-chosen error.
struct StubSource(Option<ToolExecutionError>);

#[async_trait]
impl DynamicToolSource for StubSource {
    fn id(&self) -> &str {
        "stub"
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, causa_kernel::SourceError> {
        Ok(vec![ToolDefinition {
            name: "mcp_srv_echo".into(),
            description: "stub".into(),
            parameters: json!({"type": "object"}),
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
                media: Vec::new(),
            })),
        }
    }
}

/// Direct execution pairs the result with the call id and needs nothing
/// beyond the executor itself.
#[tokio::test]
async fn executor_executes_a_static_tool_without_a_runner() {
    let executor = ToolExecutor::from_vec(vec![Arc::new(EchoTool)]);
    let payload = call("echo", json!({"q": 1}));
    let outcome = executor
        .execute_with_limits(
            payload.clone(),
            ctrl(),
            None,
            None,
            ToolOutputLimits { max_tokens: 10_000 },
        )
        .await;
    assert_eq!(outcome.result.call_id, payload.call_id);
    assert_eq!(outcome.result.status, ToolResultStatus::Succeeded);
    assert_eq!(outcome.result.output.content, json!({"q": 1}));
    assert_eq!(outcome.policy, causa_kernel::UnknownOutcomePolicy::Stop);
}

/// Snapshot semantics: a `(source, definition)` pair bridged once becomes a
/// static tool; the bridge's error mapping rides along (unknown name →
/// `Rejected`, timeout → `UnknownOutcome`).
#[tokio::test]
async fn bridged_dynamic_tool_executes_and_maps_errors() {
    let source = Arc::new(StubSource(None));
    let bridge = ToolBridge::new(
        source,
        ToolDefinition {
            name: "mcp_srv_echo".into(),
            description: "stub".into(),
            parameters: json!({"type": "object"}),
        },
    );
    let executor = ToolExecutor::from_vec(vec![Arc::new(bridge)]);
    let outcome = executor
        .execute_with_limits(
            call("mcp_srv_echo", json!({"hello": "world"})),
            ctrl(),
            None,
            None,
            ToolOutputLimits { max_tokens: 10_000 },
        )
        .await;
    assert_eq!(outcome.result.status, ToolResultStatus::Succeeded);
    assert_eq!(outcome.result.output.content, json!({"hello": "world"}));

    // Out-of-catalog source error maps to Rejected with a model-readable copy.
    let failing = Arc::new(StubSource(Some(ToolExecutionError::UnknownTool(
        "gone".into(),
    ))));
    let bridge = ToolBridge::new(
        failing,
        ToolDefinition {
            name: "mcp_srv_echo".into(),
            description: "stub".into(),
            parameters: json!({"type": "object"}),
        },
    );
    let executor = ToolExecutor::from_vec(vec![Arc::new(bridge)]);
    let outcome = executor
        .execute_with_limits(
            call("mcp_srv_echo", json!({})),
            ctrl(),
            None,
            None,
            ToolOutputLimits { max_tokens: 10_000 },
        )
        .await;
    assert_eq!(outcome.result.status, ToolResultStatus::Rejected);
    assert_eq!(outcome.policy, causa_kernel::UnknownOutcomePolicy::Stop);
}

/// A tool's own limit declaration overrides the global fallback (the
/// executor takes `tool.output_limits().unwrap_or(global)`), and the
/// default unknown-outcome policy is `Stop` — both unchanged.
#[tokio::test]
async fn tool_limit_declaration_overrides_the_global_fallback() {
    struct BigLimitTool;

    #[async_trait]
    impl Tool for BigLimitTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "big".into(),
                description: "big".into(),
                parameters: json!({"type": "object"}),
            }
        }
        fn output_limits(&self) -> Option<ToolOutputLimits> {
            Some(ToolOutputLimits { max_tokens: 10_000 })
        }
        async fn execute(
            &self,
            ctx: &ToolCallContext,
            _control: &CallControl,
        ) -> ToolExecutionOutcome {
            ToolExecutionOutcome::new(ToolResultPayload {
                call_id: ctx.call_id.clone(),
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(json!({ "text": "x".repeat(4000) })),
                media: Vec::new(),
            })
        }
    }

    let executor = ToolExecutor::from_vec(vec![Arc::new(BigLimitTool)]);
    // Global limit 100 would truncate; the tool's 10,000 wins and the
    // output passes through whole.
    let outcome = executor
        .execute_with_limits(
            call("big", json!({})),
            ctrl(),
            None,
            None,
            ToolOutputLimits { max_tokens: 100 },
        )
        .await;
    assert_eq!(outcome.result.status, ToolResultStatus::Succeeded);
    assert_eq!(
        outcome.result.output.truncation,
        causa_kernel::Truncation::None
    );
}
