//! Standalone executor evidence: tools execute directly through
//! `ToolExecutor::execute_with_limits` — no `TurnRunner`, no
//! `ConversationState`, no session — with the same result pairing, error
//! mapping, and limit semantics the reference driver gets. The executor
//! returns the recorded result only; the unknown-outcome action is the
//! caller's configuration.

use async_trait::async_trait;
use causa_kernel::{
    CallControl, DynamicToolSource, Tool, ToolCallContext, ToolCallId, ToolCallPayload,
    ToolDefinition, ToolExecutionError, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use causa_runtime::{ToolBridge, ToolExecutor, ToolOutputLimits};
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

fn limits(max_tokens: usize) -> ToolOutputLimits {
    ToolOutputLimits { max_tokens }
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
    async fn execute(&self, ctx: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(ctx.arguments.clone()),
            media: Vec::new(),
        }
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
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        match &self.0 {
            Some(e) => Err(e.clone()),
            None => Ok(ToolResultPayload {
                call_id: call.call_id.clone(),
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(call.arguments.clone()),
                media: Vec::new(),
            }),
        }
    }
}

/// Direct execution pairs the result with the call id and needs nothing
/// beyond the executor itself — the recorded result is all a standalone
/// caller gets, and it chooses any continuation itself.
#[tokio::test]
async fn executor_executes_a_static_tool_without_a_runner() {
    let executor = ToolExecutor::from_vec(vec![Arc::new(EchoTool)]);
    let payload = call("echo", json!({"q": 1}));
    let result = executor
        .execute_with_limits(payload.clone(), ctrl(), None, None, limits(10_000))
        .await;
    assert_eq!(result.call_id, payload.call_id);
    assert_eq!(result.status, ToolResultStatus::Succeeded);
    assert_eq!(result.output.content, json!({"q": 1}));
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
    let result = executor
        .execute_with_limits(
            call("mcp_srv_echo", json!({"hello": "world"})),
            ctrl(),
            None,
            None,
            limits(10_000),
        )
        .await;
    assert_eq!(result.status, ToolResultStatus::Succeeded);
    assert_eq!(result.output.content, json!({"hello": "world"}));

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
    let result = executor
        .execute_with_limits(
            call("mcp_srv_echo", json!({})),
            ctrl(),
            None,
            None,
            limits(10_000),
        )
        .await;
    assert_eq!(result.status, ToolResultStatus::Rejected);
}

/// The last argument IS the per-call limit — the executor never consults
/// the tool object. A standalone caller passes exactly the limit it wants
/// (the runner resolves fallback vs per-tool-name override before dispatch).
#[tokio::test]
async fn the_passed_limit_is_the_effective_limit() {
    // 4k bytes under a 100-token limit truncates…
    let executor = ToolExecutor::from_vec(vec![Arc::new(EchoTool)]);
    let result = executor
        .execute_with_limits(
            call("echo", json!({"text": "a".repeat(4000)})),
            ctrl(),
            None,
            None,
            limits(100),
        )
        .await;
    assert_eq!(result.output.truncation, causa_kernel::Truncation::Middle);
    // …and the same output under usize::MAX (the default) passes whole.
    let result = executor
        .execute_with_limits(
            call("echo", json!({"text": "a".repeat(4000)})),
            ctrl(),
            None,
            None,
            ToolOutputLimits::default(),
        )
        .await;
    assert_eq!(result.output.truncation, causa_kernel::Truncation::None);
}
