//! Shared fixtures for the test split. `mod.rs` is required here: Cargo
//! auto-discovers `tests/*.rs` as standalone targets, but a shared module
//! must live in a subdirectory. Each test target compiles its own copy, so
//! fixtures used by only some targets would trip dead_code.

#![allow(dead_code)]

use async_trait::async_trait;
use reimagine_context_kernel::{
    AttemptControl, CallControl, CancellationToken, Compaction, CompactionError, CompactionInput,
    CompactionOutput, ContextFrame, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRequest, ModelResponse, ModelStopReason, RunControl, TextPayload, Tool,
    ToolCallContext, ToolCallDraft, ToolDefinition, ToolExecutionOutcome, ToolExecutor, ToolOutput,
    ToolResultPayload, ToolResultStatus, Truncation, TurnContext, TurnId, TurnLimits, TurnPolicy,
    TurnRunOptions, TurnRunner, UnknownOutcomePolicy,
};
use std::sync::{Arc, Mutex};

// ---- ids and model-output constructors --------------------------------------

pub fn turn_id(s: &str) -> TurnId {
    TurnId::new(s)
}

pub fn ctx(s: &str) -> TurnContext {
    TurnContext::new(turn_id(s))
}

pub fn endturn_output(text: &str) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: vec![],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    }
}

pub fn tooluse_output(text: &str, tool_name: &str, args: serde_json::Value) -> ModelOutput {
    tooluse_calls_output(text, vec![draft(tool_name, args)])
}

pub fn tooluse_calls_output(text: &str, calls: Vec<ToolCallDraft>) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: calls,
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    }
}

pub fn draft(tool_name: &str, args: serde_json::Value) -> ToolCallDraft {
    ToolCallDraft {
        tool_name: tool_name.into(),
        arguments: args,
        provider_call_id: None,
    }
}

// ---- control planes and options ---------------------------------------------

pub fn ctrl() -> RunControl {
    RunControl::new(CancellationToken::new(), None)
}

pub fn limits(r: u32, t: u32) -> TurnLimits {
    TurnLimits {
        max_model_rounds: r,
        max_tool_calls: t,
    }
}

pub fn options_with_limits(r: u32, t: u32) -> TurnRunOptions {
    TurnRunOptions {
        policy: TurnPolicy {
            limits: limits(r, t),
            ..Default::default()
        },
        ..Default::default()
    }
}

pub fn runner_with(gateway: Arc<dyn ModelGateway>, tools: Vec<Arc<dyn Tool>>) -> TurnRunner {
    TurnRunner::new(gateway, Arc::new(ToolExecutor::from_vec(tools)))
}

// ---- test-only dedup hook -----------------------------------------------------
//
// The framework's `DedupFilter` lives in `agent-runtime`, but kernel tests
// must not depend on the layer above. This is a test-only local copy for
// the single `dedup_same_batch_rejected*` coverage.

pub struct TestDedupHook;

#[async_trait]
impl reimagine_context_kernel::ToolUseHook for TestDedupHook {
    async fn apply(
        &self,
        calls: Vec<reimagine_context_kernel::ToolCallPayload>,
        _ctx: &reimagine_context_kernel::HookCtx<'_>,
    ) -> reimagine_context_kernel::HookOutcome {
        use reimagine_context_kernel::{ToolOutput, ToolResultPayload, ToolResultStatus};
        use std::collections::HashMap;
        let mut seen: HashMap<(String, serde_json::Value), ()> = HashMap::new();
        let mut to_execute = Vec::new();
        let mut rejected = Vec::new();
        for payload in calls {
            let key = (payload.tool_name.clone(), payload.arguments.clone());
            if seen.insert(key, ()).is_some() {
                rejected.push(reimagine_context_kernel::ToolExecutionOutcome::new(
                    ToolResultPayload {
                        call_id: payload.call_id.clone(),
                        status: ToolResultStatus::Rejected,
                        output: ToolOutput::new(
                            serde_json::json!({"error": "duplicate tool call"}),
                        ),
                    },
                ));
            } else {
                to_execute.push(payload);
            }
        }
        reimagine_context_kernel::HookOutcome {
            to_execute,
            rejected,
        }
    }
}

/// Like `runner_with`, but with the test dedup hook installed.
/// Only for tests that want the historical dedup behavior;
/// normal callers compose filters explicitly via the framework layer.
pub fn runner_with_dedup(gateway: Arc<dyn ModelGateway>, tools: Vec<Arc<dyn Tool>>) -> TurnRunner {
    TurnRunner::with_hook(
        gateway,
        Arc::new(ToolExecutor::from_vec(tools)),
        Arc::new(TestDedupHook),
    )
}

// ---- scripted gateway --------------------------------------------------------

/// A gateway that replays canned outcomes and records every request.
///
/// - `scripted`: pops outputs in order; an empty queue is a Permanent error.
/// - `repeating_last`: pops until one remains, then repeats it forever —
///   for single-output completion flows.
pub struct RecordingGateway {
    outputs: Mutex<Vec<Result<ModelOutput, ModelInvokeErrorKind>>>,
    recorded: Mutex<Vec<ModelRequest>>,
    repeat_last: bool,
}

impl RecordingGateway {
    pub fn scripted(outputs: Vec<Result<ModelOutput, ModelInvokeErrorKind>>) -> Arc<Self> {
        Arc::new(Self {
            outputs: Mutex::new(outputs),
            recorded: Mutex::new(vec![]),
            repeat_last: false,
        })
    }

    pub fn repeating_last(outputs: Vec<Result<ModelOutput, ModelInvokeErrorKind>>) -> Arc<Self> {
        Arc::new(Self {
            outputs: Mutex::new(outputs),
            recorded: Mutex::new(vec![]),
            repeat_last: true,
        })
    }

    pub fn recorded(&self) -> std::sync::MutexGuard<'_, Vec<ModelRequest>> {
        self.recorded.lock().unwrap()
    }

    pub fn frames(&self) -> Vec<ContextFrame> {
        self.recorded
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.frame.clone())
            .collect()
    }
}

#[async_trait]
impl ModelGateway for RecordingGateway {
    async fn invoke(
        &self,
        req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.recorded.lock().unwrap().push(req.clone());
        let mut g = self.outputs.lock().unwrap();
        let out = if self.repeat_last {
            if g.len() > 1 {
                g.remove(0)
            } else {
                g[0].clone()
            }
        } else if g.is_empty() {
            Err(ModelInvokeErrorKind::Permanent)
        } else {
            g.remove(0)
        };
        out.map_err(|kind| ModelInvokeError::new(kind, "scripted"))
    }
}

// ---- tool fakes ---------------------------------------------------------------

pub struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echo".into(),
            parameters: serde_json::json!({"type":"object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput {
                content: serde_json::json!({"echo": ctx.arguments}),
                truncation: Truncation::None,
                meta: None,
                artifact: None,
            },
        })
    }
}

pub struct FailTool;

#[async_trait]
impl Tool for FailTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "fail".into(),
            description: "fail".into(),
            parameters: serde_json::json!({"type":"object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(serde_json::json!({"err": "fail"})),
        })
    }
}

pub struct UnknownStopTool;

#[async_trait]
impl Tool for UnknownStopTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "unk".into(),
            description: "unk".into(),
            parameters: serde_json::json!({"type":"object"}),
        }
    }
    fn unknown_outcome_policy(&self) -> UnknownOutcomePolicy {
        UnknownOutcomePolicy::Stop
    }
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::UnknownOutcome,
            output: ToolOutput::new(serde_json::json!({"unk": true})),
        })
        .with_policy(UnknownOutcomePolicy::Stop)
    }
}

// ---- compaction fake ----------------------------------------------------------

pub struct DropAllCompaction;

#[async_trait]
impl Compaction for DropAllCompaction {
    async fn compact(&self, _input: CompactionInput) -> Result<CompactionOutput, CompactionError> {
        Ok(CompactionOutput {
            blocks: Vec::new(),
            summary: None,
            truncated: true,
        })
    }
}
