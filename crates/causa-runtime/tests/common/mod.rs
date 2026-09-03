//! Shared fixtures for the agent-runtime test split. `mod.rs` is required
//! here: Cargo auto-discovers `tests/*.rs` as standalone targets, but a
//! shared module must live in a subdirectory. Each test target compiles
//! its own copy, so fixtures used by only some targets would trip
//! dead_code.
//!
//! Graduated from context-kernel's test fixtures (Slice 12) together with
//! the driver stack they exercise; the kernel keeps only fact-machine
//! fixtures now.

#![allow(dead_code)]

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, CallControl, Compaction, CompactionError, CompactionInput, CompactionOutput,
    ContextFrame, ConversationState, ModelGateway, ModelInvokeError, ModelInvokeErrorKind,
    ModelOutput, ModelRequest, ModelResponse, ModelStopReason, ModelStream, SealedResult,
    StreamDelta, TextPayload, Tool, ToolCallContext, ToolCallDraft, ToolDefinition,
    ToolExecutionOutcome, ToolOutput, ToolResultPayload, ToolResultStatus, Truncation, TurnContext,
    TurnId, UnknownOutcomePolicy,
};
use causa_runtime::{RunControl, ToolExecutor, TurnLimits, TurnPolicy, TurnRunOptions, TurnRunner};
use std::sync::{Arc, Mutex};

// ---- ids and model-output constructors --------------------------------------

pub fn turn_id(s: &str) -> TurnId {
    TurnId::new(s)
}

pub fn ctx(s: &str) -> TurnContext {
    TurnContext::new(turn_id(s))
}

/// Canonical driver dance: begin_turn → append_input → seal_turn → commit.
/// Single input + `SealedResult` (Completed or Interrupted) covers both
/// branches; tests that need either pass the appropriate variant.
pub fn commit_sealed(state: &mut ConversationState, turn_label: &str, result: SealedResult) {
    state.begin_turn(TurnId::new(turn_label)).unwrap();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    state.seal_turn(TurnId::new(turn_label), result).unwrap();
    state.commit(TurnId::new(turn_label)).unwrap();
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
    RunControl::new(tokio_util::sync::CancellationToken::new(), None)
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

// ---- scripted gateways ---------------------------------------------------------

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

/// Minimal scripted gateway — consumes canned outputs in order. The
/// former `FakeGateway` of the kernel's staged fakes module (Slice 12).
pub struct FakeGateway {
    pub outputs: Mutex<Vec<Result<ModelOutput, ModelInvokeErrorKind>>>,
}
impl FakeGateway {
    pub fn new(outputs: Vec<Result<ModelOutput, ModelInvokeErrorKind>>) -> Self {
        Self {
            outputs: Mutex::new(outputs),
        }
    }
}
#[async_trait]
impl ModelGateway for FakeGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut guard = self.outputs.lock().unwrap();
        if guard.is_empty() {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                "no more fake outputs",
            ));
        }
        match guard.remove(0) {
            Ok(o) => Ok(o),
            Err(k) => Err(ModelInvokeError::new(k, "fake error")),
        }
    }
}

// ---- test-only dedup hook -----------------------------------------------------
//
// The framework's `DedupFilter` lives in the lib, but these integration
// targets pin the historical dedup behavior independently of it.

pub struct TestDedupHook;

#[async_trait]
impl causa_runtime::ToolUseHook for TestDedupHook {
    async fn apply(
        &self,
        calls: Vec<causa_kernel::ToolCallPayload>,
        _ctx: &causa_runtime::HookCtx<'_>,
    ) -> causa_runtime::HookOutcome {
        use std::collections::HashMap;
        let mut seen: HashMap<(String, serde_json::Value), ()> = HashMap::new();
        let mut to_execute = Vec::new();
        let mut rejected = Vec::new();
        for payload in calls {
            let key = (payload.tool_name.clone(), payload.arguments.clone());
            if seen.insert(key, ()).is_some() {
                rejected.push(ToolExecutionOutcome::new(ToolResultPayload {
                    call_id: payload.call_id.clone(),
                    status: ToolResultStatus::Rejected,
                    output: ToolOutput::new(serde_json::json!({"error": "duplicate tool call"})),
                }));
            } else {
                to_execute.push(payload);
            }
        }
        causa_runtime::HookOutcome {
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

// ---- streaming fixtures (Slice 6) ----------------------------------------------

/// One scripted `stream()` call: either the call itself fails
/// (transport-level, mapped like `invoke` errors) or it yields a delta
/// sequence.
pub type StreamScript = Result<Vec<StreamDelta>, ModelInvokeErrorKind>;

/// A gateway whose `stream()` pops one script per attempt and records
/// every request — the streaming counterpart of `RecordingGateway`.
/// `invoke` is never wired here: streaming tests drive `stream()` only.
pub struct RecordingStreamingGateway {
    scripts: Mutex<Vec<StreamScript>>,
    recorded: Mutex<Vec<ModelRequest>>,
}

impl RecordingStreamingGateway {
    pub fn scripted(scripts: Vec<StreamScript>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(scripts),
            recorded: Mutex::new(vec![]),
        })
    }

    /// Number of `stream()` calls made so far — one per attempt.
    pub fn attempts(&self) -> usize {
        self.recorded.lock().unwrap().len()
    }
}

#[async_trait]
impl ModelGateway for RecordingStreamingGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Permanent,
            "streaming fixture has no invoke script",
        ))
    }

    async fn stream(
        &self,
        req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        self.recorded.lock().unwrap().push(req.clone());
        let mut scripts = self.scripts.lock().unwrap();
        match (!scripts.is_empty()).then(|| scripts.remove(0)) {
            None => Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                "no more streaming scripts",
            )),
            Some(Err(kind)) => Err(ModelInvokeError::new(kind, "scripted")),
            Some(Ok(deltas)) => Ok(Box::pin(futures_util::stream::iter(deltas))),
        }
    }
}

/// `[TextDelta(..text..), Done(endturn_output(text))]` — the minimal
/// text-only completion script.
pub fn text_script(text: &str) -> StreamScript {
    Ok(vec![
        StreamDelta::TextDelta {
            delta: text.to_string(),
        },
        StreamDelta::Done {
            stop_reason: ModelStopReason::EndTurn,
            final_output: endturn_output(text),
        },
    ])
}

/// A tool-use round script: text, one tool call split across a name delta
/// and two argument deltas, then `Done` carrying the assembled drafts.
pub fn tooluse_script(text: &str, tool: &str, args: serde_json::Value) -> StreamScript {
    Ok(vec![
        StreamDelta::TextDelta {
            delta: text.to_string(),
        },
        StreamDelta::ToolCallDelta {
            call_index: 0,
            provider_call_id: Some("prov-1".into()),
            name_delta: Some(tool.to_string()),
            arguments_delta: None,
        },
        StreamDelta::ToolCallDelta {
            call_index: 0,
            provider_call_id: None,
            name_delta: None,
            arguments_delta: Some(args.to_string()),
        },
        StreamDelta::Done {
            stop_reason: ModelStopReason::ToolUse,
            final_output: tooluse_output(text, tool, args),
        },
    ])
}
