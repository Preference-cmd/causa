//! Shared fixtures for the runtime test split. `mod.rs` is required
//! here: Cargo auto-discovers `tests/*.rs` as standalone targets, but a
//! shared module must live in a subdirectory. Each test target compiles
//! its own copy, so fixtures used by only some targets would trip
//! dead_code.

#![allow(dead_code)]

pub mod coordinator;

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, CallControl, ContextFrame, ConversationId, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelOutput, ModelRequest, ModelResponse, ModelStopReason, ModelStream,
    StreamDelta, TextPayload, Tool, ToolCallContext, ToolCallDraft, ToolDefinition, ToolOutput,
    ToolResultPayload, ToolResultStatus, Truncation, TurnContext, TurnId,
};
use causa_runtime::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, ConversationState, RunControl,
    SealedResult, Session, SessionConfig, ToolExecutor, TurnLimits, TurnPolicy, TurnRunOptions,
    TurnRunner, new_block_id,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
        .append_input(new_block_id(), TextPayload::new("hi"), "user")
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

/// A session over `id`, driving `gateway` with the default config.
pub fn idle_session(id: &str, gateway: Arc<dyn ModelGateway>) -> Session {
    Session::new(
        ConversationState::new(ConversationId(id.into())),
        Arc::new(runner_with(gateway, vec![])),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("an empty state is an idle session base")
}

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

/// A gateway whose model call parks until the test releases it, so the work is
/// observably `Running` for as long as the assertions need.
///
/// With `honour_cancel`, the parked call races the session's token and reports
/// `Cancelled` — the driver maps that to `ExplicitCancellation`. Without it,
/// the call ignores the token and returns its end-turn output when released,
/// so a completion that is already in flight wins the race against the cancel.
pub struct GatedGateway {
    text: String,
    calls: AtomicUsize,
    /// One permit per `invoke` entry — the work reached the model, not merely
    /// the session's accept path.
    entered: tokio::sync::Semaphore,
    /// One permit per `release()`, consumed by the parked `invoke`.
    release: tokio::sync::Semaphore,
    honour_cancel: bool,
}

impl GatedGateway {
    pub fn new(text: &str, honour_cancel: bool) -> Arc<Self> {
        Arc::new(Self {
            text: text.to_string(),
            calls: AtomicUsize::new(0),
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
            honour_cancel,
        })
    }

    /// How many model calls have been entered.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    /// Wait until the worker has entered the gated model call. Bounded so a
    /// regression fails loudly instead of hanging the suite.
    pub async fn wait_entered(&self) {
        tokio::time::timeout(Duration::from_secs(5), self.entered.acquire())
            .await
            .expect("the gated gateway is entered within 5s")
            .expect("the entry semaphore stays open")
            .forget();
    }

    /// Open the gate: unblock the parked call.
    pub fn release(&self) {
        self.release.add_permits(1);
    }
}

#[async_trait]
impl ModelGateway for GatedGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.entered.add_permits(1);
        if self.honour_cancel {
            tokio::select! {
                // `biased` makes the ordering deterministic: once the token has
                // fired, the cancel branch wins even if the test released the
                // gate too.
                biased;
                _ = ctrl.cancellation_token().cancelled() => Err(ModelInvokeError::new(
                    ModelInvokeErrorKind::Cancelled,
                    "gated gateway: cancelled by the session",
                )),
                permit = self.release.acquire() => {
                    permit.expect("the release semaphore stays open").forget();
                    Ok(endturn_output(&self.text))
                }
            }
        } else {
            let permit = self
                .release
                .acquire()
                .await
                .expect("the release semaphore stays open");
            permit.forget();
            Ok(endturn_output(&self.text))
        }
    }
}

/// A gateway that panics inside `invoke` — the worker's `catch_unwind` turns
/// this into an observable `Faulted` work, never a permanently `Running` one.
pub struct PanickingGateway;

#[async_trait]
impl ModelGateway for PanickingGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        panic!("boom: panicking gateway");
    }
}

/// A gateway that oversleeps any work deadline and then emits one tool-use
/// round, forcing the driver back to its loop top where the passed deadline is
/// seen and the turn ends `Interrupted { TurnDeadlineExceeded }`.
pub struct SlowGateway;

#[async_trait]
impl ModelGateway for SlowGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _ctrl: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        tokio::time::sleep(Duration::from_millis(250)).await;
        Ok(tooluse_output("late", "echo", serde_json::json!({})))
    }
}

/// Minimal scripted gateway — consumes canned outputs in order.
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
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput {
                content: serde_json::json!({"echo": ctx.input.arguments}),
                truncation: Truncation::None,
                meta: None,
                artifact: None,
            },
            media: Vec::new(),
            notes: Vec::new(),
        }
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
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(serde_json::json!({"err": "fail"})),
            media: Vec::new(),
            notes: Vec::new(),
        }
    }
}

/// Returns `UnknownOutcome` — its continuation action comes from the host's
/// unknown-outcome configuration, not a tool declaration.
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
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::UnknownOutcome,
            output: ToolOutput::new(serde_json::json!({"unk": true})),
            media: Vec::new(),
            notes: Vec::new(),
        }
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

// ---- streaming fixtures --------------------------------------------------------

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

    pub fn frames(&self) -> Vec<ContextFrame> {
        self.recorded
            .lock()
            .unwrap()
            .iter()
            .map(|request| request.frame.clone())
            .collect()
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
