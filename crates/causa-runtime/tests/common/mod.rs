//! Shared fixtures for the runtime test split. `mod.rs` is required
//! here: Cargo auto-discovers `tests/*.rs` as standalone targets, but a
//! shared module must live in a subdirectory. Each test target compiles
//! its own copy, so fixtures used by only some targets would trip
//! dead_code.

#![allow(dead_code)]

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, CallControl, ContentPart, ContextFrame, ConversationId, ModelGateway,
    ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRequest, ModelResponse,
    ModelStopReason, ModelStream, RoundId, StreamDelta, TextPayload, Tool, ToolCallContext,
    ToolCallDraft, ToolCallId, ToolCallPayload, ToolDefinition, ToolOutput, ToolResultPayload,
    ToolResultStatus, Truncation, TurnContext, TurnId,
};
use causa_runtime::{
    BatchDecision, Compaction, CompactionError, CompactionInput, CompactionOutput,
    ConversationState, HookOutcome, ResumeRequest, RunControl, SealedResult, Session,
    SessionConfig, SessionHandle, SubmitRequest, ToolExecutor, TurnInteraction, TurnLimits,
    TurnPolicy, TurnRunOptions, TurnRunner, WaitEnd, WorkObservation, WorkReceipt, WorkRef,
    WorkState,
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

/// A `TurnInteraction` that pauses on the first tool-use batch — the host
/// approval gate. The turn suspends with its active context left open, so the
/// driver returns `TurnResult::Paused` and the session publishes
/// `WorkState::Paused`.
pub struct PausingInteraction;

#[async_trait]
impl TurnInteraction for PausingInteraction {
    async fn decide_batch(&self, _calls: &[ToolCallPayload]) -> BatchDecision {
        BatchDecision::Pause { deadline: None }
    }
}

/// A session with no interaction seam over `id`, driving `gateway` with the
/// default config — work runs straight through the driver.
pub fn idle_session(id: &str, gateway: Arc<dyn ModelGateway>) -> Session {
    Session::new(
        ConversationState::new(ConversationId(id.into())),
        Arc::new(runner_with(gateway, vec![])),
        TurnRunOptions::default(),
        SessionConfig::default(),
    )
    .expect("an empty state is an idle session base")
}

/// A session over `id` whose interaction pauses on the first tool-use batch,
/// driving `gateway` with `tools` under `config`.
pub fn pausing_session(
    id: &str,
    gateway: Arc<dyn ModelGateway>,
    tools: Vec<Arc<dyn Tool>>,
    config: SessionConfig,
) -> Session {
    let options = TurnRunOptions {
        interaction: Arc::new(PausingInteraction),
        ..Default::default()
    };
    Session::new(
        ConversationState::new(ConversationId(id.into())),
        Arc::new(runner_with(gateway, tools)),
        options,
        config,
    )
    .expect("an empty state is an idle session base")
}

/// One text submission part.
pub fn session_req(key: &str, text: &str) -> SubmitRequest {
    SubmitRequest {
        request_key: key.into(),
        parts: vec![ContentPart::Text(TextPayload::new(text))],
    }
}

/// Submit one work and wait for it to pause, returning its receipt and the
/// paused observation (whose `revision` is the paused revision).
pub async fn submit_to_pause(handle: &SessionHandle, key: &str) -> (WorkReceipt, WorkObservation) {
    let receipt = handle
        .submit(session_req(key, "go"))
        .expect("an idle session accepts the work");
    let waited = handle
        .wait(&receipt.work, Duration::from_secs(5))
        .await
        .expect("the accepted work is observable");
    assert_eq!(waited.end, WaitEnd::ReachedState);
    assert_eq!(waited.observation.state, WorkState::Paused);
    (receipt, waited.observation)
}

/// One paused work over `id`: the first model call emits a tool-use batch
/// that the interaction pauses. Returns the owning session (dropping it
/// would close submission), a handle, and the paused work's ref.
pub async fn paused_work(
    id: &str,
    gateway: Arc<RecordingGateway>,
) -> (Session, SessionHandle, WorkRef) {
    let session = pausing_session(
        id,
        gateway,
        vec![Arc::new(EchoTool)],
        SessionConfig::default(),
    );
    let handle = session.handle();
    let (receipt, _) = submit_to_pause(&handle, "pause").await;
    (session, handle, receipt.work)
}

/// The paused round's unanswered call: the model emitted `echo` in round 0,
/// position 0, with empty arguments — the kernel mints its id from exactly
/// those. `passthrough` of this single call covers the awaiting batch once.
pub fn awaiting_echo() -> ToolCallPayload {
    ToolCallPayload {
        call_id: ToolCallId::generate(RoundId(0), "echo", &serde_json::json!({}), 0),
        tool_name: "echo".into(),
        arguments: serde_json::json!({}),
    }
}

/// Approve `calls` unchanged, with no extra injection.
pub fn approve(calls: Vec<ToolCallPayload>) -> ResumeRequest {
    ResumeRequest {
        decision: Some(HookOutcome::passthrough(calls)),
        inject: Vec::new(),
    }
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

// ---- test-only dedup hook -----------------------------------------------------
//
// The framework's `DedupFilter` lives in the lib; these integration
// targets pin dedup behavior independently of it.

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
                rejected.push(ToolResultPayload {
                    call_id: payload.call_id.clone(),
                    status: ToolResultStatus::Rejected,
                    output: ToolOutput::new(serde_json::json!({"error": "duplicate tool call"})),
                    media: Vec::new(),
                });
            } else {
                to_execute.push(payload);
            }
        }
        causa_runtime::HookOutcome {
            to_execute,
            rejected,
            unknown_decisions: Vec::new(),
        }
    }
}

/// Like `runner_with`, but with the test dedup hook installed. Normal
/// callers compose filters explicitly via the framework layer.
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
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput {
                content: serde_json::json!({"echo": ctx.arguments}),
                truncation: Truncation::None,
                meta: None,
                artifact: None,
            },
            media: Vec::new(),
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
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(serde_json::json!({"err": "fail"})),
            media: Vec::new(),
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
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::UnknownOutcome,
            output: ToolOutput::new(serde_json::json!({"unk": true})),
            media: Vec::new(),
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
