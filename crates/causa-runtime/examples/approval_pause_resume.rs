//! Approval pause/resume — a tool batch arrives, the host suspends the
//! turn behind a human decision, then resumes it with the withheld
//! verdict (`resume_turn`).
//!
//! Runnable offline: a scripted gateway emits one tool call, the
//! interaction gate pauses every batch, and "the human" auto-approves.
//!
//! ```text
//! cargo run --example approval_pause_resume -p causa-runtime
//! ```
//!
//! The decision vocabulary is deliberately not new: approve / reject /
//! rewrite are the three constructions of `HookOutcome`, so the gate's
//! pause (`BatchDecision::Pause`) and the release (`resume_turn` with
//! the withheld `HookOutcome`) are two halves of one mechanism. A
//! steering resume works the same way — queue `TextPayload`s into
//! `ResumeRequest::inject` instead of (or alongside) a decision.

use async_trait::async_trait;
use causa_kernel::{
    ConversationId, ModelGateway, ModelInvokeError, ModelOutput, ModelRequest, ModelResponse,
    ModelStopReason, TextPayload, Tool, ToolCallContext, ToolCallDraft, ToolDefinition, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnId,
};
use causa_runtime::{
    BatchDecision, ConversationOutcome, ConversationState, HookOutcome, PausePoint, ResumeRequest,
    RunControl, ToolExecutor, TurnInteraction, TurnResult, TurnRunOptions, TurnRunner, resume_turn,
};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Scripted stub gateway: one tool-use round, then the closing round.
struct ScriptedGateway(Mutex<Vec<ModelOutput>>);

impl ScriptedGateway {
    fn new(outputs: Vec<ModelOutput>) -> Arc<Self> {
        Arc::new(Self(Mutex::new(outputs)))
    }
}

#[async_trait]
impl ModelGateway for ScriptedGateway {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &causa_kernel::AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut outputs = self.0.lock().await;
        if outputs.is_empty() {
            return Err(ModelInvokeError::new(
                causa_kernel::ModelInvokeErrorKind::Permanent,
                "script exhausted",
            ));
        }
        Ok(outputs.remove(0))
    }
}

/// A deliberately sensitive local tool — the kind a host gates behind
/// approval.
struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".into(),
            description: "Read a file from the host filesystem.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        }
    }

    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &causa_kernel::CallControl,
    ) -> ToolResultPayload {
        ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(serde_json::json!(
                { "contents": "the promised file contents" }
            )),
            media: Vec::new(),
        }
    }
}

/// The approval gate: pause every batch, no matter what. A real host
/// would surface `calls` to a human (or a policy engine) and answer
/// `Proceed` / `Reject` / `Rewrite` without leaving the turn.
struct PauseAll;

#[async_trait]
impl TurnInteraction for PauseAll {
    async fn decide_batch(&self, _calls: &[causa_kernel::ToolCallPayload]) -> BatchDecision {
        BatchDecision::Pause { deadline: None }
    }
}

fn tool_use_round() -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new("Let me read that file first."),
            tool_calls: vec![ToolCallDraft {
                tool_name: "read_file".into(),
                arguments: serde_json::json!({ "path": "notes.txt" }),
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    }
}

fn end_turn(text: &str) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: Vec::<ToolCallDraft>::new(),
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gateway = ScriptedGateway::new(vec![
        tool_use_round(),
        end_turn("The file says: the promised file contents."),
    ]);
    let runner = TurnRunner::new(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(ReadFile)])),
    );
    let options = TurnRunOptions {
        interaction: Arc::new(PauseAll),
        ..Default::default()
    };

    // --- run: the batch arrives and the gate suspends the turn -------------
    let mut state = ConversationState::new(ConversationId("approval-demo".into()));
    state.begin_turn(TurnId::new("turn-1"))?;
    state
        .active_turn_mut()
        .expect("begin_turn just opened it")
        .append_input(TextPayload::new("Read notes.txt for me."), "user")?;

    let paused_outcome = runner
        .run_in_conversation(
            state,
            options.clone(),
            RunControl::new(Default::default(), None),
        )
        .await?;

    let (awaiting, deadline) = match &paused_outcome.result {
        TurnResult::Paused { continuation } => match &continuation.pause_point {
            PausePoint::AwaitingApproval { prepared, deadline } => {
                (prepared.awaiting.clone(), *deadline)
            }
            other => return Err(format!("expected an approval pause, got {other:?}").into()),
        },
        other => return Err(format!("expected a paused turn, got {other:?}").into()),
    };
    println!("paused behind {} call(s):", awaiting.len());
    for call in &awaiting {
        println!("  {} {}", call.tool_name, call.arguments);
    }

    // --- the human decides: approve the batch verbatim ---------------------
    // Approve = `HookOutcome::passthrough`; reject = all-rejected;
    // rewrite = edited `to_execute`. Same vocabulary, three verdicts. The
    // paused outcome is the checkpoint: the resume consumes it whole and
    // only the new decision rides on the request.
    println!("decision: approve (deadline was {deadline:?})");
    let resumed = resume_turn(
        &runner,
        paused_outcome,
        ResumeRequest {
            decision: Some(HookOutcome::passthrough(awaiting)),
            inject: Vec::new(),
        },
        options,
        RunControl::new(Default::default(), None),
    )
    .await?;

    let ConversationOutcome {
        mut state,
        result,
        trace,
    } = resumed;
    match result {
        TurnResult::Completed { final_output } => {
            println!("final: {}", final_output.response.text.0);
        }
        other => return Err(format!("resumed turn did not complete: {other:?}").into()),
    }
    state.commit(TurnId::new("turn-1"))?;
    println!(
        "turn completed across {} round(s), {} tool call(s)",
        trace.rounds.len(),
        trace.tool_calls_total
    );
    Ok(())
}
