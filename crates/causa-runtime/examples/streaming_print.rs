//! Streaming output — drive a turn with `run_streaming` and print each
//! provider delta as it arrives through the `TurnInteraction` seam.
//!
//! Runnable offline: the gateway is a scripted streaming stub (text
//! deltas, then `Done`). In production you would use a streaming
//! provider adapter from `causa-provider` — the driver loop and the
//! observation seam are identical.
//!
//! ```text
//! cargo run --example streaming_print -p causa-runtime
//! ```
//!
//! Contract worth remembering: `StreamDelta::Done` carries the fully
//! assembled `ModelOutput` — the deltas are advisory observations, never
//! the source of truth. A retried attempt re-streams the same frame;
//! presenting the partial-then-reset flow is the host's call.

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, ModelGateway, ModelInvokeError, ModelOutput, ModelRequest, ModelResponse,
    ModelStopReason, RoundId, StreamDelta, TextPayload, ToolCallDraft, TurnContext, TurnId,
};
use causa_runtime::{
    RunControl, ToolExecutor, TurnInteraction, TurnResult, TurnRunOptions, TurnRunner,
};
use std::io::Write as _;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Scripted streaming gateway: one delta script per `stream()` call.
/// The driver calls `stream()` (not `invoke`) for the streaming entry.
struct ScriptedStreamGateway(Mutex<Vec<Vec<StreamDelta>>>);

impl ScriptedStreamGateway {
    fn new(scripts: Vec<Vec<StreamDelta>>) -> Arc<Self> {
        Arc::new(Self(Mutex::new(scripts)))
    }
}

#[async_trait]
impl ModelGateway for ScriptedStreamGateway {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        Err(ModelInvokeError::new(
            causa_kernel::ModelInvokeErrorKind::Permanent,
            "streaming fixture has no invoke script",
        ))
    }

    async fn stream(
        &self,
        _request: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<causa_kernel::ModelStream, ModelInvokeError> {
        let mut scripts = self.0.lock().await;
        if scripts.is_empty() {
            return Err(ModelInvokeError::new(
                causa_kernel::ModelInvokeErrorKind::Permanent,
                "no more streaming scripts",
            ));
        }
        let deltas = scripts.remove(0);
        Ok(Box::pin(futures_util::stream::iter(deltas)))
    }
}

/// The host side of streaming: every provider delta lands here, tagged
/// with the round it belongs to. Deltas are advisory — print them raw.
struct PrintDeltas;

#[async_trait]
impl TurnInteraction for PrintDeltas {
    async fn on_delta(&self, _round_id: RoundId, delta: &StreamDelta) {
        match delta {
            StreamDelta::TextDelta { delta } => {
                print!("{delta}");
                std::io::stdout().flush().ok();
            }
            StreamDelta::ToolCallDelta { .. } => {
                println!("[tool call streaming in…]");
            }
            StreamDelta::Done { .. } => println!(),
            _ => {}
        }
    }
}

fn turn_output(text: &str) -> ModelOutput {
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
    let text = "Causa streams: facts and contracts in the kernel, \
                the loop in the runtime, opinions where you can see them.";
    let gateway = ScriptedStreamGateway::new(vec![vec![
        StreamDelta::TextDelta {
            delta: text.to_string(),
        },
        StreamDelta::Done {
            stop_reason: ModelStopReason::EndTurn,
            final_output: turn_output(text),
        },
    ]]);

    let runner = TurnRunner::new(gateway, Arc::new(ToolExecutor::from_vec(Vec::new())));

    let mut context = TurnContext::new(TurnId::new("streaming-demo"));
    context.append_input(TextPayload::new("Say something about streaming."), "user")?;

    let options = TurnRunOptions {
        interaction: Arc::new(PrintDeltas),
        ..Default::default()
    };

    let outcome = runner
        .run_streaming(context, options, RunControl::new(Default::default(), None))
        .await;
    match outcome.result {
        TurnResult::Completed { final_output } => {
            // `Done` carried this same output; print it once more as the
            // canonical record (a UI would render it incrementally).
            println!("--- canonical output ---");
            println!("{}", final_output.response.text.0);
        }
        other => return Err(format!("turn did not complete: {other:?}").into()),
    }
    Ok(())
}
