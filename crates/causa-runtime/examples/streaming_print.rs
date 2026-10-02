//! Streaming output — drive a turn with `run_streaming` and print each
//! provider delta as it arrives through the `RunEvent` seam.
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
//! the source of truth. Retries, if needed, belong to the gateway wrapper;
//! partial output is never committed by the runner.

use async_trait::async_trait;
use causa_kernel::{
    CallControl, ModelGateway, ModelInvokeError, ModelOutput, ModelRequest, ModelResponse,
    ModelStopReason, StreamDelta, TextPayload, ToolCallDraft, TurnId,
};
use causa_runtime::{
    RunControl, RunEvent, ToolExecutor, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
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
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        Err(ModelInvokeError::new(
            causa_kernel::ModelInvokeErrorKind::Permanent,
            "streaming fixture has no invoke script",
        ))
    }

    async fn stream(
        &self,
        _request: &ModelRequest,
        _control: &CallControl,
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

/// The host chooses how to display borrowed observations.
fn print_observation(event: &RunEvent<'_>) {
    match event {
        RunEvent::ModelDelta {
            delta: StreamDelta::TextDelta { delta },
            ..
        } => {
            print!("{delta}");
            std::io::stdout().flush().ok();
        }
        RunEvent::ModelDelta {
            delta: StreamDelta::ToolCallDelta { .. },
            ..
        } => println!("[tool call streaming in…]"),
        RunEvent::ModelOutput { .. } => println!(),
        _ => {}
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

    let runner = TurnRunner::new(
        gateway,
        Arc::new(ToolExecutor::new(Vec::new(), Default::default())?),
    );

    let mut context = causa_kernel::Context::new();
    context
        .edit()
        .append([causa_kernel::ContextBlock::new(
            new_block_id(),
            causa_kernel::BlockContent::Parts(vec![causa_kernel::ContentPart::Text(
                TextPayload::new("Say something about streaming."),
            )]),
            causa_kernel::BlockMeta {
                source: Some("user".into()),
                ..Default::default()
            },
        )])
        .commit()?;

    let mut options = TurnRunOptions::new(causa_kernel::ModelRef::new("offline-stream"));
    options.observer = Some(Arc::new(print_observation));

    let outcome = runner
        .run_streaming(
            TurnId::new("streaming-demo"),
            context,
            options,
            RunControl::new(Default::default(), None),
        )
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
