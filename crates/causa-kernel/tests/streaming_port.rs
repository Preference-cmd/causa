//! Streaming-port tests (Slice 6 Phase A): the `ModelGateway::stream`
//! default degeneration, `completed_model_stream`, and the
//! `TurnInteraction` contract as seen from the kernel alone.

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, AttemptNumber, InvocationId, ModelGateway, ModelInvokeError, ModelOutput,
    ModelRef, ModelRequest, ModelResponse, ModelStopReason, ModelStream, ModelUsage,
    ReasoningPayload, RoundId, StreamDelta, TextPayload, ToolSurface, TurnContext, TurnId,
    TurnInteraction, completed_model_stream,
};
use futures_util::StreamExt;
use std::sync::{Arc, Mutex};

fn request(ctx: &TurnContext) -> ModelRequest {
    ModelRequest {
        invocation_id: InvocationId {
            turn_id: ctx.turn_id(),
            round_id: RoundId(0),
        },
        attempt: AttemptNumber(1),
        frame: ctx.frame(RoundId(0)),
        model: ModelRef::new("fake"),
        tool_surface: ToolSurface::empty(),
        generation: Default::default(),
        cache: causa_kernel::CacheDirective::None,
    }
}

fn output(text: &str) -> ModelOutput {
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

/// A batch-only gateway: never overrides `stream`.
struct InvokeOnlyGateway {
    output: ModelOutput,
}

#[async_trait]
impl ModelGateway for InvokeOnlyGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        Ok(self.output.clone())
    }
}

#[tokio::test]
async fn default_stream_degenerates_to_single_done() {
    let mut ctx = TurnContext::new(TurnId::new("t1"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();
    let gateway = InvokeOnlyGateway {
        output: output("hello"),
    };
    let mut stream = gateway.stream(&request(&ctx), &no_ctrl()).await.unwrap();
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    assert_eq!(items.len(), 1, "degenerate stream is exactly one Done");
    match &items[0] {
        StreamDelta::Done {
            stop_reason,
            final_output,
        } => {
            assert_eq!(*stop_reason, ModelStopReason::EndTurn);
            assert_eq!(final_output.response.text.0, "hello");
        }
        other => panic!("expected Done, got {other:?}"),
    }
}

#[tokio::test]
async fn completed_model_stream_wraps_one_done() {
    let mut items = Vec::new();
    let mut stream: ModelStream = completed_model_stream(output("wrapped"));
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    assert_eq!(items.len(), 1);
    assert!(matches!(items[0], StreamDelta::Done { .. }));
}

#[tokio::test]
async fn turn_interaction_default_methods_are_noop_and_implementable() {
    /// A host-side observer counting what it sees through the port.
    struct CountingInteraction {
        text_deltas: Mutex<usize>,
    }
    #[async_trait]
    impl TurnInteraction for CountingInteraction {
        async fn on_delta(&self, _round_id: RoundId, delta: &StreamDelta) {
            if matches!(delta, StreamDelta::TextDelta { .. }) {
                *self.text_deltas.lock().unwrap() += 1;
            }
        }
    }

    let interaction = Arc::new(CountingInteraction {
        text_deltas: Mutex::new(0),
    });
    // Default no-op: the trait's own methods are callable on Arc<dyn _>.
    let noop: Arc<dyn TurnInteraction> = interaction.clone();
    noop.on_delta(
        RoundId(0),
        &StreamDelta::ReasoningDelta {
            delta: "thinking".into(),
        },
    )
    .await;
    interaction
        .on_delta(
            RoundId(3),
            &StreamDelta::TextDelta {
                delta: "token".into(),
            },
        )
        .await;
    assert_eq!(*interaction.text_deltas.lock().unwrap(), 1);
}

fn no_ctrl() -> AttemptControl {
    AttemptControl::new(causa_kernel::CancellationToken::new(), None)
}

// Silence unused warnings for fields only some tests touch.
#[allow(unused)]
fn _touch(_u: Option<ModelUsage>, _r: Option<ReasoningPayload>) {}
