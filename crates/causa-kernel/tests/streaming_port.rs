//! Streaming-port tests (Slice 6 Phase A): the `ModelGateway::stream`
//! default degeneration and `completed_model_stream`. (The
//! `TurnInteraction` contract moved to `causa-runtime` in Slice 13 — its
//! tests live there now.)

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, AttemptNumber, InvocationId, ModelGateway, ModelInvokeError, ModelOutput,
    ModelRef, ModelRequest, ModelResponse, ModelStopReason, ModelStream, ModelUsage,
    ReasoningPayload, RoundId, StreamDelta, TextPayload, ToolSurface, TurnContext, TurnId,
    completed_model_stream,
};
use futures_util::StreamExt;

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

fn no_ctrl() -> AttemptControl {
    AttemptControl::new(causa_kernel::CancellationToken::new(), None)
}

// Silence unused warnings for fields only some tests touch.
#[allow(unused)]
fn _touch(_u: Option<ModelUsage>, _r: Option<ReasoningPayload>) {}
