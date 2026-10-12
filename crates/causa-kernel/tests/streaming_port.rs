//! Streaming-port tests: the `ModelGateway::stream` default degeneration
//! and `completed_model_stream`. Input admission belongs to the caller.

mod common;

use async_trait::async_trait;
use causa_kernel::{
    CallControl, Context, InvocationId, ModelGateway, ModelInvokeError, ModelItem, ModelOutput,
    ModelRef, ModelRequest, ModelResponse, ModelStopReason, ModelStream, ModelUsage,
    ReasoningPayload, RoundId, StreamDelta, TextPayload, TurnId, completed_model_stream,
};
use futures_util::StreamExt;

fn request(ctx: &Context) -> ModelRequest {
    ModelRequest::new(
        InvocationId {
            turn_id: TurnId::new("t1"),
            round_id: RoundId(0),
        },
        ctx.frame(),
        ModelRef::new("fake"),
    )
}

fn output(text: &str) -> ModelOutput {
    ModelOutput::new(
        ModelResponse {
            items: vec![ModelItem::text(text)],
        },
        ModelStopReason::EndTurn,
    )
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
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        Ok(self.output.clone())
    }
}

#[tokio::test]
async fn default_stream_degenerates_to_single_done() {
    let mut ctx = Context::new();
    ctx.apply(
        Vec::new(),
        vec![common::text_block(
            common::block_id(),
            TextPayload::new("hi"),
            "user",
        )],
    )
    .unwrap();
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
        StreamDelta::Done { final_output } => {
            assert_eq!(final_output.stop_reason, ModelStopReason::EndTurn);
            assert_eq!(final_output.response.text(), "hello");
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

fn no_ctrl() -> CallControl {
    CallControl::new(causa_kernel::CancellationToken::new(), None)
}

// Silence unused warnings for fields only some tests touch.
#[allow(unused)]
fn _touch(_u: Option<ModelUsage>, _r: Option<ReasoningPayload>) {}
