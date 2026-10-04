//! Public facade consumer using the built-in gateways, with no live API keys.
#[path = "../../../../crates/causa-provider/tests/runtime_streaming_support/mod.rs"]
mod support;

use std::sync::Arc;
use std::time::Duration;

use causa::kernel::{
    BlockContent, BlockMeta, CacheDirective, CallControl, ContentPart, Context, ContextBlock,
    GenerationOptions, InvocationId, ModelGateway, ModelRef, ModelRequest, RoundId, StreamDelta,
    TextPayload, ToolSurface, TurnId,
};
use causa::providers::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway, StreamLimits,
};
use futures_util::StreamExt;
use support::*;

fn input() -> Context {
    Context::from_blocks(vec![ContextBlock::new(
        causa::runtime::new_block_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("input"))]),
        BlockMeta {
            source: Some("user".into()),
            ..Default::default()
        },
    )])
    .unwrap()
}

#[tokio::test]
async fn facade_runtime_receives_borrowed_delta_before_the_real_http_terminal() {
    use causa::runtime::{
        RunControl, RunEvent, ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions,
        TurnRunner,
    };
    let (round, release) = Round::gated(responses_prefix("preview"), responses_terminal("saved"));
    let mut server = Server::start(vec![round]);
    let runner = TurnRunner::new(
        Arc::new(OpenAiResponsesGateway::new("fixture-key").with_base_url(&server.url)),
        Arc::new(ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap()),
    );
    let mut options = TurnRunOptions::new(ModelRef::new("fixture"));
    options.observer = Some(Arc::new(move |event| {
        if let RunEvent::ModelDelta {
            delta: StreamDelta::TextDelta { delta },
            ..
        } = event
        {
            assert_eq!(delta, "preview");
            release.send(()).unwrap();
        }
    }));
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        runner.run_streaming(
            TurnId::new("facade-runtime"),
            input(),
            options,
            RunControl::new(Default::default(), None),
        ),
    )
    .await
    .unwrap();
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    let saved = serde_json::to_string(&outcome.context).unwrap();
    assert!(saved.contains("saved"));
    assert!(!saved.contains("preview"));
    server.finish();
    assert_eq!(server.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn facade_direct_stream_observes_all_three_gateways_before_terminal() {
    for protocol in 0..3 {
        let (prefix, suffix) = match protocol {
            0 => (chat_text("你好🙂"), chat_terminal("stop")),
            1 => (responses_prefix("你好🙂"), responses_terminal("你好🙂")),
            _ => (anthropic_prefix("你好🙂"), anthropic_terminal()),
        };
        let (round, release) = Round::gated(prefix, suffix);
        let mut server = Server::start(vec![round]);
        let gateway: Arc<dyn ModelGateway> = match protocol {
            0 => Arc::new(
                OpenAiChatCompletionsGateway::new("fixture-key")
                    .with_base_url(&server.url)
                    .with_stream_limits(StreamLimits::default()),
            ),
            1 => Arc::new(
                OpenAiResponsesGateway::new("fixture-key")
                    .with_base_url(&server.url)
                    .with_stream_limits(StreamLimits::default()),
            ),
            _ => Arc::new(
                AnthropicMessagesGateway::new("fixture-key")
                    .with_base_url(&server.url)
                    .with_stream_limits(StreamLimits::default()),
            ),
        };
        let request = ModelRequest {
            invocation_id: InvocationId {
                turn_id: TurnId::new("public-stream"),
                round_id: RoundId(0),
            },
            frame: input().frame(),
            model: ModelRef::new("fixture"),
            tool_surface: ToolSurface::empty(),
            generation: GenerationOptions::default(),
            cache: CacheDirective::None,
        };
        let mut stream = gateway
            .stream(&request, &CallControl::new(Default::default(), None))
            .await
            .unwrap();
        let mut observed_text = false;
        while !observed_text {
            let delta = tokio::time::timeout(Duration::from_secs(3), stream.next())
                .await
                .expect("real text must arrive before the server releases terminal bytes")
                .unwrap();
            match delta {
                StreamDelta::TextDelta { delta } => {
                    assert_eq!(delta, "你好🙂");
                    observed_text = true;
                }
                StreamDelta::Usage(_) => {}
                other => panic!("terminal or unexpected observation before text: {other:?}"),
            }
        }
        release.send(()).unwrap();
        let mut terminals = 0;
        while let Some(delta) = stream.next().await {
            match delta {
                StreamDelta::Done { final_output, .. } => {
                    assert_eq!(final_output.response.text.0, "你好🙂");
                    terminals += 1;
                }
                StreamDelta::Error { kind, message } => panic!("{kind:?}: {message}"),
                _ => {}
            }
        }
        assert_eq!(terminals, 1);
        assert!(stream.next().await.is_none());
        server.finish();
        assert_eq!(server.requests.lock().unwrap().len(), 1);
    }
}
