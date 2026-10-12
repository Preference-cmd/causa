//! Real HTTP streaming through the material-returning runtime, without live APIs.
mod runtime_streaming_support;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use causa_kernel::*;
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use causa_runtime::*;
use runtime_streaming_support::*;
use serde_json::json;

fn runner(gateway: Arc<dyn ModelGateway>, tools: Vec<Arc<dyn Tool>>) -> TurnRunner {
    TurnRunner::new(
        gateway,
        Arc::new(ToolExecutor::new(tools, ToolExecutorOptions::default()).unwrap()),
    )
}

fn input() -> Context {
    Context::from_blocks(vec![ContextBlock::new(
        new_block_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("original"))]),
        BlockMeta {
            source: Some("user".into()),
            ..Default::default()
        },
    )])
    .unwrap()
}

fn invocation(round: u32) -> InvocationId {
    InvocationId {
        turn_id: TurnId::new("http-stream"),
        round_id: RoundId(round),
    }
}

fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

#[tokio::test]
async fn all_builtin_gateways_observe_http_text_before_terminal_and_commit_only_final_output() {
    for protocol in 0..3 {
        let (prefix, suffix, expected) = match protocol {
            0 => (chat_text("你好🙂"), chat_terminal("stop"), "你好🙂"),
            1 => (
                responses_prefix("advisory"),
                responses_terminal("authoritative"),
                "authoritative",
            ),
            _ => (anthropic_prefix("你好🙂"), anthropic_terminal(), "你好🙂"),
        };
        let (round, release) = Round::gated(prefix, suffix);
        let mut server = Server::start(vec![round]);
        let gateway: Arc<dyn ModelGateway> = match protocol {
            0 => Arc::new(
                OpenAiChatCompletionsGateway::new("key")
                    .with_base_url(format!("{}/v1", server.url)),
            ),
            1 => Arc::new(
                OpenAiResponsesGateway::new("key").with_base_url(format!("{}/v1", server.url)),
            ),
            _ => Arc::new(
                AnthropicMessagesGateway::new("key").with_base_url(format!("{}/v1", server.url)),
            ),
        };
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let mut options = TurnRunOptions::new(ModelRef::new("fixture"));
        // The Anthropic leg requires an explicit output limit.
        options.generation.max_tokens = Some(1024);
        options.observer = Some(Arc::new(move |event| match event {
            RunEvent::ModelDelta {
                invocation_id,
                delta: StreamDelta::TextDelta { delta },
            } => {
                assert_eq!(*invocation_id, &invocation(0));
                assert_eq!(recorded.lock().unwrap().as_slice(), [] as [&str; 0]);
                assert!(!delta.is_empty());
                recorded.lock().unwrap().push("delta");
                release.send(()).unwrap();
            }
            RunEvent::ModelOutput {
                invocation_id,
                output,
            } => {
                assert_eq!(*invocation_id, &invocation(0));
                assert_eq!(output.response.text(), expected);
                recorded.lock().unwrap().push("output");
            }
            RunEvent::BlocksCommitted { .. } => recorded.lock().unwrap().push("commit"),
            _ => {}
        }));
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            runner(gateway, vec![]).run_streaming(
                TurnId::new("http-stream"),
                input(),
                options,
                control(),
            ),
        )
        .await
        .expect("terminal gate must be released by an actual delta");
        assert!(
            matches!(outcome.result, TurnResult::Completed { .. }),
            "{:#?}",
            outcome.result
        );
        assert_eq!(
            seen.lock().unwrap().as_slice(),
            ["delta", "output", "commit"]
        );
        let material = serde_json::to_string(&outcome.context).unwrap();
        assert!(material.contains(expected));
        assert!(!material.contains("advisory"));
        assert_eq!(outcome.context.blocks().len(), 2);
        assert!(outcome.uncommitted_tool_batch.is_none());
        server.finish();
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["stream"], true);
    }
}

struct Echo {
    calls: Arc<Mutex<Vec<ToolCallContext>>>,
    status: ToolResultStatus,
}

#[async_trait]
impl Tool for Echo {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echo".into(),
            parameters: json!({"type":"object"}),
        }
    }
    async fn execute(&self, call: &ToolCallContext, _: &CallControl) -> ToolResultPayload {
        self.calls.lock().unwrap().push(call.clone());
        ToolResultPayload {
            call_block_id: call.call_block_id,
            status: self.status.clone(),
            output: ToolOutput::new(call.input.arguments.clone()),
            media: vec![],
            notes: vec![],
        }
    }
}

fn echo(calls: Arc<Mutex<Vec<ToolCallContext>>>) -> Arc<dyn Tool> {
    Arc::new(Echo {
        calls,
        status: ToolResultStatus::Succeeded,
    })
}

#[tokio::test]
async fn terminal_tool_executes_once_with_round_identity_and_provider_id_then_next_round_streams() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (round, release) = Round::gated(
        chat_tool("{\"message\":\"你好🙂\"}"),
        chat_terminal("tool_calls"),
    );
    let mut server = Server::start(vec![
        round,
        Round::immediate(chat_text("final") + &chat_terminal("stop")),
    ]);
    let mut options = TurnRunOptions::new(ModelRef::new("fixture"));
    let observed = calls.clone();
    let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let released_in_observer = released.clone();
    options.observer = Some(Arc::new(move |event| match event {
        RunEvent::ModelDelta {
            invocation_id,
            delta:
                StreamDelta::ToolCallDelta {
                    call_index,
                    provider_call_id,
                    ..
                },
        } => {
            assert_eq!(*invocation_id, &invocation(0));
            assert_eq!(*call_index, 0);
            assert_eq!(provider_call_id.as_deref(), Some("call_wire"));
            assert!(
                observed.lock().unwrap().is_empty(),
                "draft tool must not execute"
            );
            if !released_in_observer.swap(true, std::sync::atomic::Ordering::SeqCst) {
                release.send(()).unwrap();
            }
        }
        RunEvent::ToolBatchReady { invocation_id, .. } => {
            assert_eq!(*invocation_id, &invocation(0));
            assert!(released_in_observer.load(std::sync::atomic::Ordering::SeqCst));
        }
        RunEvent::ModelDelta {
            invocation_id,
            delta: StreamDelta::TextDelta { .. },
        } => assert_eq!(*invocation_id, &invocation(1)),
        _ => {}
    }));
    let gateway = Arc::new(
        OpenAiChatCompletionsGateway::new("key").with_base_url(format!("{}/v1", server.url)),
    );
    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        runner(gateway, vec![echo(calls.clone())]).run_streaming(
            TurnId::new("http-stream"),
            input(),
            options,
            control(),
        ),
    )
    .await
    .unwrap();
    assert!(
        matches!(outcome.result, TurnResult::Completed { .. }),
        "{:#?}",
        outcome.result
    );
    assert_eq!(calls.lock().unwrap().len(), 1);
    let call = calls.lock().unwrap()[0].clone();
    assert_eq!(call.input.arguments, json!({"message":"你好🙂"}));
    let declaration = outcome
        .context
        .blocks()
        .iter()
        .find(|block| block.id() == call.call_block_id)
        .unwrap();
    assert_eq!(
        declaration.meta().provider_call_id.as_deref(),
        Some("call_wire")
    );
    assert_eq!(outcome.context.blocks().len(), 4);
    assert!(outcome.uncommitted_tool_batch.is_none());
    server.finish();
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1]["messages"][2]["tool_call_id"], "call_wire");
}

#[tokio::test]
async fn stream_error_missing_terminal_and_truncated_tool_leave_original_context_and_no_execution()
{
    for (body, kind) in [
        (chat_tool("{"), ModelInvokeErrorKind::UnknownOutcome),
        (
            chat_tool("{") + &chat_terminal("tool_calls"),
            ModelInvokeErrorKind::Undecodable,
        ),
        (
            chat_tool("{")
                + &event(json!({"error":{"type":"server_error","message":"fixture failure"}})),
            ModelInvokeErrorKind::Transient,
        ),
    ] {
        let mut server = Server::start(vec![Round::immediate(body)]);
        let original = input();
        let saved = serde_json::to_value(&original).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let outcome = runner(
            Arc::new(
                OpenAiChatCompletionsGateway::new("key")
                    .with_base_url(format!("{}/v1", server.url)),
            ),
            vec![echo(calls.clone())],
        )
        .run_streaming(
            TurnId::new("http-stream"),
            original,
            TurnRunOptions::new(ModelRef::new("fixture")),
            control(),
        )
        .await;
        let TurnResult::Interrupted {
            cause:
                TurnInterruption::ModelCallFailed {
                    invocation_id,
                    error,
                },
        } = outcome.result
        else {
            panic!("expected classified model failure: {:?}", outcome.result);
        };
        assert_eq!(invocation_id, invocation(0));
        assert_eq!(error.kind, kind);
        assert_eq!(serde_json::to_value(outcome.context).unwrap(), saved);
        assert!(calls.lock().unwrap().is_empty());
        assert!(outcome.uncommitted_tool_batch.is_none());
        server.finish();
        assert_eq!(server.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn observer_cancelled_partial_second_round_preserves_completed_first_round_and_no_pending_batch()
 {
    let (second, release) = Round::gated(chat_tool("{"), String::new());
    let mut server = Server::start(vec![
        Round::immediate(chat_tool("{}") + &chat_terminal("tool_calls")),
        second,
    ]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let token = CancellationToken::new();
    let cancel = token.clone();
    let mut options = TurnRunOptions::new(ModelRef::new("fixture"));
    options.observer = Some(Arc::new(move |event| {
        if let RunEvent::ModelDelta {
            invocation_id,
            delta: StreamDelta::ToolCallDelta { .. },
        } = event
            && invocation_id.round_id == RoundId(1)
        {
            cancel.cancel();
            let _ = release.send(());
        }
    }));
    let outcome = runner(
        Arc::new(
            OpenAiChatCompletionsGateway::new("key").with_base_url(format!("{}/v1", server.url)),
        ),
        vec![echo(calls.clone())],
    )
    .run_streaming(
        TurnId::new("http-stream"),
        input(),
        options,
        CallControl::new(token, None),
    )
    .await;
    assert!(
        matches!(outcome.result, TurnResult::Interrupted { cause: TurnInterruption::Cancelled { invocation_id: Some(ref id) } } if id == &invocation(1))
    );
    assert_eq!(calls.lock().unwrap().len(), 1);
    assert_eq!(outcome.context.blocks().len(), 3);
    assert!(
        outcome
            .context
            .blocks()
            .iter()
            .any(|b| matches!(b.content(), BlockContent::ToolResult(_)))
    );
    assert!(outcome.uncommitted_tool_batch.is_none());
    server.finish();
    assert_eq!(server.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn cancellation_after_declaration_commit_returns_complete_pending_tool_batch() {
    let mut server = Server::start(vec![Round::immediate(
        chat_tool("{}") + &chat_terminal("tool_calls"),
    )]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let token = CancellationToken::new();
    let cancel = token.clone();
    let mut options = TurnRunOptions::new(ModelRef::new("fixture"));
    options.observer = Some(Arc::new(move |event| {
        if let RunEvent::BlocksCommitted { blocks, .. } = event
            && blocks
                .iter()
                .any(|block| matches!(block.content(), BlockContent::ToolCall(_)))
        {
            cancel.cancel();
        }
    }));
    let outcome = runner(
        Arc::new(
            OpenAiChatCompletionsGateway::new("key").with_base_url(format!("{}/v1", server.url)),
        ),
        vec![echo(calls.clone())],
    )
    .run_streaming(
        TurnId::new("http-stream"),
        input(),
        options,
        CallControl::new(token, None),
    )
    .await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::Cancelled { .. }
        }
    ));
    assert_eq!(outcome.context.blocks().len(), 2);
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(outcome.uncommitted_tool_batch.unwrap().calls().len(), 1);
    server.finish();
}

#[tokio::test]
async fn failed_later_round_preserves_completed_tool_material_and_original_block_identity() {
    let mut server = Server::start(vec![
        Round::immediate(chat_tool("{}") + &chat_terminal("tool_calls")),
        Round::immediate(chat_tool("{")),
    ]);
    let original = input();
    let first = original.blocks()[0].clone();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let outcome = runner(
        Arc::new(
            OpenAiChatCompletionsGateway::new("key").with_base_url(format!("{}/v1", server.url)),
        ),
        vec![echo(calls.clone())],
    )
    .run_streaming(
        TurnId::new("http-stream"),
        original,
        TurnRunOptions::new(ModelRef::new("fixture")),
        control(),
    )
    .await;
    assert!(
        matches!(outcome.result, TurnResult::Interrupted { cause: TurnInterruption::ModelCallFailed { invocation_id, error } } if invocation_id == invocation(1) && error.kind == ModelInvokeErrorKind::UnknownOutcome)
    );
    assert_eq!(outcome.context.blocks()[0], first);
    assert_eq!(outcome.context.blocks().len(), 3);
    assert_eq!(calls.lock().unwrap().len(), 1);
    assert!(outcome.uncommitted_tool_batch.is_none());
    server.finish();
    assert_eq!(server.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn unknown_tool_result_commits_and_stops_without_another_model_round() {
    let mut server = Server::start(vec![Round::immediate(
        chat_tool("{}") + &chat_terminal("tool_calls"),
    )]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let tool = Arc::new(Echo {
        calls: calls.clone(),
        status: ToolResultStatus::UnknownOutcome,
    });
    let outcome = runner(
        Arc::new(
            OpenAiChatCompletionsGateway::new("key").with_base_url(format!("{}/v1", server.url)),
        ),
        vec![tool],
    )
    .run_streaming(
        TurnId::new("http-stream"),
        input(),
        TurnRunOptions::new(ModelRef::new("fixture")),
        control(),
    )
    .await;
    assert!(
        matches!(outcome.result, TurnResult::Interrupted { cause: TurnInterruption::UnknownToolOutcome { invocation_id, .. } } if invocation_id == invocation(0))
    );
    assert_eq!(calls.lock().unwrap().len(), 1);
    assert_eq!(outcome.context.blocks().len(), 3);
    assert!(outcome.uncommitted_tool_batch.is_none());
    server.finish();
    assert_eq!(server.requests.lock().unwrap().len(), 1);
}
