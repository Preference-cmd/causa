//! Direct execution properties extracted from the former driver/session consumers.
mod common;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::*;
use common::*;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

fn cause(out: &TurnOutcome) -> &TurnInterruption {
    match &out.result {
        TurnResult::Interrupted { cause } => cause,
        _ => panic!("expected interruption"),
    }
}

#[tokio::test]
async fn complete_output_and_actual_options_are_preserved() {
    let mut output = endturn_output(" done ");
    output.usage = Some(ModelUsage {
        input_tokens: 12,
        output_tokens: 3,
        ..Default::default()
    });
    output.reasoning = Some(ReasoningPayload {
        text: "reason".into(),
        signature: Some("signature".into()),
    });
    let gateway = RecordingGateway::scripted(vec![Ok(output)]);
    let runner = runner_with(gateway.clone(), vec![]);
    let mut opts = options();
    opts.generation.max_tokens = Some(123);
    opts.cache = CacheDirective::StablePrefix;
    let out = runner
        .run(TurnId::new("explicit"), input("hello"), opts, ctrl())
        .await;
    assert_eq!(out.turn_id, TurnId::new("explicit"));
    let TurnResult::Completed { final_output } = out.result else {
        panic!("must complete")
    };
    assert_eq!(final_output.response.text.0, " done ");
    assert_eq!(final_output.usage.unwrap().input_tokens, 12);
    assert_eq!(
        final_output.reasoning.unwrap().signature.as_deref(),
        Some("signature")
    );
    assert_eq!(out.context.blocks().len(), 2);
    assert!(out.uncommitted_tool_batch.is_none());
    let requests = gateway.recorded.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].invocation_id.round_id, RoundId(0));
    assert_eq!(requests[0].model.0, "fixture-model");
    assert_eq!(requests[0].generation.max_tokens, Some(123));
    assert_eq!(requests[0].cache, CacheDirective::StablePrefix);
}

#[tokio::test]
async fn context_reuse_and_import_never_reexecute_old_declarations() {
    let calls = Arc::new(AtomicUsize::new(0));
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("", "echo", json!({"x":1}))),
        Ok(endturn_output("first")),
        Ok(endturn_output("second")),
    ]);
    let runner = runner_with(gateway.clone(), vec![Arc::new(CountingTool(calls.clone()))]);
    let first = runner
        .run(TurnId::new("first"), input("hi"), options(), ctrl())
        .await;
    let first_saved = first.context.clone();
    let encoded = serde_json::to_vec(&first.context).unwrap();
    let mut restored: Context = serde_json::from_slice(&encoded).unwrap();
    restored
        .edit()
        .append([text_block("more")])
        .commit()
        .unwrap();
    let second = runner
        .run(TurnId::new("second"), restored, options(), ctrl())
        .await;
    assert!(matches!(second.result, TurnResult::Completed { .. }));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        &second.context.blocks()[..first_saved.blocks().len()],
        first_saved.blocks()
    );
    {
        let requests = gateway.recorded.lock().unwrap();
        assert_eq!(requests[2].invocation_id.turn_id, TurnId::new("second"));
        assert_eq!(
            requests[2].frame.blocks.len(),
            first_saved.blocks().len() + 1
        );
    }
    // Context can also carry an unmatched old declaration; it is never a queue.
    let old = ModelResponse {
        text: TextPayload::new(""),
        tool_calls: vec![draft("echo", json!({}))],
    }
    .to_blocks(ModelStopReason::ToolUse, &[new_block_id()])
    .unwrap();
    let imported = Context::from_blocks(old).unwrap();
    let runner = runner_with(
        RecordingGateway::scripted(vec![Ok(endturn_output("done"))]),
        vec![Arc::new(CountingTool(calls.clone()))],
    );
    let out = runner
        .run(TurnId::new("import"), imported, options(), ctrl())
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
struct CountingTool(Arc<AtomicUsize>);
#[async_trait]
impl Tool for CountingTool {
    fn definition(&self) -> ToolDefinition {
        EchoTool.definition()
    }
    async fn execute(&self, c: &ToolCallContext, k: &CallControl) -> ToolResultPayload {
        self.0.fetch_add(1, Ordering::SeqCst);
        EchoTool.execute(c, k).await
    }
}
#[test]
fn saved_context_is_independent_and_duplicate_ids_are_rejected() {
    let mut original = input("saved");
    let bytes = serde_json::to_vec(&original).unwrap();
    let saved: Context = serde_json::from_slice(&bytes).unwrap();
    original
        .apply(
            vec![Replacement {
                range: 0..1,
                with: vec![text_block("summary")],
            }],
            vec![],
        )
        .unwrap();
    assert_ne!(saved.blocks(), original.blocks());
    let block = saved.blocks()[0].clone();
    let wire = json!({"blocks":[block.clone(),block]});
    assert!(serde_json::from_value::<Context>(wire).is_err());
}
#[tokio::test]
async fn logical_gateway_errors_are_not_retried_or_reclassified_as_parent_control() {
    for kind in [
        ModelInvokeErrorKind::Transient,
        ModelInvokeErrorKind::Permanent,
        ModelInvokeErrorKind::TimedOut,
        ModelInvokeErrorKind::Cancelled,
        ModelInvokeErrorKind::UnknownOutcome,
        ModelInvokeErrorKind::InvalidRequest,
    ] {
        let gateway = RecordingGateway::scripted(vec![
            Err(kind.clone()),
            Ok(endturn_output("should not run")),
        ]);
        let out = runner_with(gateway.clone(), vec![])
            .run(TurnId::new("error"), input("retained"), options(), ctrl())
            .await;
        let TurnInterruption::ModelCallFailed { error, .. } = cause(&out) else {
            panic!("wrong cause: {:?}", out.result)
        };
        assert_eq!(error.kind, kind);
        assert_eq!(error.message, "scripted error");
        assert_eq!(gateway.recorded.lock().unwrap().len(), 1);
        assert_eq!(out.context.blocks().len(), 1);
    }
}
#[tokio::test]
async fn limits_count_declarations_and_return_new_batch() {
    let gateway = RecordingGateway::scripted(vec![Ok(tooluse_calls_output(
        "",
        vec![draft("echo", json!({})), draft("absent", json!({}))],
    ))]);
    let calls = Arc::new(AtomicUsize::new(0));
    let runner = runner_with(gateway.clone(), vec![Arc::new(CountingTool(calls.clone()))]);
    let mut opts = options();
    opts.limits.max_model_rounds = 0;
    let out = runner
        .run(TurnId::new("zero"), Context::new(), opts, ctrl())
        .await;
    assert!(matches!(
        cause(&out),
        TurnInterruption::MaxModelRounds { limit: 0 }
    ));
    assert!(gateway.recorded.lock().unwrap().is_empty());
    let mut opts = options();
    opts.limits.max_tool_calls = 1;
    let out = runner
        .run(TurnId::new("limit"), Context::new(), opts, ctrl())
        .await;
    assert!(matches!(
        cause(&out),
        TurnInterruption::MaxToolCalls {
            declared_before: 0,
            declared_this_round: 2,
            limit: 1,
            ..
        }
    ));
    assert_eq!(out.context.blocks().len(), 2);
    assert_eq!(out.uncommitted_tool_batch.unwrap().calls().len(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn model_round_limit_stops_before_another_binding_or_request() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("", "echo", json!({}))),
        Ok(endturn_output("extra")),
    ]);
    let mut opts = options();
    opts.limits.max_model_rounds = 1;
    let out = runner_with(gateway.clone(), vec![Arc::new(EchoTool)])
        .run(TurnId::new("round-limit"), Context::new(), opts, ctrl())
        .await;
    assert!(matches!(
        cause(&out),
        TurnInterruption::MaxModelRounds { limit: 1 }
    ));
    assert_eq!(gateway.recorded.lock().unwrap().len(), 1);
    assert_eq!(out.context.blocks().len(), 2);
    assert!(out.uncommitted_tool_batch.is_none());
}
#[tokio::test]
async fn malformed_model_output_retains_output_ids_and_provider_metadata() {
    let mut output = tooluse_output("text", " ", json!({}));
    output.response.tool_calls[0].provider_call_id = Some("provider-id".into());
    let gateway = RecordingGateway::scripted(vec![Ok(output)]);
    let out = runner_with(gateway, vec![])
        .run(TurnId::new("invalid"), Context::new(), options(), ctrl())
        .await;
    let TurnInterruption::ModelConversionFailed {
        error,
        output,
        block_ids,
        ..
    } = cause(&out)
    else {
        panic!("typed conversion failure expected")
    };
    assert!(matches!(
        error,
        ModelBlockError::EmptyToolName { tool_index: 0 }
    ));
    assert_eq!(
        output.response.tool_calls[0].provider_call_id.as_deref(),
        Some("provider-id")
    );
    assert_eq!(block_ids.len(), 2);
    assert!(out.context.blocks().is_empty());
    assert!(out.uncommitted_tool_batch.is_none());
}

struct ParkedGateway;
#[async_trait]
impl ModelGateway for ParkedGateway {
    async fn invoke(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        std::future::pending().await
    }
}
#[tokio::test]
async fn cancellation_and_deadline_return_existing_materials_without_cancelling_parent_on_timeout()
{
    for deadline in [false, true] {
        let token = CancellationToken::new();
        let control = RunControl::new(
            token.clone(),
            deadline.then(|| Instant::now() + Duration::from_millis(5)),
        );
        let runner = runner_with(Arc::new(ParkedGateway), vec![]);
        let future = runner.run(TurnId::new("parked"), input("kept"), options(), control);
        let out = if deadline {
            future.await
        } else {
            tokio::join!(future, async {
                tokio::task::yield_now().await;
                token.cancel();
            })
            .0
        };
        assert_eq!(out.context.blocks().len(), 1);
        assert!(out.uncommitted_tool_batch.is_none());
        if deadline {
            assert!(matches!(
                cause(&out),
                TurnInterruption::DeadlineExceeded {
                    invocation_id: Some(_)
                }
            ));
            assert!(!token.is_cancelled());
        } else {
            assert!(matches!(
                cause(&out),
                TurnInterruption::Cancelled {
                    invocation_id: Some(_)
                }
            ));
        }
    }
}

struct ResolveWithId(BlockId);
#[async_trait]
impl ToolBatchProcessor for ResolveWithId {
    async fn process(
        &self,
        b: &mut ToolBatch,
        _: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let call = b.calls()[0].call().clone();
        b.resolve_at(
            0,
            self.0,
            ToolResultPayload {
                call_block_id: call.call_block_id,
                status: ToolResultStatus::Rejected,
                output: ToolOutput::new(json!({"kept":true})),
                media: vec![MediaRef::new("image/png", "asset")],
                notes: vec![TextPayload::new("kept note")],
            },
        )
        .unwrap();
        Ok(())
    }
}
#[tokio::test]
async fn tool_commit_failure_returns_complete_batch_and_actual_edit_input() {
    let context = input("existing");
    let collision = context.blocks()[0].id();
    let executor = ToolExecutor::new(
        vec![Arc::new(EchoTool)],
        ToolExecutorOptions {
            before: vec![Arc::new(ResolveWithId(collision))],
            ..Default::default()
        },
    )
    .unwrap();
    let runner = TurnRunner::new(
        RecordingGateway::scripted(vec![Ok(tooluse_output(
            "",
            "echo",
            json!({"original":true}),
        ))]),
        Arc::new(executor),
    );
    let observed = Arc::new(Mutex::new(vec![]));
    let events = observed.clone();
    let token = CancellationToken::new();
    let callback_token = token.clone();
    let mut opts = options();
    opts.observer = Some(Arc::new(move |event| {
        if let RunEvent::BlocksCommitted { blocks, .. } = event {
            events
                .lock()
                .unwrap()
                .extend(blocks.iter().map(ContextBlock::id));
        }
        if let RunEvent::ToolBatchReturned { result, .. } = event {
            assert!(result.is_ok());
        }
        let _ = &callback_token;
    }));
    let out = runner
        .run(
            TurnId::new("collision"),
            context,
            opts,
            RunControl::new(token.clone(), None),
        )
        .await;
    let TurnInterruption::ToolCommitFailed { error, .. } = cause(&out) else {
        panic!("{:?}", out.result)
    };
    assert_eq!(error.appended.len(), 1);
    assert_eq!(error.appended[0].id(), collision);
    assert!(error.replacements.is_empty());
    let batch = out.uncommitted_tool_batch.as_ref().unwrap();
    assert_eq!(batch.completed_len(), 1);
    let (_, result) = batch.results()[0].result().unwrap();
    assert_eq!(result.notes[0].0, "kept note");
    assert_eq!(result.media[0].reference, "asset");
    assert_eq!(
        batch.results()[0].call().input.arguments,
        json!({"original":true})
    );
    assert_eq!(out.context.blocks().len(), 2);
    assert_eq!(observed.lock().unwrap().len(), 1);
    assert!(!token.is_cancelled());
}

struct FailedProcessor;
#[async_trait]
impl ToolBatchProcessor for FailedProcessor {
    async fn process(
        &self,
        b: &mut ToolBatch,
        _: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        b.calls_mut()[0].push_note(TextPayload::new("before failure"));
        Err(ProcessorError::Failed("precise failure".into()))
    }
}
#[tokio::test]
async fn processor_error_is_selected_before_returned_callback_cancels() {
    let executor = ToolExecutor::new(
        vec![Arc::new(EchoTool)],
        ToolExecutorOptions {
            before: vec![Arc::new(FailedProcessor)],
            ..Default::default()
        },
    )
    .unwrap();
    let runner = TurnRunner::new(
        RecordingGateway::scripted(vec![Ok(tooluse_output("", "echo", json!({})))]),
        Arc::new(executor),
    );
    let token = CancellationToken::new();
    let cancel = token.clone();
    let mut opts = options();
    opts.observer = Some(Arc::new(move |e| {
        if let RunEvent::ToolBatchReturned { result, .. } = e {
            assert!(result.is_err());
            cancel.cancel();
        }
    }));
    let out = runner
        .run(
            TurnId::new("processor"),
            Context::new(),
            opts,
            RunControl::new(token, None),
        )
        .await;
    assert!(matches!(
        cause(&out),
        TurnInterruption::ToolProcessingFailed {
            error: ToolProcessingError::Processor {
                phase: ToolProcessorPhase::Before,
                index: 0,
                ..
            },
            ..
        }
    ));
    assert_eq!(out.uncommitted_tool_batch.unwrap().calls().len(), 1);
    assert_eq!(out.context.blocks().len(), 1);
}

struct UnknownTool;
#[async_trait]
impl Tool for UnknownTool {
    fn definition(&self) -> ToolDefinition {
        EchoTool.definition()
    }
    async fn execute(&self, c: &ToolCallContext, k: &CallControl) -> ToolResultPayload {
        let mut r = EchoTool.execute(c, k).await;
        r.status = ToolResultStatus::UnknownOutcome;
        r
    }
}
#[tokio::test]
async fn observer_cancellation_respects_each_commit_boundary() {
    for stage in [
        "request",
        "output",
        "declaration",
        "ready",
        "returned",
        "result",
    ] {
        let gateway = RecordingGateway::scripted(vec![
            Ok(tooluse_output("", "echo", json!({}))),
            Ok(endturn_output("next")),
        ]);
        let calls = Arc::new(AtomicUsize::new(0));
        let runner = runner_with(gateway.clone(), vec![Arc::new(CountingTool(calls.clone()))]);
        let token = CancellationToken::new();
        let cancel = token.clone();
        let notices = Arc::new(Mutex::new(vec![]));
        let seen = notices.clone();
        let mut opts = options();
        opts.observer = Some(Arc::new(move |e| {
            let name = match e {
                RunEvent::ModelRequestReady { .. } => "request",
                RunEvent::ModelOutput { .. } => "output",
                RunEvent::BlocksCommitted { blocks, .. } => {
                    if blocks
                        .iter()
                        .any(|b| matches!(b.content(), BlockContent::ToolResult(_)))
                    {
                        "result"
                    } else {
                        "declaration"
                    }
                }
                RunEvent::ToolBatchReady { .. } => "ready",
                RunEvent::ToolBatchReturned { result, .. } => {
                    assert!(result.is_ok());
                    "returned"
                }
                RunEvent::ModelDelta { .. } => "delta",
            };
            seen.lock().unwrap().push(name);
            if name == stage {
                cancel.cancel();
            }
        }));
        let out = runner
            .run(
                TurnId::new(stage),
                Context::new(),
                opts,
                RunControl::new(token, None),
            )
            .await;
        assert!(matches!(cause(&out), TurnInterruption::Cancelled { .. }));
        assert_eq!(
            gateway.recorded.lock().unwrap().len(),
            usize::from(stage != "request")
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            usize::from(matches!(stage, "returned" | "result"))
        );
        assert_eq!(
            out.context.blocks().len(),
            match stage {
                "request" | "output" => 0,
                "result" => 2,
                _ => 1,
            }
        );
        assert_eq!(
            out.uncommitted_tool_batch.is_some(),
            matches!(stage, "declaration" | "ready" | "returned")
        );
        if matches!(stage, "declaration" | "ready") {
            assert!(!notices.lock().unwrap().contains(&"returned"));
        }
    }
}
#[tokio::test]
async fn terminal_results_survive_observer_cancellation_and_unknown_is_not_returned_twice() {
    for stop in [
        ModelStopReason::EndTurn,
        ModelStopReason::MaxTokens,
        ModelStopReason::Refusal,
        ModelStopReason::ToolUse,
    ] {
        let mut output = if stop == ModelStopReason::ToolUse {
            tooluse_output("", "echo", json!({}))
        } else {
            endturn_output("retained body")
        };
        output.stop_reason = stop;
        let gateway = RecordingGateway::scripted(vec![Ok(output)]);
        let token = CancellationToken::new();
        let cancel = token.clone();
        let mut opts = options();
        opts.observer = Some(Arc::new(move |e| match e {
            RunEvent::ModelOutput { .. }
                if matches!(stop, ModelStopReason::MaxTokens | ModelStopReason::Refusal) =>
            {
                cancel.cancel()
            }
            RunEvent::BlocksCommitted { blocks, .. }
                if stop == ModelStopReason::EndTurn
                    || blocks
                        .iter()
                        .any(|b| matches!(b.content(), BlockContent::ToolResult(_))) =>
            {
                cancel.cancel()
            }
            _ => (),
        }));
        let out = runner_with(gateway, vec![Arc::new(UnknownTool)])
            .run(
                TurnId::new("terminal"),
                Context::new(),
                opts,
                RunControl::new(token, None),
            )
            .await;
        match stop {
            ModelStopReason::EndTurn => assert!(matches!(out.result, TurnResult::Completed { .. })),
            ModelStopReason::MaxTokens => assert!(
                matches!(cause(&out),TurnInterruption::ModelMaxTokens {output,..} if output.response.text.0=="retained body")
            ),
            ModelStopReason::Refusal => {
                assert!(matches!(cause(&out), TurnInterruption::ModelRefusal { .. }))
            }
            ModelStopReason::ToolUse => {
                assert!(matches!(
                    cause(&out),
                    TurnInterruption::UnknownToolOutcome { .. }
                ));
                assert_eq!(out.context.blocks().len(), 2);
            }
        }
        assert!(out.uncommitted_tool_batch.is_none());
    }
    // Observer absence does not change unknown outcome policy.
    let out = runner_with(
        RecordingGateway::scripted(vec![Ok(tooluse_output("", "echo", json!({})))]),
        vec![Arc::new(UnknownTool)],
    )
    .run(TurnId::new("silent"), Context::new(), options(), ctrl())
    .await;
    assert!(matches!(
        cause(&out),
        TurnInterruption::UnknownToolOutcome { .. }
    ));
    assert_eq!(out.context.blocks().len(), 2);
}

#[tokio::test]
async fn tool_limit_is_selected_before_declaration_commit_callback_cancels() {
    let gateway = RecordingGateway::scripted(vec![Ok(tooluse_output("", "echo", json!({})))]);
    let token = CancellationToken::new();
    let cancel = token.clone();
    let mut opts = options();
    opts.limits.max_tool_calls = 0;
    opts.observer = Some(Arc::new(move |e| match e {
        RunEvent::BlocksCommitted { .. } => cancel.cancel(),
        RunEvent::ToolBatchReady { .. } | RunEvent::ToolBatchReturned { .. } => {
            panic!("over-limit declarations cannot enter processing")
        }
        _ => (),
    }));
    let out = runner_with(gateway, vec![Arc::new(EchoTool)])
        .run(
            TurnId::new("limit-cancel"),
            Context::new(),
            opts,
            RunControl::new(token, None),
        )
        .await;
    assert!(matches!(
        cause(&out),
        TurnInterruption::MaxToolCalls {
            limit: 0,
            declared_before: 0,
            declared_this_round: 1,
            ..
        }
    ));
    assert_eq!(out.context.blocks().len(), 1);
    assert_eq!(out.uncommitted_tool_batch.unwrap().calls().len(), 1);
}
