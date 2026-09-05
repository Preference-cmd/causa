//! Streaming driver tests (Slice 6): `run_streaming` /
//! `run_in_conversation_streaming` through scripted streaming gateways,
//! plus the shared retry-backoff behavior (Phase E).

mod common;

use causa_kernel::{
    CancellationToken, ConversationId, ConversationState, ModelGateway, ModelInvokeError,
    ModelInvokeErrorKind, ModelRequest, ModelStream, RoundId, StreamDelta, TextPayload,
    TurnContext, TurnId,
};
use causa_runtime::{
    NoopInteraction, RetryPolicy, RunControl, StreamEventCollector, TurnOutcome, TurnPolicy,
    TurnResult, TurnRunOptions, TurnRunner, project_streaming_turn,
};
use common::{
    EchoTool, RecordingStreamingGateway, StreamScript, ctrl, text_script, tooluse_script,
};
use futures_util::StreamExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn streaming_options(max_retries: u32, backoff_base_ms: u64) -> TurnRunOptions {
    TurnRunOptions {
        policy: TurnPolicy {
            retry: RetryPolicy {
                max_retries,
                retry_timeouts: false,
                backoff_base_ms,
                backoff_max_ms: backoff_base_ms.max(1),
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn collector_for(ctx: &TurnContext) -> Arc<StreamEventCollector> {
    Arc::new(StreamEventCollector::new(ctx.turn_id(), None))
}

// ---- Phase D: driver streaming ------------------------------------------------

#[tokio::test]
async fn streaming_run_completes_and_observes_text_deltas() {
    let ctx = ctx();
    let collector = collector_for(&ctx);
    let mut options = streaming_options(0, 0);
    options.interaction = collector.clone();
    let gateway = RecordingStreamingGateway::scripted(vec![text_script("hello world")]);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![])),
    );

    let out: TurnOutcome = runner.run_streaming(ctx, options, ctrl()).await;

    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 1);
    // The Done output (not the deltas) is the fact source: input block
    // plus one assistant text block carrying the assembled text.
    let blocks = out.context.blocks();
    assert_eq!(blocks.len(), 2);
    assert!(matches!(
        &blocks[1].content,
        causa_kernel::BlockContent::Parts(parts)
            if matches!(
                parts.as_slice(),
                [causa_kernel::ContentPart::Text(causa_kernel::TextPayload(t))] if t == "hello world"
            )
    ));
    // The interaction observed the advisory deltas, never the terminal ones.
    let events = collector.events();
    assert_eq!(events.len(), 1);
    match &events[0].kind {
        causa_runtime::ContextEventKind::TextDelta { round_id, delta } => {
            assert_eq!(*round_id, RoundId(0));
            assert_eq!(delta, "hello world");
        }
        other => panic!("expected TextDelta event, got {other:?}"),
    }
}

#[tokio::test]
async fn streaming_tool_round_drives_next_round() {
    let ctx = ctx();
    let collector = collector_for(&ctx);
    let mut options = streaming_options(0, 0);
    options.interaction = collector.clone();
    options.invocation = Default::default();
    let gateway = RecordingStreamingGateway::scripted(vec![
        tooluse_script("calling echo", "echo", serde_json::json!({"a": 1})),
        text_script("all done"),
    ]);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![Arc::new(
            EchoTool,
        )])),
    );

    let out: TurnOutcome = runner.run_streaming(ctx, options, ctrl()).await;

    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.trace.rounds.len(), 2);
    assert_eq!(out.trace.tool_calls_total, 1);
    assert_eq!(gateway.attempts(), 2);
    // Deltas from both rounds observed, in round order.
    let events = collector.events();
    let rounds: Vec<RoundId> = events
        .iter()
        .filter_map(|e| match &e.kind {
            causa_runtime::ContextEventKind::TextDelta { round_id, .. } => Some(*round_id),
            _ => None,
        })
        .collect();
    assert_eq!(rounds, vec![RoundId(0), RoundId(1)]);
}

#[tokio::test]
async fn streaming_retry_after_error_delta_succeeds() {
    let ctx = ctx();
    let gateway = RecordingStreamingGateway::scripted(vec![
        // Attempt 1: mid-stream provider error (retryable, same frame).
        Ok(vec![
            StreamDelta::TextDelta {
                delta: "partial".into(),
            },
            StreamDelta::Error {
                kind: ModelInvokeErrorKind::Transient,
                message: "provider hiccup".into(),
            },
        ]),
        // Attempt 2: clean completion.
        text_script("recovered"),
    ]);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![])),
    );

    let out: TurnOutcome = runner
        .run_streaming(ctx, streaming_options(1, 0), ctrl())
        .await;

    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(gateway.attempts(), 2);
    // Both attempts share one round's attempt trace.
    let attempts = &out.trace.rounds[0].attempts;
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].kind, Some(ModelInvokeErrorKind::Transient));
    assert!(attempts[0].is_retryable);
    assert_eq!(attempts[1].kind, None);
}

#[tokio::test]
async fn streaming_transport_error_maps_like_invoke() {
    let ctx = ctx();
    let gateway = RecordingStreamingGateway::scripted(vec![
        Err(ModelInvokeErrorKind::Transient),
        Err(ModelInvokeErrorKind::Transient),
        text_script("third time"),
    ]);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![])),
    );

    let out: TurnOutcome = runner
        .run_streaming(ctx, streaming_options(2, 0), ctrl())
        .await;

    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(gateway.attempts(), 3);
    assert_eq!(out.trace.rounds[0].attempts.len(), 3);
}

/// A stream that yields one delta and then never resolves — the
/// cancellation-race fixture.
struct PendAfterFirstDelta;
#[async_trait::async_trait]
impl ModelGateway for PendAfterFirstDelta {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _ctrl: &causa_kernel::AttemptControl,
    ) -> Result<causa_kernel::ModelOutput, ModelInvokeError> {
        Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Permanent,
            "fixture",
        ))
    }
    async fn stream(
        &self,
        _req: &ModelRequest,
        _ctrl: &causa_kernel::AttemptControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        let first = futures_util::stream::iter(vec![StreamDelta::TextDelta {
            delta: "started".into(),
        }]);
        Ok(Box::pin(first.chain(futures_util::stream::pending())))
    }
}

#[tokio::test]
async fn cancellation_mid_stream_interrupts_explicitly() {
    let token = CancellationToken::new();
    struct CancelOnFirstDelta {
        token: CancellationToken,
    }
    #[async_trait::async_trait]
    impl causa_kernel::TurnInteraction for CancelOnFirstDelta {
        async fn on_delta(&self, _round_id: RoundId, _delta: &StreamDelta) {
            self.token.cancel();
        }
    }
    let ctx = ctx();
    let mut options = streaming_options(3, 0);
    options.interaction = Arc::new(CancelOnFirstDelta {
        token: token.clone(),
    });
    let runner = TurnRunner::new(
        Arc::new(PendAfterFirstDelta),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![])),
    );

    let out = runner
        .run_streaming(ctx, options, RunControl::new(token, None))
        .await;

    match out.result {
        TurnResult::Interrupted {
            cause: causa_runtime::TurnInterruption::ExplicitCancellation,
        } => {}
        other => panic!("expected ExplicitCancellation, got {other:?}"),
    }
}

#[tokio::test]
async fn conversation_streaming_entry_completes_and_stamps() {
    let mut state = ConversationState::new(ConversationId("conv-stream".into()));
    state.begin_turn(TurnId::new("t1")).unwrap();
    state
        .active_turn_mut()
        .unwrap()
        .append_input(TextPayload::new("hi"), "user")
        .unwrap();
    let turn_id = state.active_turn().unwrap().turn_id();
    let collector = Arc::new(StreamEventCollector::new(
        turn_id,
        Some(ConversationId("conv-stream".into())),
    ));
    let mut options = streaming_options(0, 0);
    options.interaction = collector.clone();
    let gateway = RecordingStreamingGateway::scripted(vec![text_script("done")]);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![])),
    );

    let mut out = runner
        .run_in_conversation_streaming(state, options, ctrl())
        .await
        .unwrap();

    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert!(out.state.active_turn().unwrap().is_sealed());
    // The interaction observed the delta with the conversation envelope.
    let events = collector.events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].conversation_id,
        Some(ConversationId("conv-stream".into()))
    );
    // Host loop: commit receives the sealed turn.
    let snap = out.state.commit(TurnId::new("t1")).unwrap();
    assert_eq!(snap.turn_sequence.0, 0);
}

#[tokio::test]
async fn project_streaming_turn_interleaves_deltas_with_dispatches() {
    // Drive a real two-round streaming turn, then project it.
    let ctx = ctx();
    let collector = collector_for(&ctx);
    let mut options = streaming_options(0, 0);
    options.interaction = collector.clone();
    let gateway = RecordingStreamingGateway::scripted(vec![
        tooluse_script("calling echo", "echo", serde_json::json!({"a": 1})),
        text_script("all done"),
    ]);
    let runner = TurnRunner::new(
        gateway,
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![Arc::new(
            EchoTool,
        )])),
    );
    let out: TurnOutcome = runner.run_streaming(ctx, options, ctrl()).await;
    let deltas = collector.events();

    let events = project_streaming_turn(&out.context, &out.result, &out.trace, None, deltas);
    // Canonical order: TurnStarted, round-0 deltas (text plus the two
    // tool-call increments — name, then arguments) then
    // ToolBatchDispatched, round-1 delta, TurnOutcome.
    let kinds: Vec<&str> = events
        .iter()
        .map(|e| match &e.kind {
            causa_runtime::ContextEventKind::TurnStarted => "started",
            causa_runtime::ContextEventKind::TextDelta { .. } => "text",
            causa_runtime::ContextEventKind::ToolCallDelta { .. } => "call_delta",
            causa_runtime::ContextEventKind::ToolBatchDispatched { .. } => "dispatch",
            causa_runtime::ContextEventKind::TurnOutcome { .. } => "outcome",
            causa_runtime::ContextEventKind::ReasoningDelta { .. } => "reasoning",
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            "started",
            "text",
            "call_delta",
            "call_delta",
            "dispatch",
            "text",
            "outcome",
        ]
    );
    // The dispatch event carries the committed (pre-execution) payloads.
    match &events[4].kind {
        causa_runtime::ContextEventKind::ToolBatchDispatched {
            round_id, calls, ..
        } => {
            assert_eq!(*round_id, RoundId(0));
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].tool_name, "echo");
        }
        other => panic!("expected dispatch, got {other:?}"),
    }
}

// ---- Phase E: backoff -----------------------------------------------------------

#[tokio::test]
async fn backoff_delays_increase_and_respect_the_cap() {
    let policy = RetryPolicy {
        max_retries: 4,
        retry_timeouts: false,
        backoff_base_ms: 10,
        backoff_max_ms: 1000,
    };
    // Jitter is ±20%, so these adjacent steps cannot overlap.
    let d2 = policy.backoff_delay(2);
    let d3 = policy.backoff_delay(3);
    let d4 = policy.backoff_delay(4);
    assert!(
        d2 >= Duration::from_millis(8) && d2 <= Duration::from_millis(12),
        "d2={d2:?}"
    );
    assert!(d3 > d2, "d3={d3:?} must exceed d2={d2:?}");
    assert!(d4 > d3, "d4={d4:?} must exceed d3={d3:?}");
    // The cap applies before jitter: 10ms base, attempt 5 → 10*2^3 = 80ms,
    // under the 120ms cap; jitter keeps it within 64..96ms.
    let capped_policy = RetryPolicy {
        max_retries: 4,
        retry_timeouts: false,
        backoff_base_ms: 10,
        backoff_max_ms: 120,
    };
    let d5 = capped_policy.backoff_delay(5);
    assert!(
        d5 >= Duration::from_millis(64) && d5 <= Duration::from_millis(96),
        "d5={d5:?}"
    );
    // A cap below the raw value: 100ms base, attempt 4 → 400ms → capped at
    // 150ms; jitter keeps it within 120..180ms.
    let hard_cap = RetryPolicy {
        max_retries: 4,
        retry_timeouts: false,
        backoff_base_ms: 100,
        backoff_max_ms: 150,
    };
    let d4 = hard_cap.backoff_delay(4);
    assert!(
        d4 >= Duration::from_millis(120) && d4 <= Duration::from_millis(180),
        "d4={d4:?}"
    );
}

#[tokio::test]
async fn backoff_sleep_cancels_immediately() {
    let token = CancellationToken::new();
    let gateway = RecordingStreamingGateway::scripted(vec![Err(ModelInvokeErrorKind::Transient)]);
    let runner = TurnRunner::new(
        gateway,
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![])),
    );
    // 60s backoff; the test cancels 50ms in.
    let mut options = streaming_options(1, 60_000);
    options.interaction = Arc::new(NoopInteraction);
    let started = Instant::now();
    tokio::spawn({
        let token = token.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            token.cancel();
        }
    });
    let out = runner
        .run_streaming(ctx(), options, RunControl::new(token, None))
        .await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "backoff must be cancellation-aware, took {:?}",
        started.elapsed()
    );
    match out.result {
        TurnResult::Interrupted {
            cause: causa_runtime::TurnInterruption::ExplicitCancellation,
        } => {}
        other => panic!("expected ExplicitCancellation, got {other:?}"),
    }
}

#[tokio::test]
async fn backoff_base_zero_keeps_immediate_retry_behavior() {
    let gateway = RecordingStreamingGateway::scripted(vec![
        Err(ModelInvokeErrorKind::Transient),
        text_script("fast"),
    ]);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![])),
    );
    let started = Instant::now();
    let out: TurnOutcome = runner
        .run_streaming(ctx(), streaming_options(1, 0), ctrl())
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(gateway.attempts(), 2);
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "base 0 must not sleep, took {:?}",
        started.elapsed()
    );
}

fn ctx() -> TurnContext {
    let mut ctx = TurnContext::new(TurnId::new("t-stream"));
    ctx.append_input(TextPayload::new("hi"), "user").unwrap();
    ctx
}

// Silence the unused import when StreamScript is only used via aliases.
#[allow(unused)]
fn _assert_script_type(_: &StreamScript) {}
