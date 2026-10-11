//! Behavior of the optional [`RetryingGateway`] wrapper.
//!
//! Coverage: the policy surface, the
//! retryable kinds, the same request on every attempt, stream visibility,
//! server delays, backoff jitter and the caller's cancellation/deadline.

use async_trait::async_trait;
use causa_kernel::{
    CacheDirective, CallControl, CancellationToken, Context, GenerationOptions, InvocationId,
    ModelGateway, ModelInvokeError, ModelInvokeErrorKind, ModelItem, ModelOutput, ModelRef,
    ModelRequest, ModelResponse, ModelStopReason, ModelStream, ModelUsage, RoundId, StreamDelta,
    ToolSurface, TurnId,
};
use causa_runtime::{RetryPolicy, RetryingGateway};
use futures_util::{StreamExt, stream};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

fn output() -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            items: vec![ModelItem::text("complete")],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        raw_stop_reason: None,
        reasoning: None,
    }
}

fn done() -> StreamDelta {
    StreamDelta::Done {
        final_output: output(),
    }
}

fn request() -> ModelRequest {
    ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("retry-wrapper"),
            round_id: RoundId(2),
        },
        frame: Context::new().frame(),
        model: ModelRef::new("fixed-model"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions::default(),
        cache: CacheDirective::None,
    }
}

fn error(kind: ModelInvokeErrorKind) -> ModelInvokeError {
    ModelInvokeError::new(kind, "scripted lower gateway failure")
}

fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

struct State {
    invokes: Mutex<VecDeque<Result<ModelOutput, ModelInvokeError>>>,
    streams: Mutex<VecDeque<Result<Vec<StreamDelta>, ModelInvokeError>>>,
    calls: AtomicUsize,
    requests: Mutex<Vec<usize>>,
    at: Mutex<Vec<tokio::time::Instant>>,
    entered: Notify,
}

struct Scripted {
    state: Arc<State>,
}

fn scripted(
    invokes: Vec<Result<ModelOutput, ModelInvokeError>>,
    streams: Vec<Result<Vec<StreamDelta>, ModelInvokeError>>,
) -> (Scripted, Arc<State>) {
    let state = Arc::new(State {
        invokes: Mutex::new(invokes.into()),
        streams: Mutex::new(streams.into()),
        calls: AtomicUsize::new(0),
        requests: Mutex::new(Vec::new()),
        at: Mutex::new(Vec::new()),
        entered: Notify::new(),
    });
    (
        Scripted {
            state: state.clone(),
        },
        state,
    )
}

impl Scripted {
    fn record(&self, request: &ModelRequest) {
        self.state.calls.fetch_add(1, Ordering::SeqCst);
        self.state
            .requests
            .lock()
            .unwrap()
            .push(std::ptr::from_ref(request) as usize);
        self.state
            .at
            .lock()
            .unwrap()
            .push(tokio::time::Instant::now());
        self.state.entered.notify_waiters();
    }
}

#[async_trait]
impl ModelGateway for Scripted {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.record(request);
        self.state
            .invokes
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected invoke")
    }

    async fn stream(
        &self,
        request: &ModelRequest,
        _control: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        self.record(request);
        let deltas = self
            .state
            .streams
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected stream")?;
        Ok(Box::pin(stream::iter(deltas)))
    }
}

fn calls(state: &State) -> usize {
    state.calls.load(Ordering::SeqCst)
}

/// Virtual waits between consecutive attempts, from the paused clock.
fn waits(state: &State) -> Vec<Duration> {
    let at = state.at.lock().unwrap();
    at.windows(2)
        .map(|pair| pair[1].duration_since(pair[0]))
        .collect()
}

fn plain(policy: RetryPolicy) -> RetryPolicy {
    // Retries under test should not spend their (virtual) backoff budget.
    policy.with_base_delay(Duration::ZERO)
}

#[test]
fn policy_starts_at_the_reference_delays_and_states_every_field() {
    // No `Default` exists: the values below are the whole surface.
    let policy = RetryPolicy::new(3);
    assert_eq!(policy.max_attempts, 3);
    assert_eq!(policy.base_delay, Duration::from_millis(500));
    assert_eq!(policy.max_delay, Duration::from_secs(8));
    assert_eq!(policy.max_server_delay, Duration::from_secs(60));
    // max_attempts counts the first call; below one is raised to one.
    assert_eq!(RetryPolicy::new(0).max_attempts, 1);
    assert_eq!(RetryPolicy::new(1).max_attempts, 1);
    let custom = RetryPolicy::new(2)
        .with_base_delay(Duration::from_millis(10))
        .with_max_delay(Duration::from_millis(20))
        .with_max_server_delay(Duration::from_millis(30));
    assert_eq!(custom.base_delay, Duration::from_millis(10));
    assert_eq!(custom.max_delay, Duration::from_millis(20));
    assert_eq!(custom.max_server_delay, Duration::from_millis(30));
}

#[tokio::test]
async fn invoke_retries_transient_and_timed_out_until_success() {
    let (inner, state) = scripted(
        vec![
            Err(error(ModelInvokeErrorKind::Transient)),
            Err(error(ModelInvokeErrorKind::TimedOut)),
            Ok(output()),
        ],
        vec![],
    );
    let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(3)));
    let result = wrapper.invoke(&request(), &control()).await.unwrap();
    assert_eq!(result.response.text(), "complete");
    assert_eq!(calls(&state), 3);
}

#[tokio::test]
async fn every_attempt_sends_the_same_request_value() {
    let (inner, state) = scripted(
        vec![
            Err(error(ModelInvokeErrorKind::Transient)),
            Err(error(ModelInvokeErrorKind::TimedOut)),
            Ok(output()),
        ],
        vec![],
    );
    let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(3)));
    let request = request();
    wrapper.invoke(&request, &control()).await.unwrap();
    let seen = state.requests.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert!(seen.windows(2).all(|pair| pair[0] == pair[1]));
}

#[tokio::test]
async fn attempts_are_bounded_and_the_last_error_is_returned() {
    let (inner, state) = scripted(
        vec![
            Err(error(ModelInvokeErrorKind::Transient)),
            Err(error(ModelInvokeErrorKind::TimedOut)),
            Err(error(ModelInvokeErrorKind::Transient)),
            Ok(output()),
        ],
        vec![],
    );
    let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(3)));
    let failure = wrapper.invoke(&request(), &control()).await.unwrap_err();
    assert_eq!(failure.kind, ModelInvokeErrorKind::Transient);
    assert_eq!(calls(&state), 3);
}

#[tokio::test]
async fn a_single_attempt_policy_never_retries() {
    let (inner, state) = scripted(
        vec![Err(error(ModelInvokeErrorKind::Transient)), Ok(output())],
        vec![],
    );
    let wrapper = RetryingGateway::new(inner, RetryPolicy::new(1));
    let failure = wrapper.invoke(&request(), &control()).await.unwrap_err();
    assert_eq!(failure.kind, ModelInvokeErrorKind::Transient);
    assert_eq!(calls(&state), 1);
}

#[tokio::test]
async fn non_retryable_kinds_are_never_retried() {
    for kind in [
        ModelInvokeErrorKind::Permanent,
        ModelInvokeErrorKind::InvalidRequest,
        ModelInvokeErrorKind::Cancelled,
        ModelInvokeErrorKind::UnknownOutcome,
    ] {
        let (inner, state) = scripted(vec![Err(error(kind.clone())), Ok(output())], vec![]);
        let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(3)));
        let failure = wrapper.invoke(&request(), &control()).await.unwrap_err();
        assert_eq!(failure.kind, kind);
        assert_eq!(calls(&state), 1, "invoke {kind:?}");

        let (inner, state) = scripted(vec![], vec![Err(error(kind.clone())), Ok(vec![done()])]);
        let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(3)));
        let failure = wrapper
            .stream(&request(), &control())
            .await
            .err()
            .expect("stream establishment must fail");
        assert_eq!(failure.kind, kind);
        assert_eq!(calls(&state), 1, "stream establishment {kind:?}");

        // A non-retryable first item is handed on, never retried.
        let (inner, state) = scripted(
            vec![],
            vec![
                Ok(vec![StreamDelta::Error {
                    error: error(kind.clone()),
                }]),
                Ok(vec![done()]),
            ],
        );
        let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(3)));
        let mut stream = wrapper.stream(&request(), &control()).await.unwrap();
        assert!(matches!(
            stream.next().await,
            Some(StreamDelta::Error { .. })
        ));
        assert!(stream.next().await.is_none());
        assert_eq!(calls(&state), 1, "stream first item {kind:?}");
    }
}

#[tokio::test]
async fn stream_retries_an_error_before_establishment() {
    let (inner, state) = scripted(
        vec![],
        vec![
            Err(error(ModelInvokeErrorKind::Transient)),
            Ok(vec![done()]),
        ],
    );
    let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(2)));
    let mut stream = wrapper.stream(&request(), &control()).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(StreamDelta::Done { .. })
    ));
    assert!(stream.next().await.is_none());
    assert_eq!(calls(&state), 2);
}

#[tokio::test]
async fn stream_retries_a_first_item_error_before_any_visibility() {
    let (inner, state) = scripted(
        vec![],
        vec![
            Ok(vec![StreamDelta::Error {
                error: error(ModelInvokeErrorKind::Transient),
            }]),
            Ok(vec![StreamDelta::TextDelta {
                delta: "recovered".into(),
            }]),
        ],
    );
    let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(2)));
    let deltas: Vec<_> = wrapper
        .stream(&request(), &control())
        .await
        .unwrap()
        .collect()
        .await;
    assert_eq!(deltas.len(), 1);
    assert!(matches!(&deltas[0], StreamDelta::TextDelta { delta } if delta == "recovered"));
    assert_eq!(calls(&state), 2);
}

#[tokio::test(start_paused = true)]
async fn first_item_error_keeps_its_server_delay() {
    let (inner, state) = scripted(
        vec![],
        vec![
            Ok(vec![StreamDelta::Error {
                error: error(ModelInvokeErrorKind::Transient)
                    .with_retry_delay(Duration::from_secs(2)),
            }]),
            Ok(vec![done()]),
        ],
    );
    let wrapper = RetryingGateway::new(inner, RetryPolicy::new(2));
    let mut stream = wrapper.stream(&request(), &control()).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(StreamDelta::Done { .. })
    ));
    assert_eq!(waits(&state), vec![Duration::from_secs(2)]);
    assert_eq!(calls(&state), 2);
}

#[tokio::test]
async fn a_failure_after_a_visible_delta_is_never_retried() {
    let visible = [
        StreamDelta::TextDelta {
            delta: "partial".into(),
        },
        StreamDelta::ReasoningDelta {
            delta: "thinking".into(),
        },
        StreamDelta::ToolCallDelta {
            call_index: 0,
            provider_call_id: Some("call-1".into()),
            name_delta: Some("tool".into()),
            arguments_delta: Some("{".into()),
        },
        StreamDelta::Usage(ModelUsage {
            input_tokens: 5,
            ..Default::default()
        }),
    ];
    for delta in visible {
        let (inner, state) = scripted(
            vec![],
            vec![
                Ok(vec![
                    delta,
                    StreamDelta::Error {
                        error: error(ModelInvokeErrorKind::Transient),
                    },
                ]),
                Ok(vec![done()]),
            ],
        );
        let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(3)));
        let deltas: Vec<_> = wrapper
            .stream(&request(), &control())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(deltas.len(), 2);
        assert!(matches!(deltas[1], StreamDelta::Error { .. }));
        assert_eq!(calls(&state), 1);
    }
}

#[tokio::test]
async fn a_stream_that_ends_before_any_item_is_unknown_outcome_and_not_retried() {
    let (inner, state) = scripted(vec![], vec![Ok(vec![]), Ok(vec![done()])]);
    let wrapper = RetryingGateway::new(inner, plain(RetryPolicy::new(3)));
    let failure = wrapper
        .stream(&request(), &control())
        .await
        .err()
        .expect("stream establishment must fail");
    assert_eq!(failure.kind, ModelInvokeErrorKind::UnknownOutcome);
    assert_eq!(calls(&state), 1);
}

#[tokio::test(start_paused = true)]
async fn own_backoff_for_attempt_n_stays_between_75_and_100_percent() {
    let base = Duration::from_secs(1);
    let (inner, state) = scripted(
        vec![
            Err(error(ModelInvokeErrorKind::Transient)),
            Err(error(ModelInvokeErrorKind::Transient)),
            Ok(output()),
        ],
        vec![],
    );
    let wrapper = RetryingGateway::new(inner, RetryPolicy::new(3).with_base_delay(base));
    wrapper.invoke(&request(), &control()).await.unwrap();
    let waits = waits(&state);
    assert_eq!(waits.len(), 2, "{waits:?}");
    // Attempt 1 backs off base * 2^0, attempt 2 base * 2^1, each jittered
    // into [75%, 100%] of the capped value.
    assert!(
        waits[0] >= base.mul_f64(0.75) && waits[0] <= base,
        "{:?}",
        waits[0]
    );
    let doubled = base * 2;
    assert!(
        waits[1] >= doubled.mul_f64(0.75) && waits[1] <= doubled,
        "{:?}",
        waits[1]
    );
    assert_eq!(calls(&state), 3);
}

#[tokio::test(start_paused = true)]
async fn a_server_delay_is_waited_as_given_even_above_the_backoff_cap() {
    let (inner, state) = scripted(
        vec![
            Err(error(ModelInvokeErrorKind::Transient).with_retry_delay(Duration::from_secs(5))),
            Ok(output()),
        ],
        vec![],
    );
    // max_delay (1 s) caps only the wrapper's own backoff.
    let wrapper = RetryingGateway::new(
        inner,
        RetryPolicy::new(2)
            .with_base_delay(Duration::from_millis(1))
            .with_max_delay(Duration::from_secs(1)),
    );
    wrapper.invoke(&request(), &control()).await.unwrap();
    assert_eq!(waits(&state), vec![Duration::from_secs(5)]);
}

#[tokio::test(start_paused = true)]
async fn a_server_delay_above_the_ceiling_fails_without_waiting() {
    let start = tokio::time::Instant::now();
    let (inner, state) = scripted(
        vec![
            Err(error(ModelInvokeErrorKind::Transient).with_retry_delay(Duration::from_secs(61))),
            Ok(output()),
        ],
        vec![],
    );
    let wrapper = RetryingGateway::new(inner, RetryPolicy::new(3));
    let failure = wrapper.invoke(&request(), &control()).await.unwrap_err();
    assert_eq!(failure.kind, ModelInvokeErrorKind::Transient);
    assert_eq!(failure.retry_delay, Some(Duration::from_secs(61)));
    assert_eq!(calls(&state), 1);
    assert_eq!(start.elapsed(), Duration::ZERO);
}

#[tokio::test(start_paused = true)]
async fn a_wait_that_would_pass_the_parent_deadline_returns_the_last_error() {
    let start = tokio::time::Instant::now();
    let (inner, state) = scripted(
        vec![Err(error(ModelInvokeErrorKind::Transient)), Ok(output())],
        vec![],
    );
    let wrapper = RetryingGateway::new(
        inner,
        RetryPolicy::new(3)
            .with_base_delay(Duration::from_secs(30))
            .with_max_delay(Duration::from_secs(30)),
    );
    let control = CallControl::new(
        CancellationToken::new(),
        Some(std::time::Instant::now() + Duration::from_secs(10)),
    );
    let failure = wrapper.invoke(&request(), &control).await.unwrap_err();
    assert_eq!(failure.kind, ModelInvokeErrorKind::Transient);
    assert_eq!(calls(&state), 1);
    assert_eq!(start.elapsed(), Duration::ZERO);
    assert!(control.check().is_ok(), "the parent deadline is untouched");
}

#[tokio::test(start_paused = true)]
async fn cancellation_during_backoff_returns_cancelled_without_another_attempt() {
    let (inner, state) = scripted(
        vec![Err(error(ModelInvokeErrorKind::Transient)), Ok(output())],
        vec![],
    );
    let wrapper = RetryingGateway::new(
        inner,
        RetryPolicy::new(3).with_base_delay(Duration::from_secs(30)),
    );
    let token = CancellationToken::new();
    let control = CallControl::new(token.clone(), None);
    let request = request();
    let future = wrapper.invoke(&request, &control);
    tokio::pin!(future);
    tokio::select! {
        _ = state.entered.notified() => {}
        result = &mut future => panic!("ended before the backoff: {result:?}"),
    }
    // Let the wrapper enter its wait, then cancel mid-wait.
    tokio::time::sleep(Duration::from_millis(50)).await;
    token.cancel();
    let failure = future.await.unwrap_err();
    assert_eq!(failure.kind, ModelInvokeErrorKind::Cancelled);
    assert_eq!(calls(&state), 1);
    assert!(!control.is_cancelled() || control.check().is_err());
}

#[tokio::test]
async fn success_is_returned_after_exactly_one_call() {
    let (inner, state) = scripted(vec![Ok(output())], vec![Ok(vec![done()])]);
    let wrapper = RetryingGateway::new(inner, RetryPolicy::new(3));
    assert_eq!(
        wrapper
            .invoke(&request(), &control())
            .await
            .unwrap()
            .response
            .text(),
        "complete"
    );
    let mut stream = wrapper.stream(&request(), &control()).await.unwrap();
    assert!(matches!(
        stream.next().await,
        Some(StreamDelta::Done { .. })
    ));
    assert_eq!(calls(&state), 2);
}
