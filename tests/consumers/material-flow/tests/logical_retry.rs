//! C6: a caller-owned finite gateway wrapper; no runtime retry policy.

use async_trait::async_trait;
use causa::kernel::{
    CacheDirective, CallControl, CancellationToken, Context, ControlError, GenerationOptions,
    InvocationId, ModelGateway, ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRef,
    ModelRequest, ModelResponse, ModelStopReason, ModelStream, ModelUsage, RoundId, StreamDelta,
    TextPayload, ToolSurface, TurnId,
};
use futures_util::{StreamExt, stream};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

fn output() -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new("complete"),
            tool_calls: vec![],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    }
}
fn request() -> ModelRequest {
    ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("retry-consumer"),
            round_id: RoundId(2),
        },
        frame: Context::new().frame(),
        model: ModelRef::new("fixed-model"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions {
            max_tokens: Some(16),
            ..Default::default()
        },
        cache: CacheDirective::StablePrefix,
    }
}
fn fingerprint(request: &ModelRequest) -> String {
    format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{}",
        request.invocation_id,
        request.model,
        request.tool_surface,
        request.generation,
        request.cache,
        serde_json::to_string(&request.frame.blocks).unwrap()
    )
}
fn error(kind: ModelInvokeErrorKind) -> ModelInvokeError {
    ModelInvokeError::new(kind, "scripted lower gateway failure")
}
fn control_error(value: ControlError) -> ModelInvokeError {
    error(match value {
        ControlError::Cancelled => ModelInvokeErrorKind::Cancelled,
        ControlError::TimedOut => ModelInvokeErrorKind::TimedOut,
    })
}
async fn stop(control: &CallControl) -> ControlError {
    tokio::select! {
        biased;
        _ = control.cancellation_token().cancelled() => ControlError::Cancelled,
        _ = async {
            match control.deadline() {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        } => ControlError::TimedOut,
    }
}

struct Scripted {
    invokes: Mutex<VecDeque<Result<ModelOutput, ModelInvokeError>>>,
    streams: Mutex<VecDeque<Result<Vec<StreamDelta>, ModelInvokeError>>>,
    requests: Mutex<Vec<String>>,
}
#[async_trait]
impl ModelGateway for Scripted {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.requests.lock().unwrap().push(fingerprint(request));
        self.invokes
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected invoke")
    }
    async fn stream(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        self.requests.lock().unwrap().push(fingerprint(request));
        let deltas = self
            .streams
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected stream")?;
        Ok(Box::pin(stream::iter(deltas)))
    }
}
struct FiniteRetry {
    inner: Arc<dyn ModelGateway>,
    backoff: Option<Arc<Notify>>,
    attempt_timeout: Option<Duration>,
}
impl FiniteRetry {
    fn new(inner: Arc<dyn ModelGateway>) -> Self {
        Self {
            inner,
            backoff: None,
            attempt_timeout: None,
        }
    }
    fn attempt_control(&self, control: &CallControl) -> CallControl {
        match self.attempt_timeout {
            Some(timeout) => control.with_timeout(timeout),
            None => control.clone(),
        }
    }
    async fn backoff(&self, control: &CallControl) -> Result<(), ModelInvokeError> {
        if let Some(entered) = &self.backoff {
            entered.notify_one();
            // A caller's finite wait is independently cancellable; no parent token mutation.
            tokio::select! {
                biased;
                cause = stop(control) => return Err(control_error(cause)),
                _ = tokio::time::sleep(Duration::from_secs(30)) => {},
            }
        }
        control.check().map_err(control_error)
    }
}
#[async_trait]
impl ModelGateway for FiniteRetry {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        for attempt in 0..3 {
            control.check().map_err(control_error)?;
            let attempt_control = self.attempt_control(control);
            attempt_control.check().map_err(control_error)?;
            let result = tokio::select! {
                biased;
                cause = stop(&attempt_control) => Err(control_error(cause)),
                result = self.inner.invoke(request, &attempt_control) => result,
            };
            match result {
                Err(error) if attempt < 2 && error.kind == ModelInvokeErrorKind::Transient => {
                    self.backoff(control).await?
                }
                result => return result,
            }
        }
        unreachable!()
    }
    async fn stream(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        for attempt in 0..3 {
            control.check().map_err(control_error)?;
            let attempt_control = self.attempt_control(control);
            attempt_control.check().map_err(control_error)?;
            let result = tokio::select! {
                biased;
                cause = stop(&attempt_control) => Err(control_error(cause)),
                result = async {
                    let mut source = self.inner.stream(request, &attempt_control).await?;
                    let first = source.next().await;
                    match first {
                        Some(StreamDelta::Error { kind, message }) => Err(ModelInvokeError::new(kind, message)),
                        None => Err(error(ModelInvokeErrorKind::UnknownOutcome)),
                        Some(first) => {
                            // Returning this stream marks the irrevocable visibility boundary.
                            // Its remainder has no path back into the retry loop.
                            let emitted: ModelStream = Box::pin(stream::once(async move { first }).chain(source));
                            Ok(emitted)
                        },
                    }
                } => result,
            };
            match result {
                Err(error)
                    if attempt < 2
                        && matches!(
                            error.kind,
                            ModelInvokeErrorKind::Transient | ModelInvokeErrorKind::UnknownOutcome
                        ) =>
                {
                    self.backoff(control).await?
                }
                result => return result,
            }
        }
        unreachable!()
    }
}
fn scripted(
    invokes: Vec<Result<ModelOutput, ModelInvokeError>>,
    streams: Vec<Result<Vec<StreamDelta>, ModelInvokeError>>,
) -> Arc<Scripted> {
    Arc::new(Scripted {
        invokes: Mutex::new(invokes.into()),
        streams: Mutex::new(streams.into()),
        requests: Mutex::new(Vec::new()),
    })
}

#[tokio::test]
async fn invoke_retries_real_failures_with_every_request_field_unchanged() {
    let inner = scripted(
        vec![
            Err(error(ModelInvokeErrorKind::Transient)),
            Err(error(ModelInvokeErrorKind::Transient)),
            Ok(output()),
        ],
        vec![],
    );
    let wrapper = FiniteRetry::new(inner.clone());
    let output = wrapper
        .invoke(
            &request(),
            &CallControl::new(CancellationToken::new(), None),
        )
        .await
        .unwrap();
    assert_eq!(output.response.text.0, "complete");
    let records = inner.requests.lock().unwrap();
    assert_eq!(records.len(), 3);
    assert!(records.windows(2).all(|pair| pair[0] == pair[1]));
}

#[tokio::test]
async fn stream_can_retry_before_yield_after_establishment_error_delta_or_empty_stream() {
    let done = StreamDelta::Done {
        stop_reason: ModelStopReason::EndTurn,
        final_output: output(),
    };
    for failed in [
        Err(error(ModelInvokeErrorKind::Transient)),
        Ok(vec![StreamDelta::Error {
            kind: ModelInvokeErrorKind::Transient,
            message: "before yield".into(),
        }]),
        Ok(vec![]),
    ] {
        let inner = scripted(vec![], vec![failed, Ok(vec![done.clone()])]);
        let wrapper = FiniteRetry::new(inner.clone());
        let mut stream = wrapper
            .stream(
                &request(),
                &CallControl::new(CancellationToken::new(), None),
            )
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await,
            Some(StreamDelta::Done { .. })
        ));
        assert!(stream.next().await.is_none());
        assert_eq!(inner.requests.lock().unwrap().len(), 2);
    }
}

#[tokio::test]
async fn any_visible_delta_prevents_transparent_retry_even_without_an_observer() {
    let visible = [
        StreamDelta::TextDelta {
            delta: "partial".into(),
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
        StreamDelta::ReasoningDelta {
            delta: "thinking".into(),
        },
    ];
    for delta in visible {
        let inner = scripted(
            vec![],
            vec![
                Ok(vec![
                    delta,
                    StreamDelta::Error {
                        kind: ModelInvokeErrorKind::Transient,
                        message: "after visibility".into(),
                    },
                ]),
                Ok(vec![StreamDelta::Done {
                    stop_reason: ModelStopReason::EndTurn,
                    final_output: output(),
                }]),
            ],
        );
        let wrapper = FiniteRetry::new(inner.clone());
        let deltas = wrapper
            .stream(
                &request(),
                &CallControl::new(CancellationToken::new(), None),
            )
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        assert_eq!(deltas.len(), 2);
        assert!(matches!(deltas[1], StreamDelta::Error { .. }));
        assert_eq!(inner.requests.lock().unwrap().len(), 1);
    }
    let inner = scripted(
        vec![],
        vec![Ok(vec![StreamDelta::TextDelta {
            delta: "partial".into(),
        }])],
    );
    let deltas = FiniteRetry::new(inner.clone())
        .stream(
            &request(),
            &CallControl::new(CancellationToken::new(), None),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(deltas.len(), 1);
    assert!(
        !deltas
            .iter()
            .any(|delta| matches!(delta, StreamDelta::Done { .. }))
    );
    assert_eq!(inner.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn cancellation_during_backoff_stops_following_requests() {
    let inner = scripted(
        vec![Err(error(ModelInvokeErrorKind::Transient)), Ok(output())],
        vec![],
    );
    let entered = Arc::new(Notify::new());
    let wrapper = FiniteRetry {
        inner: inner.clone(),
        backoff: Some(entered.clone()),
        attempt_timeout: None,
    };
    let token = CancellationToken::new();
    let control = CallControl::new(token.clone(), None);
    let request = request();
    let future = wrapper.invoke(&request, &control);
    tokio::pin!(future);
    tokio::select! { _ = entered.notified() => {}, result = &mut future => panic!("ended before cancellation: {result:?}") }
    token.cancel();
    assert_eq!(
        future.await.unwrap_err().kind,
        ModelInvokeErrorKind::Cancelled
    );
    assert_eq!(inner.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn local_timeout_remains_a_model_error_and_does_not_cancel_parent() {
    let inner = scripted(vec![Ok(output())], vec![]);
    let wrapper = FiniteRetry {
        inner: inner.clone(),
        backoff: None,
        attempt_timeout: Some(Duration::ZERO),
    };
    let control = CallControl::new(
        CancellationToken::new(),
        Some(Instant::now() + Duration::from_secs(30)),
    );
    assert_eq!(
        wrapper.invoke(&request(), &control).await.unwrap_err().kind,
        ModelInvokeErrorKind::TimedOut
    );
    assert!(control.check().is_ok());
    assert!(!control.is_cancelled());
    assert!(inner.requests.lock().unwrap().is_empty());
}

struct WaitingStream {
    entered: Arc<Notify>,
    wait_in_establishment: bool,
    calls: std::sync::atomic::AtomicUsize,
}
#[async_trait]
impl ModelGateway for WaitingStream {
    async fn invoke(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        unreachable!("stream-only script")
    }
    async fn stream(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.entered.notify_one();
        if self.wait_in_establishment {
            std::future::pending::<()>().await;
        }
        Ok(Box::pin(stream::pending()))
    }
}

#[tokio::test]
async fn cancellation_stops_stream_establishment_and_wait_for_first_delta() {
    for wait_in_establishment in [true, false] {
        let entered = Arc::new(Notify::new());
        let inner = Arc::new(WaitingStream {
            entered: entered.clone(),
            wait_in_establishment,
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let wrapper = FiniteRetry::new(inner.clone());
        let token = CancellationToken::new();
        let control = CallControl::new(token.clone(), None);
        let request = request();
        let future = wrapper.stream(&request, &control);
        tokio::pin!(future);
        tokio::select! { _ = entered.notified() => {}, result = &mut future => panic!("stream completed before cancellation: {}", result.is_ok()) }
        token.cancel();
        match future.await {
            Err(error) => assert_eq!(error.kind, ModelInvokeErrorKind::Cancelled),
            Ok(_) => panic!("expected cancellation"),
        }
        assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn parent_deadline_bounds_stream_wait_and_retry_backoff() {
    for wait_in_establishment in [true, false] {
        let inner = Arc::new(WaitingStream {
            entered: Arc::new(Notify::new()),
            wait_in_establishment,
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let wrapper = FiniteRetry::new(inner.clone());
        let control = CallControl::new(
            CancellationToken::new(),
            Some(Instant::now() + Duration::from_millis(10)),
        );
        match wrapper.stream(&request(), &control).await {
            Err(error) => assert_eq!(error.kind, ModelInvokeErrorKind::TimedOut),
            Ok(_) => panic!("expected deadline"),
        }
        assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(!control.is_cancelled());
    }
    let inner = scripted(
        vec![Err(error(ModelInvokeErrorKind::Transient)), Ok(output())],
        vec![],
    );
    let wrapper = FiniteRetry {
        inner: inner.clone(),
        backoff: Some(Arc::new(Notify::new())),
        attempt_timeout: None,
    };
    let control = CallControl::new(
        CancellationToken::new(),
        Some(Instant::now() + Duration::from_millis(10)),
    );
    assert_eq!(
        wrapper.invoke(&request(), &control).await.unwrap_err().kind,
        ModelInvokeErrorKind::TimedOut
    );
    assert_eq!(inner.requests.lock().unwrap().len(), 1);
    assert!(!control.is_cancelled());
}
