//! Retrying `ModelGateway` wrapper: attempt loop, backoff and control.

use std::future::Future;
use std::time::Instant;

use async_trait::async_trait;
use causa_kernel::{
    CallControl, ControlError, ModelGateway, ModelInvokeError, ModelInvokeErrorKind, ModelOutput,
    ModelRequest, ModelStream, StreamDelta,
};
use futures_util::StreamExt;

use super::RetryPolicy;

/// A [`ModelGateway`] that retries an inner gateway under a [`RetryPolicy`].
///
/// The wrapper is an ordinary gateway — no new trait, no runner option, off
/// unless the host builds one — and every attempt sends the same
/// [`ModelRequest`].
///
/// - Only [`ModelInvokeErrorKind::Transient`] and
///   [`ModelInvokeErrorKind::TimedOut`] failures are retried. `Permanent`,
///   `Undecodable`, `InvalidRequest`, `Cancelled` and `UnknownOutcome` are
///   never retried,
///   and there is no setting for that.
/// - `invoke` and `stream` share the policy; each logical call counts its
///   own attempts.
/// - `stream` reads the inner stream's first item before returning. An inner
///   error, or a first item that is a [`StreamDelta::Error`] with a retryable
///   kind, starts another attempt. Any other first item is handed on followed
///   by the rest of the stream, and the call is never retried from then on —
///   a failure after the first delta reaches the caller unchanged. An inner
///   stream that ends before yielding any item is classified `UnknownOutcome`
///   and never retried.
/// - A wait is raced against the caller's [`CallControl`]: cancellation during
///   a wait returns `Cancelled`, and a wait that would reach the parent
///   deadline returns the last error instead of waiting. The wrapper never
///   extends the parent deadline and never cancels the parent token.
/// - A server-requested delay ([`ModelInvokeError::retry_delay`]) is waited as
///   given while it is at or below `max_server_delay`; above the ceiling the
///   call fails at once. Without one, the wrapper uses its own jittered
///   exponential backoff.
/// - One `tracing` event per retry records the attempt, the failure kind and
///   the wait.
#[derive(Clone)]
pub struct RetryingGateway<G: ModelGateway> {
    inner: G,
    policy: RetryPolicy,
}

impl<G: ModelGateway> std::fmt::Debug for RetryingGateway<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryingGateway")
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl<G: ModelGateway> RetryingGateway<G> {
    /// Wraps `inner` with `policy`. The wrapped value stays reachable
    /// through [`RetryingGateway::inner`]; a host that wants a shared trait
    /// object builds the wrapper around the concrete gateway and coerces
    /// `Arc::new(RetryingGateway::new(gateway, policy))` to
    /// `Arc<dyn ModelGateway>`.
    pub fn new(inner: G, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    /// The retry policy in force.
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// The wrapped gateway.
    pub fn inner(&self) -> &G {
        &self.inner
    }

    /// Whether `error` may start another attempt after `attempt` calls.
    fn retryable(&self, error: &ModelInvokeError, attempt: u32) -> bool {
        attempt < self.policy.max_attempts
            && matches!(
                error.kind,
                ModelInvokeErrorKind::Transient | ModelInvokeErrorKind::TimedOut
            )
    }

    /// Wait before the next attempt. `Err` means the call gives up and
    /// reports that error instead: a server delay above the ceiling, or a
    /// wait that would reach the parent deadline.
    async fn wait(
        &self,
        attempt: u32,
        error: &ModelInvokeError,
        control: &CallControl,
    ) -> Result<(), ModelInvokeError> {
        let delay = match error.retry_delay {
            Some(server) if server <= self.policy.max_server_delay => server,
            // The provider asked for longer than the host allows: do not wait.
            Some(_) => return Err(error.clone()),
            None => self.policy.backoff(attempt),
        };
        if let Some(deadline) = control.deadline()
            && Instant::now()
                .checked_add(delay)
                .is_none_or(|end| end >= deadline)
        {
            // The wait would reach or pass the parent deadline: report the
            // last error now rather than extend the call.
            return Err(error.clone());
        }
        tracing::debug!(
            target: "causa_runtime::retry",
            attempt,
            max_attempts = self.policy.max_attempts,
            kind = ?error.kind,
            wait_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
            "retrying model invocation"
        );
        tokio::select! {
            biased;
            _ = control.cancellation_token().cancelled() => {
                Err(control_error(ControlError::Cancelled))
            }
            _ = tokio::time::sleep(delay) => match control.check() {
                Ok(()) => Ok(()),
                Err(ControlError::Cancelled) => Err(control_error(ControlError::Cancelled)),
                // The deadline arrived while waiting: the last error stands
                // and no further attempt starts.
                Err(ControlError::TimedOut) => Err(error.clone()),
            },
        }
    }
}

#[async_trait]
impl<G: ModelGateway> ModelGateway for RetryingGateway<G> {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut attempt = 1;
        loop {
            check_control(control)?;
            match under_control(control, self.inner.invoke(request, control)).await {
                Attempt::Stopped(error) => return Err(error),
                Attempt::Ran(Ok(output)) => return Ok(output),
                Attempt::Ran(Err(error)) => {
                    if !self.retryable(&error, attempt) {
                        return Err(error);
                    }
                    self.wait(attempt, &error, control).await?;
                    attempt += 1;
                }
            }
        }
    }

    async fn stream(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        let mut attempt = 1;
        loop {
            check_control(control)?;
            let first = under_control(control, async {
                let mut source = self.inner.stream(request, control).await?;
                Ok(source.next().await.map(|delta| (delta, source)))
            })
            .await;
            match first {
                Attempt::Stopped(error) => return Err(error),
                Attempt::Ran(Err(error)) => {
                    if !self.retryable(&error, attempt) {
                        return Err(error);
                    }
                    self.wait(attempt, &error, control).await?;
                    attempt += 1;
                }
                // No item means no Done either: the wrapper classifies the
                // empty stream itself and never retries UnknownOutcome.
                Attempt::Ran(Ok(None)) => {
                    return Err(ModelInvokeError::new(
                        ModelInvokeErrorKind::UnknownOutcome,
                        "model stream ended without Done before yielding any item",
                    ));
                }
                Attempt::Ran(Ok(Some((first, source)))) => {
                    if let StreamDelta::Error { error } = &first
                        && self.retryable(error, attempt)
                    {
                        let error = error.clone();
                        // The failed attempt never reached the caller, so the
                        // policy applies exactly as before establishment.
                        self.wait(attempt, &error, control).await?;
                        attempt += 1;
                        continue;
                    }
                    // Returning this stream is the irrevocable visibility
                    // boundary: nothing after the first item is retried.
                    return Ok(Box::pin(
                        futures_util::stream::once(async move { first }).chain(source),
                    ));
                }
            }
        }
    }
}

/// One attempt's outcome, or the control stop that ended the call.
enum Attempt<T> {
    /// The attempt ran; its result applies.
    Ran(Result<T, ModelInvokeError>),
    /// The caller's control stopped the call first; never retried.
    Stopped(ModelInvokeError),
}

/// Race one attempt against the caller's cancellation and deadline. Racing
/// here keeps the wrapper's own waits and attempts bounded even around an
/// inner gateway that does not check the control plane itself.
async fn under_control<T>(
    control: &CallControl,
    work: impl Future<Output = Result<T, ModelInvokeError>>,
) -> Attempt<T> {
    tokio::select! {
        biased;
        cause = stop(control) => Attempt::Stopped(control_error(cause)),
        result = work => Attempt::Ran(result),
    }
}

/// The caller's cancellation or deadline, whichever arrives first.
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

fn check_control(control: &CallControl) -> Result<(), ModelInvokeError> {
    control.check().map_err(control_error)
}

fn control_error(cause: ControlError) -> ModelInvokeError {
    match cause {
        ControlError::Cancelled => ModelInvokeError::new(
            ModelInvokeErrorKind::Cancelled,
            "model invocation cancelled",
        ),
        ControlError::TimedOut => ModelInvokeError::new(
            ModelInvokeErrorKind::TimedOut,
            "model invocation deadline exceeded",
        ),
    }
}
