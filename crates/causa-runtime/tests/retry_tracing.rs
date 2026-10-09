//! The wrapper's tracing baseline: exactly one event per retry.
//!
//! Deliberately its own test binary: the capture is process-global, so no
//! other test may emit retry events concurrently with it.

use async_trait::async_trait;
use causa_kernel::{
    CacheDirective, CallControl, CancellationToken, Context, GenerationOptions, InvocationId,
    ModelGateway, ModelInvokeError, ModelInvokeErrorKind, ModelItem, ModelOutput, ModelRef,
    ModelRequest, ModelResponse, ModelStopReason, RoundId, ToolSurface, TurnId,
};
use causa_runtime::{RetryPolicy, RetryingGateway};
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

struct Sink(Arc<Mutex<String>>);
impl io::Write for Sink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .unwrap()
            .push_str(&String::from_utf8_lossy(buf));
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The process-wide capturing subscriber. Interest in the retry callsite is
/// cached globally, so a thread-scoped subscriber can race with tests that
/// run without one; a global default has no such race.
fn capture() -> &'static Arc<Mutex<String>> {
    static LOG: OnceLock<Arc<Mutex<String>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let log: Arc<Mutex<String>> = Arc::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .with_writer({
                let sink = log.clone();
                move || Sink(sink.clone())
            })
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("the tracing baseline owns the global subscriber");
        log
    })
}

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

fn request() -> ModelRequest {
    ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("retry-tracing"),
            round_id: RoundId(0),
        },
        frame: Context::new().frame(),
        model: ModelRef::new("fixed-model"),
        tool_surface: ToolSurface::empty(),
        generation: GenerationOptions::default(),
        cache: CacheDirective::None,
    }
}

/// Fails twice with a retryable kind, then succeeds.
struct Scripted(AtomicUsize);

#[async_trait]
impl ModelGateway for Scripted {
    async fn invoke(
        &self,
        _request: &ModelRequest,
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        match self.0.fetch_add(1, Ordering::SeqCst) {
            0 => Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Transient,
                "first failure",
            )),
            1 => Err(ModelInvokeError::new(
                ModelInvokeErrorKind::TimedOut,
                "second failure",
            )),
            _ => Ok(output()),
        }
    }
}

#[tokio::test]
async fn one_event_per_retry_records_attempt_kind_and_wait() {
    let log = capture();
    let wrapper = RetryingGateway::new(
        Scripted(AtomicUsize::new(0)),
        RetryPolicy::new(3).with_base_delay(Duration::from_millis(1)),
    );
    wrapper
        .invoke(
            &request(),
            &CallControl::new(CancellationToken::new(), None),
        )
        .await
        .unwrap();
    let text = log.lock().unwrap().clone();
    assert_eq!(
        text.matches("retrying model invocation").count(),
        2,
        "{text}"
    );
    assert!(text.contains("attempt=1"), "{text}");
    assert!(text.contains("attempt=2"), "{text}");
    assert!(text.contains("max_attempts=3"), "{text}");
    assert!(text.contains("kind=Transient"), "{text}");
    assert!(text.contains("kind=TimedOut"), "{text}");
    assert!(text.contains("wait_ms="), "{text}");
}
