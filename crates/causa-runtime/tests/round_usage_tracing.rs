//! Tracing integration tests for `agent.round` token usage fields.

mod common;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::*;
use common::{ctrl, endturn_output, options, runner_with};
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context as LayerContext;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

struct FieldVisitor<'a>(&'a mut BTreeMap<String, String>);

impl<'a> Visit for FieldVisitor<'a> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }
}

#[derive(Clone, Debug)]
struct SpanRecord {
    name: &'static str,
    fields: Arc<Mutex<BTreeMap<String, String>>>,
}

#[derive(Clone, Default)]
struct CaptureLayer {
    spans: Arc<Mutex<Vec<SpanRecord>>>,
}

impl CaptureLayer {
    fn new() -> (Self, Arc<Mutex<Vec<SpanRecord>>>) {
        let spans = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                spans: spans.clone(),
            },
            spans,
        )
    }
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: LayerContext<'_, S>) {
        let mut fields = BTreeMap::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        let record = SpanRecord {
            name: attrs.metadata().name(),
            fields: Arc::new(Mutex::new(fields)),
        };
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(record.clone());
        }
        self.spans.lock().unwrap().push(record);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: LayerContext<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let Some(record) = span.extensions().get::<SpanRecord>().cloned() else {
            return;
        };
        let mut fields = record.fields.lock().unwrap();
        values.record(&mut FieldVisitor(&mut fields));
    }
}

fn find_round_fields(spans: &[SpanRecord]) -> BTreeMap<String, String> {
    let round_span = spans
        .iter()
        .find(|s| s.name == "agent.round")
        .expect("agent.round span must be emitted");
    round_span.fields.lock().unwrap().clone()
}

struct StreamingGateway {
    deltas: Mutex<Vec<Vec<StreamDelta>>>,
}

impl StreamingGateway {
    fn new(deltas: Vec<Vec<StreamDelta>>) -> Arc<Self> {
        Arc::new(Self {
            deltas: Mutex::new(deltas),
        })
    }
}

#[async_trait]
impl ModelGateway for StreamingGateway {
    async fn invoke(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        panic!("streaming uses stream only")
    }
    async fn stream(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        let batch = self.deltas.lock().unwrap().remove(0);
        Ok(Box::pin(futures_util::stream::iter(batch)))
    }
}

#[tokio::test(flavor = "current_thread")]
async fn round_with_full_usage_records_all_four_fields() {
    let (layer, spans) = CaptureLayer::new();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut output = endturn_output("done");
    let mut usage = ModelUsage::new(7, 5);
    usage.cache_read_tokens = Some(11);
    usage.cache_write_tokens = Some(3);
    output.usage = Some(usage);

    let runner = runner_with(common::RecordingGateway::scripted(vec![Ok(output)]), vec![]);
    let outcome = runner
        .run(TurnId::new("t-full"), Context::new(), options(), ctrl())
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));

    let records = spans.lock().unwrap().clone();
    let fields = find_round_fields(&records);

    assert_eq!(fields.get("usage.input").map(String::as_str), Some("7"));
    assert_eq!(
        fields.get("usage.cache_read").map(String::as_str),
        Some("11")
    );
    assert_eq!(
        fields.get("usage.cache_write").map(String::as_str),
        Some("3")
    );
    assert_eq!(fields.get("usage.output").map(String::as_str), Some("5"));
}

#[tokio::test(flavor = "current_thread")]
async fn round_with_uncached_usage_omits_cache_fields() {
    let (layer, spans) = CaptureLayer::new();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut output = endturn_output("done");
    output.usage = Some(ModelUsage::new(7, 5));

    let runner = runner_with(common::RecordingGateway::scripted(vec![Ok(output)]), vec![]);
    let outcome = runner
        .run(TurnId::new("t-uncached"), Context::new(), options(), ctrl())
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));

    let records = spans.lock().unwrap().clone();
    let fields = find_round_fields(&records);

    assert_eq!(fields.get("usage.input").map(String::as_str), Some("7"));
    assert_eq!(fields.get("usage.output").map(String::as_str), Some("5"));
    assert!(
        !fields.contains_key("usage.cache_read"),
        "usage.cache_read must be absent when None: {fields:?}"
    );
    assert!(
        !fields.contains_key("usage.cache_write"),
        "usage.cache_write must be absent when None: {fields:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn round_without_usage_records_no_usage_fields() {
    let (layer, spans) = CaptureLayer::new();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let output = endturn_output("done");
    assert!(output.usage.is_none());

    let runner = runner_with(common::RecordingGateway::scripted(vec![Ok(output)]), vec![]);
    let outcome = runner
        .run(TurnId::new("t-no-usage"), Context::new(), options(), ctrl())
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));

    let records = spans.lock().unwrap().clone();
    let fields = find_round_fields(&records);

    assert!(
        !fields.contains_key("usage.input"),
        "usage.input must be absent when usage is None: {fields:?}"
    );
    assert!(
        !fields.contains_key("usage.output"),
        "usage.output must be absent when usage is None: {fields:?}"
    );
    assert!(
        !fields.contains_key("usage.cache_read"),
        "usage.cache_read must be absent when usage is None: {fields:?}"
    );
    assert!(
        !fields.contains_key("usage.cache_write"),
        "usage.cache_write must be absent when usage is None: {fields:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn streaming_round_records_usage_on_done() {
    let (layer, spans) = CaptureLayer::new();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut output = endturn_output("done streaming");
    let mut usage = ModelUsage::new(7, 5);
    usage.cache_read_tokens = Some(11);
    usage.cache_write_tokens = Some(3);
    output.usage = Some(usage);

    let gateway = StreamingGateway::new(vec![vec![
        StreamDelta::TextDelta {
            delta: "done streaming".into(),
        },
        StreamDelta::Done {
            final_output: output,
        },
    ]]);

    let runner = runner_with(gateway, vec![]);
    let outcome = runner
        .run_streaming(TurnId::new("t-stream"), Context::new(), options(), ctrl())
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));

    let records = spans.lock().unwrap().clone();
    let fields = find_round_fields(&records);

    assert_eq!(fields.get("usage.input").map(String::as_str), Some("7"));
    assert_eq!(
        fields.get("usage.cache_read").map(String::as_str),
        Some("11")
    );
    assert_eq!(
        fields.get("usage.cache_write").map(String::as_str),
        Some("3")
    );
    assert_eq!(fields.get("usage.output").map(String::as_str), Some("5"));
}

#[tokio::test(flavor = "current_thread")]
async fn failed_round_records_no_usage_fields() {
    let (layer, spans) = CaptureLayer::new();
    let subscriber = tracing_subscriber::registry().with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let runner = runner_with(
        common::RecordingGateway::scripted(vec![Err(ModelInvokeErrorKind::Permanent)]),
        vec![],
    );
    let outcome = runner
        .run(TurnId::new("t-err"), Context::new(), options(), ctrl())
        .await;
    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ModelCallFailed { .. },
            ..
        }
    ));

    let records = spans.lock().unwrap().clone();
    let fields = find_round_fields(&records);

    assert!(
        !fields.contains_key("usage.input"),
        "usage.input must not be recorded on error: {fields:?}"
    );
    assert!(
        !fields.contains_key("usage.output"),
        "usage.output must not be recorded on error: {fields:?}"
    );
    assert!(
        !fields.contains_key("usage.cache_read"),
        "usage.cache_read must not be recorded on error: {fields:?}"
    );
    assert!(
        !fields.contains_key("usage.cache_write"),
        "usage.cache_write must not be recorded on error: {fields:?}"
    );
}
