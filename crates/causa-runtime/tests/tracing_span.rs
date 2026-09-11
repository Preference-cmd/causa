//! Tracing-baseline tests: the driver emits the
//! `agent.turn` / `agent.round` / `agent.attempt` / `agent.tool` span
//! hierarchy with id-and-name fields — observable through a capturing
//! subscriber, never carrying message payloads.

mod common;

use std::io;
use std::sync::{Arc, Mutex, OnceLock};

use causa_runtime::TurnResult;
use common::{
    EchoTool, RecordingGateway, ctrl, ctx, endturn_output, options_with_limits, runner_with,
    tooluse_output,
};

/// Appends every write into the shared capture buffer.
struct SharedSink(Arc<Mutex<String>>);
impl io::Write for SharedSink {
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

/// The process-wide capturing subscriber: the global default is
/// first-set-wins, so a single test asserts the whole span baseline.
/// Other tests' warning-path events may interleave — every assertion
/// here is a `contains`.
fn capture() -> &'static Arc<Mutex<String>> {
    static LOG: OnceLock<Arc<Mutex<String>>> = OnceLock::new();
    LOG.get_or_init(|| {
        let log: Arc<Mutex<String>> = Arc::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::NEW)
            .with_writer({
                let sink = log.clone();
                move || SharedSink(sink.clone())
            })
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("tracing-span test owns the global subscriber");
        log
    })
}

#[tokio::test]
async fn driver_spans_turn_round_attempt_and_tool_dispatch() {
    let log = capture();
    let c = ctx("t1");
    let runner = runner_with(
        RecordingGateway::scripted(vec![
            Ok(tooluse_output(
                "call echo",
                "echo",
                serde_json::json!({"a": 1}),
            )),
            Ok(endturn_output("done")),
        ]),
        vec![Arc::new(EchoTool)],
    );
    let out = runner.run(c, options_with_limits(5, 10), ctrl()).await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));

    let text = log.lock().unwrap().clone();
    assert!(text.contains("agent.turn"), "missing agent.turn: {text}");
    assert!(
        text.contains(r#"scope="turn""#),
        "missing scope field: {text}"
    );
    assert!(text.contains("agent.round"), "missing agent.round: {text}");
    assert!(
        text.contains("round_id=0"),
        "missing round_id field: {text}"
    );
    assert!(
        text.contains("agent.attempt"),
        "missing agent.attempt: {text}"
    );
    assert!(text.contains("model=fake"), "missing model field: {text}");
    assert!(text.contains("agent.tool"), "missing agent.tool: {text}");
    assert!(
        text.contains("tool_name=echo"),
        "missing tool_name field: {text}"
    );
    // Field discipline: no tool arguments or message bodies in spans.
    assert!(
        !text.contains("{\"a\":1}"),
        "arguments leaked into spans: {text}"
    );
    assert!(
        !text.contains("\"done\""),
        "message text leaked into spans: {text}"
    );
}
