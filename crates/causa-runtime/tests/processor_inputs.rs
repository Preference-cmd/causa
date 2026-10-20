//! Tool processors read the bound tools' declared hints and the host's run
//! data from their context. The binding fixes the hints; the data passes
//! through as given, so a shared handle in it is read live.
mod common;
mod tool_fixtures;

use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{ToolExecutorOptions, TurnResult, TurnRunner, new_block_id};
use serde_json::json;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use tokio::sync::{mpsc, oneshot};

/// A static tool that declares hints.
struct HintedTool {
    name: &'static str,
    hints: ToolHints,
}

#[async_trait]
impl Tool for HintedTool {
    fn definition(&self) -> ToolDefinition {
        tool_fixtures::definition(self.name)
    }
    fn hints(&self) -> ToolHints {
        self.hints.clone()
    }
    async fn execute(
        &self,
        call: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        tool_fixtures::result(call, ToolResultStatus::Succeeded, json!("static"))
    }
}

/// A dynamic source whose one tool declares hints that can change. A call
/// flips the tool to destructive, as a server that re-annotates would, and
/// bumps the version, as the listing contract asks.
struct HintedSource {
    hints: Mutex<ToolHints>,
    version: AtomicU64,
    describes: AtomicUsize,
    flip_on_call: bool,
}

impl HintedSource {
    fn new(hints: ToolHints, flip_on_call: bool) -> Arc<Self> {
        Arc::new(Self {
            hints: Mutex::new(hints),
            version: AtomicU64::new(0),
            describes: AtomicUsize::new(0),
            flip_on_call,
        })
    }

    fn set_hints(&self, hints: ToolHints) {
        *self.hints.lock().unwrap() = hints;
        self.version.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl DynamicToolSource for HintedSource {
    fn id(&self) -> &str {
        "remote"
    }
    fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        Ok(vec![tool_fixtures::definition("remote_lookup")])
    }
    async fn describe(&self) -> Result<Vec<ToolDescription>, SourceError> {
        self.describes.fetch_add(1, Ordering::SeqCst);
        Ok(vec![
            ToolDescription::new(tool_fixtures::definition("remote_lookup"))
                .with_hints(self.hints.lock().unwrap().clone()),
        ])
    }
    async fn invoke(
        &self,
        call: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        if self.flip_on_call {
            self.set_hints(
                ToolHints::default()
                    .with_read_only(false)
                    .with_destructive(true),
            );
        }
        Ok(tool_fixtures::result(
            call,
            ToolResultStatus::Succeeded,
            json!("remote"),
        ))
    }
}

#[derive(Debug, PartialEq)]
struct Tag(&'static str);

/// What one processor saw for one call.
#[derive(Debug, Clone, PartialEq)]
struct Seen {
    phase: &'static str,
    round: u32,
    tool: String,
    hints: Option<ToolHints>,
    tag: Option<&'static str>,
    other_type: bool,
}

struct Recorder {
    phase: &'static str,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Recorder {
    fn pair(seen: &Arc<Mutex<Vec<Seen>>>) -> ToolExecutorOptions {
        ToolExecutorOptions::default()
            .with_before(vec![Arc::new(Recorder {
                phase: "before",
                seen: seen.clone(),
            })])
            .with_after(vec![Arc::new(Recorder {
                phase: "after",
                seen: seen.clone(),
            })])
    }
}

#[async_trait]
impl ToolBatchProcessor for Recorder {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let mut seen = self.seen.lock().unwrap();
        for entry in batch.results().iter().chain(batch.calls()) {
            let tool = entry.call().input.tool_name.clone();
            seen.push(Seen {
                phase: self.phase,
                round: ctx.round_id.0,
                hints: ctx.hints(&tool).cloned(),
                tool,
                tag: ctx.data::<Tag>().map(|tag| tag.0),
                other_type: ctx.data::<String>().is_some(),
            });
        }
        Ok(())
    }
}

fn read_only() -> ToolHints {
    ToolHints::default()
        .with_read_only(true)
        .with_open_world(false)
}

fn destructive() -> ToolHints {
    ToolHints::default()
        .with_read_only(false)
        .with_destructive(true)
}

#[tokio::test]
async fn tools_and_sources_that_declare_nothing_bind_with_default_hints() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let executor = tool_fixtures::executor(
        vec![tool_fixtures::NamedTool::new("local")],
        Recorder::pair(&seen),
    );
    executor
        .register_dynamic(tool_fixtures::Source::new("src", "dynamic", &["remote"]))
        .unwrap();
    let mut batch = tool_fixtures::batch(&["local", "remote"]);
    executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 4);
    for entry in seen.iter() {
        assert_eq!(entry.hints, Some(ToolHints::default()), "{entry:?}");
        assert_eq!(entry.tag, None);
    }
}

#[tokio::test]
async fn before_and_after_processors_see_static_and_dynamic_hints() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let executor = tool_fixtures::executor(
        vec![Arc::new(HintedTool {
            name: "read_file",
            hints: read_only(),
        })],
        Recorder::pair(&seen),
    );
    executor
        .register_dynamic(HintedSource::new(destructive(), false))
        .unwrap();
    let bound = executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap();
    let mut batch = tool_fixtures::batch(&["read_file", "remote_lookup"]);
    bound.process(&mut batch).await.unwrap();
    let hints_in = |phase: &str, tool: &str| {
        seen.lock()
            .unwrap()
            .iter()
            .find(|entry| entry.phase == phase && entry.tool == tool)
            .and_then(|entry| entry.hints.clone())
    };
    for phase in ["before", "after"] {
        assert_eq!(hints_in(phase, "read_file"), Some(read_only()), "{phase}");
        assert_eq!(
            hints_in(phase, "remote_lookup"),
            Some(destructive()),
            "{phase}"
        );
    }
}

#[tokio::test]
async fn a_binding_keeps_the_hints_it_bound() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let source = HintedSource::new(read_only(), false);
    let executor = tool_fixtures::executor(vec![], Recorder::pair(&seen));
    executor.register_dynamic(source.clone()).unwrap();

    let bound = executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap();
    source.set_hints(destructive());
    let mut batch = tool_fixtures::batch(&["remote_lookup"]);
    bound.process(&mut batch).await.unwrap();
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .all(|entry| entry.hints == Some(read_only()))
    );

    seen.lock().unwrap().clear();
    let mut batch = tool_fixtures::batch(&["remote_lookup"]);
    executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .all(|entry| entry.hints == Some(destructive()))
    );
    assert_eq!(source.describes.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_change_in_hints_alone_records_no_tool_change() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let source = HintedSource::new(read_only(), true);
    let executor = tool_fixtures::executor(vec![], Recorder::pair(&seen));
    executor.register_dynamic(source.clone()).unwrap();
    let gateway = common::RecordingGateway::scripted(vec![
        Ok(common::tooluse_output("one", "remote_lookup", json!({}))),
        Ok(common::tooluse_output("two", "remote_lookup", json!({}))),
        Ok(common::endturn_output("done")),
    ]);
    let runner = TurnRunner::new(gateway.clone(), Arc::new(executor));
    let outcome = runner
        .run(
            TurnId::new("hints"),
            common::input("look it up"),
            common::options(),
            common::ctrl(),
        )
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));

    // The second round bound the new hints ...
    let before: Vec<_> = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|entry| entry.phase == "before")
        .map(|entry| (entry.round, entry.hints.clone()))
        .collect();
    assert_eq!(
        before,
        vec![(0, Some(read_only())), (1, Some(destructive()))]
    );
    assert!(source.describes.load(Ordering::SeqCst) >= 2);

    // ... and the context and every request still hold one tool declaration.
    let tool_changes = |blocks: &[ContextBlock]| {
        blocks
            .iter()
            .filter(|block| matches!(block.content(), BlockContent::ToolChange(_)))
            .count()
    };
    assert_eq!(tool_changes(outcome.context.blocks()), 1);
    let requests = gateway.recorded.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let declared = current_tools(&requests[0].frame.blocks);
    for request in requests.iter() {
        assert_eq!(tool_changes(&request.frame.blocks), 1);
        assert_eq!(current_tools(&request.frame.blocks), declared);
    }
    // Each request extends the one before it, so the rendered prefix of an
    // earlier round is unchanged in the next.
    for pair in requests.windows(2) {
        assert!(pair[1].frame.blocks.starts_with(&pair[0].frame.blocks));
    }
}

#[tokio::test]
async fn run_data_reaches_every_processor_of_every_round() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let executor = tool_fixtures::executor(
        vec![tool_fixtures::NamedTool::new("local")],
        Recorder::pair(&seen),
    );
    let gateway = common::RecordingGateway::scripted(vec![
        Ok(common::tooluse_output("one", "local", json!({}))),
        Ok(common::tooluse_output("two", "local", json!({}))),
        Ok(common::endturn_output("done")),
        Ok(common::tooluse_output("three", "local", json!({}))),
        Ok(common::endturn_output("done")),
    ]);
    let runner = TurnRunner::new(gateway, Arc::new(executor));
    let outcome = runner
        .run(
            TurnId::new("with-data"),
            common::input("go"),
            common::options().with_processor_data(ProcessorData::new(Tag("run-a"))),
            common::ctrl(),
        )
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 4);
        for entry in seen.iter() {
            assert_eq!(entry.tag, Some("run-a"), "{entry:?}");
            assert!(!entry.other_type);
        }
        assert_eq!(
            seen.iter().map(|entry| entry.round).collect::<Vec<_>>(),
            vec![0, 0, 1, 1]
        );
    }

    // The same executor serves a run without data.
    seen.lock().unwrap().clear();
    let outcome = runner
        .run(
            TurnId::new("without-data"),
            outcome.context,
            common::options(),
            common::ctrl(),
        )
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|entry| entry.tag.is_none()));
}

#[tokio::test]
async fn a_binding_carries_data_without_the_runner() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let executor = tool_fixtures::executor(
        vec![tool_fixtures::NamedTool::new("local")],
        Recorder::pair(&seen),
    );
    let mut batch = tool_fixtures::batch(&["local"]);
    executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap()
        .with_processor_data(ProcessorData::new(Tag("standalone")))
        .process(&mut batch)
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|entry| entry.tag == Some("standalone")));
}

/// The host's mode: `true` runs every call, `false` asks first.
type AllowAll = Arc<AtomicBool>;

/// Decides calls one by one, reading the mode each time. Asking sends the
/// call's name to the host and waits for the answer.
struct Ask(mpsc::Sender<(String, oneshot::Sender<bool>)>);

#[async_trait]
impl ToolBatchProcessor for Ask {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        let mut offset = 0;
        while offset < batch.calls().len() {
            let allow_all = ctx
                .data::<AllowAll>()
                .ok_or_else(|| ProcessorError::Failed("no mode".into()))?;
            if allow_all.load(Ordering::SeqCst) {
                offset += 1;
                continue;
            }
            let call = batch.calls()[offset].call();
            let (name, id) = (call.input.tool_name.clone(), call.call_block_id);
            let (reply, answer) = oneshot::channel();
            self.0
                .send((name, reply))
                .await
                .map_err(|error| ProcessorError::Failed(error.to_string()))?;
            if answer
                .await
                .map_err(|error| ProcessorError::Failed(error.to_string()))?
            {
                offset += 1;
            } else {
                batch
                    .resolve_at(
                        batch.completed_len() + offset,
                        new_block_id(),
                        ToolResultPayload::new(
                            id,
                            ToolResultStatus::Rejected,
                            ToolOutput::new(json!({"error": "declined"})),
                        ),
                    )
                    .map_err(|error| ProcessorError::Failed(error.to_string()))?;
            }
        }
        Ok(())
    }
}

#[tokio::test]
async fn a_handle_in_the_data_is_read_live_within_one_batch() {
    let (asks, mut asked) = mpsc::channel(4);
    let tool = tool_fixtures::NamedTool::new("write");
    let executor = tool_fixtures::executor(
        vec![tool.clone()],
        ToolExecutorOptions::default().with_before(vec![Arc::new(Ask(asks))]),
    );
    let allow_all: AllowAll = Arc::new(AtomicBool::new(false));
    // The host declines the first call and, from the same prompt, switches
    // the run to allow everything.
    let host = {
        let allow_all = allow_all.clone();
        tokio::spawn(async move {
            let mut asked_about = Vec::new();
            while let Some((name, reply)) = asked.recv().await {
                asked_about.push(name);
                allow_all.store(true, Ordering::SeqCst);
                reply.send(false).unwrap();
            }
            asked_about
        })
    };
    let mut batch = tool_fixtures::batch(&["write", "write"]);
    executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap()
        .with_processor_data(ProcessorData::new(allow_all.clone()))
        .process(&mut batch)
        .await
        .unwrap();
    drop(executor);
    assert_eq!(host.await.unwrap(), vec!["write".to_owned()]);
    assert_eq!(
        tool_fixtures::statuses(&batch),
        vec![ToolResultStatus::Rejected, ToolResultStatus::Succeeded]
    );
    assert_eq!(tool.count.load(Ordering::SeqCst), 1);
}
