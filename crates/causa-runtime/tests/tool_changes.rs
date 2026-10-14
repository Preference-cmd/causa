//! The runner records the tools it offers as `ToolChange` facts in the
//! context, and refuses a prepared frame whose tool facts disagree.
mod common;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::*;
use common::*;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(name, name, json!({"type": "object"}))
}

/// A catalog whose `grow` tool offers one more tool once it has run, the way
/// an MCP server announces `tools/list_changed`.
struct GrowingSource {
    definitions: Mutex<Vec<ToolDefinition>>,
    version: AtomicU64,
}
#[async_trait]
impl DynamicToolSource for GrowingSource {
    fn id(&self) -> &str {
        "growing"
    }
    fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        Ok(self.definitions.lock().unwrap().clone())
    }
    async fn invoke(
        &self,
        call: &ToolCallContext,
        _: &CallControl,
        _: Option<&dyn ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        self.definitions.lock().unwrap().push(tool("extra"));
        self.version.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResultPayload::new(
            call.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!("grown")),
        ))
    }
}

fn growing_runner(gateway: Arc<RecordingGateway>) -> TurnRunner {
    let executor = ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap();
    executor
        .register_dynamic(Arc::new(GrowingSource {
            definitions: Mutex::new(vec![tool("grow")]),
            version: AtomicU64::new(0),
        }))
        .unwrap();
    TurnRunner::new(gateway, Arc::new(executor))
}

fn changes(blocks: &[ContextBlock]) -> Vec<ToolChange> {
    blocks
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolChange(change) => Some(change.clone()),
            _ => None,
        })
        .collect()
}

fn names(tools: &[ToolDefinition]) -> Vec<&str> {
    tools.iter().map(|tool| tool.name.as_str()).collect()
}

fn recording_options(committed: Arc<Mutex<Vec<ToolChange>>>) -> TurnRunOptions {
    let mut opts = options();
    opts.observer = Some(Arc::new(move |event| {
        if let RunEvent::BlocksCommitted { blocks, .. } = event {
            committed.lock().unwrap().extend(changes(blocks));
        }
    }));
    opts
}

#[tokio::test]
async fn the_first_round_declares_and_a_changed_catalog_appends_one_change() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("", "grow", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let committed = Arc::new(Mutex::new(vec![]));
    let out = growing_runner(gateway.clone())
        .run(
            TurnId::new("growing"),
            input("hi"),
            recording_options(committed.clone()),
            ctrl(),
        )
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));

    // Input, declaration, call, result, change, answer.
    let blocks = out.context.blocks();
    assert_eq!(blocks.len(), 6);
    assert!(matches!(blocks[1].content(), BlockContent::ToolChange(_)));
    assert!(matches!(blocks[4].content(), BlockContent::ToolChange(_)));
    let recorded = changes(blocks);
    assert_eq!(names(&recorded[0].added), ["grow"]);
    assert_eq!(names(&recorded[1].added), ["extra"]);
    assert!(recorded[1].removed.is_empty());
    assert_eq!(*committed.lock().unwrap(), recorded);

    let requests = gateway.recorded.lock().unwrap();
    assert_eq!(names(&current_tools(&requests[0].frame.blocks)), ["grow"]);
    assert_eq!(
        names(&current_tools(&requests[1].frame.blocks)),
        ["grow", "extra"]
    );
}

#[tokio::test]
async fn unchanged_tools_append_nothing_across_rounds_and_runs() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("", "echo", json!({}))),
        Ok(endturn_output("first")),
        Ok(endturn_output("second")),
    ]);
    let runner = runner_with(gateway, vec![Arc::new(EchoTool)]);
    let committed = Arc::new(Mutex::new(vec![]));
    let first = runner
        .run(
            TurnId::new("first"),
            input("hi"),
            recording_options(committed.clone()),
            ctrl(),
        )
        .await;
    assert!(matches!(first.result, TurnResult::Completed { .. }));
    assert_eq!(changes(first.context.blocks()).len(), 1);

    let mut context = first.context;
    context
        .edit()
        .append([text_block("again")])
        .commit()
        .unwrap();
    let second = runner
        .run(
            TurnId::new("second"),
            context,
            recording_options(committed.clone()),
            ctrl(),
        )
        .await;
    assert!(matches!(second.result, TurnResult::Completed { .. }));
    assert_eq!(changes(second.context.blocks()).len(), 1);
    assert_eq!(committed.lock().unwrap().len(), 1);
}

/// Drops every tool fact from the frame it hands to the model.
struct DropToolFacts;
#[async_trait]
impl ContextPreparer for DropToolFacts {
    async fn prepare(
        &self,
        context: &mut Context,
        _: &RoundInfo<'_>,
        _: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        let mut frame = context.frame();
        frame
            .blocks
            .retain(|block| !matches!(block.content(), BlockContent::ToolChange(_)));
        Ok(frame)
    }
}

#[tokio::test]
async fn a_frame_without_the_bound_tools_stops_the_turn() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("unreached"))]);
    let mut opts = options();
    opts.preparer = Some(Arc::new(DropToolFacts));
    let out = runner_with(gateway.clone(), vec![Arc::new(EchoTool)])
        .run(TurnId::new("dropped"), input("hi"), opts, ctrl())
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::PrepareFailed { .. }
        }
    ));
    assert!(gateway.recorded.lock().unwrap().is_empty());
    // The declaration stays: it records what the runner bound.
    assert_eq!(changes(out.context.blocks()).len(), 1);
}
