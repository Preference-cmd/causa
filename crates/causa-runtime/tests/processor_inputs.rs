//! Tool processors read the bound tools' declared metadata, attach
//! ephemeral per-call state for execution, and read the host's run data. The
//! binding fixes the metadata; the data passes through as given, so a shared
//! handle in it is read live.
mod common;
mod tool_fixtures;

use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{
    ToolExecutorOptions, ToolProcessingError, TurnResult, TurnRunner, new_block_id,
};
use serde_json::json;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::time::Duration;
use tokio::sync::{Notify, mpsc, oneshot};

/// Host-defined tool classification carried via `ToolMetadata`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Effect {
    ReadOnly,
    Destructive,
}

/// Call-scoped attachment handed from a before-processor to tool execution.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ApprovedPlan(&'static str);

struct DropSentinel(Arc<AtomicUsize>);

impl Drop for DropSentinel {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A static tool that carries host-defined metadata.
struct MetadataTool {
    name: &'static str,
    effect: Effect,
}

#[async_trait]
impl Tool for MetadataTool {
    fn definition(&self) -> ToolDefinition {
        tool_fixtures::definition(self.name)
    }
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata::new().with(self.effect)
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

/// A dynamic source whose one tool declares metadata that can change. A call
/// flips the tool to destructive and bumps the version, as the listing
/// contract asks.
struct MetadataSource {
    effect: Mutex<Effect>,
    version: AtomicU64,
    describes: AtomicUsize,
    flip_on_call: bool,
}

impl MetadataSource {
    fn new(effect: Effect, flip_on_call: bool) -> Arc<Self> {
        Arc::new(Self {
            effect: Mutex::new(effect),
            version: AtomicU64::new(0),
            describes: AtomicUsize::new(0),
            flip_on_call,
        })
    }

    fn set_effect(&self, effect: Effect) {
        *self.effect.lock().unwrap() = effect;
        self.version.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl DynamicToolSource for MetadataSource {
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
        let effect = *self.effect.lock().unwrap();
        Ok(vec![
            ToolDescription::new(tool_fixtures::definition("remote_lookup")).with_meta(effect),
        ])
    }
    async fn invoke(
        &self,
        call: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        if self.flip_on_call {
            self.set_effect(Effect::Destructive);
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
    has_description: bool,
    effect: Option<Effect>,
    wrong_metadata_type: bool,
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
                has_description: ctx.tool(&tool).is_some(),
                effect: ctx.tool_metadata::<Effect>(&tool).copied(),
                wrong_metadata_type: ctx.tool_metadata::<u64>(&tool).is_some(),
                tool,
                tag: ctx.data::<Tag>().map(|tag| tag.0),
                other_type: ctx.data::<String>().is_some(),
            });
        }
        Ok(())
    }
}

#[tokio::test]
async fn tools_and_sources_that_declare_nothing_bind_with_empty_metadata() {
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
        assert!(entry.has_description, "{entry:?}");
        assert_eq!(entry.effect, None, "{entry:?}");
        assert!(!entry.wrong_metadata_type, "{entry:?}");
        assert_eq!(entry.tag, None);
    }
}

#[tokio::test]
async fn before_and_after_processors_see_static_and_dynamic_metadata() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let executor = tool_fixtures::executor(
        vec![Arc::new(MetadataTool {
            name: "read_file",
            effect: Effect::ReadOnly,
        })],
        Recorder::pair(&seen),
    );
    executor
        .register_dynamic(MetadataSource::new(Effect::Destructive, false))
        .unwrap();
    let bound = executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap();
    let mut batch = tool_fixtures::batch(&["read_file", "remote_lookup"]);
    bound.process(&mut batch).await.unwrap();
    let effect_in = |phase: &str, tool: &str| {
        seen.lock()
            .unwrap()
            .iter()
            .find(|entry| entry.phase == phase && entry.tool == tool)
            .and_then(|entry| entry.effect)
    };
    for phase in ["before", "after"] {
        assert_eq!(
            effect_in(phase, "read_file"),
            Some(Effect::ReadOnly),
            "{phase}"
        );
        assert_eq!(
            effect_in(phase, "remote_lookup"),
            Some(Effect::Destructive),
            "{phase}"
        );
    }
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .all(|entry| !entry.wrong_metadata_type && entry.has_description)
    );
}

#[tokio::test]
async fn a_binding_keeps_the_metadata_it_bound() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let source = MetadataSource::new(Effect::ReadOnly, false);
    let executor = tool_fixtures::executor(vec![], Recorder::pair(&seen));
    executor.register_dynamic(source.clone()).unwrap();

    let bound = executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap();
    source.set_effect(Effect::Destructive);
    let mut batch = tool_fixtures::batch(&["remote_lookup"]);
    bound.process(&mut batch).await.unwrap();
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .all(|entry| entry.effect == Some(Effect::ReadOnly))
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
            .all(|entry| entry.effect == Some(Effect::Destructive))
    );
    assert_eq!(source.describes.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_change_in_metadata_alone_records_no_tool_change() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let source = MetadataSource::new(Effect::ReadOnly, true);
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
            TurnId::new("metadata"),
            common::input("look it up"),
            common::options(),
            common::ctrl(),
        )
        .await;
    assert!(matches!(outcome.result, TurnResult::Completed { .. }));

    // The second round bound the new metadata ...
    let before: Vec<_> = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|entry| entry.phase == "before")
        .map(|entry| (entry.round, entry.effect))
        .collect();
    assert_eq!(
        before,
        vec![(0, Some(Effect::ReadOnly)), (1, Some(Effect::Destructive))]
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
async fn call_attachments_round_trip_through_static_and_dynamic_tools() {
    struct AttachmentEchoTool(&'static str);

    #[async_trait]
    impl Tool for AttachmentEchoTool {
        fn definition(&self) -> ToolDefinition {
            tool_fixtures::definition(self.0)
        }
        async fn execute(
            &self,
            ctx: &ToolCallContext,
            _control: &CallControl,
            _store: Option<&dyn ArtifactStore>,
        ) -> ToolResultPayload {
            tool_fixtures::result(
                ctx,
                ToolResultStatus::Succeeded,
                json!({
                    "plan": ctx.attachment::<ApprovedPlan>().map(|p| p.0),
                    "wrong_type": ctx.attachment::<u64>().is_some(),
                }),
            )
        }
    }

    struct AttachmentEchoSource;

    #[async_trait]
    impl DynamicToolSource for AttachmentEchoSource {
        fn id(&self) -> &str {
            "dynamic_attach"
        }
        async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
            Ok(vec![tool_fixtures::definition("dyn_attached")])
        }
        async fn invoke(
            &self,
            call: &ToolCallContext,
            _control: &CallControl,
            _store: Option<&dyn ArtifactStore>,
        ) -> Result<ToolResultPayload, ToolExecutionError> {
            Ok(tool_fixtures::result(
                call,
                ToolResultStatus::Succeeded,
                json!({
                    "plan": call.attachment::<ApprovedPlan>().map(|p| p.0),
                    "wrong_type": call.attachment::<u64>().is_some(),
                }),
            ))
        }
    }

    struct AttachPlans;

    #[async_trait]
    impl ToolBatchProcessor for AttachPlans {
        async fn process(
            &self,
            batch: &mut ToolBatch,
            _ctx: &ProcessorContext<'_>,
        ) -> Result<(), ProcessorError> {
            for entry in batch.calls_mut() {
                match entry.call().input.tool_name.as_str() {
                    "static_attached" => {
                        assert!(entry.attach(ApprovedPlan("static-plan")));
                    }
                    "dyn_attached" => {
                        assert!(entry.attach(ApprovedPlan("dynamic-plan")));
                    }
                    _ => {}
                }
            }
            Ok(())
        }
    }

    struct VerifyResolvedCleared(Arc<AtomicBool>);

    #[async_trait]
    impl ToolBatchProcessor for VerifyResolvedCleared {
        async fn process(
            &self,
            batch: &mut ToolBatch,
            _ctx: &ProcessorContext<'_>,
        ) -> Result<(), ProcessorError> {
            for entry in batch.results_mut() {
                assert!(entry.attachments_mut().is_none());
                assert!(entry.call().attachments().is_empty());
            }
            self.0.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    let after_ran = Arc::new(AtomicBool::new(false));
    let executor = tool_fixtures::executor(
        vec![
            Arc::new(AttachmentEchoTool("static_attached")),
            Arc::new(AttachmentEchoTool("unattached")),
        ],
        ToolExecutorOptions::default()
            .with_before(vec![Arc::new(AttachPlans)])
            .with_after(vec![Arc::new(VerifyResolvedCleared(after_ran.clone()))]),
    );
    executor
        .register_dynamic(Arc::new(AttachmentEchoSource))
        .unwrap();

    let mut batch = tool_fixtures::batch(&["static_attached", "dyn_attached", "unattached"]);
    executor
        .bind(tool_fixtures::invocation(), tool_fixtures::control())
        .await
        .unwrap()
        .process(&mut batch)
        .await
        .unwrap();

    assert!(after_ran.load(Ordering::SeqCst));
    let content_for = |name: &str| {
        batch
            .results()
            .iter()
            .find(|entry| entry.call().input.tool_name == name)
            .and_then(|entry| entry.result())
            .map(|(_, result)| result.output.content.clone())
            .unwrap()
    };
    assert_eq!(
        content_for("static_attached"),
        json!({ "plan": "static-plan", "wrong_type": false })
    );
    assert_eq!(
        content_for("dyn_attached"),
        json!({ "plan": "dynamic-plan", "wrong_type": false })
    );
    assert_eq!(
        content_for("unattached"),
        json!({ "plan": null, "wrong_type": false })
    );
}

#[tokio::test]
async fn attachments_drop_deterministically_across_rejection_completion_cancellation_and_early_drop()
 {
    // (a) Rejected in before-processor: dropped immediately upon `resolve_at`.
    {
        let drops = Arc::new(AtomicUsize::new(0));
        let dispatched = Arc::new(AtomicUsize::new(0));

        struct RejectAfterAttach {
            drops: Arc<AtomicUsize>,
            observed_drops_before_return: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl ToolBatchProcessor for RejectAfterAttach {
            async fn process(
                &self,
                batch: &mut ToolBatch,
                _ctx: &ProcessorContext<'_>,
            ) -> Result<(), ProcessorError> {
                assert!(batch.calls_mut()[0].attach(DropSentinel(self.drops.clone())));
                assert_eq!(self.drops.load(Ordering::SeqCst), 0);
                let id = batch.calls()[0].call().call_block_id;
                batch
                    .resolve_at(
                        batch.completed_len(),
                        new_block_id(),
                        ToolResultPayload::new(
                            id,
                            ToolResultStatus::Rejected,
                            ToolOutput::new(json!({"error": "rejected"})),
                        ),
                    )
                    .unwrap();
                self.observed_drops_before_return
                    .store(self.drops.load(Ordering::SeqCst), Ordering::SeqCst);
                Ok(())
            }
        }

        struct CountingTool(Arc<AtomicUsize>);

        #[async_trait]
        impl Tool for CountingTool {
            fn definition(&self) -> ToolDefinition {
                tool_fixtures::definition("work")
            }
            async fn execute(
                &self,
                call: &ToolCallContext,
                _control: &CallControl,
                _store: Option<&dyn ArtifactStore>,
            ) -> ToolResultPayload {
                self.0.fetch_add(1, Ordering::SeqCst);
                tool_fixtures::result(call, ToolResultStatus::Succeeded, json!(null))
            }
        }

        let observed_before_return = Arc::new(AtomicUsize::new(0));
        let executor = tool_fixtures::executor(
            vec![Arc::new(CountingTool(dispatched.clone()))],
            ToolExecutorOptions::default().with_before(vec![Arc::new(RejectAfterAttach {
                drops: drops.clone(),
                observed_drops_before_return: observed_before_return.clone(),
            })]),
        );
        let mut batch = tool_fixtures::batch(&["work"]);
        executor
            .bind(tool_fixtures::invocation(), tool_fixtures::control())
            .await
            .unwrap()
            .process(&mut batch)
            .await
            .unwrap();
        assert_eq!(observed_before_return.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(dispatched.load(Ordering::SeqCst), 0);
    }

    // (b) Normal completion in dispatch: dropped before after-processor runs.
    {
        let drops = Arc::new(AtomicUsize::new(0));
        let observed_in_after = Arc::new(AtomicUsize::new(0));

        struct AttachOnly(Arc<AtomicUsize>);
        #[async_trait]
        impl ToolBatchProcessor for AttachOnly {
            async fn process(
                &self,
                batch: &mut ToolBatch,
                _ctx: &ProcessorContext<'_>,
            ) -> Result<(), ProcessorError> {
                assert!(batch.calls_mut()[0].attach(DropSentinel(self.0.clone())));
                Ok(())
            }
        }

        struct CheckDroppedInAfter {
            drops: Arc<AtomicUsize>,
            observed: Arc<AtomicUsize>,
        }
        #[async_trait]
        impl ToolBatchProcessor for CheckDroppedInAfter {
            async fn process(
                &self,
                _batch: &mut ToolBatch,
                _ctx: &ProcessorContext<'_>,
            ) -> Result<(), ProcessorError> {
                self.observed
                    .store(self.drops.load(Ordering::SeqCst), Ordering::SeqCst);
                Ok(())
            }
        }

        let executor = tool_fixtures::executor(
            vec![tool_fixtures::NamedTool::new("work")],
            ToolExecutorOptions::default()
                .with_before(vec![Arc::new(AttachOnly(drops.clone()))])
                .with_after(vec![Arc::new(CheckDroppedInAfter {
                    drops: drops.clone(),
                    observed: observed_in_after.clone(),
                })]),
        );
        let mut batch = tool_fixtures::batch(&["work"]);
        executor
            .bind(tool_fixtures::invocation(), tool_fixtures::control())
            .await
            .unwrap()
            .process(&mut batch)
            .await
            .unwrap();
        assert_eq!(observed_in_after.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    // (c) Cancelled dispatch settling as `UnknownOutcome`: dropped on settlement.
    {
        let drops = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(Notify::new());

        struct AttachSentinel(Arc<AtomicUsize>);
        #[async_trait]
        impl ToolBatchProcessor for AttachSentinel {
            async fn process(
                &self,
                batch: &mut ToolBatch,
                _ctx: &ProcessorContext<'_>,
            ) -> Result<(), ProcessorError> {
                assert!(batch.calls_mut()[0].attach(DropSentinel(self.0.clone())));
                Ok(())
            }
        }

        struct HangingTool(Arc<Notify>);
        #[async_trait]
        impl Tool for HangingTool {
            fn definition(&self) -> ToolDefinition {
                tool_fixtures::definition("hang")
            }
            async fn execute(
                &self,
                ctx: &ToolCallContext,
                _control: &CallControl,
                _store: Option<&dyn ArtifactStore>,
            ) -> ToolResultPayload {
                assert!(ctx.attachment::<DropSentinel>().is_some());
                self.0.notify_one();
                std::future::pending::<()>().await;
                unreachable!()
            }
        }

        let executor = Arc::new(tool_fixtures::executor(
            vec![Arc::new(HangingTool(started.clone()))],
            ToolExecutorOptions::default()
                .with_before(vec![Arc::new(AttachSentinel(drops.clone()))]),
        ));
        let token = CancellationToken::new();
        let control = CallControl::new(token.clone(), None);
        let mut batch = tool_fixtures::batch(&["hang"]);
        let handle = tokio::spawn({
            let executor = executor.clone();
            let control = control.clone();
            async move {
                let err = executor
                    .bind(tool_fixtures::invocation(), control)
                    .await
                    .unwrap()
                    .process(&mut batch)
                    .await
                    .unwrap_err();
                (err, batch)
            }
        });
        started.notified().await;
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        token.cancel();
        let (err, batch) = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("cancelled dispatch finishes")
            .unwrap();
        assert!(matches!(err, ToolProcessingError::Control(_)));
        assert_eq!(
            tool_fixtures::statuses(&batch),
            vec![ToolResultStatus::UnknownOutcome]
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    // (c2) Timed-out dispatch settling as `UnknownOutcome`: dropped on settlement.
    {
        let drops = Arc::new(AtomicUsize::new(0));

        struct AttachSentinel(Arc<AtomicUsize>);
        #[async_trait]
        impl ToolBatchProcessor for AttachSentinel {
            async fn process(
                &self,
                batch: &mut ToolBatch,
                _ctx: &ProcessorContext<'_>,
            ) -> Result<(), ProcessorError> {
                assert!(batch.calls_mut()[0].attach(DropSentinel(self.0.clone())));
                Ok(())
            }
        }

        struct SleepingTool;
        #[async_trait]
        impl Tool for SleepingTool {
            fn definition(&self) -> ToolDefinition {
                tool_fixtures::definition("sleep")
            }
            async fn execute(
                &self,
                ctx: &ToolCallContext,
                _control: &CallControl,
                _store: Option<&dyn ArtifactStore>,
            ) -> ToolResultPayload {
                assert!(ctx.attachment::<DropSentinel>().is_some());
                std::future::pending::<()>().await;
                unreachable!()
            }
        }

        let executor = tool_fixtures::executor(
            vec![Arc::new(SleepingTool)],
            ToolExecutorOptions::default()
                .with_call_timeout(Duration::from_millis(20))
                .with_before(vec![Arc::new(AttachSentinel(drops.clone()))]),
        );
        let mut batch = tool_fixtures::batch(&["sleep"]);
        executor
            .bind(tool_fixtures::invocation(), tool_fixtures::control())
            .await
            .unwrap()
            .process(&mut batch)
            .await
            .unwrap();
        assert_eq!(
            tool_fixtures::statuses(&batch),
            vec![ToolResultStatus::UnknownOutcome]
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    // (d) `ToolBatch` dropped before completion: dropped immediately.
    {
        let drops = Arc::new(AtomicUsize::new(0));

        struct AttachThenFail(Arc<AtomicUsize>);
        #[async_trait]
        impl ToolBatchProcessor for AttachThenFail {
            async fn process(
                &self,
                batch: &mut ToolBatch,
                _ctx: &ProcessorContext<'_>,
            ) -> Result<(), ProcessorError> {
                assert!(batch.calls_mut()[0].attach(DropSentinel(self.0.clone())));
                Err(ProcessorError::Failed("abort".into()))
            }
        }

        let executor = tool_fixtures::executor(
            vec![tool_fixtures::NamedTool::new("work")],
            ToolExecutorOptions::default()
                .with_before(vec![Arc::new(AttachThenFail(drops.clone()))]),
        );
        let mut batch = tool_fixtures::batch(&["work"]);
        let err = executor
            .bind(tool_fixtures::invocation(), tool_fixtures::control())
            .await
            .unwrap()
            .process(&mut batch)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolProcessingError::Processor { .. }));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(batch);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
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
