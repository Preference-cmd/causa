//! Consumer test proving all six types can be built through the facade with setters only.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use causa::kernel::{
    ArtifactHint, ArtifactRef, ArtifactStore, BlockMeta, CacheDirective, CallControl, Context,
    ContextFrame, ContextPreparer, GenerationOptions, MediaRef, ModelRef, PrepareError,
    ProcessorContext, ProcessorError, RoundInfo, StoreError, TextPayload, ToolBatch,
    ToolBatchProcessor, ToolCallDraft, ToolOutput, ToolResultPayload, ToolResultStatus, WireMeta,
};
use causa::runtime::{RunObserver, ToolExecutorOptions, TurnLimits, TurnRunOptions, new_block_id};

struct DummyArtifactStore;

#[async_trait]
impl ArtifactStore for DummyArtifactStore {
    async fn persist(&self, _data: &[u8], _hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        unimplemented!()
    }

    async fn read(
        &self,
        _id: &str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError> {
        unimplemented!()
    }
}

struct DummyContextPreparer;

#[async_trait]
impl ContextPreparer for DummyContextPreparer {
    async fn prepare(
        &self,
        _context: &mut Context,
        _round: &RoundInfo<'_>,
        _control: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        unimplemented!()
    }
}

struct DummyProcessor;

#[async_trait]
impl ToolBatchProcessor for DummyProcessor {
    async fn process(
        &self,
        _batch: &mut ToolBatch,
        _ctx: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        unimplemented!()
    }
}

#[test]
fn six_most_built_types_construct_through_facade_with_setters() {
    // 1. GenerationOptions
    let mut extra = serde_json::Map::new();
    extra.insert("custom_key".to_string(), serde_json::json!("custom_val"));
    let schema = serde_json::json!({"type": "object"});
    let gen_opts = GenerationOptions::default()
        .with_temperature(0.7)
        .with_max_tokens(4096)
        .with_reasoning_effort("high")
        .with_extra_body(extra.clone())
        .with_output_schema(schema.clone());

    assert_eq!(gen_opts.temperature, Some(0.7));
    assert_eq!(gen_opts.max_tokens, Some(4096));
    assert_eq!(gen_opts.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(gen_opts.extra_body, extra);
    assert_eq!(gen_opts.output_schema, Some(schema));

    // 2. BlockMeta
    let wire = WireMeta::new("format", serde_json::json!({"residue": 1}));
    let meta = BlockMeta::default()
        .with_provider_call_id("call_p1")
        .with_source("assistant")
        .with_wire(wire.clone());

    assert_eq!(meta.provider_call_id.as_deref(), Some("call_p1"));
    assert_eq!(meta.source.as_deref(), Some("assistant"));
    assert_eq!(meta.wire, Some(wire));

    // 3. ToolResultPayload
    let call_id = new_block_id();
    let media = vec![MediaRef::new("image/png", "asset://img1")];
    let notes = vec![TextPayload::new("executed successfully")];
    let res = ToolResultPayload::new(
        call_id,
        ToolResultStatus::Succeeded,
        ToolOutput::new(serde_json::json!({"status": "ok"})),
    )
    .with_media(media.clone())
    .with_notes(notes.clone());

    assert_eq!(res.call_block_id, call_id);
    assert_eq!(res.status, ToolResultStatus::Succeeded);
    assert_eq!(res.media, media);
    assert_eq!(res.notes, notes);

    // 4. ToolCallDraft
    let draft = ToolCallDraft::new("search_docs", serde_json::json!({"q": "test"}))
        .with_provider_call_id("call_draft_99");

    assert_eq!(draft.tool_name, "search_docs");
    assert_eq!(draft.arguments, serde_json::json!({"q": "test"}));
    assert_eq!(draft.provider_call_id.as_deref(), Some("call_draft_99"));

    // 5. ToolExecutorOptions
    let first: Arc<dyn ToolBatchProcessor> = Arc::new(DummyProcessor);
    let second: Arc<dyn ToolBatchProcessor> = Arc::new(DummyProcessor);
    let store: Arc<dyn ArtifactStore> = Arc::new(DummyArtifactStore);
    let exec_opts = ToolExecutorOptions::default()
        .with_before(vec![first.clone()])
        .with_after(vec![second.clone(), first.clone()])
        .with_call_timeout(Duration::from_secs(30))
        .with_artifact_store(store);

    assert_eq!(exec_opts.before.len(), 1);
    assert!(Arc::ptr_eq(&exec_opts.before[0], &first));
    assert_eq!(exec_opts.after.len(), 2);
    assert!(Arc::ptr_eq(&exec_opts.after[0], &second));
    assert_eq!(exec_opts.call_timeout, Some(Duration::from_secs(30)));
    assert!(exec_opts.artifact_store.is_some());

    // 6. TurnRunOptions
    let preparer: Arc<dyn ContextPreparer> = Arc::new(DummyContextPreparer);
    let observer: Arc<RunObserver> = Arc::new(|_| {});
    let run_opts = TurnRunOptions::new(ModelRef::new("test-model"))
        .with_generation(gen_opts.clone())
        .with_cache(CacheDirective::StablePrefix)
        .with_limits(TurnLimits::new(5, 32))
        .with_preparer(preparer)
        .with_observer(observer);

    assert_eq!(run_opts.model.0, "test-model");
    assert_eq!(run_opts.generation, gen_opts);
    assert_eq!(run_opts.cache, CacheDirective::StablePrefix);
    assert_eq!(run_opts.limits.max_model_rounds, 5);
    assert_eq!(run_opts.limits.max_tool_calls, 32);
    assert!(run_opts.preparer.is_some());
    assert!(run_opts.observer.is_some());
}
