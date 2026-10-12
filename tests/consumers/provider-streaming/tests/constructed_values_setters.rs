//! Consumer test proving all six types can be built through the facade with setters only.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use causa::kernel::{
    ArtifactHint, ArtifactRef, ArtifactStore, BlockMeta, CacheDirective, CallControl, Context,
    ContextFrame, ContextPreparer, GenerationOptions, MediaRef, ModelRef, PrepareError, RoundInfo,
    StoreError, TextPayload, ToolBatchProcessor, ToolCallDraft, ToolOutput, ToolResultPayload,
    ToolResultStatus, WireMeta,
};
use causa::runtime::{
    new_block_id, RunObserver, ToolExecutorOptions, TurnLimits, TurnRunOptions,
};

struct DummyArtifactStore;
impl ArtifactStore for DummyArtifactStore {
    fn persist<'life0, 'life1, 'async_trait>(
        &'life0 self,
        _data: &'life1 [u8],
        _hint: ArtifactHint,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ArtifactRef, StoreError>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async { unimplemented!() })
    }

    fn read<'life0, 'life1, 'async_trait>(
        &'life0 self,
        _id: &'life1 str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, StoreError>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async { unimplemented!() })
    }
}

struct DummyContextPreparer;
impl ContextPreparer for DummyContextPreparer {
    fn prepare<'life0, 'life1, 'life2, 'life3, 'life4, 'async_trait>(
        &'life0 self,
        _context: &'life1 mut Context,
        _round: &'life2 RoundInfo<'life3>,
        _control: &'life4 CallControl,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ContextFrame, PrepareError>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        'life2: 'async_trait,
        'life3: 'async_trait,
        'life4: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async { unimplemented!() })
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
    let before: Vec<Arc<dyn ToolBatchProcessor>> = vec![];
    let after: Vec<Arc<dyn ToolBatchProcessor>> = vec![];
    let store: Arc<dyn ArtifactStore> = Arc::new(DummyArtifactStore);
    let exec_opts = ToolExecutorOptions::default()
        .with_before(before)
        .with_after(after)
        .with_call_timeout(Duration::from_secs(30))
        .with_artifact_store(store);

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
