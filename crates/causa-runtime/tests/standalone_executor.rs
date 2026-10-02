//! Independent binding consumers: concrete objects, identity, and host storage.
mod tool_fixtures;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{
    ToolBridge, ToolCatalogError, ToolExecutor, ToolExecutorOptions, ToolRegistryError,
};
use serde_json::json;
use std::sync::{Arc, Mutex, atomic::Ordering};
use tool_fixtures::*;

#[tokio::test]
async fn standalone_batch_dispatch_preserves_identity_and_duplicate_calls() {
    let tool = NamedTool::new("echo");
    let executor = executor(vec![tool.clone()], ToolExecutorOptions::default());
    let binding = executor.bind(invocation(), control()).await.unwrap();
    assert_eq!(binding.surface().definitions, vec![definition("echo")]);
    let mut batch = batch(&["echo", "echo"]);
    let ids = batch.declaration_ids();
    binding.process(&mut batch).await.unwrap();
    assert_eq!(tool.count.load(Ordering::SeqCst), 2);
    assert_eq!(statuses(&batch), vec![ToolResultStatus::Succeeded; 2]);
    for id in ids {
        assert!(
            batch
                .results()
                .iter()
                .any(|entry| entry.result().unwrap().1.call_block_id == id)
        );
    }
    let mut next = tool_fixtures::batch(&["echo", "echo"]);
    executor
        .bind(invocation(), control())
        .await
        .unwrap()
        .process(&mut next)
        .await
        .unwrap();
    assert_eq!(
        tool.count.load(Ordering::SeqCst),
        4,
        "no hidden cross-binding deduplication"
    );
}

#[tokio::test]
async fn catalog_order_is_static_then_registration_then_list_order() {
    let executor = executor(
        vec![NamedTool::new("z"), NamedTool::new("a")],
        ToolExecutorOptions::default(),
    );
    executor
        .register_dynamic(Source::new("two", "two", &["d", "c"]))
        .unwrap();
    executor
        .register_dynamic(Source::new("one", "one", &["b"]))
        .unwrap();
    assert_eq!(executor.dynamic_ids(), ["two", "one"]);
    let binding = executor.bind(invocation(), control()).await.unwrap();
    assert_eq!(
        binding
            .surface()
            .definitions
            .iter()
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>(),
        ["z", "a", "d", "c", "b"]
    );
}

#[tokio::test]
async fn old_binding_keeps_actual_source_after_unregister_and_replacement() {
    let old = Source::new("source", "old object", &["work"]);
    let new = Source::new("source", "new object", &["work"]);
    let executor = executor(vec![], ToolExecutorOptions::default());
    executor.register_dynamic(old.clone()).unwrap();
    let old_binding = executor.bind(invocation(), control()).await.unwrap();
    executor.unregister_dynamic("source").unwrap();
    executor.register_dynamic(new.clone()).unwrap();
    let new_binding = executor.bind(invocation(), control()).await.unwrap();
    let mut old_batch = batch(&["work"]);
    let mut new_batch = batch(&["work"]);
    old_binding.process(&mut old_batch).await.unwrap();
    new_binding.process(&mut new_batch).await.unwrap();
    assert_eq!(
        old_batch.results()[0].result().unwrap().1.output.content["object"],
        "old object"
    );
    assert_eq!(
        new_batch.results()[0].result().unwrap().1.output.content["object"],
        "new object"
    );
    assert_eq!(old.invocations.load(Ordering::SeqCst), 1);
    assert_eq!(new.invocations.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn static_name_drift_fails_new_binding_but_does_not_change_old_target() {
    let tool = NamedTool::new("old");
    let executor = executor(vec![tool.clone()], ToolExecutorOptions::default());
    let bound = executor.bind(invocation(), control()).await.unwrap();
    *tool.name.lock().unwrap() = "new".into();
    assert!(
        matches!(executor.bind(invocation(), control()).await, Err(ToolCatalogError::StaticNameChanged { registered, actual }) if registered == "old" && actual == "new")
    );
    let mut batch = batch(&["old"]);
    bound.process(&mut batch).await.unwrap();
    assert_eq!(tool.count.load(Ordering::SeqCst), 1);
}

#[test]
fn registry_rejects_duplicate_static_names_and_source_ids() {
    let tool = NamedTool::new("same");
    assert!(
        matches!(ToolExecutor::new(vec![tool.clone(), tool], ToolExecutorOptions::default()), Err(ToolRegistryError::DuplicateTool { name }) if name == "same")
    );
    let executor = executor(vec![], ToolExecutorOptions::default());
    let source = Source::new("same", "same", &[]);
    executor.register_dynamic(source.clone()).unwrap();
    assert_eq!(source.lists.load(Ordering::SeqCst), 0);
    assert!(
        matches!(executor.register_dynamic(source), Err(ToolRegistryError::DuplicateSource { source_id }) if source_id == "same")
    );
    assert!(
        matches!(executor.unregister_dynamic("absent"), Err(ToolRegistryError::UnknownSource { source_id }) if source_id == "absent")
    );
}

#[tokio::test]
async fn bind_rejects_same_source_cross_source_and_static_dynamic_duplicates() {
    for (static_names, sources) in [
        (vec![], vec![("a", vec!["same", "same"])]),
        (vec![], vec![("a", vec!["same"]), ("b", vec!["same"])]),
        (vec!["same"], vec![("a", vec!["same"])]),
    ] {
        let executor = executor(
            static_names
                .iter()
                .map(|name| NamedTool::new(name) as Arc<dyn Tool>)
                .collect(),
            ToolExecutorOptions::default(),
        );
        for (id, names) in sources {
            executor
                .register_dynamic(Source::new(id, id, &names))
                .unwrap();
        }
        assert!(
            matches!(executor.bind(invocation(), control()).await, Err(ToolCatalogError::DuplicateTool { name }) if name == "same")
        );
    }
}

#[derive(Default)]
struct Store(Mutex<Vec<(Vec<u8>, BlockId)>>);
#[async_trait]
impl ArtifactStore for Store {
    async fn persist(&self, data: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        self.0
            .lock()
            .unwrap()
            .push((data.to_vec(), hint.call_block_id));
        Ok(ArtifactRef {
            id: "image".into(),
            size_bytes: data.len(),
            kind: hint.kind,
            persisted: true,
        })
    }
    async fn read(
        &self,
        _id: &str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError> {
        unreachable!()
    }
}
struct MediaSource;
#[async_trait]
impl DynamicToolSource for MediaSource {
    fn id(&self) -> &str {
        "media"
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        Ok(vec![definition("image")])
    }
    async fn invoke(
        &self,
        _call: &ToolCallContext,
        _control: &CallControl,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        panic!("store-aware path required")
    }
    async fn invoke_with_store(
        &self,
        call: &ToolCallContext,
        _control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        let artifact = store
            .unwrap()
            .persist(
                &[1, 2, 3],
                ArtifactHint {
                    tool_name: "image".into(),
                    call_block_id: call.call_block_id,
                    kind: ArtifactKind::Binary,
                },
            )
            .await
            .unwrap();
        let mut output = result(call, ToolResultStatus::Succeeded, json!("x".repeat(4000)));
        output.media.push(MediaRef::new("image/png", artifact.id));
        output.notes.push(TextPayload::new("kept note"));
        Ok(output)
    }
}
struct Shorten;
#[async_trait]
impl ToolBatchProcessor for Shorten {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        for entry in batch.results_mut() {
            entry.output_mut().unwrap().content = json!("short");
        }
        Ok(())
    }
}
#[tokio::test]
async fn artifact_store_passes_through_dynamic_bridge_and_output_edits_keep_media() {
    for dynamic in [false, true] {
        let store = Arc::new(Store::default());
        let source: Arc<dyn DynamicToolSource> = Arc::new(MediaSource);
        let tools: Vec<Arc<dyn Tool>> = if dynamic {
            vec![]
        } else {
            vec![Arc::new(ToolBridge::new(
                source.clone(),
                definition("image"),
            ))]
        };
        let executor = executor(
            tools,
            ToolExecutorOptions {
                artifact_store: Some(store.clone()),
                after: vec![Arc::new(Shorten)],
                ..Default::default()
            },
        );
        if dynamic {
            executor.register_dynamic(source).unwrap();
        }
        let mut batch = batch(&["image"]);
        let id = batch.declaration_ids()[0];
        executor
            .bind(invocation(), control())
            .await
            .unwrap()
            .process(&mut batch)
            .await
            .unwrap();
        assert_eq!(*store.0.lock().unwrap(), vec![(vec![1, 2, 3], id)]);
        let result = batch.results()[0].result().unwrap().1;
        assert_eq!(result.output.content, json!("short"));
        assert_eq!(result.media, vec![MediaRef::new("image/png", "image")]);
        assert_eq!(result.notes, vec![TextPayload::new("kept note")]);
    }
}

#[tokio::test]
async fn empty_bind_and_empty_batch_complete_without_helpers() {
    let executor = executor(vec![], ToolExecutorOptions::default());
    let bound = executor.bind(invocation(), control()).await.unwrap();
    assert!(bound.surface().definitions.is_empty());
    bound.process(&mut batch(&[])).await.unwrap();
}

struct DescribedTool(Mutex<ToolDefinition>);
#[async_trait]
impl Tool for DescribedTool {
    fn definition(&self) -> ToolDefinition {
        self.0.lock().unwrap().clone()
    }
    async fn execute(&self, call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        result(
            call,
            ToolResultStatus::Succeeded,
            json!("same concrete object"),
        )
    }
}
#[tokio::test]
async fn static_description_and_schema_change_only_enter_new_bindings() {
    let tool = Arc::new(DescribedTool(Mutex::new(definition("static"))));
    let executor = executor(vec![tool.clone()], ToolExecutorOptions::default());
    let old = executor.bind(invocation(), control()).await.unwrap();
    let old_definition = old.surface().definitions[0].clone();
    tool.0.lock().unwrap().description = "new description".into();
    tool.0.lock().unwrap().parameters =
        json!({"type": "object", "properties": {"new": {"type": "boolean"}}});
    let new = executor.bind(invocation(), control()).await.unwrap();
    assert_eq!(old.surface().definitions[0], old_definition);
    assert_eq!(new.surface().definitions[0].description, "new description");
    assert_eq!(
        new.surface().definitions[0].parameters,
        tool.0.lock().unwrap().parameters
    );
    let mut old_batch = batch(&["static"]);
    old.process(&mut old_batch).await.unwrap();
    assert_eq!(
        old_batch.results()[0].result().unwrap().1.output.content,
        json!("same concrete object")
    );
}
