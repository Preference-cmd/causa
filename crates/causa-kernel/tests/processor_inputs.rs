//! What a tool processor can read besides the batch: the bound tools' declared
//! metadata, call-scoped attachments, and the host's run data.

use async_trait::async_trait;
use causa_kernel::{
    ArtifactStore, BlockId, CallAttachments, CallControl, CancellationToken, DynamicToolSource,
    ProcessorContext, ProcessorData, RoundId, SourceError, Tool, ToolBatch, ToolCallContext,
    ToolCallPayload, ToolDefinition, ToolDescription, ToolExecutionError, ToolMetadata, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnId,
};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

fn definition(name: &str) -> ToolDefinition {
    ToolDefinition::new(name, "fixture", json!({"type": "object"}))
}

fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

fn block_id(n: u128) -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(n))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HostCap(&'static str);

#[derive(Debug, Clone, PartialEq, Eq)]
struct OtherTag(u32);

struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

/// A tool and a source that do not override `metadata()` or `describe()`.
struct Plain;

#[async_trait]
impl Tool for Plain {
    fn definition(&self) -> ToolDefinition {
        definition("plain")
    }
    async fn execute(
        &self,
        ctx: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> ToolResultPayload {
        ToolResultPayload::new(
            ctx.call_block_id,
            ToolResultStatus::Succeeded,
            ToolOutput::new(json!(null)),
        )
    }
}

#[async_trait]
impl DynamicToolSource for Plain {
    fn id(&self) -> &str {
        "plain"
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        Ok(vec![definition("plain_a"), definition("plain_b")])
    }
    async fn invoke(
        &self,
        _call: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        Err(ToolExecutionError::Unavailable("fixture".into()))
    }
}

#[test]
fn tool_metadata_supports_typed_insert_replace_get_and_debug() {
    let mut meta = ToolMetadata::new();
    assert!(meta.is_empty());
    assert!(ToolMetadata::default().is_empty());

    assert!(meta.insert(HostCap("read")).is_none());
    assert!(!meta.is_empty());
    assert_eq!(meta.get::<HostCap>(), Some(&HostCap("read")));
    assert_eq!(meta.get::<OtherTag>(), None);

    let previous = meta.insert(HostCap("write"));
    assert_eq!(previous.as_deref(), Some(&HostCap("read")));
    assert_eq!(meta.get::<HostCap>(), Some(&HostCap("write")));

    let chained = ToolMetadata::new()
        .with(HostCap("chained"))
        .with(OtherTag(7));
    assert_eq!(chained.get::<HostCap>(), Some(&HostCap("chained")));
    assert_eq!(chained.get::<OtherTag>(), Some(&OtherTag(7)));
    assert_eq!(format!("{chained:?}"), "ToolMetadata { len: 2, .. }");
}

#[test]
fn a_tool_that_does_not_override_metadata_has_empty_metadata() {
    assert!(Plain.metadata().is_empty());
    let description = ToolDescription::new(definition("plain"));
    assert!(description.metadata.is_empty());

    let with_meta = description.with_meta(HostCap("safe"));
    assert_eq!(with_meta.metadata.get::<HostCap>(), Some(&HostCap("safe")));
    assert_eq!(with_meta.definition, definition("plain"));

    let replaced = with_meta.with_metadata(ToolMetadata::new().with(OtherTag(42)));
    assert_eq!(replaced.metadata.get::<HostCap>(), None);
    assert_eq!(replaced.metadata.get::<OtherTag>(), Some(&OtherTag(42)));
}

#[tokio::test]
async fn the_default_describe_lists_with_empty_metadata() {
    let described = Plain.describe().await.expect("describe");
    assert_eq!(described.len(), 2);
    assert_eq!(described[0].definition, definition("plain_a"));
    assert!(described[0].metadata.is_empty());
    assert_eq!(described[1].definition, definition("plain_b"));
    assert!(described[1].metadata.is_empty());
}

#[test]
fn a_context_built_with_new_has_no_tools_and_no_data() {
    let turn = TurnId::new("t");
    let order: Vec<BlockId> = Vec::new();
    let control = control();
    let ctx = ProcessorContext::new(&turn, RoundId(0), &order, &control);
    assert!(ctx.tools.is_empty());
    assert!(ctx.data.is_none());
    assert!(ctx.tool("plain").is_none());
    assert!(ctx.tool_metadata::<HostCap>("plain").is_none());
    assert!(ctx.data::<u8>().is_none());
}

#[test]
fn tool_metadata_resolves_by_name_and_type_and_data_by_exact_type() {
    struct Mode(&'static str);

    let turn = TurnId::new("t");
    let order: Vec<BlockId> = Vec::new();
    let control = control();
    let tools = vec![
        ToolDescription::new(definition("read")).with_meta(HostCap("read-only")),
        ToolDescription::new(definition("write")),
    ];
    let data = ProcessorData::new(Mode("ask"));
    let ctx = ProcessorContext::new(&turn, RoundId(2), &order, &control)
        .with_tools(&tools)
        .with_data(&data);

    assert_eq!(
        ctx.tool("read")
            .and_then(|tool| tool.metadata.get::<HostCap>()),
        Some(&HostCap("read-only"))
    );
    assert_eq!(
        ctx.tool_metadata::<HostCap>("read"),
        Some(&HostCap("read-only"))
    );
    assert_eq!(ctx.tool_metadata::<OtherTag>("read"), None);
    assert!(ctx.tool("write").is_some());
    assert_eq!(ctx.tool_metadata::<HostCap>("write"), None);
    assert!(ctx.tool("missing").is_none());
    assert_eq!(ctx.tool_metadata::<HostCap>("missing"), None);

    assert_eq!(ctx.data::<Mode>().map(|mode| mode.0), Some("ask"));
    assert!(ctx.data::<String>().is_none());
    assert!(data.get::<Mode>().is_some());
    assert_eq!(format!("{data:?}"), "ProcessorData { .. }");
}

#[test]
fn call_attachments_support_crud_and_tool_call_context_access() {
    let mut attachments = CallAttachments::new();
    assert!(attachments.is_empty());
    assert!(CallAttachments::default().is_empty());

    assert!(attachments.insert(HostCap("plan-a")).is_none());
    assert!(!attachments.is_empty());
    assert_eq!(attachments.get::<HostCap>(), Some(&HostCap("plan-a")));
    assert_eq!(attachments.get::<OtherTag>(), None);

    let prev = attachments.insert(HostCap("plan-b"));
    assert_eq!(prev.as_deref(), Some(&HostCap("plan-a")));
    assert_eq!(attachments.get::<HostCap>(), Some(&HostCap("plan-b")));

    let removed = attachments.remove::<HostCap>();
    assert_eq!(removed.as_deref(), Some(&HostCap("plan-b")));
    assert!(attachments.is_empty());

    let chained = CallAttachments::new().with(HostCap("x")).with(OtherTag(9));
    assert_eq!(format!("{chained:?}"), "CallAttachments { len: 2, .. }");

    let mut cleared = chained.clone();
    cleared.clear();
    assert!(cleared.is_empty());

    let ctx = ToolCallContext::new(block_id(1), ToolCallPayload::new("read", json!({})))
        .with_attachment(HostCap("unit-test"))
        .with_attachment(OtherTag(3));
    assert_eq!(ctx.attachment::<HostCap>(), Some(&HostCap("unit-test")));
    assert_eq!(ctx.attachment::<OtherTag>(), Some(&OtherTag(3)));
    assert_eq!(ctx.attachment::<String>(), None);
    assert!(!ctx.attachments().is_empty());
}

#[test]
fn batch_entry_attachments_clear_on_resolve_and_survive_refused_resolve() {
    let drops = Arc::new(AtomicUsize::new(0));
    let id_a = block_id(10);
    let id_b = block_id(11);
    let mut batch = ToolBatch::new(vec![
        ToolCallContext::new(id_a, ToolCallPayload::new("a", json!({}))),
        ToolCallContext::new(id_b, ToolCallPayload::new("b", json!({}))),
    ])
    .unwrap();

    assert!(batch.calls_mut()[0].attach(DropCounter(drops.clone())));
    assert!(batch.calls_mut()[1].attach(DropCounter(drops.clone())));
    assert_eq!(drops.load(Ordering::SeqCst), 0);

    // A refused `resolve_at` leaves attachments intact.
    let err = batch
        .resolve_at(
            0,
            block_id(12),
            ToolResultPayload::new(
                id_b,
                ToolResultStatus::Succeeded,
                ToolOutput::new(json!(null)),
            ),
        )
        .unwrap_err();
    assert_eq!(err.result.call_block_id, id_b);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(!batch.calls()[0].call().attachments().is_empty());

    // An accepted `resolve_at` drops attachments immediately, and resolved
    // entries refuse further attachments.
    batch
        .resolve_at(
            0,
            block_id(13),
            ToolResultPayload::new(
                id_a,
                ToolResultStatus::Rejected,
                ToolOutput::new(json!({"error": "denied"})),
            ),
        )
        .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(batch.results()[0].call().attachments().is_empty());
    assert!(batch.results_mut()[0].attachments_mut().is_none());
    assert!(!batch.results_mut()[0].attach(HostCap("late")));

    // Dropping the batch drops attachments on remaining pending entries.
    drop(batch);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn data_is_shared_not_copied() {
    let mode = Arc::new(AtomicU8::new(0));
    let data = ProcessorData::new(mode.clone());
    let copy = data.clone();
    mode.store(3, Ordering::SeqCst);
    let read = |data: &ProcessorData| {
        data.get::<Arc<AtomicU8>>()
            .map(|mode| mode.load(Ordering::SeqCst))
    };
    assert_eq!(read(&data), Some(3));
    assert_eq!(read(&copy), Some(3));
    // The exact type is required: the value is an `Arc<AtomicU8>`.
    assert!(data.get::<AtomicU8>().is_none());
}
