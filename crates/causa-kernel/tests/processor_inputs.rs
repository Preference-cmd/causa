//! What a tool processor can read besides the batch: the bound tools' declared
//! hints and the host's data. Both are optional and default to nothing.

use async_trait::async_trait;
use causa_kernel::{
    ArtifactStore, BlockId, CallControl, CancellationToken, DynamicToolSource, ProcessorContext,
    ProcessorData, RoundId, SourceError, Tool, ToolCallContext, ToolDefinition, ToolDescription,
    ToolExecutionError, ToolHints, ToolOutput, ToolResultPayload, ToolResultStatus, TurnId,
};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

fn definition(name: &str) -> ToolDefinition {
    ToolDefinition::new(name, "fixture", json!({"type": "object"}))
}

fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

/// A tool and a source written before hints existed: neither overrides them.
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
fn hints_default_to_undeclared_and_setters_declare_one_field_each() {
    let hints = ToolHints::default();
    assert_eq!(
        (
            hints.read_only,
            hints.destructive,
            hints.idempotent,
            hints.open_world
        ),
        (None, None, None, None)
    );
    let hints = ToolHints::default()
        .with_read_only(false)
        .with_destructive(true);
    assert_eq!(hints.read_only, Some(false));
    assert_eq!(hints.destructive, Some(true));
    assert_eq!(hints.idempotent, None);
    assert_eq!(hints.open_world, None);
    assert_eq!(
        ToolHints::default()
            .with_idempotent(true)
            .with_open_world(false),
        {
            let mut expected = ToolHints::default();
            expected.idempotent = Some(true);
            expected.open_world = Some(false);
            expected
        }
    );
}

#[test]
fn a_tool_that_does_not_override_hints_declares_none() {
    assert_eq!(Plain.hints(), ToolHints::default());
    let description = ToolDescription::new(definition("plain"));
    assert_eq!(description.hints, ToolHints::default());
    let hinted = description.with_hints(ToolHints::default().with_read_only(true));
    assert_eq!(hinted.hints.read_only, Some(true));
    assert_eq!(hinted.definition, definition("plain"));
}

#[tokio::test]
async fn the_default_describe_lists_with_no_hints() {
    let described = Plain.describe().await.expect("describe");
    assert_eq!(
        described,
        vec![
            ToolDescription::new(definition("plain_a")),
            ToolDescription::new(definition("plain_b")),
        ]
    );
}

#[test]
fn a_context_built_with_new_has_no_tools_and_no_data() {
    let turn = TurnId::new("t");
    let order: Vec<BlockId> = Vec::new();
    let control = control();
    let ctx = ProcessorContext::new(&turn, RoundId(0), &order, &control);
    assert!(ctx.tools.is_empty());
    assert!(ctx.data.is_none());
    assert_eq!(ctx.hints("plain"), None);
    assert!(ctx.data::<u8>().is_none());
}

#[test]
fn hints_resolve_by_name_and_data_by_exact_type() {
    struct Mode(&'static str);

    let turn = TurnId::new("t");
    let order: Vec<BlockId> = Vec::new();
    let control = control();
    let tools = vec![
        ToolDescription::new(definition("read"))
            .with_hints(ToolHints::default().with_read_only(true)),
        ToolDescription::new(definition("write")),
    ];
    let data = ProcessorData::new(Mode("ask"));
    let ctx = ProcessorContext::new(&turn, RoundId(2), &order, &control)
        .with_tools(&tools)
        .with_data(&data);
    assert_eq!(ctx.hints("read").and_then(|h| h.read_only), Some(true));
    assert_eq!(ctx.hints("write"), Some(&ToolHints::default()));
    assert_eq!(ctx.hints("missing"), None);
    assert_eq!(ctx.data::<Mode>().map(|mode| mode.0), Some("ask"));
    assert!(ctx.data::<String>().is_none());
    assert!(data.get::<Mode>().is_some());
    assert_eq!(format!("{data:?}"), "ProcessorData { .. }");
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
