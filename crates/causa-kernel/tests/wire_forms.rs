//! The serialized form of the kernel's enums is snake_case throughout. A host
//! stores context material, so a change to any string below changes what a
//! stored context reads as and has to be a recorded break.
mod common;

use causa_kernel::{
    ArtifactKind, ArtifactRef, BlockContent, BlockMeta, ContextBlock, ModelInvokeErrorKind,
    ModelItem, ToolCallDraft, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use serde_json::json;

#[test]
fn tool_result_statuses_are_snake_case() {
    for (status, wire) in [
        (ToolResultStatus::Succeeded, "succeeded"),
        (ToolResultStatus::Failed, "failed"),
        (ToolResultStatus::Rejected, "rejected"),
        (ToolResultStatus::Cancelled, "cancelled"),
        (ToolResultStatus::TimedOut, "timed_out"),
        (ToolResultStatus::UnknownOutcome, "unknown_outcome"),
    ] {
        assert_eq!(serde_json::to_value(&status).unwrap(), json!(wire));
        assert_eq!(
            serde_json::from_value::<ToolResultStatus>(json!(wire)).unwrap(),
            status
        );
    }
    // The PascalCase form written before this rule is not read.
    assert!(serde_json::from_value::<ToolResultStatus>(json!("Succeeded")).is_err());
}

#[test]
fn artifact_kinds_are_snake_case() {
    for (kind, wire) in [
        (ArtifactKind::FullOutput, "full_output"),
        (ArtifactKind::PipeCache, "pipe_cache"),
        (ArtifactKind::Binary, "binary"),
    ] {
        assert_eq!(serde_json::to_value(&kind).unwrap(), json!(wire));
        assert_eq!(
            serde_json::from_value::<ArtifactKind>(json!(wire)).unwrap(),
            kind
        );
    }
    assert!(serde_json::from_value::<ArtifactKind>(json!("FullOutput")).is_err());
}

#[test]
fn a_stored_tool_result_block_carries_both_in_snake_case() {
    let mut output = ToolOutput::new(json!("ok"));
    output.artifact = Some(ArtifactRef::new("asset", 3, ArtifactKind::FullOutput, true));
    let block = ContextBlock::new(
        common::block_id(),
        BlockContent::ToolResult(ToolResultPayload::new(
            common::block_id(),
            ToolResultStatus::UnknownOutcome,
            output,
        )),
        BlockMeta::default(),
    );
    let wire = serde_json::to_value(&block).unwrap();
    let text = wire.to_string();
    assert!(text.contains(r#""status":"unknown_outcome""#), "{text}");
    assert!(text.contains(r#""kind":"full_output""#), "{text}");
    assert!(text.contains(r#""shape":"tool_result""#), "{text}");
    assert_eq!(serde_json::from_value::<ContextBlock>(wire).unwrap(), block);
}

#[test]
fn model_item_content_is_snake_case() {
    let parts = serde_json::to_value(ModelItem::text("hi")).unwrap();
    assert!(parts["content"].get("parts").is_some(), "{parts}");
    let call =
        serde_json::to_value(ModelItem::tool_call(ToolCallDraft::new("read", json!({})))).unwrap();
    assert!(call["content"].get("tool_call").is_some(), "{call}");
}

#[test]
fn model_invoke_error_kinds_are_snake_case() {
    for (kind, wire) in [
        (ModelInvokeErrorKind::Transient, "transient"),
        (ModelInvokeErrorKind::TimedOut, "timed_out"),
        (ModelInvokeErrorKind::Cancelled, "cancelled"),
        (ModelInvokeErrorKind::Permanent, "permanent"),
        (ModelInvokeErrorKind::Undecodable, "undecodable"),
        (ModelInvokeErrorKind::InvalidRequest, "invalid_request"),
        (ModelInvokeErrorKind::UnknownOutcome, "unknown_outcome"),
    ] {
        assert_eq!(serde_json::to_value(&kind).unwrap(), json!(wire));
    }
}
