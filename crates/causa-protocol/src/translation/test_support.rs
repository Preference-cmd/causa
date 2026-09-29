//! Shared test fixtures for the three kernel-face renderers. `#[cfg(test)]`
//! only: one copy of the block/text/call/result/frame builders so the
//! per-protocol test modules cannot drift apart.

#![cfg(test)]

use serde_json::Value;

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, ContentPart, ContextBlock, ContextFrame, ContextVersion,
    FrameScope, MediaRef, ModelContext, RoundId, TextPayload, ToolCallPayload, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnId,
};

pub(crate) fn label_id(label: &str) -> BlockId {
    let mut value = 0_u128;
    for byte in label.bytes() {
        value = value.wrapping_mul(257).wrapping_add(u128::from(byte));
    }
    BlockId::new(uuid::Uuid::from_u128(value))
}

pub(crate) fn block(
    seq: u64,
    content: BlockContent,
    source: Option<&str>,
    provider_call_id: Option<&str>,
) -> ContextBlock {
    ContextBlock::new(
        BlockId::new(uuid::Uuid::from_u128(u128::from(seq) + 1)),
        content,
        BlockMeta {
            provider_call_id: provider_call_id.map(String::from),
            source: source.map(String::from),
        },
    )
}

pub(crate) fn text(seq: u64, text: &str, source: Option<&str>) -> ContextBlock {
    parts(seq, vec![ContentPart::Text(TextPayload::new(text))], source)
}

/// A Parts block from explicit content parts — the general message shape.
pub(crate) fn parts(seq: u64, parts: Vec<ContentPart>, source: Option<&str>) -> ContextBlock {
    block(seq, BlockContent::Parts(parts), source, None)
}

/// A single-text part convenience over [`parts`].
pub(crate) fn text_part(text: &str) -> ContentPart {
    ContentPart::Text(TextPayload::new(text))
}

/// A media part pointing at a host-side asset reference.
pub(crate) fn media_part(media_type: &str, reference: &str) -> ContentPart {
    ContentPart::Media(MediaRef::new(media_type, reference))
}

pub(crate) fn call(
    _seq: u64,
    call_id: &str,
    provider: Option<&str>,
    name: &str,
    arguments: Value,
) -> ContextBlock {
    ContextBlock::new(
        label_id(call_id),
        BlockContent::ToolCall(ToolCallPayload {
            tool_name: name.into(),
            arguments,
        }),
        BlockMeta {
            provider_call_id: provider.map(String::from),
            source: None,
        },
    )
}

pub(crate) fn result(
    seq: u64,
    call_id: &str,
    status: ToolResultStatus,
    content: Value,
) -> ContextBlock {
    result_with_media(seq, call_id, status, content, Vec::new())
}

/// A tool result block with media attachments (references only).
pub(crate) fn result_with_media(
    seq: u64,
    call_id: &str,
    status: ToolResultStatus,
    content: Value,
    media: Vec<MediaRef>,
) -> ContextBlock {
    block(
        seq,
        BlockContent::ToolResult(ToolResultPayload {
            call_block_id: label_id(call_id),
            status,
            output: ToolOutput::new(content),
            media,
            notes: Vec::new(),
        }),
        None,
        None,
    )
}

pub(crate) fn frame(blocks: Vec<ContextBlock>) -> ContextFrame {
    let scope = FrameScope::Turn {
        turn_id: TurnId::new("t1"),
        source_version: ContextVersion(3),
    };
    ContextFrame {
        scope,
        round_id: RoundId(0),
        model_context: ModelContext { blocks },
    }
}
