//! Shared test fixtures for the three kernel-face renderers. `#[cfg(test)]`
//! only: one copy of the block/text/call/result/frame builders so the
//! per-protocol test modules cannot drift apart.

#![cfg(test)]

use serde_json::Value;

use reimagine_context_kernel::{
    BlockContent, BlockId, BlockMeta, BlockSequence, ContextBlock, ContextFrame, ContextVersion,
    FrameId, FrameScope, ModelContext, RoundId, TextPayload, ToolCallId, ToolCallPayload,
    ToolOutput, ToolResultPayload, ToolResultStatus, TurnId,
};

pub(crate) fn block(
    seq: u64,
    content: BlockContent,
    source: Option<&str>,
    provider_call_id: Option<&str>,
) -> ContextBlock {
    ContextBlock {
        id: BlockId {
            turn_id: TurnId::new("t1"),
            sequence: BlockSequence(seq),
        },
        sequence: BlockSequence(seq),
        content,
        meta: BlockMeta {
            provider_call_id: provider_call_id.map(String::from),
            source: source.map(String::from),
        },
    }
}

pub(crate) fn text(seq: u64, text: &str, source: Option<&str>) -> ContextBlock {
    block(
        seq,
        BlockContent::Text(TextPayload::new(text)),
        source,
        None,
    )
}

pub(crate) fn call(
    seq: u64,
    call_id: &str,
    provider: Option<&str>,
    name: &str,
    arguments: Value,
) -> ContextBlock {
    block(
        seq,
        BlockContent::ToolCall(ToolCallPayload {
            call_id: ToolCallId::new(call_id),
            tool_name: name.into(),
            arguments,
        }),
        None,
        provider,
    )
}

pub(crate) fn result(
    seq: u64,
    call_id: &str,
    status: ToolResultStatus,
    content: Value,
) -> ContextBlock {
    block(
        seq,
        BlockContent::ToolResult(ToolResultPayload {
            call_id: ToolCallId::new(call_id),
            status,
            output: ToolOutput::new(content),
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
        frame_id: FrameId::from_scope(&scope, RoundId(0)),
        scope,
        round_id: RoundId(0),
        model_context: ModelContext { blocks },
    }
}
