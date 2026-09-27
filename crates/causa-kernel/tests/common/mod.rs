//! Shared fixtures for the kernel's fact-machine test split. `mod.rs` is
//! required here: Cargo auto-discovers `tests/*.rs` as standalone targets,
//! but a shared module must live in a subdirectory. Each test target
//! compiles its own copy, so fixtures used by only some targets would trip
//! dead_code.

#![allow(dead_code)]

use causa_kernel::{
    BlockId, ModelOutput, ModelResponse, ModelStopReason, TextPayload, ToolCallDraft, TurnContext,
    TurnId,
};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_BLOCK_ID: AtomicU64 = AtomicU64::new(1);

pub fn block_id() -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(
        NEXT_BLOCK_ID.fetch_add(1, Ordering::Relaxed) as u128,
    ))
}

pub fn block_ids_for(response: &ModelResponse) -> Vec<BlockId> {
    let count = causa_kernel::model_output_block_count(response);
    (0..count).map(|_| block_id()).collect()
}

// ---- ids and model-output constructors --------------------------------------

pub fn turn_id(s: &str) -> TurnId {
    TurnId::new(s)
}

pub fn ctx(s: &str) -> TurnContext {
    TurnContext::new(turn_id(s))
}

pub fn endturn_output(text: &str) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: vec![],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    }
}

pub fn draft(tool_name: &str, args: serde_json::Value) -> ToolCallDraft {
    ToolCallDraft {
        tool_name: tool_name.into(),
        arguments: args,
        provider_call_id: None,
    }
}
