//! Shared fixtures for the kernel's fact-machine test split. `mod.rs` is
//! required here: Cargo auto-discovers `tests/*.rs` as standalone targets,
//! but a shared module must live in a subdirectory. Each test target
//! compiles its own copy, so fixtures used by only some targets would trip
//! dead_code.
//!
//! Driver-side fixtures (gateways, tools, runner helpers) graduated to
//! `agent-runtime/tests/common` with the driver stack (Slice 12); the
//! kernel keeps only what fact tests need.

#![allow(dead_code)]

use async_trait::async_trait;
use causa_kernel::{
    Compaction, CompactionError, CompactionInput, CompactionOutput, ModelOutput, ModelResponse,
    ModelStopReason, TextPayload, ToolCallDraft, TurnContext, TurnId,
};

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

// ---- compaction fake ----------------------------------------------------------

pub struct DropAllCompaction;

#[async_trait]
impl Compaction for DropAllCompaction {
    async fn compact(&self, _input: CompactionInput) -> Result<CompactionOutput, CompactionError> {
        Ok(CompactionOutput {
            blocks: Vec::new(),
            summary: None,
            truncated: true,
        })
    }
}
