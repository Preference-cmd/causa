//! Shared fixtures for the Slice 1.5 test split. `mod.rs` is required here:
//! Cargo auto-discovers `tests/*.rs` as standalone targets, but a shared
//! module must live in a subdirectory. Each test target compiles its own
//! copy, so fixtures used by only some targets would trip dead_code.

#![allow(dead_code)]

use reimagine_context_kernel::{
    ModelOutput, ModelResponse, ModelStopReason, TextPayload, ToolCallDraft, TurnContext, TurnId,
};

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

pub fn tooluse_output(text: &str, tool_name: &str, args: serde_json::Value) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: vec![ToolCallDraft {
                tool_name: tool_name.into(),
                arguments: args,
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    }
}
