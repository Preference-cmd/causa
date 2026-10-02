//! Consumer fixtures; no framework strategy implementation is published.
#![allow(dead_code)]
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::{
    RunControl, ToolExecutor, ToolExecutorOptions, TurnRunOptions, TurnRunner, new_block_id,
};
use std::sync::{Arc, Mutex};
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
    tooluse_calls_output(text, vec![draft(tool_name, args)])
}

pub fn tooluse_calls_output(text: &str, calls: Vec<ToolCallDraft>) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: calls,
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
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

pub fn ctrl() -> RunControl {
    RunControl::new(Default::default(), None)
}
pub fn options() -> TurnRunOptions {
    TurnRunOptions::new(ModelRef::new("fixture-model"))
}
pub fn runner_with(gateway: Arc<dyn ModelGateway>, tools: Vec<Arc<dyn Tool>>) -> TurnRunner {
    TurnRunner::new(
        gateway,
        Arc::new(ToolExecutor::new(tools, ToolExecutorOptions::default()).unwrap()),
    )
}
pub fn text_block(text: &str) -> ContextBlock {
    ContextBlock::new(
        new_block_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
        BlockMeta {
            source: Some("user".into()),
            ..Default::default()
        },
    )
}
pub fn input(text: &str) -> Context {
    Context::from_blocks(vec![text_block(text)]).unwrap()
}
pub struct RecordingGateway {
    pub outputs: Mutex<Vec<Result<ModelOutput, ModelInvokeErrorKind>>>,
    pub recorded: Mutex<Vec<ModelRequest>>,
}
impl RecordingGateway {
    pub fn scripted(outputs: Vec<Result<ModelOutput, ModelInvokeErrorKind>>) -> Arc<Self> {
        Arc::new(Self {
            outputs: Mutex::new(outputs),
            recorded: Mutex::new(vec![]),
        })
    }
}
#[async_trait]
impl ModelGateway for RecordingGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        self.recorded.lock().unwrap().push(request.clone());
        let mut outputs = self.outputs.lock().unwrap();
        if outputs.is_empty() {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                "script exhausted",
            ));
        }
        outputs
            .remove(0)
            .map_err(|kind| ModelInvokeError::new(kind, "scripted error"))
    }
}
pub struct EchoTool;
#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".into(),
            description: "echo".into(),
            parameters: serde_json::json!({"type":"object"}),
        }
    }
    async fn execute(&self, call: &ToolCallContext, _: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: call.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(call.input.arguments.clone()),
            media: vec![],
            notes: vec![],
        }
    }
}
