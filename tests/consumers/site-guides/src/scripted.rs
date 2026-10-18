//! A model for the offline programs. The site never shows this module: it
//! stands in for a provider so a guide's program runs without a key.

use async_trait::async_trait;
use causa::kernel::{
    BlockContent, CallControl, ModelGateway, ModelInvokeError, ModelItem, ModelOutput,
    ModelRequest, ModelResponse, ModelStopReason, ToolCallDraft, ToolResultPayload,
};

/// A gateway whose closure decides each reply from the request.
pub struct ScriptedModel<F>(pub F);

#[async_trait]
impl<F> ModelGateway for ScriptedModel<F>
where
    F: Fn(&ModelRequest) -> ModelOutput + Send + Sync,
{
    async fn invoke(
        &self,
        request: &ModelRequest,
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        Ok((self.0)(request))
    }
}

/// A final text answer.
pub fn reply(text: impl Into<String>) -> ModelOutput {
    ModelOutput::new(
        ModelResponse {
            items: vec![ModelItem::text(text)],
        },
        ModelStopReason::EndTurn,
    )
}

/// One tool call, to be answered in the next round.
pub fn call(tool: &str, arguments: serde_json::Value) -> ModelOutput {
    ModelOutput::new(
        ModelResponse {
            items: vec![ModelItem::tool_call(ToolCallDraft::new(tool, arguments))],
        },
        ModelStopReason::ToolUse,
    )
}

/// The tool results the request carries, oldest first.
pub fn tool_results(request: &ModelRequest) -> Vec<&ToolResultPayload> {
    request
        .frame
        .blocks
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect()
}
