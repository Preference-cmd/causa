//! `ModelGateway` port — the model-invocation seam drivers call into.
//! Transport-free; concrete gateways live outside the kernel.

use async_trait::async_trait;

use crate::context::ids::{AttemptNumber, InvocationId};
use crate::context::model::{
    GenerationOptions, ModelInvokeError, ModelOutput, ModelRef, ToolSurface,
};
use crate::context::turn::ContextFrame;
use crate::ports::control::AttemptControl;

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub invocation_id: InvocationId,
    pub attempt: AttemptNumber,
    pub frame: ContextFrame,
    pub model: ModelRef,
    pub tool_surface: ToolSurface,
    pub generation: GenerationOptions,
}

#[async_trait]
pub trait ModelGateway: Send + Sync {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError>;
}
