use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

use crate::control::AttemptControl;
use crate::ids::{AttemptNumber, InvocationId};
use crate::model::{
    GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRef, RetryPolicy,
    ToolSurface,
};
use crate::turn::ContextFrame;

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

pub struct FakeGateway {
    pub outputs: std::sync::Mutex<Vec<Result<ModelOutput, ModelInvokeErrorKind>>>,
}
impl FakeGateway {
    pub fn new(outputs: Vec<Result<ModelOutput, ModelInvokeErrorKind>>) -> Self {
        Self {
            outputs: std::sync::Mutex::new(outputs),
        }
    }
}
#[async_trait]
impl ModelGateway for FakeGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut guard = self.outputs.lock().unwrap();
        if guard.is_empty() {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                "no more fake outputs",
            ));
        }
        match guard.remove(0) {
            Ok(o) => Ok(o),
            Err(k) => Err(ModelInvokeError::new(k, "fake error")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TurnLimits {
    pub max_model_rounds: u32,
    pub max_tool_calls: u32,
}
impl Default for TurnLimits {
    fn default() -> Self {
        Self {
            max_model_rounds: 10,
            max_tool_calls: 64,
        }
    }
}

pub struct TurnRunConfig {
    pub model: ModelRef,
    pub tool_surface: ToolSurface,
    pub generation: GenerationOptions,
    pub retry: RetryPolicy,
    pub limits: TurnLimits,
    pub tool_output_limits: crate::tool::ToolOutputLimits,
    pub artifact_store: Option<Arc<dyn crate::tool::ArtifactStore>>,
    pub window_budget: crate::turn::WindowBudget,
    pub compaction: Option<Arc<dyn crate::turn::Compaction>>,
    pub token_counter: Option<Arc<dyn crate::turn::TokenCounter>>,
    pub call_timeout: Option<Duration>,
    pub attempt_timeout: Option<Duration>,
}
impl Default for TurnRunConfig {
    fn default() -> Self {
        Self {
            model: ModelRef::new("fake"),
            tool_surface: ToolSurface::empty(),
            generation: GenerationOptions::default(),
            retry: RetryPolicy::default(),
            limits: TurnLimits::default(),
            tool_output_limits: crate::tool::ToolOutputLimits::default(),
            artifact_store: None,
            window_budget: crate::turn::WindowBudget::default(),
            compaction: None,
            token_counter: None,
            call_timeout: None,
            attempt_timeout: None,
        }
    }
}
