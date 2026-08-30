//! Scripted gateway for tests — consumes canned outputs in order.

use async_trait::async_trait;

use crate::context::model::{ModelInvokeError, ModelInvokeErrorKind, ModelOutput};
use crate::ports::control::AttemptControl;
use crate::ports::gateway::ModelGateway;
use crate::ports::gateway::ModelRequest;

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
