#![cfg(feature = "runtime")]

use async_trait::async_trait;
use causa::kernel::*;
use causa::runtime::*;
use std::sync::{Arc, Mutex};

struct Refusal;
#[async_trait]
impl ModelGateway for Refusal {
    async fn invoke(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        Ok(ModelOutput {
            response: ModelResponse {
                text: TextPayload::new("retained refusal"),
                tool_calls: vec![],
            },
            stop_reason: ModelStopReason::Refusal,
            usage: Some(ModelUsage {
                input_tokens: 7,
                ..Default::default()
            }),
            reasoning: None,
        })
    }
}
#[tokio::test]
async fn host_can_copy_selected_observations_and_recover_output_without_them() {
    for observe in [false, true] {
        let records = Arc::new(Mutex::new(vec![]));
        let token = CancellationToken::new();
        let mut options = TurnRunOptions::new(ModelRef::new("host-model"));
        if observe {
            let copies = records.clone();
            let cancel = token.clone();
            options.observer = Some(Arc::new(move |event| {
                if let RunEvent::ModelOutput {
                    invocation_id,
                    output,
                } = event
                {
                    copies
                        .lock()
                        .unwrap()
                        .push(((*invocation_id).clone(), output.response.text.0.clone()));
                    cancel.cancel();
                }
            }));
        }
        let runner = TurnRunner::new(
            Arc::new(Refusal),
            Arc::new(ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap()),
        );
        let out = runner
            .run(
                TurnId::new("refusal"),
                Context::new(),
                options,
                RunControl::new(token, None),
            )
            .await;
        let TurnResult::Interrupted {
            cause: TurnInterruption::ModelRefusal { output, .. },
        } = out.result
        else {
            panic!("refusal must survive callback cancellation")
        };
        assert_eq!(output.response.text.0, "retained refusal");
        assert_eq!(output.usage.unwrap().input_tokens, 7);
        assert!(out.context.blocks().is_empty());
        assert!(out.uncommitted_tool_batch.is_none());
        assert_eq!(records.lock().unwrap().len(), usize::from(observe));
    }
}
