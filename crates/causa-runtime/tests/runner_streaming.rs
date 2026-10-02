mod common;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::*;
use common::*;
use serde_json::json;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

struct Gateway {
    scripts: Mutex<Vec<Result<Vec<StreamDelta>, ModelInvokeErrorKind>>>,
    requests: Mutex<Vec<ModelRequest>>,
}
#[async_trait]
impl ModelGateway for Gateway {
    async fn invoke(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        panic!("streaming uses stream only")
    }
    async fn stream(
        &self,
        r: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        self.requests.lock().unwrap().push(r.clone());
        self.scripts
            .lock()
            .unwrap()
            .remove(0)
            .map(|d| Box::pin(futures_util::stream::iter(d)) as ModelStream)
            .map_err(|k| ModelInvokeError::new(k, "stream setup failed"))
    }
}
fn gateway(scripts: Vec<Result<Vec<StreamDelta>, ModelInvokeErrorKind>>) -> Arc<Gateway> {
    Arc::new(Gateway {
        scripts: Mutex::new(scripts),
        requests: Mutex::new(vec![]),
    })
}
fn done(output: ModelOutput) -> StreamDelta {
    StreamDelta::Done {
        stop_reason: output.stop_reason,
        final_output: output,
    }
}
#[tokio::test]
async fn streaming_tools_share_material_commit_loop() {
    let g = gateway(vec![
        Ok(vec![
            StreamDelta::TextDelta {
                delta: "advisory".into(),
            },
            StreamDelta::ToolCallDelta {
                call_index: 0,
                provider_call_id: Some("wire".into()),
                name_delta: Some("echo".into()),
                arguments_delta: Some("{}".into()),
            },
            done(tooluse_output("actual", "echo", json!({}))),
        ]),
        Ok(vec![
            StreamDelta::Usage(ModelUsage::default()),
            done(endturn_output("final")),
        ]),
    ]);
    let events = Arc::new(Mutex::new(vec![]));
    let seen = events.clone();
    let mut opts = options();
    opts.observer = Some(Arc::new(move |event| {
        seen.lock().unwrap().push(match event {
            RunEvent::ModelRequestReady { .. } => "request",
            RunEvent::ModelDelta { delta, .. } => {
                assert!(!matches!(
                    delta,
                    StreamDelta::Done { .. } | StreamDelta::Error { .. }
                ));
                "delta"
            }
            RunEvent::ModelOutput { .. } => "output",
            RunEvent::BlocksCommitted { .. } => "commit",
            RunEvent::ToolBatchReady { .. } => "ready",
            RunEvent::ToolBatchReturned { result, .. } => {
                assert!(result.is_ok());
                "returned"
            }
        })
    }));
    let out = runner_with(g.clone(), vec![Arc::new(EchoTool)])
        .run_streaming(TurnId::new("stream"), Context::new(), opts, ctrl())
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(out.context.blocks().len(), 4);
    assert_eq!(g.requests.lock().unwrap()[1].frame.blocks.len(), 3);
    assert_eq!(
        *events.lock().unwrap(),
        vec![
            "request", "delta", "delta", "output", "commit", "ready", "returned", "commit",
            "request", "delta", "output", "commit"
        ]
    );
    let wire = serde_json::to_string(&out.context).unwrap();
    assert!(!wire.contains("advisory"));
    assert!(wire.contains("actual"));
}
#[tokio::test]
async fn stream_setup_error_error_delta_and_missing_done_never_commit_partial_output() {
    for (script, kind) in [
        (
            Err(ModelInvokeErrorKind::Permanent),
            ModelInvokeErrorKind::Permanent,
        ),
        (
            Ok(vec![
                StreamDelta::TextDelta {
                    delta: "partial".into(),
                },
                StreamDelta::Error {
                    kind: ModelInvokeErrorKind::Transient,
                    message: "failure".into(),
                },
            ]),
            ModelInvokeErrorKind::Transient,
        ),
        (
            Ok(vec![StreamDelta::TextDelta {
                delta: "partial".into(),
            }]),
            ModelInvokeErrorKind::UnknownOutcome,
        ),
    ] {
        let g = gateway(vec![script]);
        let out = runner_with(g.clone(), vec![])
            .run_streaming(TurnId::new("failed"), input("original"), options(), ctrl())
            .await;
        let TurnResult::Interrupted {
            cause: TurnInterruption::ModelCallFailed { error, .. },
        } = out.result
        else {
            panic!("model failure")
        };
        assert_eq!(error.kind, kind);
        assert_eq!(out.context.blocks().len(), 1);
        assert_eq!(g.requests.lock().unwrap().len(), 1);
        assert!(out.uncommitted_tool_batch.is_none());
    }
}
#[tokio::test]
async fn streaming_observer_cancellation_prevents_done_and_half_tool_execution() {
    let g = gateway(vec![Ok(vec![
        StreamDelta::ToolCallDelta {
            call_index: 0,
            provider_call_id: None,
            name_delta: Some("echo".into()),
            arguments_delta: Some("{".into()),
        },
        done(tooluse_output("", "echo", json!({}))),
    ])]);
    let token = CancellationToken::new();
    let cancel = token.clone();
    let mut opts = options();
    opts.observer = Some(Arc::new(move |e| {
        if matches!(e, RunEvent::ModelDelta { .. }) {
            cancel.cancel()
        }
    }));
    let out = runner_with(g, vec![Arc::new(EchoTool)])
        .run_streaming(
            TurnId::new("cancel"),
            Context::new(),
            opts,
            RunControl::new(token, None),
        )
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::Cancelled { .. }
        }
    ));
    assert!(out.context.blocks().is_empty());
    assert!(out.uncommitted_tool_batch.is_none());
}

struct PollCountingGateway(Arc<std::sync::atomic::AtomicUsize>);
#[async_trait]
impl ModelGateway for PollCountingGateway {
    async fn invoke(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        panic!("streaming test")
    }
    async fn stream(
        &self,
        _: &ModelRequest,
        _: &CallControl,
    ) -> Result<ModelStream, ModelInvokeError> {
        let polls = self.0.clone();
        Ok(Box::pin(futures_util::stream::poll_fn(move |_| {
            let index = polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::task::Poll::Ready(Some(if index == 0 {
                StreamDelta::TextDelta {
                    delta: "first".into(),
                }
            } else {
                done(endturn_output("must not poll"))
            }))
        })))
    }
}

#[tokio::test]
async fn callback_cancellation_stops_before_polling_the_stream_again() {
    let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let token = CancellationToken::new();
    let cancel = token.clone();
    let mut opts = options();
    opts.observer = Some(Arc::new(move |event| {
        if matches!(event, RunEvent::ModelDelta { .. }) {
            cancel.cancel();
        }
    }));
    let out = runner_with(Arc::new(PollCountingGateway(polls.clone())), vec![])
        .run_streaming(
            TurnId::new("stop-polling"),
            Context::new(),
            opts,
            RunControl::new(token, None),
        )
        .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::Cancelled { .. }
        }
    ));
    assert!(out.context.blocks().is_empty());
    assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 1);
}
