//! Concrete preparation consumers own input, projection and budget policy.
mod common;
use async_trait::async_trait;
use causa_kernel::*;
use causa_runtime::*;
use common::*;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

struct InputProjection {
    queue: Mutex<VecDeque<ContextBlock>>,
    calls: AtomicUsize,
    round_configs: Mutex<Vec<(InvocationId, String, ToolSurface, GenerationOptions)>>,
}
#[async_trait]
impl ContextPreparer for InputProjection {
    async fn prepare(
        &self,
        context: &mut Context,
        round: &RoundInfo<'_>,
        _: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        self.round_configs.lock().unwrap().push((
            round.invocation_id.clone(),
            round.model.0.clone(),
            round.tool_surface.clone(),
            round.generation.clone(),
        ));
        if let Some(block) = self.queue.lock().unwrap().pop_front() {
            context.edit().append([block]).commit().unwrap();
        }
        let mut frame = context.frame();
        frame
            .blocks
            .insert(0, text_block(&format!("temporary environment {n}")));
        Ok(frame)
    }
}
#[tokio::test]
async fn persistent_inputs_arrive_once_temporary_observations_refresh_and_actual_config_matches() {
    let preparer = Arc::new(InputProjection {
        queue: Mutex::new(vec![text_block("initial input"), text_block("running input")].into()),
        calls: AtomicUsize::new(0),
        round_configs: Mutex::new(vec![]),
    });
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("", "echo", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let mut opts = options();
    opts.preparer = Some(preparer.clone());
    opts.generation.max_tokens = Some(19);
    let out = runner_with(gateway.clone(), vec![Arc::new(EchoTool)])
        .run(TurnId::new("prepared"), Context::new(), opts, ctrl())
        .await;
    assert!(matches!(out.result, TurnResult::Completed { .. }));
    assert_eq!(preparer.calls.load(Ordering::SeqCst), 2);
    assert_eq!(out.context.blocks().len(), 5);
    let encoded = serde_json::to_string(&out.context).unwrap();
    assert!(!encoded.contains("temporary environment"));
    assert_eq!(encoded.matches("running input").count(), 1);
    let requests = gateway.recorded.lock().unwrap();
    let configs = preparer.round_configs.lock().unwrap();
    for (i, request) in requests.iter().enumerate() {
        assert_eq!(configs[i].0, request.invocation_id);
        assert_eq!(configs[i].1, request.model.0);
        assert_eq!(configs[i].2, request.tool_surface);
        assert_eq!(configs[i].3, request.generation);
        assert_eq!(request.tool_surface.definitions[0].name, "echo");
    }
    assert_ne!(
        requests[0].frame.blocks[0].id(),
        requests[1].frame.blocks[0].id()
    );
    assert_eq!(requests[1].frame.blocks.len(), 5);
}
struct EditingFailure {
    cancel: Option<CancellationToken>,
    park: bool,
}
#[async_trait]
impl ContextPreparer for EditingFailure {
    async fn prepare(
        &self,
        context: &mut Context,
        _: &RoundInfo<'_>,
        _: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        context
            .edit()
            .append([text_block("committed before failure")])
            .commit()
            .unwrap();
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        if self.park {
            std::future::pending::<()>().await;
        }
        Err(PrepareError {
            message: "preparation failed after edit".into(),
        })
    }
}
#[tokio::test]
async fn prepare_failure_and_control_stop_preserve_committed_edits_without_sending() {
    for mode in [0, 1, 2] {
        let token = CancellationToken::new();
        let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("never"))]);
        let mut opts = options();
        opts.preparer = Some(Arc::new(EditingFailure {
            cancel: (mode == 1).then(|| token.clone()),
            park: mode == 2,
        }));
        let control = RunControl::new(
            token,
            (mode == 2).then(|| Instant::now() + Duration::from_millis(5)),
        );
        let out = runner_with(gateway.clone(), vec![])
            .run(TurnId::new("failure"), input("before"), opts, control)
            .await;
        assert_eq!(out.context.blocks().len(), 2);
        assert!(gateway.recorded.lock().unwrap().is_empty());
        assert!(out.uncommitted_tool_batch.is_none());
        match out.result {
            TurnResult::Interrupted {
                cause: TurnInterruption::PrepareFailed { error, .. },
            } => {
                assert_eq!(mode, 0);
                assert_eq!(error.message, "preparation failed after edit");
            }
            TurnResult::Interrupted {
                cause: TurnInterruption::Cancelled { .. },
            } => assert_eq!(mode, 1),
            TurnResult::Interrupted {
                cause: TurnInterruption::DeadlineExceeded { .. },
            } => assert_eq!(mode, 2),
            other => panic!("unexpected {other:?}"),
        }
    }
}
#[tokio::test]
async fn pre_cancelled_run_never_binds_prepares_or_calls_model() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("never"))]);
    let p = Arc::new(InputProjection {
        queue: Mutex::new(VecDeque::new()),
        calls: AtomicUsize::new(0),
        round_configs: Mutex::new(vec![]),
    });
    let mut opts = options();
    opts.preparer = Some(p.clone());
    let token = CancellationToken::new();
    token.cancel();
    let out = runner_with(gateway.clone(), vec![])
        .run(
            TurnId::new("cancelled"),
            input("unchanged"),
            opts,
            RunControl::new(token, None),
        )
        .await;
    assert_eq!(out.context.blocks().len(), 1);
    assert_eq!(p.calls.load(Ordering::SeqCst), 0);
    assert!(gateway.recorded.lock().unwrap().is_empty());
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::Cancelled {
                invocation_id: None
            }
        }
    ));
}
// A concrete budget policy measures the actual frame and configuration. It is
// intentionally a consumer, not a published runtime counter/compaction policy.
struct BudgetedFrame {
    limit: usize,
}
fn request_cost(
    frame: &ContextFrame,
    surface: &ToolSurface,
    generation: &GenerationOptions,
) -> usize {
    serde_json::to_vec(&frame.blocks).unwrap().len()
        + serde_json::to_vec(surface).unwrap().len()
        + generation.max_tokens.unwrap_or(0) as usize
}
#[async_trait]
impl ContextPreparer for BudgetedFrame {
    async fn prepare(
        &self,
        c: &mut Context,
        r: &RoundInfo<'_>,
        _: &CallControl,
    ) -> Result<ContextFrame, PrepareError> {
        let mut frame = c.frame();
        while request_cost(&frame, r.tool_surface, r.generation) > self.limit {
            if frame.blocks.is_empty() {
                return Err(PrepareError {
                    message: "fixed tools/output budget exceeds request allowance".into(),
                });
            }
            frame.blocks.remove(0);
        }
        Ok(frame)
    }
}
#[tokio::test]
async fn concrete_budget_changes_actual_request_without_editing_retained_material() {
    let context = input(&"long input ".repeat(100));
    let retained = context.blocks().to_vec();
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("short"))]);
    let mut opts = options();
    opts.generation.max_tokens = Some(10);
    opts.preparer = Some(Arc::new(BudgetedFrame { limit: 200 }));
    let out = runner_with(gateway.clone(), vec![])
        .run(TurnId::new("budget"), context, opts, ctrl())
        .await;
    assert_eq!(&out.context.blocks()[..1], retained);
    let requests = gateway.recorded.lock().unwrap();
    assert!(
        request_cost(
            &requests[0].frame,
            &requests[0].tool_surface,
            &requests[0].generation
        ) <= 200
    );
    assert!(requests[0].frame.blocks.is_empty());
}
#[tokio::test]
async fn per_run_preparers_do_not_leak_input_between_runs_on_same_runner() {
    let runner = runner_with(
        RecordingGateway::scripted(vec![Ok(endturn_output("a")), Ok(endturn_output("b"))]),
        vec![],
    );
    for text in ["input A", "input B"] {
        let mut opts = options();
        opts.preparer = Some(Arc::new(InputProjection {
            queue: Mutex::new(vec![text_block(text)].into()),
            calls: AtomicUsize::new(0),
            round_configs: Mutex::new(vec![]),
        }));
        let out = runner
            .run(TurnId::new(text), Context::new(), opts, ctrl())
            .await;
        let wire = serde_json::to_string(&out.context).unwrap();
        assert!(wire.contains(text));
        assert!(!wire.contains(if text == "input A" {
            "input B"
        } else {
            "input A"
        }));
    }
}

struct DriftingTool(AtomicUsize);
#[async_trait]
impl Tool for DriftingTool {
    fn definition(&self) -> ToolDefinition {
        let mut definition = EchoTool.definition();
        if self.0.fetch_add(1, Ordering::SeqCst) > 0 {
            definition.name = "drifted".into();
        }
        definition
    }
    async fn execute(&self, _: &ToolCallContext, _: &CallControl) -> ToolResultPayload {
        panic!("catalog failure cannot execute")
    }
}
#[tokio::test]
async fn binding_failure_precedes_preparation_and_model_request() {
    let gateway = RecordingGateway::scripted(vec![Ok(endturn_output("never"))]);
    let preparer = Arc::new(InputProjection {
        queue: Mutex::new(vec![text_block("must stay queued")].into()),
        calls: AtomicUsize::new(0),
        round_configs: Mutex::new(vec![]),
    });
    let mut opts = options();
    opts.preparer = Some(preparer.clone());
    let out = runner_with(
        gateway.clone(),
        vec![Arc::new(DriftingTool(AtomicUsize::new(0)))],
    )
    .run(
        TurnId::new("catalog-failed"),
        input("unchanged"),
        opts,
        ctrl(),
    )
    .await;
    assert!(matches!(
        out.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ToolCatalogFailed {
                error: ToolCatalogError::StaticNameChanged { .. },
                ..
            }
        }
    ));
    assert_eq!(preparer.calls.load(Ordering::SeqCst), 0);
    assert_eq!(preparer.queue.lock().unwrap().len(), 1);
    assert_eq!(out.context.blocks().len(), 1);
    assert!(gateway.recorded.lock().unwrap().is_empty());
}
