//! Interruption tests keep the still-supported model failure, round-budget,
//! and cancellation outcomes observable at the turn boundary.

mod common;

use causa_kernel::ModelInvokeErrorKind;
use causa_runtime::{
    RunControl, TurnInterruption, TurnLimits, TurnPolicy, TurnResult, TurnRunOptions, TurnRunner,
};
use common::{
    GatedGateway, RecordingGateway, ctrl, ctx, endturn_output, runner_with, tooluse_output,
};
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn permanent_model_failure_is_returned_as_retry_exhausted() {
    let gateway = RecordingGateway::scripted(vec![Err(ModelInvokeErrorKind::Permanent)]);
    let runner = runner_with(gateway, vec![]);

    let outcome = runner
        .run(ctx("retry-exhausted"), TurnRunOptions::default(), ctrl())
        .await;

    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::RetryExhausted {
                last_kind: ModelInvokeErrorKind::Permanent,
                ..
            }
        }
    ));
    assert!(outcome.context.is_sealed());
}

#[tokio::test]
async fn model_round_limit_interrupts_before_an_extra_model_call() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("first", "missing", serde_json::json!({}))),
        Ok(endturn_output("not called")),
    ]);
    let runner = runner_with(gateway.clone(), vec![]);
    let options = TurnRunOptions {
        policy: TurnPolicy {
            limits: TurnLimits {
                max_model_rounds: 1,
                max_tool_calls: 2,
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let outcome = runner.run(ctx("round-limit"), options, ctrl()).await;

    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::MaxModelRounds { limit: 1 }
        }
    ));
    assert_eq!(gateway.recorded().len(), 1);
}

#[tokio::test]
async fn explicit_cancellation_interrupts_a_parked_model_call() {
    let gateway = GatedGateway::new("late", true);
    let runner = TurnRunner::new(
        gateway.clone(),
        Arc::new(causa_runtime::ToolExecutor::from_vec(vec![])),
    );
    let token = tokio_util::sync::CancellationToken::new();
    let run = tokio::spawn({
        let token = token.clone();
        async move {
            runner
                .run(
                    ctx("cancelled"),
                    TurnRunOptions::default(),
                    RunControl::new(token, None),
                )
                .await
        }
    });

    tokio::time::timeout(Duration::from_secs(5), gateway.wait_entered())
        .await
        .expect("model call entered");
    token.cancel();
    let outcome = tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("cancelled turn returned")
        .unwrap();

    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::ExplicitCancellation
        }
    ));
    assert!(outcome.context.is_sealed());
}
