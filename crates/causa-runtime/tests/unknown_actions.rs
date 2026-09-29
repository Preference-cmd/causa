//! Unknown tool outcomes are recorded facts; runtime policy decides whether
//! the driver may continue after recording them.

mod common;

use async_trait::async_trait;
use causa_kernel::{
    BlockContent, CallControl, ProcessorContext, ProcessorError, Tool, ToolBatch,
    ToolBatchProcessor, ToolCallContext, ToolDefinition, ToolOutput, ToolResultPayload,
    ToolResultStatus,
};
use causa_runtime::{
    ToolExecutor, ToolProcessingChain, TurnInterruption, TurnPolicy, TurnResult, TurnRunOptions,
    TurnRunner, UnknownOutcomeConfig, UnknownOutcomePolicy,
};
use common::{RecordingGateway, ctrl, ctx, endturn_output, runner_with, tooluse_output};
use serde_json::json;
use std::sync::Arc;

struct UnknownTool;

#[async_trait]
impl Tool for UnknownTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "lookup".into(),
            description: "returns an externally unresolved result".into(),
            parameters: json!({"type": "object"}),
        }
    }

    async fn execute(&self, call: &ToolCallContext, _control: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: call.call_block_id,
            status: ToolResultStatus::UnknownOutcome,
            output: ToolOutput::new(json!({"state": "unknown"})),
            media: Vec::new(),
            notes: Vec::new(),
        }
    }
}

fn options(policy: UnknownOutcomePolicy) -> TurnRunOptions {
    let mut unknown_outcome = UnknownOutcomeConfig::default();
    unknown_outcome.overrides.insert("lookup".into(), policy);
    TurnRunOptions {
        policy: TurnPolicy {
            unknown_outcome,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[tokio::test]
async fn stop_policy_records_result_then_interrupts_without_another_round() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("check", "lookup", json!({}))),
        Ok(endturn_output("must not be requested")),
    ]);
    let runner = runner_with(gateway.clone(), vec![Arc::new(UnknownTool)]);

    let outcome = runner
        .run(
            ctx("stop-unknown"),
            options(UnknownOutcomePolicy::Stop),
            ctrl(),
        )
        .await;

    assert!(matches!(
        outcome.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::UnsafeUnknownOutcome { .. }
        }
    ));
    assert_eq!(gateway.recorded().len(), 1);
    let result = outcome
        .context
        .blocks()
        .iter()
        .find_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .expect("the unknown result is committed before policy stops the turn");
    assert_eq!(result.status, ToolResultStatus::UnknownOutcome);
}

#[tokio::test]
async fn continue_policy_allows_a_later_model_round() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("check", "lookup", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let runner = runner_with(gateway.clone(), vec![Arc::new(UnknownTool)]);

    let outcome = runner
        .run(
            ctx("continue-unknown"),
            options(UnknownOutcomePolicy::Continue),
            ctrl(),
        )
        .await;

    assert!(matches!(outcome.result, TurnResult::Completed { .. }));
    assert_eq!(gateway.recorded().len(), 2);
    assert_eq!(outcome.trace.rounds.len(), 2);
}

struct RenameBeforeDispatch;

#[async_trait]
impl ToolBatchProcessor for RenameBeforeDispatch {
    async fn process(
        &self,
        batch: &mut ToolBatch,
        _context: &ProcessorContext<'_>,
    ) -> Result<(), ProcessorError> {
        for entry in batch.calls_mut() {
            if let Some(input) = entry.input_mut() {
                input.tool_name = "lookup".into();
            }
        }
        Ok(())
    }
}

#[tokio::test]
async fn unknown_policy_uses_the_effective_preprocessed_tool_name() {
    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("call alias", "alias", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let processors = ToolProcessingChain::builder()
        .before(Arc::new(RenameBeforeDispatch))
        .build();
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(UnknownTool)])),
        processors,
    );

    let mut config = UnknownOutcomeConfig::default();
    config
        .overrides
        .insert("alias".into(), UnknownOutcomePolicy::Continue);
    let stopped = runner
        .run(
            ctx("effective-name-stop"),
            options_with_config(config),
            ctrl(),
        )
        .await;
    assert!(matches!(
        stopped.result,
        TurnResult::Interrupted {
            cause: TurnInterruption::UnsafeUnknownOutcome { .. }
        }
    ));

    let gateway = RecordingGateway::scripted(vec![
        Ok(tooluse_output("call alias", "alias", json!({}))),
        Ok(endturn_output("done")),
    ]);
    let processors = ToolProcessingChain::builder()
        .before(Arc::new(RenameBeforeDispatch))
        .build();
    let runner = TurnRunner::with_tool_processors(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(UnknownTool)])),
        processors,
    );
    let mut config = UnknownOutcomeConfig::default();
    config
        .overrides
        .insert("alias".into(), UnknownOutcomePolicy::Stop);
    config
        .overrides
        .insert("lookup".into(), UnknownOutcomePolicy::Continue);
    let continued = runner
        .run(
            ctx("effective-name-continue"),
            options_with_config(config),
            ctrl(),
        )
        .await;
    assert!(matches!(continued.result, TurnResult::Completed { .. }));
}

fn options_with_config(unknown_outcome: UnknownOutcomeConfig) -> TurnRunOptions {
    TurnRunOptions {
        policy: TurnPolicy {
            unknown_outcome,
            ..Default::default()
        },
        ..Default::default()
    }
}
