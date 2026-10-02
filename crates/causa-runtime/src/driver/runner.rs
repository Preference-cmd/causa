//! Public run entry points and ownership of the supplied context.

use super::TurnOutcome;
use crate::{RunControl, ToolExecutor, TurnRunOptions};
use causa_kernel::{Context, ModelGateway, TurnId};
use std::sync::Arc;

/// Drives logical model rounds over caller-owned materials.
///
/// Only fresh declarations from each accepted model output are executed.
/// Preparation and observation are supplied per run; tool policy belongs to
/// the executor. Controlled stops return materials, including the complete
/// current batch when its results have not committed.
pub struct TurnRunner {
    pub(super) gateway: Arc<dyn ModelGateway>,
    pub(super) executor: Arc<ToolExecutor>,
}
impl TurnRunner {
    /// Assemble the two execution dependencies.
    pub fn new(gateway: Arc<dyn ModelGateway>, executor: Arc<ToolExecutor>) -> Self {
        Self { gateway, executor }
    }

    /// Run using complete logical gateway calls.
    pub async fn run(
        &self,
        turn_id: TurnId,
        context: Context,
        options: TurnRunOptions,
        control: RunControl,
    ) -> TurnOutcome {
        self.execute(turn_id, context, options, control, false)
            .await
    }
    /// Run using streaming gateway calls, accepting only complete `Done` output.
    pub async fn run_streaming(
        &self,
        turn_id: TurnId,
        context: Context,
        options: TurnRunOptions,
        control: RunControl,
    ) -> TurnOutcome {
        self.execute(turn_id, context, options, control, true).await
    }
    #[tracing::instrument(name = "agent.turn", skip_all, fields(turn_id = %turn_id.0, model = %options.model.0))]
    async fn execute(
        &self,
        turn_id: TurnId,
        mut context: Context,
        options: TurnRunOptions,
        control: RunControl,
        streaming: bool,
    ) -> TurnOutcome {
        let (result, uncommitted_tool_batch) = self
            .run_loop(&turn_id, &mut context, &options, &control, streaming)
            .await;
        TurnOutcome {
            turn_id,
            context,
            result,
            uncommitted_tool_batch,
        }
    }
}
