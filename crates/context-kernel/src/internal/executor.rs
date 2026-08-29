//! Tool batch dispatch — dedup-then-parallel execution with panic isolation,
//! call-deadline backstop, and token-limit truncation with artifact spill.

use crate::block::ToolCallPayload;
use crate::budget::TokenCounter;
use crate::control::CallControl;
use crate::tool::{ArtifactHint, ArtifactStore, Tool};
use crate::tool_data::{
    ArtifactKind, ArtifactRef, ToolCallContext, ToolExecutionOutcome, ToolOutput, ToolOutputLimits,
    ToolOutputMeta, ToolResultPayload, ToolResultStatus, Truncation,
};
use std::collections::HashMap;
use std::sync::Arc;

pub struct ToolExecutor {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ToolExecutor {
    pub fn from_vec(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut map = HashMap::new();
        for t in tools {
            map.insert(t.definition().name.clone(), t);
        }
        Self { tools: map }
    }

    pub fn from_map(map: HashMap<String, Arc<dyn Tool>>) -> Self {
        Self { tools: map }
    }

    /// Execute a single ToolCallPayload with panic isolation, a call-deadline
    /// backstop, and token-limit truncation.
    pub async fn execute_with_limits(
        &self,
        payload: ToolCallPayload,
        control: CallControl,
        store: Option<Arc<dyn ArtifactStore>>,
        token_counter: Option<Arc<dyn TokenCounter>>,
        global_limits: ToolOutputLimits,
    ) -> ToolExecutionOutcome {
        let tool_opt = self.tools.get(&payload.tool_name).cloned();
        let Some(tool) = tool_opt else {
            return ToolExecutionOutcome::new(ToolResultPayload {
                call_id: payload.call_id.clone(),
                status: ToolResultStatus::Rejected,
                output: ToolOutput::new(
                    serde_json::json!({"error": format!("unknown tool: {}", payload.tool_name)}),
                ),
            });
        };

        let ctx = ToolCallContext {
            call_id: payload.call_id.clone(),
            tool_name: payload.tool_name.clone(),
            arguments: payload.arguments.clone(),
        };
        let store_ref: Option<&dyn ArtifactStore> =
            store.as_deref().map(|s| s as &dyn ArtifactStore);

        // Panic isolation (Task level) plus call-deadline backstop: a tool that
        // neither returns nor observes CallControl still yields a structured
        // UnknownOutcome instead of hanging the turn.
        use futures_util::FutureExt;
        let fut = {
            let tool = tool.clone();
            let control = control.clone();
            std::panic::AssertUnwindSafe(async move {
                tool.execute_with_store(&ctx, &control, store_ref).await
            })
            .catch_unwind()
        };
        let mut outcome = match control.deadline() {
            Some(deadline) => match tokio::time::timeout_at(deadline.into(), fut).await {
                Ok(Ok(o)) => o,
                Ok(Err(_)) => Self::panicked_outcome(&payload),
                Err(_) => Self::deadline_backstop_outcome(&payload),
            },
            None => match fut.await {
                Ok(o) => o,
                Err(_) => Self::panicked_outcome(&payload),
            },
        };

        // UnknownOutcome policy always comes from the trusted tool declaration,
        // never from the outcome the tool produced itself.
        if matches!(outcome.result.status, ToolResultStatus::UnknownOutcome) {
            outcome.policy = tool.unknown_outcome_policy();
        }

        // Token-limit truncation (middle truncation + artifact spill)
        let effective_limit = tool.output_limits().unwrap_or(global_limits).max_tokens;

        let estimated = if let Some(counter) = &token_counter {
            counter.estimate_value(&outcome.result.output.content)
        } else {
            serde_json::to_string(&outcome.result.output.content)
                .map(|s| s.len() / 4)
                .unwrap_or(0)
        };

        if estimated > effective_limit {
            let content_str = serde_json::to_string(&outcome.result.output.content)
                .unwrap_or_else(|_| outcome.result.output.content.to_string());
            let data_bytes = serde_json::to_vec(&outcome.result.output.content)
                .unwrap_or_else(|_| content_str.clone().into_bytes());

            let artifact: Option<ArtifactRef> = if let Some(store_arc) = &store {
                let hint = ArtifactHint {
                    tool_name: payload.tool_name.clone(),
                    call_id: payload.call_id.clone(),
                    kind: ArtifactKind::FullOutput,
                };
                match store_arc.persist(&data_bytes, hint).await {
                    Ok(r) => Some(r),
                    Err(_) => None,
                }
            } else {
                None
            };

            let notice = if let Some(ref a) = artifact {
                format!(
                    "\n...[truncated: original {} tokens, artifact:{}]...\n",
                    estimated, a.id
                )
            } else {
                format!(
                    "\n...[truncated: original {} tokens, showing head+tail]...\n",
                    estimated
                )
            };

            // head 60% + notice + tail 40% — slice at char boundaries safely
            let total = content_str.len();
            let head_target = total * 3 / 5;
            let tail_target = total - head_target;
            let head_end = floor_char_boundary(&content_str, head_target);
            let tail_start = ceil_char_boundary(&content_str, total.saturating_sub(tail_target));
            let head = &content_str[..head_end];
            let tail = &content_str[tail_start..];
            let preview = format!("{}{}{}", head, notice, tail);

            outcome.result.output.content = serde_json::Value::String(preview);
            outcome.result.output.truncation = Truncation::Middle;
            outcome.result.output.artifact = artifact;
            let prev_meta = outcome.result.output.meta.take();
            outcome.result.output.meta = Some(ToolOutputMeta {
                duration_ms: prev_meta.as_ref().and_then(|m| m.duration_ms),
                original_tokens: Some(estimated),
                extra: prev_meta.as_ref().and_then(|m| m.extra.clone()),
            });
        }

        outcome
    }

    fn panicked_outcome(payload: &ToolCallPayload) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: payload.call_id.clone(),
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(serde_json::json!({"error": "tool panicked"})),
        })
    }

    fn deadline_backstop_outcome(payload: &ToolCallPayload) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: payload.call_id.clone(),
            status: ToolResultStatus::UnknownOutcome,
            output: ToolOutput {
                content: serde_json::json!({"error": "tool did not return before call deadline"}),
                truncation: Truncation::None,
                meta: Some(ToolOutputMeta {
                    duration_ms: None,
                    original_tokens: None,
                    extra: Some(serde_json::json!({"reason": "call_deadline_backstop"})),
                }),
                artifact: None,
            },
        })
    }
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}
fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}
