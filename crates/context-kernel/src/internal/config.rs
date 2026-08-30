//! Run configuration axes — what to ask (`TurnInvocation`), when the loop
//! gives up (`TurnPolicy`), how tools execute (`ExecutionOptions`), and how
//! context is materialized (`FramePolicy`, canonical carrier in `budget`).
//! Deliberately four focused units instead of one universal context config.

use std::sync::Arc;
use std::time::Duration;

use crate::ports::budget::{FramePolicy, TokenCounter};
use crate::ports::gateway::{GenerationOptions, ModelInvokeErrorKind, ModelRef, ToolSurface};
use crate::ports::tool::{ArtifactStore, ToolOutputLimits};

/// Retry policy — driver-side scheduling, not a kernel fact. The retryability
/// judgment lives here because interpreting error kinds is loop policy.
#[derive(Debug, Clone, Default)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub retry_timeouts: bool,
}
impl RetryPolicy {
    /// Whether this policy schedules a further attempt for the error kind.
    pub fn allows(&self, kind: &ModelInvokeErrorKind) -> bool {
        match kind {
            ModelInvokeErrorKind::Transient => true,
            ModelInvokeErrorKind::TimedOut => self.retry_timeouts,
            _ => false,
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

/// Invocation options — what the model is asked this run.
#[derive(Debug, Clone)]
pub struct TurnInvocation {
    pub model: ModelRef,
    pub tool_surface: ToolSurface,
    pub generation: GenerationOptions,
}
impl Default for TurnInvocation {
    fn default() -> Self {
        Self {
            model: ModelRef::new("fake"),
            tool_surface: ToolSurface::empty(),
            generation: GenerationOptions::default(),
        }
    }
}

/// Turn policy — when the loop retries or gives up.
#[derive(Debug, Clone, Default)]
pub struct TurnPolicy {
    pub retry: RetryPolicy,
    pub limits: TurnLimits,
    /// Per-model-attempt budget; `None` = unbounded attempt.
    pub attempt_timeout: Option<Duration>,
}

/// Execution options — how tool calls run inside a round.
#[derive(Clone, Default)]
pub struct ExecutionOptions {
    pub tool_output_limits: ToolOutputLimits,
    pub artifact_store: Option<Arc<dyn ArtifactStore>>,
    /// Counter used for tool-output truncation estimation.
    pub token_counter: Option<Arc<dyn TokenCounter>>,
    /// Per-tool-call deadline; `None` = unbounded call (backstop still
    /// applies if the turn carries a deadline).
    pub call_timeout: Option<Duration>,
}
impl std::fmt::Debug for ExecutionOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionOptions")
            .field("tool_output_limits", &self.tool_output_limits)
            .field("artifact_store", &self.artifact_store.is_some())
            .field("token_counter", &self.token_counter.is_some())
            .field("call_timeout", &self.call_timeout)
            .finish()
    }
}

/// The reference driver's input: the four configuration axes. External
/// assemblers may build any of them independently; `Default` yields the
/// placeholder/noop wiring.
#[derive(Debug, Clone, Default)]
pub struct TurnRunOptions {
    pub invocation: TurnInvocation,
    pub policy: TurnPolicy,
    pub execution: ExecutionOptions,
    pub frame: FramePolicy,
}
