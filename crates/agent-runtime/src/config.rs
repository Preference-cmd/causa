//! Run configuration axes — what to ask (`TurnInvocation`), when the loop
//! gives up (`TurnPolicy`), how tools execute (`ExecutionOptions`), and how
//! context is materialized (`FramePolicy`, canonical carrier in `budget`).
//! Deliberately four focused units instead of one universal context config.

use std::sync::Arc;
use std::time::Duration;

use reimagine_context_kernel::{ArtifactStore, ToolOutputLimits};
use reimagine_context_kernel::{FramePolicy, TokenCounter};
use reimagine_context_kernel::{GenerationOptions, ModelInvokeErrorKind, ModelRef, ToolSurface};
use reimagine_context_kernel::{StreamDelta, TurnInteraction};

/// Retry policy — driver-side scheduling, not a kernel fact. The retryability
/// judgment lives here because interpreting error kinds is loop policy.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Retries allowed per model round after the first failed attempt;
    /// `0` (the default) disables retrying entirely.
    pub max_retries: u32,
    /// Whether `TimedOut` model errors are retried (`Transient` errors
    /// always are). Default `false`.
    pub retry_timeouts: bool,
    /// Backoff base before the first retry, in milliseconds. `0` = no
    /// backoff (the pre-Slice-6 behavior: retry immediately).
    pub backoff_base_ms: u64,
    /// Backoff ceiling, in milliseconds.
    pub backoff_max_ms: u64,
}
impl Default for RetryPolicy {
    /// Conservative defaults: no automatic retries, but a 500ms→8s
    /// exponential backoff whenever retries are enabled.
    fn default() -> Self {
        Self {
            max_retries: 0,
            retry_timeouts: false,
            backoff_base_ms: 500,
            backoff_max_ms: 8_000,
        }
    }
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

    /// Wait before the (1-based) `next_attempt` fires:
    /// `min(base * 2^(next-2), max)` with ±20% jitter, shared by the
    /// `invoke` and `stream` retry paths. `ZERO` when the base is 0.
    pub fn backoff_delay(&self, next_attempt: u32) -> Duration {
        if self.backoff_base_ms == 0 {
            return Duration::ZERO;
        }
        let exp = next_attempt.saturating_sub(2).min(32);
        let raw = self.backoff_base_ms.saturating_mul(1u64 << exp);
        let capped = raw.min(self.backoff_max_ms);
        // ±20% jitter from wall-clock nanos — scheduling noise, not a
        // security primitive. 80..=120 percent.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0);
        let jitter_pct = 80 + nanos % 41;
        Duration::from_millis(capped.saturating_mul(jitter_pct) / 100)
    }
}

/// Per-turn loop bounds: model rounds and dispatched tool calls. Exceeding
/// either interrupts the turn (see [`crate::driver::TurnInterruption`]).
#[derive(Debug, Clone)]
pub struct TurnLimits {
    /// Ceiling on model rounds — checked as `round >= max_model_rounds`
    /// at the top of every round. Default: 10.
    pub max_model_rounds: u32,
    /// Ceiling on tool calls across the whole turn, counted at dispatch
    /// time (a batch paused pending approval counts when emitted).
    /// Default: 64.
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
    /// Which model the gateway invokes. Default: the `"fake"` placeholder.
    pub model: ModelRef,
    /// The tool surface advertised to the model this run. Default: empty.
    pub tool_surface: ToolSurface,
    /// Generation (sampling) parameters sent with every attempt. Default:
    /// `GenerationOptions::default()`.
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
    /// Retry schedule for failed model attempts — see [`RetryPolicy`].
    pub retry: RetryPolicy,
    /// Loop bounds for the turn — see [`TurnLimits`].
    pub limits: TurnLimits,
    /// Per-model-attempt budget; `None` = unbounded attempt.
    pub attempt_timeout: Option<Duration>,
}

/// Execution options — how tool calls run inside a round.
#[derive(Clone, Default)]
pub struct ExecutionOptions {
    /// Fallback per-output token limit for truncation; a trusted tool's
    /// own `output_limits` declaration overrides it.
    pub tool_output_limits: ToolOutputLimits,
    /// Where a truncated output's full bytes are spilled as an artifact;
    /// `None` = truncate without a retrievable original.
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

/// The reference driver's input: the four configuration axes plus the
/// single interaction seam. External assemblers may build any of them
/// independently; `Default` yields the placeholder/noop wiring.
#[derive(Clone)]
pub struct TurnRunOptions {
    /// What the model is asked this run — see [`TurnInvocation`].
    pub invocation: TurnInvocation,
    /// When the loop retries or gives up — see [`TurnPolicy`].
    pub policy: TurnPolicy,
    /// How tool calls execute — see [`ExecutionOptions`].
    pub execution: ExecutionOptions,
    /// Frame materialization policy for the bare-turn entries; inert for
    /// the conversation entries (lossless merged view).
    pub frame: FramePolicy,
    /// The one host↔driver interaction boundary for the turn. Default:
    /// [`NoopInteraction`] — observes nothing, decides nothing.
    pub interaction: Arc<dyn TurnInteraction>,
}
impl Default for TurnRunOptions {
    fn default() -> Self {
        Self {
            invocation: TurnInvocation::default(),
            policy: TurnPolicy::default(),
            execution: ExecutionOptions::default(),
            frame: FramePolicy::default(),
            interaction: Arc::new(NoopInteraction),
        }
    }
}
impl std::fmt::Debug for TurnRunOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnRunOptions")
            .field("invocation", &self.invocation)
            .field("policy", &self.policy)
            .field("execution", &self.execution)
            .field("frame", &self.frame)
            .finish_non_exhaustive()
    }
}

/// The interaction-less default: observes no deltas, makes no decisions.
/// The literal absence of behavior, not a policy.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopInteraction;

#[async_trait::async_trait]
impl TurnInteraction for NoopInteraction {
    async fn on_delta(&self, _round_id: reimagine_context_kernel::RoundId, _delta: &StreamDelta) {}
}
