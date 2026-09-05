//! Tool batch dispatch — dedup-then-parallel execution with panic isolation,
//! call-deadline backstop, and token-limit truncation with artifact spill.
//! Also the tool-catalog composition point (Slice 10): static Rust tools
//! and dynamic sources (`DynamicToolSource`, e.g. MCP servers) merge into
//! one dispatch path; the driver only consumes the assembled surface.

use crate::composition::ToolBridge;
use causa_kernel::CallControl;
use causa_kernel::TokenCounter;
use causa_kernel::ToolCallPayload;
use causa_kernel::{
    ArtifactHint, ArtifactStore, DynamicToolSource, Tool, ToolCallContext, ToolDefinition,
    ToolExecutionOutcome, ToolOutputLimits, ToolSurface,
};
use causa_kernel::{
    ArtifactKind, ArtifactRef, ToolOutput, ToolOutputMeta, ToolResultPayload, ToolResultStatus,
    Truncation,
};
use futures_util::FutureExt;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::Instrument;

/// One registered dynamic source plus its executor-side listing cache.
/// The cache is keyed by the source's own change signal
/// ([`DynamicToolSource::version`]) so a stable catalog costs no
/// round-trip on surface assembly.
struct DynamicEntry {
    source: Arc<dyn DynamicToolSource>,
    cache: std::sync::Mutex<(u64, Vec<ToolDefinition>)>,
}

/// Dynamic-source registry failures (Slice 10).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolRegistryError {
    /// A dynamic source with this id is already registered.
    #[error("dynamic source already registered: {0}")]
    DuplicateSource(String),
    /// No dynamic source registered under this id.
    #[error("no dynamic source registered: {0}")]
    UnknownSource(String),
}

/// The tool dispatcher: static tools plus registered dynamic sources
/// behind one execution path — parallel batch dispatch with per-call
/// panic isolation, a call-deadline backstop, and token-limit truncation
/// with artifact spill.
pub struct ToolExecutor {
    tools: HashMap<String, Arc<dyn Tool>>,
    /// Registered dynamic sources; interior-mutable because the executor
    /// lives in an `Arc` shared with the driver.
    dynamic: std::sync::RwLock<Vec<DynamicEntry>>,
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field(
                "dynamic",
                &self
                    .dynamic
                    .read()
                    .expect("dynamic lock")
                    .iter()
                    .map(|e| e.source.id().to_string())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ToolExecutor {
    /// Build from static tools, keyed by each tool's `definition().name`.
    pub fn from_vec(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut map = HashMap::new();
        for t in tools {
            map.insert(t.definition().name.clone(), t);
        }
        Self {
            tools: map,
            dynamic: std::sync::RwLock::new(Vec::new()),
        }
    }

    /// Build from an explicit name → tool map; the keys are the caller's
    /// responsibility.
    pub fn from_map(map: HashMap<String, Arc<dyn Tool>>) -> Self {
        Self {
            tools: map,
            dynamic: std::sync::RwLock::new(Vec::new()),
        }
    }

    /// Register a dynamic tool source (Slice 10). Its current listing
    /// enters [`ToolExecutor::tool_surface`], and dispatch routes exactly
    /// the names that listing advertised (listing membership — naming is
    /// the source's own business); a name already present in the static
    /// map keeps winning.
    pub fn register_dynamic(
        &self,
        source: Arc<dyn DynamicToolSource>,
    ) -> Result<(), ToolRegistryError> {
        let mut dynamic = self.dynamic.write().expect("dynamic lock");
        if dynamic.iter().any(|e| e.source.id() == source.id()) {
            return Err(ToolRegistryError::DuplicateSource(source.id().to_string()));
        }
        dynamic.push(DynamicEntry {
            source,
            cache: std::sync::Mutex::new((u64::MAX, Vec::new())),
        });
        Ok(())
    }

    /// Remove a dynamic source; returns it for reconnect-and-reregister
    /// flows.
    pub fn unregister_dynamic(
        &self,
        id: &str,
    ) -> Result<Arc<dyn DynamicToolSource>, ToolRegistryError> {
        let mut dynamic = self.dynamic.write().expect("dynamic lock");
        let pos = dynamic
            .iter()
            .position(|e| e.source.id() == id)
            .ok_or_else(|| ToolRegistryError::UnknownSource(id.to_string()))?;
        Ok(dynamic.remove(pos).source)
    }

    /// Ids of the registered dynamic sources.
    pub fn dynamic_ids(&self) -> Vec<String> {
        self.dynamic
            .read()
            .expect("dynamic lock")
            .iter()
            .map(|e| e.source.id().to_string())
            .collect()
    }

    /// Assemble the model-facing tool surface: every static tool plus the
    /// current listing of every reachable dynamic source. A source that
    /// fails to list keeps serving its last good listing (with a warning);
    /// a source with no cached listing yet is skipped — either way the
    /// turn proceeds and the host can unregister/reconnect. Static names
    /// win collisions, exactly as dispatch does.
    ///
    /// Lock discipline: registry and cache guards are never held across
    /// an `await` — each source is snapshot before the `list()` call and
    /// re-locked to publish the result (clippy `await_holding_lock`).
    pub async fn tool_surface(&self) -> ToolSurface {
        let mut definitions: Vec<ToolDefinition> =
            self.tools.values().map(|t| t.definition()).collect();

        // Snapshot the dynamic registry so no `RwLockReadGuard` crosses an
        // await. Each entry is cloned as `(source, cached_version, cached_defs)`.
        let snapshot: Vec<(Arc<dyn DynamicToolSource>, u64, Vec<ToolDefinition>)> = {
            let dynamic = self.dynamic.read().expect("dynamic lock");
            dynamic
                .iter()
                .map(|entry| {
                    let cache = entry.cache.lock().expect("source cache lock");
                    (entry.source.clone(), cache.0, cache.1.clone())
                })
                .collect()
        };

        for (source, cached_version, cached_defs) in snapshot {
            let observed = source.version();
            let listing = if observed == cached_version && cached_version != u64::MAX {
                Some(cached_defs)
            } else {
                match source.list().await {
                    Ok(fresh) => {
                        // Publish the refreshed listing back into the cache
                        // (re-lock by id — the source is still registered).
                        let dynamic = self.dynamic.read().expect("dynamic lock");
                        if let Some(entry) = dynamic.iter().find(|e| e.source.id() == source.id()) {
                            let mut cache = entry.cache.lock().expect("source cache lock");
                            // Another concurrent surface assembly may have
                            // already refreshed; keep the newest.
                            if cache.0 != observed {
                                cache.0 = observed;
                                cache.1 = fresh.clone();
                            }
                        }
                        Some(fresh)
                    }
                    Err(e) => {
                        tracing::warn!(
                            source = source.id(),
                            error = %e,
                            "dynamic tool source failed to list; serving its last good listing"
                        );
                        (!cached_defs.is_empty()).then_some(cached_defs)
                    }
                }
            };
            if let Some(defs) = listing {
                // Surface must match dispatch: a dynamic name that collides
                // with a static tool is not advertised (the static tool
                // keeps winning there too).
                definitions.extend(
                    defs.into_iter()
                        .filter(|d| !self.tools.contains_key(&d.name)),
                );
            }
        }
        ToolSurface::from_definitions(definitions)
    }

    /// Find the dynamic source owning `tool_name` — routing is by listing
    /// membership: a source is dispatched exactly the names it advertised
    /// in [`ToolExecutor::tool_surface`] (the same cache), so naming stays
    /// the implementor's business and the model can only call what the
    /// surface showed. Returns the source and its advertised definition
    /// for the bridge. The registry/cache locks never cross an await.
    fn find_dynamic(
        &self,
        tool_name: &str,
    ) -> Option<(Arc<dyn DynamicToolSource>, ToolDefinition)> {
        let dynamic = self.dynamic.read().expect("dynamic lock");
        dynamic.iter().find_map(|entry| {
            let cache = entry.cache.lock().expect("source cache lock");
            cache
                .1
                .iter()
                .find(|d| d.name == tool_name)
                .map(|d| (entry.source.clone(), d.clone()))
        })
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
        // Observability baseline (Slice 6.6): one `agent.tool` span per
        // dispatch, name and id only. Entered per poll via `Instrument`,
        // so the future stays `Send`.
        let span = tracing::info_span!(
            "agent.tool",
            tool_name = %payload.tool_name,
            call_id = %payload.call_id.0
        );
        self.execute_with_limits_inner(payload, control, store, token_counter, global_limits)
            .instrument(span)
            .await
    }

    async fn execute_with_limits_inner(
        &self,
        payload: ToolCallPayload,
        control: CallControl,
        store: Option<Arc<dyn ArtifactStore>>,
        token_counter: Option<Arc<dyn TokenCounter>>,
        global_limits: ToolOutputLimits,
    ) -> ToolExecutionOutcome {
        // Route: the static map first, then dynamic sources by listing
        // membership. A dynamic call is wrapped in a [`ToolBridge`] and
        // runs the exact static path below — panic isolation, call-deadline
        // backstop, unknown-outcome policy (bridge default: `Stop`),
        // truncation — one code path, one error mapping (the bridge's).
        let tool: Option<Arc<dyn Tool>> = match self.tools.get(&payload.tool_name) {
            Some(tool) => Some(tool.clone()),
            None => self
                .find_dynamic(&payload.tool_name)
                .map(|(source, definition)| {
                    Arc::new(ToolBridge::new(source, definition)) as Arc<dyn Tool>
                }),
        };
        let Some(tool) = tool else {
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

        // Token-limit truncation (middle truncation + artifact spill).
        // Shape note: on truncation `content` is REPLACED by a JSON string
        // (head + notice + tail) regardless of the original `Value` shape —
        // an object or array observation becomes a string on the wire. The
        // `truncation: Middle` marker and the artifact ref are how consumers
        // detect this.
        //
        // Budget note: the retained head+tail is sized against the DECLARED
        // limit, not the original size — the notice and the JSON-string
        // wrapping are part of the measured output. Every candidate is
        // re-estimated with the wired estimator and the data budget shrinks
        // until the result fits, so the committed content re-estimates at or
        // under `max_tokens` — with one defined floor: when the limit is
        // smaller than the notice's own estimate, the notice alone is
        // emitted (nothing smaller is representable).
        let effective_limit = tool.output_limits().unwrap_or(global_limits).max_tokens;

        let estimate = |value: &serde_json::Value| -> usize {
            if let Some(counter) = &token_counter {
                counter.estimate_value(value)
            } else {
                // The driver's fallback opinion — single home in `defaults` (7.4).
                crate::defaults::placeholder_token_estimate_value(value)
            }
        };
        let estimated = estimate(&outcome.result.output.content);

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
                store_arc.persist(&data_bytes, hint).await.ok()
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

            // Initial data budget under the chars/4 heuristic with the
            // notice's own footprint reserved; the loop re-measures every
            // candidate with the wired estimator, so any counter converges.
            // The cut floors at budget/8 (geometric convergence) and 1
            // (termination); budget 0 leaves the notice as the floor.
            let mut data_budget = effective_limit
                .saturating_sub(estimate(&serde_json::Value::String(notice.clone())))
                .saturating_mul(4);
            let mut preview_value;
            loop {
                let (head, tail) = split_head_tail(&content_str, data_budget);
                let candidate = serde_json::Value::String(format!("{head}{notice}{tail}"));
                let candidate_tokens = estimate(&candidate);
                let fits = candidate_tokens <= effective_limit;
                preview_value = candidate;
                if fits || data_budget == 0 {
                    break;
                }
                let cut = (candidate_tokens - effective_limit)
                    .saturating_mul(4)
                    .max(data_budget / 8)
                    .max(1);
                data_budget = data_budget.saturating_sub(cut);
            }

            outcome.result.output.content = preview_value;
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

/// Head+tail preview split of `s` totalling at most `budget` bytes
/// (60% head / 40% tail), each cut at a char boundary and never
/// overlapping — the retained amount follows the budget, never the
/// original size.
fn split_head_tail(s: &str, budget: usize) -> (&str, &str) {
    let kept = budget.min(s.len());
    let head_end = floor_char_boundary(s, kept * 3 / 5);
    let tail_start = ceil_char_boundary(s, s.len() - (kept - head_end));
    (&s[..head_end], &s[tail_start..])
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
