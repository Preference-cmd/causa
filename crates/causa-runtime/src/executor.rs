//! Tool dispatch and catalog composition: panic isolation and call-deadline
//! handling are centralized here; output policy belongs to batch processors.
//! Also the tool-catalog composition point: static Rust tools
//! and dynamic sources (`DynamicToolSource`, e.g. MCP servers) merge into
//! one dispatch path; the driver only consumes the assembled surface.

use crate::composition::ToolBridge;
use causa_kernel::CallControl;
use causa_kernel::{
    ArtifactStore, BlockId, DynamicToolSource, Tool, ToolCallContext, ToolDefinition, ToolOutput,
    ToolOutputMeta, ToolResultPayload, ToolResultStatus, ToolSurface, Truncation,
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

/// Dynamic-source registry failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolRegistryError {
    /// A dynamic source with this id is already registered.
    #[error("dynamic source already registered: {0}")]
    DuplicateSource(String),
    /// No dynamic source registered under this id.
    #[error("no dynamic source registered: {0}")]
    UnknownSource(String),
}

/// A tool returned a result associated with a different declaration.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolExecutionError {
    /// The result's `call_block_id` did not match the dispatched declaration.
    #[error("tool result targets declaration {actual:?}, expected {expected:?}")]
    MismatchedDeclaration {
        /// Declaration identity passed to the dispatched tool.
        expected: BlockId,
        /// Declaration identity returned by the tool.
        actual: BlockId,
    },
}

/// The tool dispatcher: static tools plus registered dynamic sources
/// behind one execution path — parallel batch dispatch with per-call
/// panic isolation and a call-deadline backstop. Output retention is
/// handled by optional post-processors.
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

    /// Register a dynamic tool source. Its current listing
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

    /// Execute one prepared call with panic isolation and a call-deadline
    /// backstop. Output retention is handled by optional post-processors.
    pub async fn execute(
        &self,
        ctx: ToolCallContext,
        control: CallControl,
        store: Option<Arc<dyn ArtifactStore>>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        let span = tracing::info_span!(
            "agent.tool",
            tool_name = %ctx.input.tool_name,
            call_block_id = ?ctx.call_block_id
        );
        self.execute_inner(ctx, control, store)
            .instrument(span)
            .await
    }

    async fn execute_inner(
        &self,
        ctx: ToolCallContext,
        control: CallControl,
        store: Option<Arc<dyn ArtifactStore>>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        let tool: Option<Arc<dyn Tool>> = match self.tools.get(&ctx.input.tool_name) {
            Some(tool) => Some(tool.clone()),
            None => self
                .find_dynamic(&ctx.input.tool_name)
                .map(|(source, definition)| {
                    Arc::new(ToolBridge::new(source, definition)) as Arc<dyn Tool>
                }),
        };
        let Some(tool) = tool else {
            return Ok(ToolResultPayload {
                call_block_id: ctx.call_block_id,
                status: ToolResultStatus::Rejected,
                output: ToolOutput::new(serde_json::json!({
                    "error": format!("unknown tool: {}", ctx.input.tool_name)
                })),
                media: Vec::new(),
                notes: Vec::new(),
            });
        };

        let store_ref: Option<&dyn ArtifactStore> =
            store.as_deref().map(|value| value as &dyn ArtifactStore);
        let panic_ctx = ctx.clone();
        let deadline_ctx = ctx.clone();
        let fut = {
            let tool = tool.clone();
            let call_ctx = ctx.clone();
            let call_control = control.clone();
            std::panic::AssertUnwindSafe(async move {
                tool.execute_with_store(&call_ctx, &call_control, store_ref)
                    .await
            })
            .catch_unwind()
        };
        let result = match control.deadline() {
            Some(deadline) => match tokio::time::timeout_at(deadline.into(), fut).await {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Self::panicked_outcome(&panic_ctx),
                Err(_) => Self::deadline_backstop_outcome(&deadline_ctx),
            },
            None => match fut.await {
                Ok(result) => result,
                Err(_) => Self::panicked_outcome(&panic_ctx),
            },
        };
        if result.call_block_id != ctx.call_block_id {
            return Err(ToolExecutionError::MismatchedDeclaration {
                expected: ctx.call_block_id,
                actual: result.call_block_id,
            });
        }
        Ok(result)
    }

    fn panicked_outcome(ctx: &ToolCallContext) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(serde_json::json!({"error": "tool panicked"})),
            media: Vec::new(),
            notes: Vec::new(),
        }
    }

    fn deadline_backstop_outcome(ctx: &ToolCallContext) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
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
            media: Vec::new(),
            notes: Vec::new(),
        }
    }
}
