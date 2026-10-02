use super::dispatch::wait_for_stop;
use super::error::{ToolCatalogError, ToolRegistryError};
use crate::composition::ToolBridge;
use causa_kernel::{
    ArtifactStore, CallControl, DynamicToolSource, InvocationId, Tool, ToolBatchProcessor,
    ToolDefinition, ToolSurface,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

/// Fixed processing configuration received when constructing an executor.
#[derive(Default)]
pub struct ToolExecutorOptions {
    /// Ordered processors invoked before tool dispatch.
    pub before: Vec<Arc<dyn ToolBatchProcessor>>,
    /// Ordered processors invoked after every call has a result.
    pub after: Vec<Arc<dyn ToolBatchProcessor>>,
    /// Optional local timeout, started when each tool call actually begins.
    pub call_timeout: Option<Duration>,
    /// Optional host storage passed to tools and dynamic-source bridges.
    pub artifact_store: Option<Arc<dyn ArtifactStore>>,
}

struct StaticEntry {
    name: String,
    tool: Arc<dyn Tool>,
}

/// Cache ownership follows this exact registration, including re-registering
/// the same source Arc. No registry lookup is used to publish refresh results.
struct DynamicEntry {
    id: String,
    source: Arc<dyn DynamicToolSource>,
    cache: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
    listing: Option<(u64, Vec<ToolDefinition>)>,
    latest_refresh: Option<Arc<()>>,
}

/// Assembles invocation catalogs and processes their newly declared batches.
///
/// Registrations affect future bindings; unregistering does not revoke an
/// existing binding. Static names come only from each tool's definition.
pub struct ToolExecutor {
    tools: Vec<StaticEntry>,
    dynamic: RwLock<Vec<Arc<DynamicEntry>>>,
    pub(super) options: ToolExecutorOptions,
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field(
                "tools",
                &self.tools.iter().map(|e| &e.name).collect::<Vec<_>>(),
            )
            .field("dynamic", &self.dynamic_ids())
            .finish()
    }
}

/// One invocation's fixed descriptions, concrete targets, and parent control.
///
/// The binding borrows the executor configuration without holding locks. It
/// cannot be cloned, and processing consumes it. It does not prevent a caller
/// from creating a new binding or independently replaying a declaration.
pub struct BoundTools<'a> {
    pub(super) executor: &'a ToolExecutor,
    pub(super) surface: ToolSurface,
    pub(super) targets: HashMap<String, Arc<dyn Tool>>,
    pub(super) invocation_id: InvocationId,
    pub(super) control: CallControl,
}

impl ToolExecutor {
    /// Constructs the sole executor configuration, rejecting static names that
    /// occur twice even when the tools or definitions are identical.
    pub fn new(
        tools: Vec<Arc<dyn Tool>>,
        options: ToolExecutorOptions,
    ) -> Result<Self, ToolRegistryError> {
        let mut names = std::collections::HashSet::new();
        let mut entries = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = tool.definition().name;
            if !names.insert(name.clone()) {
                return Err(ToolRegistryError::DuplicateTool { name });
            }
            entries.push(StaticEntry { name, tool });
        }
        Ok(Self {
            tools: entries,
            dynamic: RwLock::new(Vec::new()),
            options,
        })
    }

    /// Registers a source without listing it. Each registration receives a
    /// separate private cache, even if the source Arc was registered before.
    pub fn register_dynamic(
        &self,
        source: Arc<dyn DynamicToolSource>,
    ) -> Result<(), ToolRegistryError> {
        let id = source.id().to_owned();
        let mut entries = self.dynamic.write().expect("dynamic registry lock");
        if entries.iter().any(|entry| entry.id == id) {
            return Err(ToolRegistryError::DuplicateSource { source_id: id });
        }
        entries.push(Arc::new(DynamicEntry {
            id,
            source,
            cache: Mutex::new(Cache::default()),
        }));
        Ok(())
    }

    /// Removes a source and returns it. Existing bindings and in-flight binds
    /// retain the concrete registration they already acquired.
    pub fn unregister_dynamic(
        &self,
        id: &str,
    ) -> Result<Arc<dyn DynamicToolSource>, ToolRegistryError> {
        let mut entries = self.dynamic.write().expect("dynamic registry lock");
        let position = entries
            .iter()
            .position(|entry| entry.id == id)
            .ok_or_else(|| ToolRegistryError::UnknownSource {
                source_id: id.to_owned(),
            })?;
        Ok(entries.remove(position).source.clone())
    }

    /// Returns source identities in registration order.
    pub fn dynamic_ids(&self) -> Vec<String> {
        self.dynamic
            .read()
            .expect("dynamic registry lock")
            .iter()
            .map(|entry| entry.id.clone())
            .collect()
    }

    /// Fixes an ordered catalog and its concrete dispatch targets for one
    /// invocation. Listing failures abort acquisition rather than using stale
    /// definitions or silently omitting a source.
    pub async fn bind(
        &self,
        invocation_id: InvocationId,
        control: CallControl,
    ) -> Result<BoundTools<'_>, ToolCatalogError> {
        control.check().map_err(ToolCatalogError::Control)?;
        let registrations = self.dynamic.read().expect("dynamic registry lock").clone();
        let mut definitions = Vec::new();
        let mut targets = HashMap::new();
        for entry in &self.tools {
            control.check().map_err(ToolCatalogError::Control)?;
            let definition = entry.tool.definition();
            if definition.name != entry.name {
                return Err(ToolCatalogError::StaticNameChanged {
                    registered: entry.name.clone(),
                    actual: definition.name,
                });
            }
            targets.insert(entry.name.clone(), entry.tool.clone());
            definitions.push(definition);
        }
        for entry in registrations {
            control.check().map_err(ToolCatalogError::Control)?;
            for definition in entry.list(&control).await? {
                let name = definition.name.clone();
                if targets.contains_key(&name) {
                    return Err(ToolCatalogError::DuplicateTool { name });
                }
                targets.insert(
                    name,
                    Arc::new(ToolBridge::new(entry.source.clone(), definition.clone()))
                        as Arc<dyn Tool>,
                );
                definitions.push(definition);
            }
        }
        control.check().map_err(ToolCatalogError::Control)?;
        Ok(BoundTools {
            executor: self,
            surface: ToolSurface::from_definitions(definitions),
            targets,
            invocation_id,
            control,
        })
    }
}

impl DynamicEntry {
    async fn list(&self, control: &CallControl) -> Result<Vec<ToolDefinition>, ToolCatalogError> {
        let observed = self.source.version();
        let refresh = Arc::new(());
        {
            let mut cache = self.cache.lock().expect("source cache lock");
            if let Some((version, definitions)) = &cache.listing
                && *version == observed
            {
                return Ok(definitions.clone());
            }
            // A refresh failure must never expose the previous good listing.
            cache.listing = None;
            cache.latest_refresh = Some(refresh.clone());
        }
        let listing = tokio::select! {
            biased;
            cause = wait_for_stop(control) => return Err(ToolCatalogError::Control(cause)),
            result = self.source.list() => result,
        };
        // The source may synchronously cancel the parent during its final
        // poll. Apply stage control before selecting a listing error.
        control.check().map_err(ToolCatalogError::Control)?;
        let definitions = listing.map_err(|error| ToolCatalogError::Source {
            source_id: self.id.clone(),
            error,
        })?;
        let after = self.source.version();
        let mut cache = self.cache.lock().expect("source cache lock");
        if observed == after
            && cache
                .latest_refresh
                .as_ref()
                .is_some_and(|latest| Arc::ptr_eq(latest, &refresh))
        {
            cache.listing = Some((observed, definitions.clone()));
        }
        // A moving version still gives this invocation a concrete listing;
        // it is never mislabeled as a reusable current-version cache entry.
        Ok(definitions)
    }
}

impl BoundTools<'_> {
    /// Borrows the ordered, immutable model-facing catalog fixed by this bind.
    pub fn surface(&self) -> &ToolSurface {
        &self.surface
    }
}
