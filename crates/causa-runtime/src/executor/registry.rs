//! Static membership and ordered dynamic-source registration.

use super::{ToolRegistryError, options::ToolExecutorOptions, source::DynamicEntry};
use causa_kernel::{DynamicToolSource, Tool};
use std::sync::{Arc, RwLock};

pub(super) struct StaticEntry {
    pub(super) name: String,
    pub(super) tool: Arc<dyn Tool>,
}

/// Assembles invocation catalogs and processes their newly declared batches.
///
/// Registrations affect future bindings; unregistering does not revoke an
/// existing binding. Static names come only from each tool's definition.
pub struct ToolExecutor {
    pub(super) tools: Vec<StaticEntry>,
    pub(super) dynamic: RwLock<Vec<Arc<DynamicEntry>>>,
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
        entries.push(Arc::new(DynamicEntry::new(id, source)));
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
}
