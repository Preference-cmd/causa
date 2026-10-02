//! Acquire the definitions and concrete targets for one invocation.

use super::{ToolCatalogError, registry::ToolExecutor};
use crate::composition::ToolBridge;
use causa_kernel::{CallControl, InvocationId, Tool, ToolSurface};
use std::collections::HashMap;
use std::sync::Arc;

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

impl BoundTools<'_> {
    /// Borrows the ordered, immutable model-facing catalog fixed by this bind.
    pub fn surface(&self) -> &ToolSurface {
        &self.surface
    }
}
