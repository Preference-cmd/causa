//! Acquire the definitions and concrete targets for one invocation.

use super::{ToolCatalogError, registry::ToolExecutor};
use crate::composition::ToolBridge;
use causa_kernel::{CallControl, InvocationId, ProcessorData, Tool, ToolDescription, ToolSurface};
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
    pub(super) tools: Vec<ToolDescription>,
    pub(super) targets: HashMap<String, Arc<dyn Tool>>,
    pub(super) invocation_id: InvocationId,
    pub(super) control: CallControl,
    pub(super) data: Option<ProcessorData>,
}

impl ToolExecutor {
    /// Fixes an ordered catalog, the tools' declared hints and their concrete
    /// dispatch targets for one invocation. Static tools give
    /// [`Tool::definition`] and [`Tool::hints`]; dynamic sources give
    /// [`causa_kernel::DynamicToolSource::describe`]. Listing failures abort
    /// acquisition rather than using stale definitions or silently omitting a
    /// source.
    pub async fn bind(
        &self,
        invocation_id: InvocationId,
        control: CallControl,
    ) -> Result<BoundTools<'_>, ToolCatalogError> {
        control.check().map_err(ToolCatalogError::Control)?;
        let registrations = self.dynamic.read().expect("dynamic registry lock").clone();
        let mut tools = Vec::new();
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
            tools.push(ToolDescription::new(definition).with_hints(entry.tool.hints()));
        }
        for entry in registrations {
            control.check().map_err(ToolCatalogError::Control)?;
            for description in entry.describe(&control).await? {
                let name = description.definition.name.clone();
                if targets.contains_key(&name) {
                    return Err(ToolCatalogError::DuplicateTool { name });
                }
                let bridge = ToolBridge::new(entry.source.clone(), description.definition.clone())
                    .with_hints(description.hints.clone());
                targets.insert(name, Arc::new(bridge) as Arc<dyn Tool>);
                tools.push(description);
            }
        }
        control.check().map_err(ToolCatalogError::Control)?;
        let surface = ToolSurface::from_definitions(
            tools.iter().map(|tool| tool.definition.clone()).collect(),
        );
        Ok(BoundTools {
            executor: self,
            surface,
            tools,
            targets,
            invocation_id,
            control,
            data: None,
        })
    }
}

impl BoundTools<'_> {
    /// Borrows the ordered, immutable model-facing catalog fixed by this bind.
    pub fn surface(&self) -> &ToolSurface {
        &self.surface
    }

    /// Hands `data` to every processor of this binding's batch through
    /// [`causa_kernel::ProcessorContext::data`]. The runner sets it from
    /// [`crate::TurnRunOptions::processor_data`]; a caller that binds and
    /// processes on its own sets it here.
    pub fn with_processor_data(mut self, data: ProcessorData) -> Self {
        self.data = Some(data);
        self
    }
}
