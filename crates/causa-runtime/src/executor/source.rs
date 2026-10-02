//! Cache acquisition and publication owned by one source registration.

use super::{ToolCatalogError, dispatch::wait_for_stop};
use causa_kernel::{CallControl, DynamicToolSource, ToolDefinition};
use std::sync::{Arc, Mutex};

/// Cache ownership follows this exact registration, including re-registering
/// the same source Arc. No registry lookup is used to publish refresh results.
pub(super) struct DynamicEntry {
    pub(super) id: String,
    pub(super) source: Arc<dyn DynamicToolSource>,
    cache: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
    listing: Option<(u64, Vec<ToolDefinition>)>,
    latest_refresh: Option<Arc<()>>,
}

impl DynamicEntry {
    pub(super) fn new(id: String, source: Arc<dyn DynamicToolSource>) -> Self {
        Self {
            id,
            source,
            cache: Mutex::new(Cache::default()),
        }
    }

    pub(super) async fn list(
        &self,
        control: &CallControl,
    ) -> Result<Vec<ToolDefinition>, ToolCatalogError> {
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
