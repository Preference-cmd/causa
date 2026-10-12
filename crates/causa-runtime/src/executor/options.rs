//! Configuration fixed for the lifetime of an executor.

use causa_kernel::{ArtifactStore, ToolBatchProcessor};
use std::sync::Arc;
use std::time::Duration;

/// Fixed processing configuration received when constructing an executor.
#[derive(Default)]
#[non_exhaustive]
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

impl ToolExecutorOptions {
    /// Replaces the ordered processors invoked before tool dispatch.
    pub fn with_before(mut self, before: Vec<Arc<dyn ToolBatchProcessor>>) -> Self {
        self.before = before;
        self
    }

    /// Replaces the ordered processors invoked after every call has a result.
    pub fn with_after(mut self, after: Vec<Arc<dyn ToolBatchProcessor>>) -> Self {
        self.after = after;
        self
    }

    /// Sets the local timeout started when each tool call begins.
    pub fn with_call_timeout(mut self, call_timeout: Duration) -> Self {
        self.call_timeout = Some(call_timeout);
        self
    }

    /// Sets the host storage passed to tools and dynamic-source bridges.
    pub fn with_artifact_store(mut self, artifact_store: Arc<dyn ArtifactStore>) -> Self {
        self.artifact_store = Some(artifact_store);
        self
    }
}
