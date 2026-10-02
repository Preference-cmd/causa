//! Configuration fixed for the lifetime of an executor.

use causa_kernel::{ArtifactStore, ToolBatchProcessor};
use std::sync::Arc;
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
