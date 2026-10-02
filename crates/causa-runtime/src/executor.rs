//! Invocation-bound tool catalogs and controlled whole-batch processing.
//!
//! A binding fixes the advertised definitions and the actual dispatch targets.
//! Processing never looks up a shared catalog and never commits context material.

mod catalog;
mod collection;
pub(crate) mod dispatch;
mod error;
mod options;
mod processing;
mod registry;
mod source;

pub use catalog::BoundTools;
pub use error::{ToolCatalogError, ToolProcessingError, ToolProcessorPhase, ToolRegistryError};
pub use options::ToolExecutorOptions;
pub use registry::ToolExecutor;
