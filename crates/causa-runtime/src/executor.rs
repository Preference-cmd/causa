//! Invocation-bound tool catalogs and controlled whole-batch processing.
//!
//! A binding fixes the advertised definitions and the actual dispatch targets.
//! Processing never looks up a shared catalog and never commits context material.

mod catalog;
pub(crate) mod dispatch;
mod error;
mod processing;

pub use catalog::{BoundTools, ToolExecutor, ToolExecutorOptions};
pub use error::{ToolCatalogError, ToolProcessingError, ToolProcessorPhase, ToolRegistryError};
