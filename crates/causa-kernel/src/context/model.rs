//! Model response vocabulary and pure material conversion.

mod conversion;
mod value;

pub use conversion::ModelBlockError;
pub use value::{ModelResponse, ModelStopReason, ToolCallDraft};
