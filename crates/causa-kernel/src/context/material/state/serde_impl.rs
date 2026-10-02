//! Direct material deserialization with context-local identity validation.

use super::Context;
use crate::context::block::ContextBlock;
use serde::Deserialize;

#[derive(Deserialize)]
struct ContextFields {
    blocks: Vec<ContextBlock>,
}

impl<'de> Deserialize<'de> for Context {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let fields = ContextFields::deserialize(deserializer)?;
        Self::from_blocks(fields.blocks).map_err(serde::de::Error::custom)
    }
}
