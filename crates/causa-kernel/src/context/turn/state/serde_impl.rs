//! Direct `TurnContext` deserialization with context-local ID validation.

use super::{TurnContext, TurnLifecycle};
use crate::context::block::ContextBlock;
use crate::context::ids::TurnId;
use ::serde::Deserialize;

#[derive(Deserialize)]
struct TurnContextFields {
    turn_id: TurnId,
    blocks: Vec<ContextBlock>,
    lifecycle: TurnLifecycle,
}

impl<'de> ::serde::Deserialize<'de> for TurnContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: ::serde::Deserializer<'de>,
    {
        let fields = TurnContextFields::deserialize(deserializer)?;
        TurnContext::validate_blocks(&fields.blocks).map_err(::serde::de::Error::custom)?;
        Ok(Self {
            turn_id: fields.turn_id,
            blocks: fields.blocks,
            lifecycle: fields.lifecycle,
        })
    }
}
