//! Fact vocabulary of the model door — what the fact machine validates and
//! records. The gateway envelope (`ModelOutput`), the transport error, and
//! request parameters (`ModelRef`, `GenerationOptions`, `ToolSurface`) are
//! port vocabulary and live in `crate::ports::gateway`.

use serde::{Deserialize, Serialize};

use crate::context::block::{ContentPart, TextPayload, WireMeta};

/// A tool call as the model proposed it, before the caller assigns its declaration `BlockId`. Exactly what the gateway parsed off the wire.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCallDraft {
    /// Name of the tool the model asked to invoke.
    pub tool_name: String,
    /// Tool arguments as a JSON object; validated by the kernel before commit.
    pub arguments: serde_json::Value,
    /// Provider-issued identifier for this tool call, if the upstream model
    /// API assigned one. The kernel records it verbatim (on the envelope
    /// `BlockMeta`) and pairs tool results by the declaration `BlockId`, never by this field.
    /// `ToolCallId` remains a borrowed content key for concurrent deduplication.
    #[serde(default)]
    pub provider_call_id: Option<String>,
}
impl ToolCallDraft {
    /// A draft call with no provider call id.
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
            provider_call_id: None,
        }
    }

    /// Sets the provider-issued identifier for this tool call.
    pub fn with_provider_call_id(mut self, provider_call_id: impl Into<String>) -> Self {
        self.provider_call_id = Some(provider_call_id.into());
        self
    }
}

/// Why the model stopped producing output for this round, as reported by
/// the gateway. The kernel validates only the EndTurn/ToolUse structural
/// pairing; interpreting terminal reasons is driver policy.
///
/// Exhaustive on purpose: the runtime decides what a turn does next from this value, so a new one is a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStopReason {
    /// The model ended its turn without requesting tools; `tool_calls` must
    /// be empty.
    EndTurn,
    /// The model requested tool invocations; `tool_calls` must be non-empty.
    ToolUse,
    /// Output was cut off at the token limit.
    MaxTokens,
    /// The model (or a provider content filter) refused to produce output.
    Refusal,
    /// The provider stopped because the context window was exhausted. The
    /// output was not committed; the runner reports it as an interruption.
    ContextWindowExceeded,
    /// The provider reported a stop value this kernel has no meaning for. The
    /// output was not committed and the wire value is preserved on
    /// `ModelOutput::raw_stop_reason`.
    Unrecognized,
}

/// The model door's fact shape: what the model produced, as an ordered list
/// of items. One item becomes one block on commit, in item order. Named for
/// the participant, not for any wire role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// What the model said and asked, in the order the provider returned it.
    pub items: Vec<ModelItem>,
}

impl ModelResponse {
    /// The tool calls the model asked for, in item order; ids are assigned
    /// at commit.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCallDraft> {
        self.items.iter().filter_map(|item| match &item.content {
            ModelItemContent::ToolCall(call) => Some(call),
            ModelItemContent::Parts(_) => None,
        })
    }

    /// Every text part of every readable item, concatenated in order with no
    /// separator. Media and sealed parts contribute nothing.
    pub fn text(&self) -> String {
        let mut text = String::new();
        for item in &self.items {
            if let ModelItemContent::Parts(parts) = &item.content {
                for part in parts {
                    if let ContentPart::Text(t) = part {
                        text.push_str(&t.0);
                    }
                }
            }
        }
        text
    }
}

/// One item the model produced: its content plus the wire fields that
/// content does not hold.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelItem {
    /// What the item says or asks.
    pub content: ModelItemContent,
    /// What the provider's item carried beyond `content`; copied to
    /// [`BlockMeta::wire`](crate::BlockMeta::wire) on commit. The kernel
    /// never reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wire: Option<WireMeta>,
}
impl ModelItem {
    /// An item with no wire residue.
    pub fn new(content: ModelItemContent) -> Self {
        Self {
            content,
            wire: None,
        }
    }
}

impl ModelItem {
    /// An item of one text part and no wire fields.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: ModelItemContent::Parts(vec![ContentPart::Text(TextPayload::new(text))]),
            wire: None,
        }
    }

    /// A tool-call item with no wire fields.
    pub fn tool_call(call: ToolCallDraft) -> Self {
        Self {
            content: ModelItemContent::ToolCall(call),
            wire: None,
        }
    }
}

/// The content of a [`ModelItem`].
///
/// Exhaustive on purpose: conversion to blocks handles every shape, so a new one is a breaking change.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelItemContent {
    /// Ordered content parts (text and sealed items) that commit as one
    /// `Parts` block.
    Parts(Vec<ContentPart>),
    /// A tool call that commits as one `ToolCall` block.
    ToolCall(ToolCallDraft),
}
