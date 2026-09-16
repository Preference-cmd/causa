//! Wire-protocol discriminator for provider adapters.

use serde::{Deserialize, Serialize};

/// Discriminator for the message protocol a provider entry speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Protocol {
    /// OpenAI Chat Completions request and response format.
    #[serde(rename = "openai_chat_completions")]
    OpenAiChatCompletions,
    /// Anthropic Messages request and response format.
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
    /// OpenAI Responses request and response format.
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
}

impl Protocol {
    /// The stable protocol name used by its serde representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiChatCompletions => "openai_chat_completions",
            Self::AnthropicMessages => "anthropic_messages",
            Self::OpenAiResponses => "openai_responses",
        }
    }
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
