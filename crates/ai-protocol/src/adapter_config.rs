//! Typed adapter construction parameters.
//!
//! These shapes are the constructor arguments for the concrete provider
//! adapters: `base_url` (chat-completions protocols only), `api_key`,
//! and `default_model`. They are provider-domain values; the workspace
//! config *document* (`AgentProviderConfigDocument`) that carries them
//! on disk lives in `reimagine-app-host`, mirroring the Pi agent
//! provider / app layering.

use serde::{Deserialize, Serialize};

/// OpenAI-compatible provider config. `base_url` is required because
/// V1 supports arbitrary OpenAI-compatible endpoints, not just
/// `api.openai.com`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiChatCompletionsConfig {
    base_url: String,
    api_key: String,
    default_model: String,
}

impl OpenAiChatCompletionsConfig {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            default_model: default_model.into(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}

/// OpenAI Responses API provider config. `base_url` is required
/// because V1 supports arbitrary OpenAI-compatible endpoints, not just
/// `api.openai.com`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiResponsesConfig {
    base_url: String,
    api_key: String,
    default_model: String,
}

impl OpenAiResponsesConfig {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            default_model: default_model.into(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}

/// Anthropic provider config. V1 keeps this minimal: API key + default
/// model. `base_url` is optional; when `None`, the adapter defaults to
/// `https://api.anthropic.com`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicMessagesConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    base_url: Option<String>,
    api_key: String,
    default_model: String,
}

impl AnthropicMessagesConfig {
    pub fn new(api_key: impl Into<String>, default_model: impl Into<String>) -> Self {
        Self {
            base_url: None,
            api_key: api_key.into(),
            default_model: default_model.into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn default_model(&self) -> &str {
        &self.default_model
    }
}
