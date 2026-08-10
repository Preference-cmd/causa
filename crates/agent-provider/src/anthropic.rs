//! Anthropic provider adapter.
//!
//! Translates `AgentRequest` into an Anthropic messages payload via
//! `translation::request::to_anthropic_messages` and
//! `translation::tools::to_anthropic_tools`, runs it through a
//! `CompletionBackend`, and translates the response back via
//! `translation::response::from_anthropic_response`.
//!
//! Streaming is delegated to the backend and translated through
//! `translation::streaming::AnthropicStreamAccumulator`.

use std::sync::Arc;

use async_trait::async_trait;
use reimagine_agent_harness::{
    AgentProvider, AgentRequest, AgentResponse, AgentStream, ModelInfo, ProviderName,
};

use reimagine_ai_protocol::CompletionBackend;
use reimagine_ai_protocol::AnthropicMessagesConfig;

/// V1 adapter for Anthropic messages API.
pub struct AnthropicMessagesProvider {
    name: ProviderName,
    config: AnthropicMessagesConfig,
    backend: Arc<dyn CompletionBackend>,
}

impl std::fmt::Debug for AnthropicMessagesProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicMessagesProvider")
            .field("name", &self.name)
            .field("default_model", &self.config.default_model())
            .finish()
    }
}

impl AnthropicMessagesProvider {
    /// Construct with a custom backend (used by tests and by app-host
    /// to inject a real backend).
    pub fn with_backend(
        name: ProviderName,
        config: AnthropicMessagesConfig,
        backend: Arc<dyn CompletionBackend>,
    ) -> Self {
        Self {
            name,
            config,
            backend,
        }
    }

    /// Construct with the production `ReqwestBackend`. Unit tests
    /// inject a fake backend; the default suite does not require live
    /// provider credentials.
    pub fn new(name: ProviderName, config: AnthropicMessagesConfig) -> Self {
        Self {
            name: name.clone(),
            config: config.clone(),
            backend: crate::reqwest_backend::arc_real_anthropic_messages_backend(name, config),
        }
    }

    pub fn config(&self) -> &AnthropicMessagesConfig {
        &self.config
    }
}

#[async_trait]
impl AgentProvider for AnthropicMessagesProvider {
    fn name(&self) -> ProviderName {
        self.name.clone()
    }

    async fn complete(
        &self,
        request: AgentRequest,
    ) -> Result<AgentResponse, reimagine_agent_harness::ProviderError> {
        match self.backend.complete(request).await {
            Ok(resp) => Ok(resp),
            Err(err) => Err(err.to_provider_error(Some(self.name.clone()))),
        }
    }

    async fn stream(
        &self,
        request: AgentRequest,
    ) -> Result<Box<dyn AgentStream>, reimagine_agent_harness::ProviderError> {
        match self.backend.stream(request).await {
            Ok(stream) => Ok(stream),
            Err(err) => Err(err.to_provider_error(Some(self.name.clone()))),
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, reimagine_agent_harness::ProviderError> {
        match self.backend.list_models().await {
            Ok(models) => Ok(models),
            Err(err) => Err(err.to_provider_error(Some(self.name.clone()))),
        }
    }
}
