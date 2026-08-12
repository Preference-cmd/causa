//! Generic provider adapter delegating to a `CompletionBackend`.
//!
//! The three V1 adapters (`OpenAiChatCompletionsProvider`,
//! `AnthropicMessagesProvider`, `OpenAiResponsesProvider`) are type
//! aliases over [`BackendProvider`]; per-protocol behavior lives
//! entirely in the config type + backend construction via
//! [`ProviderConfig`], so the delegation glue is written once (AC-10).

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use reimagine_agent_harness::{
    AgentProvider, AgentRequest, AgentResponse, AgentStream, ModelInfo, ProviderName,
};
use reimagine_ai_protocol::CompletionBackend;

/// Configuration types usable with [`BackendProvider`]: construction
/// parameters plus the display fields hosts see in `Debug`.
pub trait ProviderConfig: Clone + Send + Sync + std::fmt::Debug {
    /// Build the production `ReqwestBackend` for this protocol.
    fn arc_real_backend(name: ProviderName, config: Self) -> Arc<dyn CompletionBackend>;
    /// Build the production backend rooted at `workspace_dir`, so
    /// `Url`-source file blocks resolve against the workspace.
    fn arc_real_backend_with_workspace_dir(
        name: ProviderName,
        config: Self,
        workspace_dir: PathBuf,
    ) -> Arc<dyn CompletionBackend>;
    /// `Debug` field: the base URL, or `None` when the provider uses a
    /// built-in default.
    fn base_url(&self) -> Option<&str>;
    /// `Debug` field: the default model.
    fn default_model(&self) -> &str;
}

/// V1 provider adapter over a [`CompletionBackend`].
///
/// All protocol logic lives on the protocol side (`ai-protocol`
/// translation + the typed config); this struct only delegates the
/// three `AgentProvider` methods and maps `ProviderAdapterError` into
/// the harness `ProviderError`.
pub struct BackendProvider<C> {
    name: ProviderName,
    config: C,
    backend: Arc<dyn CompletionBackend>,
}

impl<C: ProviderConfig> std::fmt::Debug for BackendProvider<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendProvider")
            .field("name", &self.name)
            .field("base_url", &self.config.base_url())
            .field("default_model", &self.config.default_model())
            .finish()
    }
}

impl<C: ProviderConfig> BackendProvider<C> {
    /// Construct with a custom backend (used by tests to inject a fake
    /// backend or a wiremock-backed HTTP client).
    pub fn with_backend(
        name: ProviderName,
        config: C,
        backend: Arc<dyn CompletionBackend>,
    ) -> Self {
        Self {
            name,
            config,
            backend,
        }
    }

    /// Construct with the production `ReqwestBackend`. Tests inject a
    /// fake backend or a local wiremock-backed HTTP client so the
    /// default suite does not require live provider credentials.
    pub fn new(name: ProviderName, config: C) -> Self {
        Self {
            name: name.clone(),
            config: config.clone(),
            backend: C::arc_real_backend(name, config),
        }
    }

    /// Rebuild the production `ReqwestBackend` rooted at
    /// `workspace_dir`, so `Url`-source file blocks resolve against the
    /// workspace. Only meaningful on [`Self::new`]-constructed
    /// providers; a custom backend injected via
    /// [`Self::with_backend`] is replaced.
    pub fn with_workspace_dir(mut self, workspace_dir: impl Into<PathBuf>) -> Self {
        self.backend = C::arc_real_backend_with_workspace_dir(
            self.name.clone(),
            self.config.clone(),
            workspace_dir.into(),
        );
        self
    }

    pub fn config(&self) -> &C {
        &self.config
    }
}

#[async_trait]
impl<C: ProviderConfig> AgentProvider for BackendProvider<C> {
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
