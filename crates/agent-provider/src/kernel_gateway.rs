//! Kernel `ModelGateway` adapters for Anthropic Messages and OpenAI Chat/Responses.
//!
//! Slice 3.5 F1: three `ModelGateway` adapters collapsed into the param-type
//! `KernelHttpGateway<C: KernelGatewayConfig>`. Each protocol's wire differences
//! (path, auth headers, render/parse) live in its `KernelGatewayConfig` impl;
//! the shared reqwest transport, error mapping, and control-plane wiring are
//! generic over `C` and reuse `crate::gateway_transport`.

use async_trait::async_trait;
use reimagine_ai_protocol::translation::anthropic::{
    parse_anthropic_response, render_anthropic_messages,
};
use reimagine_ai_protocol::translation::openai_chat::{
    parse_openai_chat_response, render_openai_chat_messages,
};
use reimagine_ai_protocol::translation::openai_responses::{
    parse_openai_responses_output, render_openai_responses_input,
};
use reimagine_context_kernel::{
    AttemptControl, ModelGateway, ModelInvokeError, ModelOutput, ModelRequest,
};
use reqwest::Client;
use serde_json::Value;

use crate::gateway_transport::{GatewayCore, finish_response, send_with_control};

/// Protocol-specific rendering, parsing, and auth decoration for a kernel gateway.
///
/// Mirrors the config-typed pattern of `BackendProvider<C: ProviderConfig>`:
/// the per-protocol quirks (endpoint, path, header shape, render/parse) are
/// isolated here so the transport loop is written once.
pub trait KernelGatewayConfig: Clone + Send + Sync + std::fmt::Debug + Default {
    fn render(&self, request: &ModelRequest) -> Result<Value, ModelInvokeError>;
    fn parse(&self, value: &Value) -> Result<ModelOutput, ModelInvokeError>;
    fn decorate_request(
        &self,
        builder: reqwest::RequestBuilder,
        api_key: &str,
    ) -> reqwest::RequestBuilder;
    const DEFAULT_ENDPOINT: &'static str;
    const PATH: &'static str;
    const PROVIDER: &'static str;
}

/// Anthropic Messages gateway config — the only variant with extra state
/// (`anthropic-version` header). Stored in `config: C` so the generic
/// struct holds all protocol state uniformly.
#[derive(Debug, Clone)]
pub struct AnthropicGatewayConfig {
    anthropic_version: String,
}

impl Default for AnthropicGatewayConfig {
    fn default() -> Self {
        Self {
            anthropic_version: "2023-06-01".into(),
        }
    }
}

impl KernelGatewayConfig for AnthropicGatewayConfig {
    fn render(&self, request: &ModelRequest) -> Result<Value, ModelInvokeError> {
        render_anthropic_messages(
            &request.frame,
            &request.tool_surface,
            &request.generation,
            &request.model,
        )
    }

    fn parse(&self, value: &Value) -> Result<ModelOutput, ModelInvokeError> {
        parse_anthropic_response(value)
    }

    fn decorate_request(
        &self,
        builder: reqwest::RequestBuilder,
        api_key: &str,
    ) -> reqwest::RequestBuilder {
        builder
            .header("x-api-key", api_key)
            .header("anthropic-version", &self.anthropic_version)
    }

    const DEFAULT_ENDPOINT: &'static str = "https://api.anthropic.com/v1/messages";
    const PATH: &'static str = "/v1/messages";
    const PROVIDER: &'static str = "anthropic";
}

/// OpenAI Chat Completions gateway config — stateless.
#[derive(Debug, Clone, Default)]
pub struct OpenAiChatGatewayConfig;

impl KernelGatewayConfig for OpenAiChatGatewayConfig {
    fn render(&self, request: &ModelRequest) -> Result<Value, ModelInvokeError> {
        render_openai_chat_messages(
            &request.frame,
            &request.tool_surface,
            &request.generation,
            &request.model,
        )
    }

    fn parse(&self, value: &Value) -> Result<ModelOutput, ModelInvokeError> {
        parse_openai_chat_response(value)
    }

    fn decorate_request(
        &self,
        builder: reqwest::RequestBuilder,
        api_key: &str,
    ) -> reqwest::RequestBuilder {
        builder.bearer_auth(api_key)
    }

    const DEFAULT_ENDPOINT: &'static str = "https://api.openai.com/v1/chat/completions";
    const PATH: &'static str = "/v1/chat/completions";
    const PROVIDER: &'static str = "openai";
}

/// OpenAI Responses gateway config — stateless.
#[derive(Debug, Clone, Default)]
pub struct OpenAiResponsesGatewayConfig;

impl KernelGatewayConfig for OpenAiResponsesGatewayConfig {
    fn render(&self, request: &ModelRequest) -> Result<Value, ModelInvokeError> {
        render_openai_responses_input(
            &request.frame,
            &request.tool_surface,
            &request.generation,
            &request.model,
        )
    }

    fn parse(&self, value: &Value) -> Result<ModelOutput, ModelInvokeError> {
        parse_openai_responses_output(value)
    }

    fn decorate_request(
        &self,
        builder: reqwest::RequestBuilder,
        api_key: &str,
    ) -> reqwest::RequestBuilder {
        builder.bearer_auth(api_key)
    }

    const DEFAULT_ENDPOINT: &'static str = "https://api.openai.com/v1/responses";
    const PATH: &'static str = "/v1/responses";
    const PROVIDER: &'static str = "openai";
}

/// Generic kernel gateway: reqwest transport + the pure translation in
/// `reimagine_ai_protocol::translation` composed with the shared error
/// mapping table and control-plane wiring from `crate::gateway_transport`.
#[derive(Debug, Clone)]
pub struct KernelHttpGateway<C: KernelGatewayConfig> {
    core: GatewayCore,
    api_key: String,
    config: C,
}

impl<C: KernelGatewayConfig> KernelHttpGateway<C> {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            core: GatewayCore::new(C::DEFAULT_ENDPOINT),
            api_key: api_key.into(),
            config: C::default(),
        }
    }

    /// Point at a non-default host (test doubles, gateways). `C::PATH`
    /// is appended to the given base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.core = self.core.with_base_url(base_url, C::PATH);
        self
    }

    /// Full endpoint override (host + path).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.core = self.core.with_endpoint(endpoint);
        self
    }

    pub fn with_http_client(mut self, http: Client) -> Self {
        self.core = self.core.with_http_client(http);
        self
    }
}

impl KernelHttpGateway<AnthropicGatewayConfig> {
    pub fn with_anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.config.anthropic_version = version.into();
        self
    }
}

#[async_trait]
impl<C: KernelGatewayConfig> ModelGateway for KernelHttpGateway<C> {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        // A frame the renderer rejects never reaches the wire.
        let body = self.config.render(request)?;
        let req = self
            .config
            .decorate_request(self.core.post(), &self.api_key)
            .json(&body);
        let (status, text) = send_with_control(req, control).await?;
        finish_response(status, &text, C::PROVIDER, |v| self.config.parse(v))
    }
}

/// Public type aliases preserve the pre-3.5 names.
pub type AnthropicMessagesGateway = KernelHttpGateway<AnthropicGatewayConfig>;
pub type OpenAiChatCompletionsGateway = KernelHttpGateway<OpenAiChatGatewayConfig>;
pub type OpenAiResponsesGateway = KernelHttpGateway<OpenAiResponsesGatewayConfig>;
