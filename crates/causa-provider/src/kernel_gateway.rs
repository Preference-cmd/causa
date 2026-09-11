//! Kernel `ModelGateway` adapters for Anthropic Messages and OpenAI Chat/Responses.
//!
//! Three `ModelGateway` adapters over the param-type
//! `KernelHttpGateway<C: KernelGatewayConfig>`. Each protocol's wire
//! differences (path, auth headers, render/parse) live in its
//! `KernelGatewayConfig` impl; the shared reqwest transport, error
//! mapping, and control-plane wiring are generic over `C` and reuse
//! `crate::gateway_transport`.

use async_trait::async_trait;
use causa_kernel::{
    AttemptControl, BlockContent, ContentPart, MediaRef, ModelGateway, ModelInvokeError,
    ModelOutput, ModelRequest,
};
use causa_protocol::translation::anthropic::{parse_anthropic_response, render_anthropic_messages};
use causa_protocol::translation::media::MediaSet;
use causa_protocol::translation::openai_chat::{
    parse_openai_chat_response, render_openai_chat_messages,
};
use causa_protocol::translation::openai_responses::{
    parse_openai_responses_output, render_openai_responses_input,
};
use reqwest::Client;
use serde_json::Value;
use std::sync::Arc;
use tracing::Instrument;

use crate::gateway_transport::{GatewayCore, finish_response, send_with_control};
use crate::media::MediaResolver;

/// Protocol-specific rendering, parsing, and auth decoration for a kernel gateway.
///
/// The per-protocol quirks (endpoint, path, header shape, render/parse)
/// are isolated here so the transport loop is written once. `media` is
/// the resolution table the gateway built from its injected resolver —
/// the config passes it through to the pure renderer.
pub trait KernelGatewayConfig: Clone + Send + Sync + std::fmt::Debug + Default {
    fn render(&self, request: &ModelRequest, media: &MediaSet) -> Result<Value, ModelInvokeError>;
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
    fn render(&self, request: &ModelRequest, media: &MediaSet) -> Result<Value, ModelInvokeError> {
        render_anthropic_messages(
            &request.frame,
            media,
            &request.tool_surface,
            &request.generation,
            &request.model,
            request.cache,
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
    fn render(&self, request: &ModelRequest, media: &MediaSet) -> Result<Value, ModelInvokeError> {
        render_openai_chat_messages(
            &request.frame,
            media,
            &request.tool_surface,
            &request.generation,
            &request.model,
            request.cache,
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
    fn render(&self, request: &ModelRequest, media: &MediaSet) -> Result<Value, ModelInvokeError> {
        render_openai_responses_input(
            &request.frame,
            media,
            &request.tool_surface,
            &request.generation,
            &request.model,
            request.cache,
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
/// `causa_protocol::translation` composed with the shared error
/// mapping table and control-plane wiring from `crate::gateway_transport`.
///
/// Media: an optional host-injected [`MediaResolver`] turns the frame's
/// fact-level references into inline payloads right before rendering —
/// batch and stream share one resolution pass. Without a resolver, every
/// reference degrades to its text placeholder.
pub struct KernelHttpGateway<C: KernelGatewayConfig> {
    core: GatewayCore,
    api_key: String,
    config: C,
    media_resolver: Option<Arc<dyn MediaResolver>>,
    /// Inline media ceiling, in decoded bytes (base64 is ~4/3 of this);
    /// payloads beyond it degrade to placeholders like resolver misses.
    /// Default: 5 MiB (the documented per-image cap of the strictest of
    /// the three wires).
    max_inline_media_bytes: usize,
}

impl<C: KernelGatewayConfig> std::fmt::Debug for KernelHttpGateway<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelHttpGateway")
            .field("provider", &C::PROVIDER)
            .field("path", &C::PATH)
            .field("media_resolver", &self.media_resolver.is_some())
            .field("max_inline_media_bytes", &self.max_inline_media_bytes)
            .finish_non_exhaustive()
    }
}

impl<C: KernelGatewayConfig> Clone for KernelHttpGateway<C> {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            api_key: self.api_key.clone(),
            config: self.config.clone(),
            media_resolver: self.media_resolver.clone(),
            max_inline_media_bytes: self.max_inline_media_bytes,
        }
    }
}

impl<C: KernelGatewayConfig> KernelHttpGateway<C> {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            core: GatewayCore::new(C::DEFAULT_ENDPOINT),
            api_key: api_key.into(),
            config: C::default(),
            media_resolver: None,
            max_inline_media_bytes: 5 * 1024 * 1024,
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

    /// Inject the media resolver: the host's asset table the gateway
    /// consults for every media reference in the frame.
    pub fn with_media_resolver(mut self, resolver: Arc<dyn MediaResolver>) -> Self {
        self.media_resolver = Some(resolver);
        self
    }

    /// Override the inline media ceiling, in decoded bytes.
    pub fn with_max_inline_media_bytes(mut self, max_bytes: usize) -> Self {
        self.max_inline_media_bytes = max_bytes;
        self
    }

    /// One resolution pass over the request's frame: every media
    /// reference (Parts blocks and tool results) the injected resolver
    /// serves within the inline ceiling enters the table. Deterministic
    /// — the same frame and asset state produce the same table.
    fn resolve_media(&self, request: &ModelRequest) -> MediaSet {
        let mut set = MediaSet::new();
        let Some(resolver) = &self.media_resolver else {
            return set;
        };
        let mut refs: Vec<MediaRef> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for block in &request.frame.model_context.blocks {
            let collect = |r: &MediaRef,
                           refs: &mut Vec<MediaRef>,
                           seen: &mut std::collections::HashSet<String>| {
                if seen.insert(r.reference.clone()) {
                    refs.push(r.clone());
                }
            };
            match &block.content {
                BlockContent::Parts(parts) => {
                    for part in parts {
                        if let ContentPart::Media(r) = part {
                            collect(r, &mut refs, &mut seen);
                        }
                    }
                }
                BlockContent::ToolResult(result) => {
                    for r in &result.media {
                        collect(r, &mut refs, &mut seen);
                    }
                }
                BlockContent::ToolCall(_) => {}
            }
        }
        for r in refs {
            if let Some(payload) = resolver.resolve(&r)
                && payload.data_base64.len() * 3 / 4 <= self.max_inline_media_bytes
            {
                set.insert(r.reference, payload);
            }
        }
        set
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
        // Observability baseline: ids and names only — never arguments,
        // message bodies, or API keys. The span is entered per poll via
        // `Instrument` so the boxed future stays `Send`.
        let span = tracing::debug_span!("agent.http", provider = C::PROVIDER, path = C::PATH);
        async {
            // A frame the renderer rejects never reaches the wire. Media
            // resolves first — the table feeds every renderer branch.
            let media = self.resolve_media(request);
            let body = self.config.render(request, &media)?;
            let req = self
                .config
                .decorate_request(self.core.post(), &self.api_key)
                .json(&body);
            let (status, text) = send_with_control(req, control).await?;
            finish_response(status, &text, C::PROVIDER, |v| self.config.parse(v))
        }
        .instrument(span)
        .await
    }
}

/// Public type aliases for the three protocols.
pub type AnthropicMessagesGateway = KernelHttpGateway<AnthropicGatewayConfig>;
pub type OpenAiChatCompletionsGateway = KernelHttpGateway<OpenAiChatGatewayConfig>;
pub type OpenAiResponsesGateway = KernelHttpGateway<OpenAiResponsesGatewayConfig>;
