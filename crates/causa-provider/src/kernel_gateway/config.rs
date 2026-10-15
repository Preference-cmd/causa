//! Protocol rendering, full-response parsing and authentication.

use causa_kernel::{ModelInvokeError, ModelOutput, ModelRequest};
use causa_protocol::translation::anthropic::{parse_anthropic_response, render_anthropic_messages};
use causa_protocol::translation::media::MediaSet;
use causa_protocol::translation::openai_chat::{
    parse_openai_chat_response, render_openai_chat_messages,
};
use causa_protocol::translation::openai_responses::{
    parse_openai_responses_output, render_openai_responses_input,
};
use causa_protocol::translation::render_options::{
    AnthropicRenderOptions, ChatRenderOptions, ResponsesRenderOptions,
};
use serde_json::Value;

mod sealed {
    use super::*;

    /// Protocol-specific rendering, parsing, and auth decoration for a kernel gateway.
    ///
    /// The per-protocol quirks (endpoint, path, header shape, render/parse)
    /// are isolated here so the transport loop is written once. `media` is
    /// the resolution table the gateway built from its injected resolver;
    /// the config passes it through to the pure renderer.
    pub trait GatewayProtocol {
        /// Wire protocol used by the pure streaming decoder.
        const PROTOCOL: causa_protocol::Protocol;
        /// Render a kernel request and its resolved media into a wire request body.
        /// Rejection prevents the gateway from sending an HTTP request.
        fn render(
            &self,
            request: &ModelRequest,
            media: &MediaSet,
        ) -> Result<Value, ModelInvokeError>;
        /// Parse a successful wire response into the kernel's model output.
        fn parse(&self, value: &Value) -> Result<ModelOutput, ModelInvokeError>;
        /// Add protocol-specific authentication and version headers to a request.
        fn decorate_request(
            &self,
            builder: reqwest::RequestBuilder,
            api_key: &str,
        ) -> reqwest::RequestBuilder;
        /// Full endpoint URL used unless the host overrides it.
        const DEFAULT_ENDPOINT: &'static str;
        /// Protocol path appended by [`crate::KernelHttpGateway::with_base_url`].
        /// It has no version segment; the base URL supplies one.
        const PATH: &'static str;
    }
}

/// Protocol configuration selecting the wire format and transport parameters
/// for [`crate::KernelHttpGateway`].
///
/// This trait is sealed: it is implemented only by the three built-in
/// configurations ([`AnthropicGatewayConfig`], [`OpenAiChatGatewayConfig`], and
/// [`OpenAiResponsesGatewayConfig`]) and cannot be implemented outside this
/// crate. Callers select a protocol through [`crate::KernelHttpGateway`] or its
/// aliases ([`crate::AnthropicMessagesGateway`],
/// [`crate::OpenAiChatCompletionsGateway`], and
/// [`crate::OpenAiResponsesGateway`]).
///
/// ```compile_fail
/// use causa_provider::KernelGatewayConfig;
///
/// #[derive(Clone, Debug, Default)]
/// struct CustomConfig;
///
/// // Cannot be implemented outside causa_provider because the trait is sealed.
/// impl KernelGatewayConfig for CustomConfig {}
/// ```
pub trait KernelGatewayConfig:
    sealed::GatewayProtocol + Clone + Send + Sync + std::fmt::Debug + Default
{
}

/// Anthropic Messages gateway config: the `anthropic-version` header, the
/// render options of that protocol, and the extra `anthropic-beta` values.
/// Stored in `config: C` so the generic struct holds all protocol state
/// uniformly. The setters live on [`crate::KernelHttpGateway`].
#[derive(Debug, Clone)]
pub struct AnthropicGatewayConfig {
    pub(super) anthropic_version: String,
    pub(super) render_options: AnthropicRenderOptions,
    /// Extra `anthropic-beta` values, sent as one comma-separated header.
    pub(super) beta: Vec<String>,
}

impl Default for AnthropicGatewayConfig {
    fn default() -> Self {
        Self {
            anthropic_version: "2023-06-01".into(),
            render_options: AnthropicRenderOptions::default(),
            beta: Vec::new(),
        }
    }
}

impl sealed::GatewayProtocol for AnthropicGatewayConfig {
    const PROTOCOL: causa_protocol::Protocol = causa_protocol::Protocol::AnthropicMessages;
    fn render(&self, request: &ModelRequest, media: &MediaSet) -> Result<Value, ModelInvokeError> {
        render_anthropic_messages(
            &request.frame,
            media,
            &request.generation,
            &request.model,
            request.cache,
            &self.render_options,
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
        let builder = builder
            .header("x-api-key", api_key)
            .header("anthropic-version", &self.anthropic_version);
        // One comma-separated header: the values the host listed, then the
        // values the enabled render options need. No model table decides headers.
        let mut beta = self.beta.clone();
        for value in self.render_options.beta_values() {
            if !beta.iter().any(|existing| existing == value) {
                beta.push(value.to_owned());
            }
        }
        if beta.is_empty() {
            builder
        } else {
            builder.header("anthropic-beta", beta.join(","))
        }
    }

    const DEFAULT_ENDPOINT: &'static str = "https://api.anthropic.com/v1/messages";
    const PATH: &'static str = "/messages";
}

impl KernelGatewayConfig for AnthropicGatewayConfig {}

/// OpenAI Chat Completions gateway config: the Chat-only render options.
#[derive(Debug, Clone, Default)]
pub struct OpenAiChatGatewayConfig {
    pub(super) render_options: ChatRenderOptions,
}

impl sealed::GatewayProtocol for OpenAiChatGatewayConfig {
    const PROTOCOL: causa_protocol::Protocol = causa_protocol::Protocol::OpenAiChatCompletions;
    fn render(&self, request: &ModelRequest, media: &MediaSet) -> Result<Value, ModelInvokeError> {
        render_openai_chat_messages(
            &request.frame,
            media,
            &request.generation,
            &request.model,
            request.cache,
            &self.render_options,
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
    const PATH: &'static str = "/chat/completions";
}

impl KernelGatewayConfig for OpenAiChatGatewayConfig {}

/// OpenAI Responses gateway config: the Responses-only render options.
#[derive(Debug, Clone, Default)]
pub struct OpenAiResponsesGatewayConfig {
    pub(super) render_options: ResponsesRenderOptions,
}

impl sealed::GatewayProtocol for OpenAiResponsesGatewayConfig {
    const PROTOCOL: causa_protocol::Protocol = causa_protocol::Protocol::OpenAiResponses;
    fn render(&self, request: &ModelRequest, media: &MediaSet) -> Result<Value, ModelInvokeError> {
        render_openai_responses_input(
            &request.frame,
            media,
            &request.generation,
            &request.model,
            request.cache,
            &self.render_options,
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
    const PATH: &'static str = "/responses";
}

impl KernelGatewayConfig for OpenAiResponsesGatewayConfig {}
