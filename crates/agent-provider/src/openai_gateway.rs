//! Kernel `ModelGateway` adapters for the OpenAI Chat Completions and
//! Responses APIs.
//!
//! Slice 3 wiring, structurally identical to
//! [`crate::anthropic_gateway`]: reqwest transport and the pure translation
//! in `reimagine_ai_protocol::translation::{openai_chat, openai_responses}`,
//! composed with the shared error mapping table and control-plane wiring
//! from [`crate::gateway_transport`]. Both adapters authenticate with
//! `Authorization: Bearer` and are read-only on the control plane;
//! `invoke` returns the complete [`ModelOutput`].

use async_trait::async_trait;
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

use crate::gateway_transport::{GatewayCore, finish_response, send_with_control};

/// OpenAI Chat Completions `ModelGateway` backed by reqwest.
#[derive(Debug, Clone)]
pub struct OpenAiChatCompletionsGateway {
    core: GatewayCore,
    api_key: String,
}

impl OpenAiChatCompletionsGateway {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            core: GatewayCore::new("https://api.openai.com/v1/chat/completions"),
            api_key: api_key.into(),
        }
    }

    /// Point at a non-default host (test doubles, gateways). The path
    /// `/v1/chat/completions` is appended to the given base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.core = self.core.with_base_url(base_url, "/v1/chat/completions");
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

#[async_trait]
impl ModelGateway for OpenAiChatCompletionsGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        // A frame the renderer rejects never reaches the wire.
        let body = render_openai_chat_messages(
            &request.frame,
            &request.tool_surface,
            &request.generation,
            &request.model,
        )?;

        let req = self.core.post().bearer_auth(&self.api_key).json(&body);

        let (status, text) = send_with_control(req, control).await?;
        finish_response(status, &text, "openai", parse_openai_chat_response)
    }
}

/// OpenAI Responses API `ModelGateway` backed by reqwest.
#[derive(Debug, Clone)]
pub struct OpenAiResponsesGateway {
    core: GatewayCore,
    api_key: String,
}

impl OpenAiResponsesGateway {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            core: GatewayCore::new("https://api.openai.com/v1/responses"),
            api_key: api_key.into(),
        }
    }

    /// Point at a non-default host (test doubles, gateways). The path
    /// `/v1/responses` is appended to the given base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.core = self.core.with_base_url(base_url, "/v1/responses");
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

#[async_trait]
impl ModelGateway for OpenAiResponsesGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        // A frame the renderer rejects never reaches the wire.
        let body = render_openai_responses_input(
            &request.frame,
            &request.tool_surface,
            &request.generation,
            &request.model,
        )?;

        let req = self.core.post().bearer_auth(&self.api_key).json(&body);

        let (status, text) = send_with_control(req, control).await?;
        finish_response(status, &text, "openai", parse_openai_responses_output)
    }
}
