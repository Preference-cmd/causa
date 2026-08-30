//! Kernel `ModelGateway` adapter for the Anthropic Messages API.
//!
//! Slice 3 wiring: reqwest transport + the pure translation in
//! `reimagine_ai_protocol::translation::anthropic` (request rendering and
//! response parsing) + the shared error mapping table and control-plane
//! wiring from [`crate::gateway_transport`]. The adapter is read-only on
//! the control plane; `invoke` returns the complete [`ModelOutput`].

use async_trait::async_trait;
use reimagine_ai_protocol::translation::anthropic::{
    parse_anthropic_response, render_anthropic_messages,
};
use reimagine_context_kernel::{
    AttemptControl, ModelGateway, ModelInvokeError, ModelOutput, ModelRequest,
};
use reqwest::Client;

use crate::gateway_transport::{GatewayCore, finish_response, send_with_control};

/// Anthropic Messages `ModelGateway` backed by reqwest.
#[derive(Debug, Clone)]
pub struct AnthropicMessagesGateway {
    core: GatewayCore,
    api_key: String,
    anthropic_version: String,
}

impl AnthropicMessagesGateway {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            core: GatewayCore::new("https://api.anthropic.com/v1/messages"),
            api_key: api_key.into(),
            anthropic_version: "2023-06-01".into(),
        }
    }

    /// Point at a non-default host (test doubles, gateways). The Messages
    /// path `/v1/messages` is appended to the given base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.core = self.core.with_base_url(base_url, "/v1/messages");
        self
    }

    /// Full endpoint override (host + path).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.core = self.core.with_endpoint(endpoint);
        self
    }

    pub fn with_anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
        self
    }

    pub fn with_http_client(mut self, http: Client) -> Self {
        self.core = self.core.with_http_client(http);
        self
    }
}

#[async_trait]
impl ModelGateway for AnthropicMessagesGateway {
    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &AttemptControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        // A frame the renderer rejects never reaches the wire.
        let body = render_anthropic_messages(
            &request.frame,
            &request.tool_surface,
            &request.generation,
            &request.model,
        )?;

        let req = self
            .core
            .post()
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .json(&body);

        let (status, text) = send_with_control(req, control).await?;
        finish_response(status, &text, "anthropic", parse_anthropic_response)
    }
}
