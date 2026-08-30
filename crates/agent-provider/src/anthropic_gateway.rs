//! Kernel `ModelGateway` adapter for the Anthropic Messages API.
//!
//! Slice 3 wiring: reqwest transport + the pure translation in
//! `reimagine_ai_protocol::translation::anthropic` (request rendering and
//! response parsing) + the Slice 3 error mapping table
//! (`ModelInvokeErrorKind` is the closed vocabulary the kernel's
//! `RetryPolicy` interprets):
//!
//! | Source                                  | Kind            |
//! |-----------------------------------------|-----------------|
//! | reqwest connect / network / body error  | `Transient`     |
//! | request timeout (incl. attempt deadline)| `TimedOut`      |
//! | cancellation token fired                | `Cancelled`     |
//! | HTTP 408 / 429 / 5xx                    | `Transient`     |
//! | HTTP 400 / 422                          | `InvalidRequest`|
//! | HTTP 401 / 403 / 404                    | `Permanent`     |
//! | 2xx body not parseable                  | `Permanent`     |
//! | anything else                           | `Permanent`     |
//!
//! The adapter is **read-only** on the control plane: `AttemptControl::deadline`
//! becomes the request timeout and the shared cancellation token is raced
//! against in-flight HTTP via `select!`. It never constructs control planes —
//! those are driver-owned (`AttemptControl::new` stays `pub(crate)`).
//!
//! `invoke` returns the complete [`ModelOutput`]; SSE streaming is an
//! adapter-internal transport detail deferred out of Slice 3.

use async_trait::async_trait;
use reimagine_ai_protocol::translation::anthropic::{
    parse_anthropic_response, render_anthropic_messages,
};
use reimagine_context_kernel::{
    AttemptControl, ModelGateway, ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelRequest,
};
use reqwest::{Client, StatusCode};
use serde_json::Value;

/// Anthropic Messages `ModelGateway` backed by reqwest.
#[derive(Debug, Clone)]
pub struct AnthropicMessagesGateway {
    http: Client,
    endpoint: String,
    api_key: String,
    anthropic_version: String,
}

impl AnthropicMessagesGateway {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: Client::new(),
            endpoint: "https://api.anthropic.com/v1/messages".into(),
            api_key: api_key.into(),
            anthropic_version: "2023-06-01".into(),
        }
    }

    /// Point at a non-default host (test doubles, gateways). The Messages
    /// path `/v1/messages` is appended to the given base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.endpoint = format!("{}/v1/messages", base_url.into().trim_end_matches('/'));
        self
    }

    /// Full endpoint override (host + path).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn with_anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
        self
    }

    pub fn with_http_client(mut self, http: Client) -> Self {
        self.http = http;
        self
    }

    fn map_response(
        &self,
        status: StatusCode,
        text: &str,
    ) -> Result<ModelOutput, ModelInvokeError> {
        if !status.is_success() {
            return Err(map_status_error(status, text));
        }
        let value: Value = serde_json::from_str(text).map_err(|e| {
            ModelInvokeError::new(
                ModelInvokeErrorKind::Permanent,
                format!("2xx response body is not valid JSON: {e}"),
            )
        })?;
        // Body-schema failures already surface as Permanent from the parser
        // (§4 row: "2xx 响应体解析失败 → Permanent").
        parse_anthropic_response(&value)
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

        if control.is_cancelled() {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Cancelled,
                "cancelled before send",
            ));
        }

        let mut req = self
            .http
            .post(&self.endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .json(&body);

        // Control-plane wiring: the attempt deadline becomes the request
        // timeout; the shared token is raced against the in-flight work.
        if let Some(deadline) = control.deadline() {
            req = req.timeout(deadline.saturating_duration_since(std::time::Instant::now()));
        }

        let http = async {
            let response = req.send().await?;
            let status = response.status();
            let text = response.text().await?;
            Ok::<_, reqwest::Error>((status, text))
        };

        tokio::select! {
            outcome = http => {
                let (status, text) = outcome.map_err(map_transport_error)?;
                self.map_response(status, &text)
            }
            _ = control.cancellation_token().cancelled() => Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Cancelled,
                "cancelled while awaiting response",
            )),
        }
    }
}

fn map_transport_error(e: reqwest::Error) -> ModelInvokeError {
    let kind = if e.is_timeout() {
        ModelInvokeErrorKind::TimedOut
    } else if e.is_connect() || e.is_body() {
        ModelInvokeErrorKind::Transient
    } else {
        ModelInvokeErrorKind::Permanent
    };
    ModelInvokeError::new(kind, format!("anthropic transport: {e}"))
}

fn map_status_error(status: StatusCode, body: &str) -> ModelInvokeError {
    let kind = match status.as_u16() {
        408 | 429 | 500..=599 => ModelInvokeErrorKind::Transient,
        400 | 422 => ModelInvokeErrorKind::InvalidRequest,
        // 401 / 403 / 404 and everything else (§4 "其余" row).
        _ => ModelInvokeErrorKind::Permanent,
    };
    ModelInvokeError::new(
        kind,
        format!(
            "anthropic error {}: {}",
            status.as_u16(),
            error_snippet(body)
        ),
    )
}

/// Prefer the provider's own error message; fall back to a truncated raw
/// body when the payload is not the documented error envelope.
fn error_snippet(body: &str) -> String {
    let from_envelope = serde_json::from_str::<Value>(body).ok().and_then(|v| {
        let message = v.get("error")?.get("message")?.as_str()?;
        Some(message.to_string())
    });
    from_envelope.unwrap_or_else(|| {
        let mut snippet: String = body.chars().take(200).collect();
        if body.chars().count() > 200 {
            snippet.push('…');
        }
        snippet
    })
}
