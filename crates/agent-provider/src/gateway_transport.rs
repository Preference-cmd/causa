//! Shared transport plumbing for kernel `ModelGateway` adapters (Slice 3).
//!
//! Owns the two adapter-wide concerns the per-protocol gateways must not
//! duplicate: control-plane wiring (`send_with_control`) and the Slice 3
//! error mapping table (`ModelInvokeErrorKind` is the closed vocabulary the
//! kernel's `RetryPolicy` interprets):
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
//! against in-flight HTTP via `select!`. Gateways never construct control
//! planes — those are driver-owned (`AttemptControl::new` stays
//! `pub(crate)`).

use reimagine_context_kernel::{
    AttemptControl, ModelInvokeError, ModelInvokeErrorKind, ModelOutput,
};
use reqwest::{Client, StatusCode};
use serde_json::Value;

/// Endpoint + HTTP client bundle shared by the per-protocol gateways.
#[derive(Debug, Clone)]
pub(crate) struct GatewayCore {
    pub http: Client,
    pub endpoint: String,
}

impl GatewayCore {
    pub fn new(default_endpoint: &str) -> Self {
        Self {
            http: Client::new(),
            endpoint: default_endpoint.into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>, path: &str) -> Self {
        self.endpoint = format!("{}{path}", base_url.into().trim_end_matches('/'));
        self
    }

    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    pub fn with_http_client(mut self, http: Client) -> Self {
        self.http = http;
        self
    }

    pub fn post(&self) -> reqwest::RequestBuilder {
        self.http.post(&self.endpoint)
    }
}

/// Send a prepared POST under the attempt's control plane: pre-cancel
/// short-circuit, deadline-as-timeout, token raced against the in-flight
/// work. Returns the raw status + body for protocol-specific parsing.
pub(crate) async fn send_with_control(
    req: reqwest::RequestBuilder,
    control: &AttemptControl,
) -> Result<(StatusCode, String), ModelInvokeError> {
    if control.is_cancelled() {
        return Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Cancelled,
            "cancelled before send",
        ));
    }

    let mut req = req;
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
        outcome = http => outcome.map_err(map_transport_error),
        _ = control.cancellation_token().cancelled() => Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Cancelled,
            "cancelled while awaiting response",
        )),
    }
}

/// Turn a raw response into a [`ModelOutput`]: non-2xx through the error
/// table, 2xx JSON-syntax failures as `Permanent`, then the
/// protocol-specific body parser (whose own schema failures already
/// surface as `Permanent` — §4 row "2xx 响应体解析失败 → Permanent").
pub(crate) fn finish_response<F>(
    status: StatusCode,
    text: &str,
    provider: &str,
    parse: F,
) -> Result<ModelOutput, ModelInvokeError>
where
    F: FnOnce(&Value) -> Result<ModelOutput, ModelInvokeError>,
{
    if !status.is_success() {
        return Err(map_status_error(status, text, provider));
    }
    let value: Value = serde_json::from_str(text).map_err(|e| {
        ModelInvokeError::new(
            ModelInvokeErrorKind::Permanent,
            format!("{provider} 2xx response body is not valid JSON: {e}"),
        )
    })?;
    parse(&value)
}

fn map_transport_error(e: reqwest::Error) -> ModelInvokeError {
    let kind = if e.is_timeout() {
        ModelInvokeErrorKind::TimedOut
    } else if e.is_connect() || e.is_body() {
        ModelInvokeErrorKind::Transient
    } else {
        ModelInvokeErrorKind::Permanent
    };
    ModelInvokeError::new(kind, format!("provider transport: {e}"))
}

fn map_status_error(status: StatusCode, body: &str, provider: &str) -> ModelInvokeError {
    let kind = match status.as_u16() {
        408 | 429 | 500..=599 => ModelInvokeErrorKind::Transient,
        400 | 422 => ModelInvokeErrorKind::InvalidRequest,
        // 401 / 403 / 404 and everything else (§4 "其余" row).
        _ => ModelInvokeErrorKind::Permanent,
    };
    ModelInvokeError::new(
        kind,
        format!(
            "{provider} error {}: {}",
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
