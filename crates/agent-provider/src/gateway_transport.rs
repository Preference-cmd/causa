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
//! | 2xx body not parseable / over cap       | `Permanent`     |
//! | anything else                           | `Permanent`     |
//!
//! The adapter is **read-only** on the control plane: `AttemptControl::deadline`
//! becomes the request timeout and the shared cancellation token is raced
//! against in-flight HTTP via `select!`. Gateways never construct control
//! planes — those are driver-owned (`AttemptControl::new` stays
//! `pub(crate)`).

use std::time::Duration;

use reimagine_context_kernel::{
    AttemptControl, ModelInvokeError, ModelInvokeErrorKind, ModelOutput,
};
use reqwest::{Client, StatusCode};
use serde_json::Value;

/// Hard cap on buffered provider response bodies. A runaway or hijacked
/// endpoint must not OOM the host process. Generous for any LLM response
/// including long tool observations.
const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

/// Baseline transport timeouts for the default client, matching the
/// frozen `ReqwestBackend` precedent (connect 10s, total 120s). A
/// per-request attempt deadline replaces the total timeout when present;
/// the connect baseline always applies.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_TOTAL_TIMEOUT: Duration = Duration::from_secs(120);

/// Endpoint + HTTP client bundle shared by the per-protocol gateways.
#[derive(Debug, Clone)]
pub(crate) struct GatewayCore {
    pub http: Client,
    pub endpoint: String,
}

impl GatewayCore {
    pub fn new(default_endpoint: &str) -> Self {
        Self {
            http: default_client(),
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

/// Default shared client: baseline connect + total timeouts so a gateway
/// without an attempt deadline cannot hang forever on a stalled
/// connection. Hosts with different baselines inject their own client
/// through `GatewayCore::with_http_client`.
fn default_client() -> Client {
    Client::builder()
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .timeout(DEFAULT_TOTAL_TIMEOUT)
        .build()
        .expect("static client configuration is valid")
}

/// Send a prepared POST under the attempt's control plane: pre-cancel
/// short-circuit, deadline-as-timeout, token raced against the in-flight
/// work. Returns the raw status + bounded body for protocol-specific
/// parsing.
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
        let response = req.send().await.map_err(map_transport_error)?;
        read_bounded(response).await
    };

    tokio::select! {
        outcome = http => outcome,
        _ = control.cancellation_token().cancelled() => Err(ModelInvokeError::new(
            ModelInvokeErrorKind::Cancelled,
            "cancelled while awaiting response",
        )),
    }
}

/// Buffer the response body under [`MAX_RESPONSE_BYTES`]. The
/// Content-Length header is rejected up front when present; the chunk
/// loop covers chunked bodies with no declared length. Mid-body network
/// failures surface through the transport table (body error →
/// `Transient`); an over-cap body or non-UTF-8 payload is `Permanent`
/// (§4 row: the provider produced an unusable response).
async fn read_bounded(
    mut response: reqwest::Response,
) -> Result<(StatusCode, String), ModelInvokeError> {
    let status = response.status();
    if let Some(len) = response.content_length()
        && len as usize > MAX_RESPONSE_BYTES
    {
        return Err(oversize_error(len as usize));
    }
    let mut body: Vec<u8> = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(map_transport_error)? {
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(oversize_error(MAX_RESPONSE_BYTES + 1));
        }
        body.extend_from_slice(&chunk);
    }
    let text = String::from_utf8(body).map_err(|_| {
        ModelInvokeError::new(
            ModelInvokeErrorKind::Permanent,
            "provider response body is not valid UTF-8",
        )
    })?;
    Ok((status, text))
}

fn oversize_error(len: usize) -> ModelInvokeError {
    ModelInvokeError::new(
        ModelInvokeErrorKind::Permanent,
        format!(
            "provider response too large: {len} bytes exceeds the {MAX_RESPONSE_BYTES}-byte cap"
        ),
    )
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
/// body when the payload is not the documented error envelope. Single
/// bounded pass — the body is already capped by [`read_bounded`], but the
/// snippet never scans more than the first 201 chars.
fn error_snippet(body: &str) -> String {
    let from_envelope = serde_json::from_str::<Value>(body).ok().and_then(|v| {
        let message = v.get("error")?.get("message")?.as_str()?;
        Some(message.to_string())
    });
    from_envelope.unwrap_or_else(|| {
        let mut chars = body.chars();
        let mut snippet: String = chars.by_ref().take(200).collect();
        if chars.next().is_some() {
            snippet.push('…');
        }
        snippet
    })
}
