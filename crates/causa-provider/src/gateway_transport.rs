//! Shared transport plumbing for kernel `ModelGateway` adapters.
//!
//! Owns the two adapter-wide concerns the per-protocol gateways must not
//! duplicate: control-plane wiring (`send_with_control`) and the error
//! mapping table (`ModelInvokeErrorKind` is the closed vocabulary the
//! caller can use when deciding whether to start another logical invocation):
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
//! The adapter is read-only on the control plane. The absolute deadline and
//! cancellation token are raced against every socket wait, independent of
//! host-client timeout configuration, and checked during local progression.

use std::{future::Future, time::Duration};

use causa_kernel::{
    CallControl, ControlError, ModelInvokeError, ModelInvokeErrorKind, ModelOutput,
};
use reqwest::{Client, StatusCode};
use serde_json::Value;

/// Hard cap on buffered provider response bodies. A runaway or hijacked
/// endpoint must not OOM the host process. Generous for any LLM response
/// including long tool observations.
const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

/// Baseline transport timeouts for the default client (connect 10s,
/// total 120s). A per-request attempt deadline replaces the total timeout
/// when present; the connect baseline always applies.
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
        .retry(reqwest::retry::never())
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("static client configuration is valid")
}

/// Send a prepared POST under the attempt's control plane: pre-cancel
/// short-circuit, deadline-as-timeout, token raced against the in-flight
/// work. Returns the raw status + bounded body for protocol-specific
/// parsing.
pub(crate) async fn send_with_control(
    req: reqwest::RequestBuilder,
    control: &CallControl,
) -> Result<(StatusCode, String), ModelInvokeError> {
    let response = send_response(req, control).await?;
    read_bounded(response, control).await
}

/// Buffer the response body under [`MAX_RESPONSE_BYTES`]. The
/// Content-Length header is rejected up front when present; the chunk
/// loop covers chunked bodies with no declared length. Mid-body network
/// failures surface through the transport table (body error →
/// `Transient`); an over-cap body or non-UTF-8 payload is `Permanent`
/// (the provider produced an unusable response).
async fn read_bounded(
    mut response: reqwest::Response,
    control: &CallControl,
) -> Result<(StatusCode, String), ModelInvokeError> {
    let status = response.status();
    if let Some(len) = response.content_length()
        && len as usize > MAX_RESPONSE_BYTES
    {
        return Err(oversize_error(len as usize));
    }
    let mut body: Vec<u8> = Vec::new();
    let mut chunks = 0;
    while let Some(chunk) = controlled(control, response.chunk())
        .await?
        .map_err(map_transport_error)?
    {
        check_control(control)?;
        chunks += 1;
        if chunks >= 32 {
            chunks = 0;
            tokio::task::yield_now().await;
            check_control(control)?;
        }
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
/// protocol-specific body parser (whose own schema failures also surface
/// as `Permanent`).
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

pub(crate) fn map_transport_error(e: reqwest::Error) -> ModelInvokeError {
    let kind = if e.is_timeout() {
        ModelInvokeErrorKind::TimedOut
    } else if e.is_connect() || e.is_body() || e.is_decode() {
        // Response::chunk wraps failed HTTP body frames as decode errors.
        // Wire JSON/UTF-8 failures are classified by our own pure parsers.
        ModelInvokeErrorKind::Transient
    } else {
        ModelInvokeErrorKind::Permanent
    };
    ModelInvokeError::new(kind, format!("provider transport: {}", e.without_url()))
}

pub(crate) fn map_status_error(status: StatusCode, body: &str, provider: &str) -> ModelInvokeError {
    let kind = match status.as_u16() {
        408 | 429 | 500..=599 => ModelInvokeErrorKind::Transient,
        400 | 422 => ModelInvokeErrorKind::InvalidRequest,
        // 401 / 403 / 404 and everything else.
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
    let envelope = serde_json::from_str::<Value>(body).ok();
    let message = envelope
        .as_ref()
        .and_then(|v| v.get("error")?.get("message")?.as_str())
        .unwrap_or(body);
    let mut chars = message.chars();
    let mut snippet: String = chars.by_ref().take(200).collect();
    if chars.next().is_some() {
        snippet.push('…');
    }
    snippet
}

/// Check local progress as well as socket waits.
pub(crate) fn check_control(control: &CallControl) -> Result<(), ModelInvokeError> {
    control.check().map_err(|error| match error {
        ControlError::Cancelled => ModelInvokeError::new(
            ModelInvokeErrorKind::Cancelled,
            "model invocation cancelled",
        ),
        ControlError::TimedOut => ModelInvokeError::new(
            ModelInvokeErrorKind::TimedOut,
            "model invocation deadline exceeded",
        ),
    })
}

/// Race a transport operation against the absolute control deadline. This
/// remains effective with host clients that have no reqwest timeout.
pub(crate) async fn controlled<T>(
    control: &CallControl,
    work: impl Future<Output = T>,
) -> Result<T, ModelInvokeError> {
    check_control(control)?;
    let deadline = async {
        match control.deadline() {
            Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        biased;
        _ = control.cancellation_token().cancelled() => Err(ModelInvokeError::new(ModelInvokeErrorKind::Cancelled, "model invocation cancelled")),
        _ = deadline => Err(ModelInvokeError::new(ModelInvokeErrorKind::TimedOut, "model invocation deadline exceeded")),
        result = work => { check_control(control)?; Ok(result) },
    }
}

/// Send one request without buffering its body.
pub(crate) async fn send_response(
    mut req: reqwest::RequestBuilder,
    control: &CallControl,
) -> Result<reqwest::Response, ModelInvokeError> {
    if let Some(deadline) = control.deadline() {
        req = req.timeout(deadline.saturating_duration_since(std::time::Instant::now()));
    }
    controlled(control, req.send())
        .await?
        .map_err(map_transport_error)
}

/// Consume an unsuccessful response with a byte bound and control checks.
/// Streaming diagnostics deliberately discard its untrusted contents, so
/// neither full-body storage nor UTF-8/JSON decoding is necessary.
pub(crate) async fn read_error_response(
    mut response: reqwest::Response,
    control: &CallControl,
) -> Result<StatusCode, ModelInvokeError> {
    let status = response.status();
    if let Some(len) = response.content_length()
        && len > MAX_RESPONSE_BYTES as u64
    {
        return Err(oversize_error(MAX_RESPONSE_BYTES + 1));
    }
    let mut bytes = 0usize;
    let mut chunks = 0usize;
    while let Some(chunk) = controlled(control, response.chunk())
        .await?
        .map_err(map_transport_error)?
    {
        bytes = bytes.saturating_add(chunk.len());
        if bytes > MAX_RESPONSE_BYTES {
            return Err(oversize_error(bytes));
        }
        chunks += 1;
        if chunks >= 32 {
            chunks = 0;
            tokio::task::yield_now().await;
            check_control(control)?;
        }
    }
    Ok(status)
}
