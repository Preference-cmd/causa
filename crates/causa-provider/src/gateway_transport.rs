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
//! | HTTP 408 / 409 / 429 / 5xx              | `Transient`     |
//! | HTTP 400 / 422                          | `InvalidRequest`|
//! | HTTP 401 / 403 / 404                    | `Permanent`     |
//! | 2xx body not parseable / over cap       | `Permanent`     |
//! | anything else                           | `Permanent`     |
//!
//! Two response headers can override the status rule: `x-should-retry: true`
//! classifies as `Transient` whatever the status, and `false` removes
//! retryability (400/422 stay `InvalidRequest`). A `retry-after-ms` or
//! `retry-after` value (seconds or an HTTP date) fills
//! [`ModelInvokeError::retry_delay`]; the transport itself never waits.
//!
//! The adapter is read-only on the control plane. The absolute deadline and
//! cancellation token are raced against every socket wait, independent of
//! host-client timeout configuration, and checked during local progression.

use std::{future::Future, time::Duration};

use causa_kernel::{
    CallControl, ControlError, ModelInvokeError, ModelInvokeErrorKind, ModelOutput,
};
use reqwest::header::HeaderMap;
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
) -> Result<BufferedResponse, ModelInvokeError> {
    let response = send_response(req, control).await?;
    read_bounded(response, control).await
}

/// A fully buffered response: the status and headers classification reads,
/// plus the bounded body the protocol parser consumes.
pub(crate) struct BufferedResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
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
) -> Result<BufferedResponse, ModelInvokeError> {
    let status = response.status();
    let headers = response.headers().clone();
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
    Ok(BufferedResponse {
        status,
        headers,
        body: text,
    })
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
    response: BufferedResponse,
    provider: &str,
    parse: F,
) -> Result<ModelOutput, ModelInvokeError>
where
    F: FnOnce(&Value) -> Result<ModelOutput, ModelInvokeError>,
{
    if !response.status.is_success() {
        return Err(map_status_error(
            response.status,
            &response.headers,
            &response.body,
            provider,
        ));
    }
    let value: Value = serde_json::from_str(&response.body).map_err(|e| {
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

/// Classify a non-2xx response. `x-should-retry` overrides the status rule
/// when it names a boolean; a `retry-after-ms` / `retry-after` header adds
/// the server's requested delay to the error.
pub(crate) fn map_status_error(
    status: StatusCode,
    headers: &HeaderMap,
    body: &str,
    provider: &str,
) -> ModelInvokeError {
    let kind = match should_retry(headers) {
        Some(true) => ModelInvokeErrorKind::Transient,
        // A refused retry never stays retryable; a rejected request keeps
        // its own kind so callers can still tell it from auth/not-found.
        Some(false) => match status.as_u16() {
            400 | 422 => ModelInvokeErrorKind::InvalidRequest,
            _ => ModelInvokeErrorKind::Permanent,
        },
        None => status_kind(status),
    };
    let error = ModelInvokeError::new(
        kind,
        format!(
            "{provider} error {}: {}",
            status.as_u16(),
            error_snippet(body)
        ),
    );
    match retry_after(headers) {
        Some(delay) => error.with_retry_delay(delay),
        None => error,
    }
}

fn status_kind(status: StatusCode) -> ModelInvokeErrorKind {
    match status.as_u16() {
        408 | 409 | 429 | 500..=599 => ModelInvokeErrorKind::Transient,
        400 | 422 => ModelInvokeErrorKind::InvalidRequest,
        // 401 / 403 / 404 and everything else.
        _ => ModelInvokeErrorKind::Permanent,
    }
}

/// The `x-should-retry` override: `Some(true)` means retry, `Some(false)`
/// means never, and `None` (absent, unreadable or any other value) leaves
/// the status rule in force.
fn should_retry(headers: &HeaderMap) -> Option<bool> {
    match header_str(headers, "x-should-retry")?.trim() {
        value if value.eq_ignore_ascii_case("true") => Some(true),
        value if value.eq_ignore_ascii_case("false") => Some(false),
        _ => None,
    }
}

/// A server-requested retry delay: `retry-after-ms` as fractional
/// milliseconds first, then `retry-after` as seconds or an HTTP date. A
/// missing, unparseable or already-past value yields `None`.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    if let Some(value) = header_str(headers, "retry-after-ms")
        && let Some(delay) = seconds_duration(value, 1_000.0)
    {
        return Some(delay);
    }
    let value = header_str(headers, "retry-after")?;
    seconds_duration(value, 1.0).or_else(|| http_date_duration(value))
}

/// Parse a non-negative decimal number of the given unit into a duration.
fn seconds_duration(value: &str, units_per_second: f64) -> Option<Duration> {
    let amount: f64 = value.trim().parse().ok()?;
    if !amount.is_finite() || amount < 0.0 {
        return None;
    }
    Duration::try_from_secs_f64(amount / units_per_second).ok()
}

/// Parse an HTTP-date as a delay from now; a past date yields `None`.
fn http_date_duration(value: &str) -> Option<Duration> {
    let date = httpdate::parse_http_date(value.trim()).ok()?;
    date.duration_since(std::time::SystemTime::now()).ok()
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name)?.to_str().ok()
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

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderName, HeaderValue};
    use std::time::SystemTime;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
                HeaderValue::from_str(value).expect("valid header value"),
            );
        }
        map
    }

    fn kind(status: u16, pairs: &[(&str, &str)]) -> ModelInvokeErrorKind {
        map_status_error(
            StatusCode::from_u16(status).expect("valid status"),
            &headers(pairs),
            "{\"error\":{\"message\":\"busy\"}}",
            "test",
        )
        .kind
    }

    #[test]
    fn status_table_classifies_retryable_and_rejected_statuses() {
        for status in [408, 409, 429, 500, 503, 599] {
            assert_eq!(
                kind(status, &[]),
                ModelInvokeErrorKind::Transient,
                "{status}"
            );
        }
        for status in [400, 422] {
            assert_eq!(
                kind(status, &[]),
                ModelInvokeErrorKind::InvalidRequest,
                "{status}"
            );
        }
        for status in [401, 403, 404, 418] {
            assert_eq!(
                kind(status, &[]),
                ModelInvokeErrorKind::Permanent,
                "{status}"
            );
        }
    }

    #[test]
    fn should_retry_header_overrides_the_status_rule() {
        assert_eq!(
            kind(400, &[("x-should-retry", "true")]),
            ModelInvokeErrorKind::Transient
        );
        assert_eq!(
            kind(401, &[("x-should-retry", "true")]),
            ModelInvokeErrorKind::Transient
        );
        assert_eq!(
            kind(429, &[("x-should-retry", "false")]),
            ModelInvokeErrorKind::Permanent
        );
        assert_eq!(
            kind(503, &[("x-should-retry", "false")]),
            ModelInvokeErrorKind::Permanent
        );
        assert_eq!(
            kind(400, &[("x-should-retry", "false")]),
            ModelInvokeErrorKind::InvalidRequest
        );
        // An unrecognized value leaves the status rule in force.
        assert_eq!(
            kind(429, &[("x-should-retry", "maybe")]),
            ModelInvokeErrorKind::Transient
        );
        assert_eq!(
            kind(404, &[("x-should-retry", "maybe")]),
            ModelInvokeErrorKind::Permanent
        );
        // So does a missing or unreadable one.
        assert_eq!(
            kind(429, &[("x-should-retry", "1")]),
            ModelInvokeErrorKind::Transient
        );
    }

    #[test]
    fn retry_after_ms_takes_precedence_over_seconds() {
        let error = map_status_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after-ms", "1500"), ("retry-after", "9")]),
            "",
            "test",
        );
        assert_eq!(error.retry_delay, Some(Duration::from_millis(1500)));

        let error = map_status_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after-ms", "250.5"), ("retry-after", "9")]),
            "",
            "test",
        );
        let delay = error.retry_delay.expect("fractional milliseconds parse");
        assert!(
            delay >= Duration::from_nanos(250_499_999)
                && delay <= Duration::from_nanos(250_500_001),
            "{delay:?}"
        );
    }

    #[test]
    fn retry_after_seconds_and_http_dates_are_parsed() {
        assert_eq!(
            retry_after(&headers(&[("retry-after", "3")])),
            Some(Duration::from_secs(3))
        );
        let future = SystemTime::now() + Duration::from_secs(30);
        let value = httpdate::fmt_http_date(future);
        let parsed = retry_after(&headers(&[("retry-after", &value)])).expect("future HTTP date");
        assert!(
            parsed >= Duration::from_secs(25) && parsed <= Duration::from_secs(31),
            "{parsed:?}"
        );
    }

    #[test]
    fn unusable_retry_after_values_leave_the_delay_unset() {
        for value in ["", "abc", "-1", "NaN", "inf"] {
            assert_eq!(
                retry_after(&headers(&[("retry-after", value)])),
                None,
                "retry-after {value:?}"
            );
            assert_eq!(
                retry_after(&headers(&[("retry-after-ms", value)])),
                None,
                "retry-after-ms {value:?}"
            );
        }
        // An unusable retry-after-ms falls back to a usable retry-after.
        assert_eq!(
            retry_after(&headers(&[
                ("retry-after-ms", "junk"),
                ("retry-after", "4")
            ])),
            Some(Duration::from_secs(4))
        );
        let past = httpdate::fmt_http_date(SystemTime::now() - Duration::from_secs(60));
        assert_eq!(retry_after(&headers(&[("retry-after", &past)])), None);
        let error = map_status_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after", "junk")]),
            "",
            "test",
        );
        assert_eq!(error.retry_delay, None);
        assert_eq!(
            ModelInvokeError::new(ModelInvokeErrorKind::Transient, "x").retry_delay,
            None
        );
    }

    #[test]
    fn documented_error_message_comes_from_the_envelope() {
        let error = map_status_error(
            StatusCode::TOO_MANY_REQUESTS,
            &HeaderMap::new(),
            "{\"error\":{\"message\":\"slow down\"}}",
            "anthropic",
        );
        assert_eq!(error.message, "anthropic error 429: slow down");
    }
}
