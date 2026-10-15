//! Shared transport plumbing for kernel `ModelGateway` adapters.
//!
//! Owns the two adapter-wide concerns the per-protocol gateways must not
//! duplicate: control-plane wiring (`send_with_control`) and the error
//! mapping table (`ModelInvokeErrorKind` is the closed vocabulary the
//! caller can use when deciding whether to start another logical invocation):
//!
//! | Source                                  | Kind            |
//! |-----------------------------------------|-----------------|
//! | reqwest connect / request / body error  | `Transient`     |
//! | request timeout (incl. attempt deadline)| `TimedOut`      |
//! | cancellation token fired                | `Cancelled`     |
//! | HTTP 408 / 409 / 429 / 5xx              | `Transient`     |
//! | HTTP 400 / 422                          | `InvalidRequest`|
//! | HTTP 401 / 403 / 404                    | `Permanent`     |
//! | any other non-2xx status                | `Permanent`     |
//! | 2xx body that is an error envelope      | by code or type |
//! | 2xx body not parseable                  | `Undecodable`   |
//! | any body over the size cap              | `Undecodable`   |
//! | 2xx stream without an SSE content type  | `Undecodable`   |
//! | a request the client could not build    | `Permanent`     |
//!
//! A request error covers a connection the peer closed before answering.
//! The request may have reached the server, so a host that cannot tolerate
//! a repeated call must not retry `Transient`. An over-cap body is
//! `Undecodable` whatever its status, and carries no status.
//!
//! A non-2xx error carries its status in [`ModelInvokeError::status`] and
//! the provider's error code or type in [`ModelInvokeError::code`], so a
//! caller can tell a rejected credential (401, 403) from an unknown model or
//! path (404) without reading the message. The message keeps at most 200
//! characters of the provider's `error.message`, on one line.
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
use causa_protocol::translation::error_envelope::{ErrorDetail, reported_error};
use reqwest::header::HeaderMap;
use reqwest::{Client, StatusCode};
use serde_json::Value;

/// Hard cap on buffered provider response bodies. A runaway or hijacked
/// endpoint must not OOM the host process. Generous for any LLM response
/// including long tool observations.
const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

/// Baseline transport timeouts for the default client (connect 10s,
/// read idle 300s). The idle timeout resets after each successful read,
/// preventing a connection from hanging indefinitely while allowing long
/// streaming or generation turns to complete. Total call duration is governed
/// by the caller's [`CallControl`] deadline, not a static HTTP timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_READ_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

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

/// Build a client with custom connect and idle-read timeouts.
fn client_with_timeouts(connect: Duration, read_idle: Duration) -> Client {
    Client::builder()
        .connect_timeout(connect)
        .read_timeout(read_idle)
        .retry(reqwest::retry::never())
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("static client configuration is valid")
}

/// Default shared client: baseline connect + read-idle timeouts so a gateway
/// without an attempt deadline cannot hang forever on a stalled
/// connection while allowing healthy long responses to complete. Total call
/// duration belongs to the caller's [`CallControl`] deadline. Hosts with
/// different baselines inject their own client through
/// `GatewayCore::with_http_client`.
fn default_client() -> Client {
    client_with_timeouts(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_IDLE_TIMEOUT)
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
#[derive(Debug)]
pub(crate) struct BufferedResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
}

/// Buffer the response body under [`MAX_RESPONSE_BYTES`]. The
/// Content-Length header is rejected up front when present; the chunk
/// loop covers chunked bodies with no declared length. Mid-body network
/// failures surface through the transport table (body error →
/// `Transient`); an over-cap body or non-UTF-8 payload is `Undecodable`
/// (the provider produced an unusable response).
pub(crate) async fn read_bounded(
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
    let text = match String::from_utf8(body) {
        Ok(t) => t,
        Err(_) if !status.is_success() => String::new(),
        Err(_) => {
            return Err(ModelInvokeError::new(
                ModelInvokeErrorKind::Undecodable,
                "provider response body is not valid UTF-8",
            ));
        }
    };
    Ok(BufferedResponse {
        status,
        headers,
        body: text,
    })
}

fn oversize_error(len: usize) -> ModelInvokeError {
    ModelInvokeError::new(
        ModelInvokeErrorKind::Undecodable,
        format!(
            "provider response too large: {len} bytes exceeds the {MAX_RESPONSE_BYTES}-byte cap"
        ),
    )
}

/// Turn a raw response into a [`ModelOutput`]: non-2xx through the error
/// table, 2xx JSON-syntax failures as `Undecodable`, a 2xx error envelope as
/// the error it reports, then the protocol-specific body parser (whose own
/// schema failures also surface as `Undecodable`).
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
            ModelInvokeErrorKind::Undecodable,
            format!("{provider} 2xx response body is not valid JSON: {e}"),
        )
    })?;
    // Some servers report a failure under a success status.
    if let Some(error) = reported_error(&value) {
        return Err(error);
    }
    parse(&value)
}

pub(crate) fn map_transport_error(e: reqwest::Error) -> ModelInvokeError {
    let kind = if e.is_timeout() {
        ModelInvokeErrorKind::TimedOut
    } else if e.is_connect() || e.is_request() || e.is_body() || e.is_decode() {
        // A request error is the exchange failing before a response arrived,
        // for example the peer closing the connection; a refused connection
        // is the same failure one step earlier. Response::chunk wraps failed
        // HTTP body frames as decode errors. Wire JSON/UTF-8 failures are
        // classified by our own pure parsers.
        ModelInvokeErrorKind::Transient
    } else {
        // What is left is a request the client could not build.
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
    let detail = ErrorDetail::read_body(body);
    let message = match &detail.message {
        Some(msg) => format!("{provider} error {}: {msg}", status.as_u16()),
        None => format!("{provider} error {}", status.as_u16()),
    };
    let mut error = ModelInvokeError::new(kind, message).with_status(status.as_u16());
    if let Some(code) = detail.code {
        error = error.with_code(code);
    }
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
///
/// Note: This stop-wait race is kept separate from `causa-runtime` because edge crates
/// (like `causa-provider`) cannot depend on the runtime per repo layering rules,
/// and `causa-kernel` holds contracts only without timers or async runtime dependencies.
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

#[cfg(test)]
mod tests {
    use super::*;
    use causa_kernel::CancellationToken;
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

    #[test]
    fn non_envelope_error_omits_body_text() {
        let error = map_status_error(
            StatusCode::BAD_GATEWAY,
            &HeaderMap::new(),
            "<html>502 Bad Gateway</html>",
            "anthropic",
        );
        assert_eq!(error.message, "anthropic error 502");

        let error2 = map_status_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &HeaderMap::new(),
            "{\"status\": \"fail\"}",
            "chat",
        );
        assert_eq!(error2.message, "chat error 500");
    }

    async fn read_request_head(socket: &mut tokio::net::TcpStream) {
        use tokio::io::AsyncReadExt;
        let mut head = Vec::new();
        loop {
            let mut b = [0u8];
            if socket.read(&mut b).await.unwrap() == 0 {
                break;
            }
            head.push(b[0]);
            if head.ends_with(b"\r\n\r\n") {
                break;
            }
        }
    }

    async fn write_chunk(socket: &mut tokio::net::TcpStream, chunk: &str) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        let frame = format!("{:x}\r\n{chunk}\r\n", chunk.len());
        socket.write_all(frame.as_bytes()).await?;
        socket.flush().await
    }

    #[tokio::test]
    async fn trickling_stream_longer_than_idle_timeout_completes_successfully() {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request_head(&mut socket).await;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                .await
                .unwrap();
            socket.flush().await.unwrap();
            for i in 0..12 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                write_chunk(&mut socket, &format!("data: chunk {i}\n\n"))
                    .await
                    .unwrap();
            }
            socket.write_all(b"0\r\n\r\n").await.unwrap();
            socket.flush().await.unwrap();
        });

        let client = client_with_timeouts(Duration::from_secs(2), Duration::from_millis(200));
        let control = CallControl::new(CancellationToken::new(), None);
        let response = send_with_control(client.post(format!("http://{addr}/test")), &control)
            .await
            .expect("stream trickling within idle timeout must complete");

        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(response.body.matches("data: chunk ").count(), 12);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn stream_stalled_longer_than_idle_timeout_fails_as_timed_out() {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request_head(&mut socket).await;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                .await
                .unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(400)).await;
        });

        let client = client_with_timeouts(Duration::from_secs(2), Duration::from_millis(200));
        let control = CallControl::new(CancellationToken::new(), None);
        let error = send_with_control(client.post(format!("http://{addr}/test")), &control)
            .await
            .expect_err("stalled stream must time out");

        assert_eq!(error.kind, ModelInvokeErrorKind::TimedOut);
        server.abort();
    }

    #[tokio::test]
    async fn call_control_deadline_shorter_than_trickling_stream_fails_at_deadline() {
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            read_request_head(&mut socket).await;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                .await
                .unwrap();
            socket.flush().await.unwrap();
            for i in 0..12 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                if write_chunk(&mut socket, &format!("data: chunk {i}\n\n"))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            let _ = socket.write_all(b"0\r\n\r\n").await;
        });

        let client = client_with_timeouts(Duration::from_secs(2), Duration::from_millis(200));
        let control = CallControl::new(CancellationToken::new(), None)
            .with_timeout(Duration::from_millis(150));
        let error = send_with_control(client.post(format!("http://{addr}/test")), &control)
            .await
            .expect_err("call control deadline must abort trickling stream");

        assert_eq!(error.kind, ModelInvokeErrorKind::TimedOut);
        assert_eq!(error.message, "model invocation deadline exceeded");
        server.abort();
    }
}
