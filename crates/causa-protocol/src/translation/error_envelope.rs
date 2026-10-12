//! The provider error envelope, read one way wherever it arrives.
//!
//! All three protocols report a failure as an object with a `message` and a
//! `code` or `type`: in the body of a non-success HTTP response under
//! `error`, and inside a successful stream as an error event. Both places go
//! through [`ErrorDetail::read`], so a host sees the same text for the same
//! envelope. An envelope that arrives inside a successful response has no
//! status to classify it, so [`provider_error`] reads the kind from the
//! envelope's `code` and `type`.
//!
//! Only two things leave the envelope. The message is cut to 200 characters
//! and loses its control characters, so it is safe to print on one line. The
//! code is kept only when it is a short identifier. Nothing else in the
//! envelope, and nothing outside it, is echoed.

use causa_kernel::{ModelInvokeError, ModelInvokeErrorKind};
use serde_json::Value;

/// Longest provider message kept, in characters.
const MAX_MESSAGE_CHARS: usize = 200;

/// What an error envelope says, reduced to what is safe to show.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ErrorDetail {
    /// The provider's `code` when it is a short identifier, otherwise its
    /// `type` under the same condition.
    pub code: Option<String>,
    /// The provider's `message`, on one line and at most 200 characters
    /// (longer text ends with an ellipsis). `None` when absent or empty.
    pub message: Option<String>,
}

impl ErrorDetail {
    /// Reads an error object: the value under `error` in an HTTP error body,
    /// or the error carried by a stream event. Only an object is read: an
    /// error given as a bare string is not the envelope, and its text is not
    /// echoed.
    pub fn read(error: &Value) -> Self {
        let code = name(error, "code")
            .or_else(|| name(error, "type"))
            .map(str::to_owned);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .and_then(snippet);
        Self { code, message }
    }

    /// Reads the body of a non-success HTTP response. A body that is not
    /// JSON with an `error` object yields an empty detail; there is no
    /// fallback to the raw text.
    pub fn read_body(body: &str) -> Self {
        serde_json::from_str::<Value>(body)
            .ok()
            .and_then(|envelope| envelope.get("error").map(Self::read))
            .unwrap_or_default()
    }
}

/// The string under `key`, when it is a short identifier.
fn name<'a>(error: &'a Value, key: &str) -> Option<&'a str> {
    error
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| is_identifier(text))
}

/// The error a successful response body reports about itself: an `error`
/// that is an object. `None` for a body that reports no error, including one
/// whose `error` is `null` or any other value that is not the envelope.
pub fn reported_error(body: &Value) -> Option<ModelInvokeError> {
    body.get("error")
        .filter(|error| error.is_object())
        .map(provider_error)
}

/// An error the provider reported inside a successful response: a stream
/// event, or a 2xx body that is an error envelope. There is no status, so
/// the kind comes from the envelope: `code` is looked up first and `type`
/// second, and the first one the table knows decides. The text is the
/// envelope's code and message, read by the rule an HTTP error body goes
/// through.
///
/// | `code` or `type` | Kind |
/// |---|---|
/// | `rate_limit_exceeded`, `rate_limit_error`, `overloaded_error`, `server_error`, `api_error`, `internal_server_error`, `timeout_error` | `Transient` |
/// | `invalid_request_error`, `invalid_request` | `InvalidRequest` |
/// | anything else, or neither present | `Permanent` |
pub fn provider_error(error: &Value) -> ModelInvokeError {
    let detail = ErrorDetail::read(error);
    let kind = [name(error, "code"), name(error, "type")]
        .into_iter()
        .flatten()
        .find_map(named_kind)
        .unwrap_or(ModelInvokeErrorKind::Permanent);
    let label = detail.code.as_deref().unwrap_or("unknown");
    let message = match &detail.message {
        Some(message) => format!("provider error ({label}): {message}"),
        None => format!("provider error ({label})"),
    };
    let error = ModelInvokeError::new(kind, message);
    match detail.code {
        Some(code) => error.with_code(code),
        None => error,
    }
}

fn named_kind(name: &str) -> Option<ModelInvokeErrorKind> {
    match name {
        "rate_limit_exceeded"
        | "rate_limit_error"
        | "overloaded_error"
        | "server_error"
        | "api_error"
        | "internal_server_error"
        | "timeout_error" => Some(ModelInvokeErrorKind::Transient),
        "invalid_request_error" | "invalid_request" => Some(ModelInvokeErrorKind::InvalidRequest),
        _ => None,
    }
}

/// Whether provider text is short and plain enough to echo as a name: 1 to
/// 64 ASCII letters, digits, dots, underscores or hyphens.
pub(crate) fn is_identifier(text: &str) -> bool {
    (1..=64).contains(&text.len())
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Characters that break a line or reorder the text around them without
/// being control characters: the Unicode line and paragraph separators and
/// the bidirectional marks, embeddings, overrides and isolates.
fn breaks_line(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{2028}' | '\u{2029}' | '\u{061C}' | '\u{200E}' | '\u{200F}'
                | '\u{202A}'..='\u{202E}'
                | '\u{2066}'..='\u{2069}'
        )
}

/// One printable line: each run of control, line-separator or bidirectional
/// formatting characters becomes a space, then the text is cut to
/// [`MAX_MESSAGE_CHARS`].
fn snippet(message: &str) -> Option<String> {
    let line = message
        .split(breaks_line)
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if line.is_empty() {
        return None;
    }
    let mut chars = line.chars();
    let mut kept: String = chars.by_ref().take(MAX_MESSAGE_CHARS).collect();
    if chars.next().is_some() {
        kept.push('…');
    }
    Some(kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn code_falls_back_to_type_and_must_be_an_identifier() {
        let detail = ErrorDetail::read(&json!({"type": "rate_limit_error", "message": "slow"}));
        assert_eq!(detail.code.as_deref(), Some("rate_limit_error"));
        let detail = ErrorDetail::read(&json!({"code": "model_not_found", "type": "x"}));
        assert_eq!(detail.code.as_deref(), Some("model_not_found"));
        // A null or numeric code is not a name; the type is used instead.
        let detail = ErrorDetail::read(&json!({"code": null, "type": "invalid_request_error"}));
        assert_eq!(detail.code.as_deref(), Some("invalid_request_error"));
        for code in ["", "two words", "a\nb", &"x".repeat(65)] {
            assert_eq!(
                ErrorDetail::read(&json!({"code": code})).code,
                None,
                "{code:?}"
            );
            // A code that is not a name does not hide a type that is.
            assert_eq!(
                ErrorDetail::read(&json!({"code": code, "type": "api_error"}))
                    .code
                    .as_deref(),
                Some("api_error"),
                "{code:?}"
            );
        }
    }

    #[test]
    fn kind_is_read_from_code_then_type() {
        use ModelInvokeErrorKind::{InvalidRequest, Permanent, Transient};
        for (envelope, kind, code) in [
            (
                json!({"type": "invalid_request_error", "code": "context_length_exceeded"}),
                InvalidRequest,
                Some("context_length_exceeded"),
            ),
            (
                json!({"type": "server_error", "code": "model_overloaded"}),
                Transient,
                Some("model_overloaded"),
            ),
            (
                json!({"type": "rate_limit_error", "code": ""}),
                Transient,
                Some("rate_limit_error"),
            ),
            (
                json!({"type": "timeout_error"}),
                Transient,
                Some("timeout_error"),
            ),
            (
                json!({"type": "authentication_error", "code": "invalid_api_key"}),
                Permanent,
                Some("invalid_api_key"),
            ),
            (json!({"message": "no name"}), Permanent, None),
        ] {
            let error = provider_error(&envelope);
            assert_eq!(error.kind, kind, "{envelope}");
            assert_eq!(error.code.as_deref(), code, "{envelope}");
        }
    }

    #[test]
    fn a_body_reports_an_error_only_through_an_object() {
        for body in [
            json!({"error": null}),
            json!({"choices": []}),
            json!({"error": 7}),
            json!({"error": "quota used up"}),
        ] {
            assert!(reported_error(&body).is_none(), "{body}");
        }
        let error =
            reported_error(&json!({"error": {"type": "overloaded_error", "message": "busy"}}))
                .expect("object error");
        assert_eq!(error.kind, ModelInvokeErrorKind::Transient);
        assert_eq!(error.message, "provider error (overloaded_error): busy");
        // A bare string is not the envelope; its text is never echoed.
        assert_eq!(
            ErrorDetail::read(&json!("quota used up")),
            ErrorDetail::default()
        );
    }

    #[test]
    fn message_is_one_line_without_control_characters() {
        let detail = ErrorDetail::read(&json!({"message": "bad\r\nkey\u{1b}[31m red \u{7}"}));
        assert_eq!(detail.message.as_deref(), Some("bad key [31m red"));
        let detail = ErrorDetail::read(
            &json!({"message": "one\u{2028}two\u{2029}three \u{202e}evil\u{2066}x\u{200f}"}),
        );
        assert_eq!(detail.message.as_deref(), Some("one two three evil x"));
        assert_eq!(ErrorDetail::read(&json!({"message": "\n\t"})).message, None);
        assert_eq!(ErrorDetail::read(&json!({"message": 7})).message, None);
    }

    #[test]
    fn message_is_cut_at_two_hundred_characters() {
        let detail = ErrorDetail::read(&json!({"message": "é".repeat(201)}));
        let message = detail.message.expect("message");
        assert_eq!(message.chars().count(), 201);
        assert!(message.ends_with('…'));
        let detail = ErrorDetail::read(&json!({"message": "é".repeat(200)}));
        assert_eq!(detail.message.expect("message").chars().count(), 200);
    }

    #[test]
    fn body_without_an_error_object_yields_nothing() {
        assert_eq!(
            ErrorDetail::read_body("<html>502</html>"),
            ErrorDetail::default()
        );
        assert_eq!(
            ErrorDetail::read_body("{\"status\":\"fail\"}"),
            ErrorDetail::default()
        );
        let detail =
            ErrorDetail::read_body("{\"error\":{\"message\":\"no\",\"type\":\"api_error\"}}");
        assert_eq!(detail.message.as_deref(), Some("no"));
        assert_eq!(detail.code.as_deref(), Some("api_error"));
    }
}
