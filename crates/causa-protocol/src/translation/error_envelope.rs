//! The provider error envelope, read one way wherever it arrives.
//!
//! All three protocols report a failure as an object with a `message` and a
//! `code` or `type`: in the body of a non-success HTTP response under
//! `error`, and inside a successful stream as an error event. Both places go
//! through [`ErrorDetail::read`], so a host sees the same text for the same
//! envelope.
//!
//! Only two things leave the envelope. The message is cut to 200 characters
//! and loses its control characters, so it is safe to print on one line. The
//! code is kept only when it is a short identifier. Nothing else in the
//! envelope, and nothing outside it, is echoed.

use serde_json::Value;

/// Longest provider message kept, in characters.
const MAX_MESSAGE_CHARS: usize = 200;

/// What an error envelope says, reduced to what is safe to show.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ErrorDetail {
    /// The provider's `code`, or its `type` when there is no string code.
    /// `None` unless the value is a short identifier.
    pub code: Option<String>,
    /// The provider's `message`, on one line and at most 200 characters
    /// (longer text ends with an ellipsis). `None` when absent or empty.
    pub message: Option<String>,
}

impl ErrorDetail {
    /// Reads an error object: the value under `error` in an HTTP error body,
    /// or the error carried by a stream event.
    pub fn read(error: &Value) -> Self {
        let code = error
            .get("code")
            .and_then(Value::as_str)
            .or_else(|| error.get("type").and_then(Value::as_str))
            .filter(|code| is_identifier(code))
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

/// Whether provider text is short and plain enough to echo as a name: 1 to
/// 64 ASCII letters, digits, dots, underscores or hyphens.
pub(crate) fn is_identifier(text: &str) -> bool {
    (1..=64).contains(&text.len())
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// One printable line: each run of control characters becomes a space, then
/// the text is cut to [`MAX_MESSAGE_CHARS`].
fn snippet(message: &str) -> Option<String> {
    let line = message
        .split(char::is_control)
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
        }
    }

    #[test]
    fn message_is_one_line_without_control_characters() {
        let detail = ErrorDetail::read(&json!({"message": "bad\r\nkey\u{1b}[31m red \u{7}"}));
        assert_eq!(detail.message.as_deref(), Some("bad key [31m red"));
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
