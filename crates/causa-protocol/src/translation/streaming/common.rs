use causa_kernel::{ModelInvokeError, ModelInvokeErrorKind, ModelOutput, StreamDelta};
use serde_json::Value;

use super::StreamLimits;
use crate::translation::error_envelope::{ErrorDetail, is_identifier};
use crate::translation::sse_parser::SseEvent;

/// A failure of the adapter to use what the provider sent.
pub(super) fn undecodable(message: impl Into<String>) -> ModelInvokeError {
    ModelInvokeError::new(ModelInvokeErrorKind::Undecodable, message)
}

/// The caller broke the decoder's contract; nothing the provider sent is at fault.
pub(super) fn misused(message: impl Into<String>) -> ModelInvokeError {
    ModelInvokeError::new(ModelInvokeErrorKind::Permanent, message)
}

pub(super) fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, ModelInvokeError> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| undecodable(format!("stream: missing or invalid {key}")))
}

pub(super) fn index(v: &Value, key: &str) -> Result<usize, ModelInvokeError> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| undecodable(format!("stream: missing or invalid {key}")))?;
    usize::try_from(n).map_err(|_| undecodable("stream index exceeds platform range"))
}

/// Appends where a decode failure happened: the 1-based position of the event
/// in the stream and, when the event names one, its type. The type is provider
/// text, so it is echoed only when it is a short identifier. An error the
/// provider reported itself is returned unchanged: its text already says what
/// went wrong, and matches the same envelope arriving as an HTTP error.
pub(super) fn located(
    mut error: ModelInvokeError,
    position: usize,
    event: &SseEvent,
) -> ModelInvokeError {
    if error.kind != ModelInvokeErrorKind::Undecodable {
        return error;
    }
    let kind = serde_json::from_str::<Value>(&event.data)
        .ok()
        .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_owned))
        .or_else(|| event.event.clone())
        .filter(|kind| is_identifier(kind));
    error.message = match kind {
        Some(kind) => format!("{} (event {position}: {kind})", error.message),
        None => format!("{} (event {position})", error.message),
    };
    error
}

pub(super) fn done(output: ModelOutput) -> StreamDelta {
    StreamDelta::Done {
        final_output: output,
    }
}

/// An error the provider reported inside a successful response. The kind
/// comes from the provider's code; the text is the envelope's code and
/// message, read by the rule an HTTP error body goes through.
pub(crate) fn provider_error(error: &Value) -> ModelInvokeError {
    let detail = ErrorDetail::read(error);
    let kind = match detail.code.as_deref() {
        Some(
            "rate_limit_exceeded"
            | "rate_limit_error"
            | "overloaded_error"
            | "server_error"
            | "api_error"
            | "internal_server_error",
        ) => ModelInvokeErrorKind::Transient,
        Some("invalid_request_error" | "invalid_request") => ModelInvokeErrorKind::InvalidRequest,
        _ => ModelInvokeErrorKind::Permanent,
    };
    let label = detail.code.as_deref().unwrap_or("unknown");
    let message = match &detail.message {
        Some(message) => format!("provider stream error ({label}): {message}"),
        None => format!("provider stream error ({label})"),
    };
    let error = ModelInvokeError::new(kind, message);
    match detail.code {
        Some(code) => error.with_code(code),
        None => error,
    }
}

pub(super) struct Budget {
    pub limits: StreamLimits,
    bytes: usize,
}

impl Budget {
    pub fn new(limits: StreamLimits) -> Self {
        Self { limits, bytes: 0 }
    }
    pub fn add(&mut self, bytes: usize) -> Result<(), ModelInvokeError> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| undecodable("stream output size overflow"))?;
        if self.bytes > self.limits.max_output_bytes {
            return Err(undecodable("stream output exceeds max_output_bytes"));
        }
        Ok(())
    }
    pub fn item(&self, count: usize) -> Result<(), ModelInvokeError> {
        if count >= self.limits.max_items {
            return Err(undecodable("stream exceeds max_items"));
        }
        Ok(())
    }
    pub fn snapshot(&self, value: &Value) -> Result<(), ModelInvokeError> {
        // Serialization into a counting sink avoids allocating another complete
        // snapshot just to measure its size.
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self.0.saturating_add(bytes.len());
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut count = Count(0);
        serde_json::to_writer(&mut count, value)
            .map_err(|_| undecodable("invalid terminal snapshot"))?;
        if count.0.saturating_add(self.bytes) > self.limits.max_output_bytes {
            return Err(undecodable("terminal snapshot exceeds max_output_bytes"));
        }
        Ok(())
    }
}

pub(super) fn merge_usage(
    target: &mut Value,
    update: &Value,
    budget: &mut Budget,
) -> Result<(), ModelInvokeError> {
    let before = serde_json::to_vec(&target)
        .map_err(|_| undecodable("invalid usage state"))?
        .len();
    let fields = update
        .as_object()
        .ok_or_else(|| undecodable("stream usage must be an object"))?;
    let target = target
        .as_object_mut()
        .ok_or_else(|| undecodable("invalid usage state"))?;
    const COUNTS: &[&str] = &[
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
    ];
    const DETAILS: &[&str] = &[
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
    ];
    for key in COUNTS {
        if let Some(value) = fields.get(*key).filter(|v| !v.is_null()) {
            let n = value
                .as_u64()
                .ok_or_else(|| undecodable("stream usage count must be an unsigned integer"))?;
            usize::try_from(n)
                .map_err(|_| undecodable("stream usage count exceeds platform range"))?;
            target.insert((*key).into(), value.clone());
        }
    }
    for key in DETAILS {
        if let Some(value) = fields.get(*key).filter(|v| !v.is_null()) {
            let values = value
                .as_object()
                .ok_or_else(|| undecodable("stream usage details must be an object"))?;
            let retained = target
                .entry(*key)
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .expect("usage detail object");
            for count in ["cached_tokens", "reasoning_tokens"] {
                if let Some(value) = values.get(count).filter(|v| !v.is_null()) {
                    let n = value.as_u64().ok_or_else(|| {
                        undecodable("stream usage count must be an unsigned integer")
                    })?;
                    usize::try_from(n)
                        .map_err(|_| undecodable("stream usage count exceeds platform range"))?;
                    retained.insert(count.into(), value.clone());
                }
            }
        }
    }
    let after = serde_json::to_vec(&target)
        .map_err(|_| undecodable("invalid usage state"))?
        .len();
    budget.add(after.saturating_sub(before))?;
    Ok(())
}
