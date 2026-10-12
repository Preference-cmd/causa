use causa_kernel::{ModelInvokeError, ModelInvokeErrorKind, ModelOutput, StreamDelta};
use serde_json::Value;

use super::StreamLimits;
use crate::translation::sse_parser::SseEvent;

pub(super) fn permanent(message: impl Into<String>) -> ModelInvokeError {
    ModelInvokeError::new(ModelInvokeErrorKind::Permanent, message)
}

pub(super) fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, ModelInvokeError> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| permanent(format!("stream: missing or invalid {key}")))
}

pub(super) fn index(v: &Value, key: &str) -> Result<usize, ModelInvokeError> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| permanent(format!("stream: missing or invalid {key}")))?;
    usize::try_from(n).map_err(|_| permanent("stream index exceeds platform range"))
}

/// Appends where a failure happened: the 1-based position of the event in the
/// stream and, when the event names one, its type. The type is provider text,
/// so it is echoed only when it is a short identifier.
pub(super) fn located(
    mut error: ModelInvokeError,
    position: usize,
    event: &SseEvent,
) -> ModelInvokeError {
    let kind = serde_json::from_str::<Value>(&event.data)
        .ok()
        .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_owned))
        .or_else(|| event.event.clone())
        .filter(|kind| {
            (1..=64).contains(&kind.len())
                && kind
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        });
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

/// Diagnostic includes the error code, never the provider's arbitrary message,
/// which can contain echoed prompts or tool arguments.
pub(crate) fn provider_error(error: &Value) -> ModelInvokeError {
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| error.get("type").and_then(Value::as_str))
        .unwrap_or("unknown");
    let kind = match code {
        "rate_limit_exceeded"
        | "rate_limit_error"
        | "overloaded_error"
        | "server_error"
        | "api_error"
        | "internal_server_error" => ModelInvokeErrorKind::Transient,
        "invalid_request_error" | "invalid_request" => ModelInvokeErrorKind::InvalidRequest,
        _ => ModelInvokeErrorKind::Permanent,
    };
    let diagnostic_code = match code {
        "rate_limit_exceeded"
        | "rate_limit_error"
        | "overloaded_error"
        | "server_error"
        | "api_error"
        | "internal_server_error"
        | "invalid_request_error"
        | "invalid_request"
        | "authentication_error"
        | "permission_error"
        | "not_found_error" => code,
        _ => "unknown",
    };
    ModelInvokeError::new(kind, format!("provider stream error ({diagnostic_code})"))
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
            .ok_or_else(|| permanent("stream output size overflow"))?;
        if self.bytes > self.limits.max_output_bytes {
            return Err(permanent("stream output exceeds max_output_bytes"));
        }
        Ok(())
    }
    pub fn item(&self, count: usize) -> Result<(), ModelInvokeError> {
        if count >= self.limits.max_items {
            return Err(permanent("stream exceeds max_items"));
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
            .map_err(|_| permanent("invalid terminal snapshot"))?;
        if count.0.saturating_add(self.bytes) > self.limits.max_output_bytes {
            return Err(permanent("terminal snapshot exceeds max_output_bytes"));
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
        .map_err(|_| permanent("invalid usage state"))?
        .len();
    let fields = update
        .as_object()
        .ok_or_else(|| permanent("stream usage must be an object"))?;
    let target = target
        .as_object_mut()
        .ok_or_else(|| permanent("invalid usage state"))?;
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
                .ok_or_else(|| permanent("stream usage count must be an unsigned integer"))?;
            usize::try_from(n)
                .map_err(|_| permanent("stream usage count exceeds platform range"))?;
            target.insert((*key).into(), value.clone());
        }
    }
    for key in DETAILS {
        if let Some(value) = fields.get(*key).filter(|v| !v.is_null()) {
            let values = value
                .as_object()
                .ok_or_else(|| permanent("stream usage details must be an object"))?;
            let retained = target
                .entry(*key)
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .expect("usage detail object");
            for count in ["cached_tokens", "reasoning_tokens"] {
                if let Some(value) = values.get(count).filter(|v| !v.is_null()) {
                    let n = value.as_u64().ok_or_else(|| {
                        permanent("stream usage count must be an unsigned integer")
                    })?;
                    usize::try_from(n)
                        .map_err(|_| permanent("stream usage count exceeds platform range"))?;
                    retained.insert(count.into(), value.clone());
                }
            }
        }
    }
    let after = serde_json::to_vec(&target)
        .map_err(|_| permanent("invalid usage state"))?
        .len();
    budget.add(after.saturating_sub(before))?;
    Ok(())
}
