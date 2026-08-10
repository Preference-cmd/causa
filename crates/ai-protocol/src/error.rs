//! Adapter-internal error types and their conversion into the
//! Reimagine-owned `reimagine_agent_harness::ProviderError`.
//!
//! V1 categories: transport, API, serialization, configuration, and
//! `streaming_unsupported`. Each variant carries the upstream message
//! verbatim so hosts can show it without losing fidelity.

use reimagine_agent_harness::{ProviderError, ProviderName};

use crate::protocol::Protocol;

/// Internal error type for the `agent-provider` crate. The public
/// provider boundary is `reimagine_agent_harness::ProviderError`; this type is
/// only used inside the crate and at the constructor boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderAdapterError {
    /// Network / transport-level failure reaching the upstream.
    Transport(String),
    /// Provider returned a structured error response.
    Api { code: String, message: String },
    /// Request or response JSON did not match the expected shape.
    Serialization(String),
    /// Local configuration was missing or invalid.
    Configuration(String),
    /// Streaming was requested but the adapter / backend does not
    /// support it. Distinct from `Transport` so the agent loop can
    /// decide whether to fall back.
    StreamingUnsupported,
    /// A provider config was missing the inner typed config matching
    /// the discriminator.
    MissingConfig {
        provider: String,
        protocol: Protocol,
    },
}

impl ProviderAdapterError {
    pub fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }

    pub fn api(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Api {
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization(message.into())
    }

    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration(message.into())
    }

    pub fn streaming_unsupported() -> Self {
        Self::StreamingUnsupported
    }

    /// Returns `true` if this error is transient and may succeed on retry.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport(_) => true,
            Self::Api { code, .. } => {
                // 429 (rate limit) and 5xx (server error) are retryable.
                code.parse::<u16>()
                    .map(|c| c == 429 || (500..600).contains(&c))
                    .unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Convert into the Reimagine-owned `ProviderError`. `provider` is
    /// attached when known so diagnostics carry it.
    pub fn to_provider_error(&self, provider: Option<ProviderName>) -> ProviderError {
        let (code, message) = match self {
            Self::Transport(m) => ("TRANSPORT".to_string(), m.clone()),
            Self::Api { code, message } => (code.clone(), message.clone()),
            Self::Serialization(m) => ("SERIALIZATION".to_string(), m.clone()),
            Self::Configuration(m) => ("CONFIGURATION".to_string(), m.clone()),
            Self::StreamingUnsupported => (
                "STREAMING_UNSUPPORTED".to_string(),
                "this provider adapter does not support streaming".to_string(),
            ),
            Self::MissingConfig { provider, protocol } => (
                "CONFIGURATION".to_string(),
                format!("provider `{provider}` is missing config for protocol `{protocol}`"),
            ),
        };
        let mut err = ProviderError::new(code, message);
        if let Some(p) = provider {
            err = err.with_provider(p);
        }
        err
    }
}

impl std::fmt::Display for ProviderAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(m) => write!(f, "[TRANSPORT] {m}"),
            Self::Api { code, message } => write!(f, "[API:{code}] {message}"),
            Self::Serialization(m) => write!(f, "[SERIALIZATION] {m}"),
            Self::Configuration(m) => write!(f, "[CONFIGURATION] {m}"),
            Self::StreamingUnsupported => write!(f, "[STREAMING_UNSUPPORTED]"),
            Self::MissingConfig { provider, protocol } => {
                write!(
                    f,
                    "[CONFIGURATION] provider `{provider}` missing config for `{protocol}`"
                )
            }
        }
    }
}

impl std::error::Error for ProviderAdapterError {}
