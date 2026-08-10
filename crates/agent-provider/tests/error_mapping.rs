use reimagine_agent_harness::{ProviderError, ProviderName};
use reimagine_ai_protocol::ProviderAdapterError;

#[test]
fn transport_error_carries_code_and_message() {
    let e = ProviderAdapterError::transport("connection refused");
    let mapped: ProviderError = e.to_provider_error(Some(ProviderName::new("openai")));
    assert_eq!(mapped.code(), "TRANSPORT");
    assert!(mapped.message().contains("connection refused"));
    assert_eq!(mapped.provider().map(|p| p.as_str()), Some("openai"));
}

#[test]
fn api_error_carries_upstream_code() {
    let e = ProviderAdapterError::api("RATE_LIMIT", "slow down");
    let mapped: ProviderError = e.to_provider_error(Some(ProviderName::new("openai")));
    assert_eq!(mapped.code(), "RATE_LIMIT");
    assert_eq!(mapped.message(), "slow down");
}

#[test]
fn serialization_error_carries_message() {
    let e = ProviderAdapterError::serialization("bad json");
    let mapped: ProviderError = e.to_provider_error(Some(ProviderName::new("openai")));
    assert_eq!(mapped.code(), "SERIALIZATION");
    assert_eq!(mapped.message(), "bad json");
}

#[test]
fn configuration_error_carries_message() {
    let e = ProviderAdapterError::configuration("missing api_key");
    let mapped: ProviderError = e.to_provider_error(Some(ProviderName::new("anthropic_messages")));
    assert_eq!(mapped.code(), "CONFIGURATION");
    assert_eq!(mapped.message(), "missing api_key");
}

#[test]
fn streaming_unsupported_carries_distinct_code() {
    let e = ProviderAdapterError::streaming_unsupported();
    let mapped: ProviderError = e.to_provider_error(Some(ProviderName::new("openai")));
    assert_eq!(mapped.code(), "STREAMING_UNSUPPORTED");
}

#[test]
fn missing_config_carries_provider_name_and_protocol() {
    let e = ProviderAdapterError::MissingConfig {
        provider: "broken".into(),
        protocol: reimagine_agent_provider::Protocol::AnthropicMessages,
    };
    let s = format!("{e}");
    assert!(s.contains("broken"));
    assert!(s.contains("anthropic_messages"));
}

#[test]
fn missing_config_maps_to_configuration_error() {
    let e = ProviderAdapterError::MissingConfig {
        provider: "broken".into(),
        protocol: reimagine_agent_provider::Protocol::AnthropicMessages,
    };
    let mapped: ProviderError = e.to_provider_error(None);
    assert_eq!(mapped.code(), "CONFIGURATION");
    assert!(mapped.message().contains("broken"));
}

#[test]
fn display_includes_code_and_message() {
    let e = ProviderAdapterError::api("AUTH", "bad key");
    let s = format!("{e}");
    assert!(s.contains("AUTH"));
    assert!(s.contains("bad key"));
}
