use serde_json::Value;

use reimagine_agent_harness::{ModelCapability, ModelInfo, ModelName};

use crate::error::ProviderAdapterError;

/// Translate an OpenAI-compatible `/models` response JSON into
/// `Vec<ModelInfo>`.
pub fn from_openai_models(value: &Value) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
    let data = value
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            ProviderAdapterError::serialization("openai listing missing `data` array")
        })?;
    let mut out = Vec::with_capacity(data.len());
    for (i, entry) in data.iter().enumerate() {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ProviderAdapterError::serialization(format!(
                    "openai listing data[{i}].id missing or not a string"
                ))
            })?
            .to_string();
        out.push(
            ModelInfo::new(ModelName::new(id))
                .with_capabilities([ModelCapability::Chat, ModelCapability::ToolUse]),
        );
    }
    Ok(out)
}

/// Translate an Anthropic `/v1/models` response JSON into `Vec<ModelInfo>`.
pub fn from_anthropic_models(value: &Value) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
    let data = value
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            ProviderAdapterError::serialization("anthropic listing missing `data` array")
        })?;
    let mut out = Vec::with_capacity(data.len());
    for (i, entry) in data.iter().enumerate() {
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ProviderAdapterError::serialization(format!(
                    "anthropic listing data[{i}].id missing or not a string"
                ))
            })?
            .to_string();
        out.push(
            ModelInfo::new(ModelName::new(id))
                .with_capabilities([ModelCapability::Chat, ModelCapability::ToolUse]),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn listing_from_openai_models_extracts_ids_with_capabilities() {
        let value = json!({
            "object": "list",
            "data": [
                { "id": "gpt-4o-mini", "object": "model" },
                { "id": "gpt-4o", "object": "model" }
            ]
        });
        let models = from_openai_models(&value).expect("ok");
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name().as_str(), "gpt-4o-mini");
        assert_eq!(models[1].name().as_str(), "gpt-4o");
        for m in &models {
            assert!(m.capabilities().contains(&ModelCapability::Chat));
            assert!(m.capabilities().contains(&ModelCapability::ToolUse));
            assert!(m.provider().is_none());
        }
    }

    #[test]
    fn listing_from_openai_models_rejects_missing_data() {
        let value = json!({ "object": "list" });
        let err = from_openai_models(&value).expect_err("must reject");
        assert_eq!(
            err,
            ProviderAdapterError::serialization("openai listing missing `data` array")
        );
    }

    #[test]
    fn listing_from_anthropic_models_extracts_id_with_capabilities() {
        let value = json!({
            "data": [
                { "id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet" }
            ]
        });
        let models = from_anthropic_models(&value).expect("ok");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name().as_str(), "claude-3-5-sonnet-20241022");
        assert!(models[0].capabilities().contains(&ModelCapability::Chat));
        assert!(models[0].capabilities().contains(&ModelCapability::ToolUse));
    }

    #[test]
    fn listing_from_anthropic_models_empty_data_yields_empty_vec() {
        let value = json!({ "data": [] });
        let models = from_anthropic_models(&value).expect("ok");
        assert!(models.is_empty());
    }
}
