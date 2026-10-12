use causa_kernel::{GenerationOptions, ModelInvokeError, ModelInvokeErrorKind};
use serde_json::Value;

fn invalid(message: &str) -> ModelInvokeError {
    ModelInvokeError::new(ModelInvokeErrorKind::InvalidRequest, message)
}

fn validate_bounds(
    generation: &GenerationOptions,
    temperature_max: f32,
) -> Result<(), ModelInvokeError> {
    if let Some(temperature) = generation.temperature
        && (!temperature.is_finite() || !(0.0..=temperature_max).contains(&temperature))
    {
        return Err(invalid(
            "temperature must be finite and within the protocol's supported range",
        ));
    }
    if generation.max_tokens == Some(0) {
        return Err(invalid("max_tokens must be greater than zero"));
    }
    Ok(())
}

pub(in crate::translation) fn validate_openai(
    generation: &GenerationOptions,
) -> Result<(), ModelInvokeError> {
    validate_bounds(generation, 2.0)
}

pub(in crate::translation) fn validate_anthropic(
    generation: &GenerationOptions,
) -> Result<(), ModelInvokeError> {
    validate_bounds(generation, 1.0)?;
    // Anthropic requires `max_tokens`; thinking counts toward it and is on by
    // default on current models, so the adapter invents no ceiling on the
    // host's spend.
    if generation.max_tokens.is_none() {
        return Err(invalid(
            "max_tokens is required by the Anthropic Messages API; the adapter sends no default",
        ));
    }
    if generation.output_schema.is_some() {
        return Err(invalid(
            "output_schema is unsupported by the Anthropic adapter",
        ));
    }
    Ok(())
}

/// Writes the host's extra request fields into the top level of a rendered
/// body. A key the renderer already wrote is left alone, so a host cannot
/// replace the model, the messages or the tools by accident; every other key
/// is copied unchanged and unchecked.
pub(in crate::translation) fn merge_extra_body(body: &mut Value, generation: &GenerationOptions) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    for (key, value) in &generation.extra_body {
        if !object.contains_key(key) {
            object.insert(key.clone(), value.clone());
        }
    }
}
