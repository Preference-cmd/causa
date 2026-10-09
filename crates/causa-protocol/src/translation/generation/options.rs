use causa_kernel::{GenerationOptions, ModelInvokeError, ModelInvokeErrorKind, ReasoningEffort};

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
    validate_bounds(generation, 2.0)?;
    if generation.temperature.is_some()
        && generation
            .reasoning_effort
            .is_some_and(|effort| effort != ReasoningEffort::None)
    {
        return Err(invalid(
            "temperature cannot be combined with active reasoning_effort",
        ));
    }
    Ok(())
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
    // The effort table maps every other value; `minimal` has no wire level.
    if generation.reasoning_effort == Some(ReasoningEffort::Minimal) {
        return Err(invalid(
            "reasoning_effort minimal has no Anthropic Messages equivalent",
        ));
    }
    if generation.temperature.is_some()
        && generation
            .reasoning_effort
            .is_some_and(|effort| effort != ReasoningEffort::None)
    {
        return Err(invalid(
            "temperature cannot be combined with active reasoning_effort",
        ));
    }
    if generation.output_schema.is_some() {
        return Err(invalid(
            "output_schema is unsupported by the Anthropic adapter",
        ));
    }
    Ok(())
}
