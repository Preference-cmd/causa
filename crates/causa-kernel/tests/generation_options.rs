//! Serializable generation choices distinguish omission from explicit effort.
use causa_kernel::{GenerationOptions, ReasoningEffort};
use serde_json::json;

#[test]
fn effort_vocabulary_round_trips_without_inventing_a_default() {
    let unspecified: GenerationOptions = serde_json::from_value(json!({})).unwrap();
    assert_eq!(unspecified, GenerationOptions::default());
    assert!(
        serde_json::to_value(&unspecified)
            .unwrap()
            .get("reasoning_effort")
            .is_none()
    );
    for (effort, wire) in [
        (ReasoningEffort::None, "none"),
        (ReasoningEffort::Minimal, "minimal"),
        (ReasoningEffort::Low, "low"),
        (ReasoningEffort::Medium, "medium"),
        (ReasoningEffort::High, "high"),
        (ReasoningEffort::XHigh, "xhigh"),
        (ReasoningEffort::Max, "max"),
    ] {
        let value = {
            let mut generation = GenerationOptions::default();
            generation.reasoning_effort = Some(effort);
            generation
        };
        let encoded = serde_json::to_value(&value).unwrap();
        assert_eq!(encoded["reasoning_effort"], wire);
        assert_eq!(
            serde_json::from_value::<GenerationOptions>(encoded).unwrap(),
            value
        );
        assert_ne!(value, unspecified);
    }
    for unknown in ["ultra", "HIGH", "default", ""] {
        assert!(
            serde_json::from_value::<GenerationOptions>(json!({"reasoning_effort": unknown}))
                .is_err()
        );
    }
}
