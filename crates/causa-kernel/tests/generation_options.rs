//! Serializable generation choices distinguish omission from an explicit
//! value and keep whatever the host supplied.
use causa_kernel::GenerationOptions;
use serde_json::json;

#[test]
fn effort_is_any_string_and_round_trips_without_inventing_a_default() {
    let unspecified: GenerationOptions = serde_json::from_value(json!({})).unwrap();
    assert_eq!(unspecified, GenerationOptions::default());
    let encoded = serde_json::to_value(&unspecified).unwrap();
    assert!(encoded.get("reasoning_effort").is_none());
    assert!(encoded.get("extra_body").is_none());
    // Values written by any version read back, known to this one or not.
    for effort in ["none", "minimal", "xhigh", "off", "ultra", "HIGH", ""] {
        let value = GenerationOptions::default().with_reasoning_effort(effort);
        let encoded = serde_json::to_value(&value).unwrap();
        assert_eq!(encoded["reasoning_effort"], effort);
        assert_eq!(
            serde_json::from_value::<GenerationOptions>(encoded).unwrap(),
            value
        );
        assert_ne!(value, unspecified);
    }
}

#[test]
fn extra_request_fields_round_trip_as_given() {
    let mut value = GenerationOptions::default();
    value.extra_body.insert(
        "thinking".into(),
        json!({"type": "adaptive", "display": "summarized"}),
    );
    value.extra_body.insert("top_k".into(), json!(40));
    let encoded = serde_json::to_value(&value).unwrap();
    assert_eq!(
        encoded["extra_body"],
        json!({"thinking": {"type": "adaptive", "display": "summarized"}, "top_k": 40})
    );
    assert_eq!(
        serde_json::from_value::<GenerationOptions>(encoded).unwrap(),
        value
    );
}
