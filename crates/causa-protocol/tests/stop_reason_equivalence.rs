//! Batch and streaming must classify a provider response identically. The
//! three parsers and their streaming decoders share one mapping table per
//! protocol (\`translation::stop_reason\`); this file pins that for every known
//! stop value and for one unknown value per protocol.

use causa_kernel::{ModelOutput, ModelStopReason, StreamDelta};
use causa_protocol::Protocol;
use causa_protocol::translation::streaming::{StreamDecoder, StreamLimits};
use serde_json::{Value, json};

/// Decode an SSE event stream (plus a Chat \`[DONE]\` terminator) and return the
/// terminal output.
fn decode(protocol: Protocol, events: &[Value]) -> ModelOutput {
    let mut bytes: Vec<u8> = events
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect::<String>()
        .into_bytes();
    if protocol == Protocol::OpenAiChatCompletions {
        bytes.extend_from_slice(b"data: [DONE]\n\n");
    }
    let mut decoder = StreamDecoder::new(protocol, StreamLimits::default()).unwrap();
    let mut done = None;
    for piece in bytes.chunks(7) {
        let mut input = piece;
        while let Some(delta) = decoder.decode(&mut input).unwrap() {
            if let StreamDelta::Done { final_output } = delta {
                done = Some(final_output);
            }
        }
    }
    decoder.finish().unwrap();
    done.expect("stream terminal")
}

fn anthropic_body(stop_reason: &str) -> Value {
    json!({"content": [], "stop_reason": stop_reason})
}

fn anthropic_events(stop_reason: &str) -> Vec<Value> {
    vec![
        json!({"type": "message_start", "message": {"content": []}}),
        json!({"type": "message_delta", "delta": {"stop_reason": stop_reason}}),
        json!({"type": "message_stop"}),
    ]
}

fn chat_body(finish_reason: &str) -> Value {
    json!({"choices": [{"message": {"content": ""}, "finish_reason": finish_reason}]})
}

fn chat_events(finish_reason: &str) -> Vec<Value> {
    vec![json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    })]
}

fn responses_body(status: &str, incomplete_reason: Option<&str>) -> Value {
    match incomplete_reason {
        Some(reason) => {
            json!({"status": status, "output": [], "incomplete_details": {"reason": reason}})
        }
        None => json!({"status": status, "output": []}),
    }
}

fn responses_events(body: &Value) -> Vec<Value> {
    let kind = match body["status"].as_str().expect("status") {
        "incomplete" => "response.incomplete",
        "completed" => "response.completed",
        other => panic!("unsupported fixture status {other}"),
    };
    vec![json!({"type": kind, "response": body})]
}

fn assert_same_classification(protocol: Protocol, batch: Value, streamed: ModelOutput) {
    let batch = match protocol {
        Protocol::AnthropicMessages => {
            causa_protocol::translation::anthropic::parse_anthropic_response(&batch).unwrap()
        }
        Protocol::OpenAiChatCompletions => {
            causa_protocol::translation::openai_chat::parse_openai_chat_response(&batch).unwrap()
        }
        Protocol::OpenAiResponses => {
            causa_protocol::translation::openai_responses::parse_openai_responses_output(&batch)
                .unwrap()
        }
    };
    assert_eq!(
        batch.stop_reason, streamed.stop_reason,
        "batch and streaming disagree on {protocol}"
    );
    assert_eq!(
        batch.raw_stop_reason, streamed.raw_stop_reason,
        "batch and streaming disagree on the raw stop value for {protocol}"
    );
}

#[test]
fn anthropic_batch_and_streaming_classify_every_value_identically() {
    for stop_reason in [
        "end_turn",
        "stop_sequence",
        "tool_use",
        "max_tokens",
        "refusal",
        "model_context_window_exceeded",
        "pause_turn",
        "unknown_value",
    ] {
        let streamed = decode(Protocol::AnthropicMessages, &anthropic_events(stop_reason));
        assert_same_classification(
            Protocol::AnthropicMessages,
            anthropic_body(stop_reason),
            streamed,
        );
        assert_eq!(
            decode(Protocol::AnthropicMessages, &anthropic_events(stop_reason))
                .raw_stop_reason
                .as_deref(),
            Some(stop_reason)
        );
    }
}

#[test]
fn chat_batch_and_streaming_classify_every_value_identically() {
    for finish_reason in [
        "stop",
        "end",
        "tool_calls",
        "function_call",
        "length",
        "content_filter",
        "unknown_value",
    ] {
        let streamed = decode(Protocol::OpenAiChatCompletions, &chat_events(finish_reason));
        assert_same_classification(
            Protocol::OpenAiChatCompletions,
            chat_body(finish_reason),
            streamed,
        );
        assert_eq!(
            decode(Protocol::OpenAiChatCompletions, &chat_events(finish_reason))
                .raw_stop_reason
                .as_deref(),
            Some(finish_reason)
        );
    }
}

#[test]
fn responses_batch_and_streaming_classify_every_value_identically() {
    for (status, reason, expected_raw) in [
        ("completed", None, "completed"),
        (
            "incomplete",
            Some("max_output_tokens"),
            "incomplete.max_output_tokens",
        ),
        (
            "incomplete",
            Some("content_filter"),
            "incomplete.content_filter",
        ),
        (
            "incomplete",
            Some("unknown_reason"),
            "incomplete.unknown_reason",
        ),
        // No reason on the wire: the raw value is the status alone. Nothing on
        // the wire said "unknown", so nothing may invent it.
        ("incomplete", None, "incomplete"),
    ] {
        let body = responses_body(status, reason);
        let streamed = decode(Protocol::OpenAiResponses, &responses_events(&body));
        assert_eq!(
            streamed.raw_stop_reason.as_deref(),
            Some(expected_raw),
            "Responses raw stop value for {status}/{reason:?}"
        );
        assert_same_classification(Protocol::OpenAiResponses, body, streamed);
    }
}

#[test]
fn unknown_values_never_commit_and_keep_their_wire_value() {
    // The kernel has no meaning for these: the output travels whole and the
    // runner does not continue. The classification is the shared marker.
    for (protocol, output) in [
        (
            Protocol::AnthropicMessages,
            decode(Protocol::AnthropicMessages, &anthropic_events("pause_turn")),
        ),
        (
            Protocol::OpenAiChatCompletions,
            decode(
                Protocol::OpenAiChatCompletions,
                &chat_events("brand_new_reason"),
            ),
        ),
        (
            Protocol::OpenAiResponses,
            decode(
                Protocol::OpenAiResponses,
                &responses_events(&responses_body("incomplete", Some("brand_new_reason"))),
            ),
        ),
    ] {
        assert_eq!(
            output.stop_reason,
            ModelStopReason::Unrecognized,
            "{protocol} must report an unknown stop value as Unrecognized"
        );
        assert!(
            output.raw_stop_reason.is_some(),
            "{protocol} must keep the wire stop value"
        );
    }
}
