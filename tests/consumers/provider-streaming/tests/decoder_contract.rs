//! Adversarial pure-decoder checks through the public facade boundary.
#[path = "../../../../crates/causa-provider/tests/runtime_streaming_support/mod.rs"]
mod support;

use causa::kernel::{ModelInvokeError, ModelInvokeErrorKind, StreamDelta};
use causa::protocol::{Protocol, translation::streaming::StreamDecoder};
use causa::providers::StreamLimits;
use serde_json::{Value, json};
use support::*;

fn decode(
    protocol: Protocol,
    wire: &str,
    limits: StreamLimits,
) -> Result<Vec<StreamDelta>, ModelInvokeError> {
    let mut decoder = StreamDecoder::new(protocol, limits)?;
    let mut deltas = Vec::new();
    // Consumer feeds one UTF-8 byte at a time; boundaries are not events.
    for byte in wire.as_bytes() {
        let mut input = std::slice::from_ref(byte);
        while let Some(delta) = decoder.decode(&mut input)? {
            deltas.push(delta);
        }
    }
    let mut empty = [].as_slice();
    while let Some(delta) = decoder.decode(&mut empty)? {
        deltas.push(delta);
    }
    decoder.finish()?;
    Ok(deltas)
}

fn response_event(value: Value) -> String {
    named_event(value["type"].as_str().unwrap(), value.clone())
}

fn added_tool(index: u64, id: &str, call_id: &str) -> Value {
    json!({"type":"response.output_item.added","output_index":index,"item":{"type":"function_call","id":id,"call_id":call_id,"name":"echo","arguments":""}})
}

#[test]
fn native_output_indices_map_to_tool_ordinals_and_terminal_identity_cannot_rebind() {
    let prefix = responses_prefix("你好🙂")
        + &response_event(added_tool(1, "fc1", "call1"))
        + &response_event(added_tool(2, "fc2", "call2"));
    let response = json!({"type":"response.completed","response":{"id":"resp1","status":"completed","output":[
        {"type":"message","id":"msg1","role":"assistant","content":[{"type":"output_text","text":"final"}]},
        {"type":"function_call","id":"fc1","call_id":"call1","name":"echo","arguments":"{}"},
        {"type":"function_call","id":"fc2","call_id":"call2","name":"echo","arguments":"{\"message\":\"🙂\"}"}
    ]}});
    let deltas = decode(
        Protocol::OpenAiResponses,
        &(prefix.clone() + &response_event(response.clone())),
        StreamLimits::default(),
    )
    .unwrap();
    let identities: Vec<_> = deltas
        .iter()
        .filter_map(|delta| match delta {
            StreamDelta::ToolCallDelta {
                call_index,
                provider_call_id,
                ..
            } => Some((*call_index, provider_call_id.as_deref())),
            _ => None,
        })
        .collect();
    assert_eq!(identities, [(0, Some("call1")), (1, Some("call2"))]);
    let mut conflict = response;
    conflict["response"]["output"][1]["call_id"] = json!("rebound");
    assert_eq!(
        decode(
            Protocol::OpenAiResponses,
            &(prefix + &response_event(conflict)),
            StreamLimits::default()
        )
        .unwrap_err()
        .kind,
        ModelInvokeErrorKind::Permanent
    );
}

#[test]
fn unsupported_choices_giant_native_indices_and_truncated_arguments_fail_explicitly() {
    for (protocol, wire) in [
        (
            Protocol::OpenAiChatCompletions,
            event(json!({"choices":[{"index":1,"delta":{"content":"bad"},"finish_reason":null}]})),
        ),
        (
            Protocol::OpenAiResponses,
            response_event(added_tool(u64::MAX, "fc1", "call1")),
        ),
        (
            Protocol::OpenAiChatCompletions,
            chat_tool("{") + &chat_terminal("tool_calls"),
        ),
    ] {
        assert_eq!(
            decode(protocol, &wire, StreamLimits::default())
                .unwrap_err()
                .kind,
            ModelInvokeErrorKind::Permanent
        );
    }
}

#[test]
fn repeated_known_part_completion_is_an_invalid_sequence() {
    let closed = response_event(
        json!({"type":"response.content_part.done","output_index":0,"item_id":"msg1","content_index":0,"part":{"type":"output_text","text":"saved"}}),
    );
    let wire = responses_prefix("preview") + &closed + &closed + &responses_terminal("saved");
    assert_eq!(
        decode(Protocol::OpenAiResponses, &wire, StreamLimits::default())
            .unwrap_err()
            .kind,
        ModelInvokeErrorKind::Permanent
    );
}

#[test]
fn malformed_supported_part_snapshot_cannot_be_hidden_by_a_valid_terminal() {
    let prefix = response_event(
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg1","content":[]}}),
    );
    let malformed = response_event(
        json!({"type":"response.content_part.added","output_index":0,"item_id":"msg1","content_index":0,"part":{"type":"output_text","text":{}}}),
    );
    let wire = prefix + &malformed + &responses_terminal("saved");
    assert_eq!(
        decode(Protocol::OpenAiResponses, &wire, StreamLimits::default())
            .unwrap_err()
            .kind,
        ModelInvokeErrorKind::Permanent
    );
}

#[test]
fn cumulative_usage_snapshots_and_unknown_wire_bytes_are_not_double_counted_or_unbounded() {
    let wire = anthropic_prefix("a")
        + &named_event(
            "content_block_stop",
            json!({"type":"content_block_stop","index":0}),
        )
        + &named_event(
            "message_delta",
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}),
        )
        + &named_event(
            "message_delta",
            json!({"type":"message_delta","delta":{},"usage":{"output_tokens":3}}),
        )
        + &named_event("message_stop", json!({"type":"message_stop"}));
    let deltas = decode(Protocol::AnthropicMessages, &wire, StreamLimits::default()).unwrap();
    let output = deltas
        .into_iter()
        .find_map(|d| match d {
            StreamDelta::Done { final_output, .. } => Some(final_output),
            _ => None,
        })
        .unwrap();
    let usage = output.usage.unwrap();
    assert_eq!((usage.input_tokens, usage.output_tokens), (7, 3));
    let limits = StreamLimits {
        max_total_bytes: 80,
        ..StreamLimits::default()
    };
    let unknown = response_event(json!({"type":"future.unknown"})).repeat(20);
    assert_eq!(
        decode(Protocol::OpenAiResponses, &unknown, limits)
            .unwrap_err()
            .kind,
        ModelInvokeErrorKind::Permanent
    );
}
