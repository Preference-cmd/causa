use causa_kernel::{
    ModelInvokeError, ModelInvokeErrorKind, ModelOutput, ModelStopReason, StreamDelta,
};
use serde_json::{Value, json};

use super::{StreamDecoder, StreamLimits, configure_stream_request};
use crate::{
    Protocol,
    translation::{
        anthropic::parse_anthropic_response, openai_chat::parse_openai_chat_response,
        openai_responses::parse_openai_responses_output,
    },
};

fn wire(events: &[Value]) -> Vec<u8> {
    events
        .iter()
        .map(|v| format!("data: {v}\n\n"))
        .collect::<String>()
        .into_bytes()
}

fn collect(
    protocol: Protocol,
    bytes: &[u8],
    chunks: usize,
    limits: StreamLimits,
) -> Result<Vec<StreamDelta>, ModelInvokeError> {
    let mut decoder = StreamDecoder::new(protocol, limits)?;
    let mut deltas = Vec::new();
    for chunk in bytes.chunks(chunks.max(1)) {
        let mut input = chunk;
        while let Some(delta) = decoder.decode(&mut input)? {
            deltas.push(delta);
        }
    }
    decoder.finish()?;
    assert!(decoder.is_terminal());
    assert!(
        decoder
            .decode(&mut b"invalid trailing data".as_slice())?
            .is_none()
    );
    Ok(deltas)
}

fn output(deltas: &[StreamDelta]) -> &ModelOutput {
    let terminals: Vec<_> = deltas
        .iter()
        .filter_map(|delta| match delta {
            StreamDelta::Done { final_output, .. } => Some(final_output),
            _ => None,
        })
        .collect();
    assert_eq!(terminals.len(), 1);
    terminals[0]
}

fn paired(protocol: Protocol, bytes: &[u8], expected: &ModelOutput) -> Vec<StreamDelta> {
    let expected = serde_json::to_value(expected).unwrap();
    let mut result = Vec::new();
    for size in [1, 2, 3, 7, 31, bytes.len()] {
        result = collect(protocol, bytes, size, StreamLimits::default()).unwrap();
        assert_eq!(
            serde_json::to_value(output(&result)).unwrap(),
            expected,
            "chunk size {size}"
        );
    }
    result
}

#[test]
fn chat_interleaved_tools_and_trailing_usage_match_full_response() {
    let events = [
        json!({"id":"chat1","choices":[{"index":0,"delta":{"content":"你好🦀"},"finish_reason":null}]}),
        json!({"id":"chat1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c0","type":"function","function":{"name":"first","arguments":"{\"a\":"}},{"index":1,"id":"c1","function":{"name":"second","arguments":"{"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"b\":2}"}},{"index":0,"function":{"arguments":"1}"}}]}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":7}}}),
    ];
    let mut bytes = wire(&events);
    bytes.extend_from_slice(b"data: [DONE]\n\n");
    let expected = parse_openai_chat_response(&json!({"choices":[{"message":{"content":"你好🦀","role":"assistant","tool_calls":[{"id":"c0","type":"function","function":{"name":"first","arguments":"{\"a\":1}"}},{"id":"c1","type":"function","function":{"name":"second","arguments":"{\"b\":2}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":7}}})).unwrap();
    let deltas = paired(Protocol::OpenAiChatCompletions, &bytes, &expected);
    assert!(deltas.iter().any(|d| matches!(d,StreamDelta::ToolCallDelta{call_index:1,arguments_delta:Some(a),..} if a=="\"b\":2}")));
    assert!(matches!(&deltas[deltas.len()-2],StreamDelta::Usage(u) if u.output_tokens==3));
}

#[test]
fn anthropic_mixed_blocks_signature_and_cumulative_usage_match_full_response() {
    let events = [
        json!({"type":"message_start","message":{"content":[],"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":4}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"你好🦀"}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"c0","name":"first","input":{}}}),
        json!({"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"c1","name":"second","input":{}}}),
        json!({"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"b\":2}"}}),
        json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}),
        json!({"type":"content_block_stop","index":2}),
        json!({"type":"content_block_stop","index":3}),
        json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}),
        json!({"type":"message_delta","delta":{},"usage":{"output_tokens":7}}),
        json!({"type":"message_stop"}),
    ];
    let expected = parse_anthropic_response(&json!({"content":[{"type":"thinking","thinking":"think","signature":"sig"},{"type":"text","text":"你好🦀"},{"type":"tool_use","id":"c0","name":"first","input":{"a":1}},{"type":"tool_use","id":"c1","name":"second","input":{"b":2}}],"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":7,"cache_read_input_tokens":4}})).unwrap();
    let deltas = paired(Protocol::AnthropicMessages, &wire(&events), &expected);
    let usages: Vec<_> = deltas
        .iter()
        .filter_map(|d| match d {
            StreamDelta::Usage(u) => Some(u.output_tokens),
            _ => None,
        })
        .collect();
    assert_eq!(usages, [1, 5, 7]);
    assert!(deltas.iter().any(|d| matches!(d,StreamDelta::ToolCallDelta{call_index:1,provider_call_id:Some(id),..} if id=="c1")));
}

fn responses_fixture() -> (Vec<Value>, Value) {
    let response = json!({"id":"r1","status":"completed","output":[{"type":"message","id":"m0","content":[{"type":"output_text","text":"authoritative"}]},{"type":"reasoning","id":"reason","summary":[{"type":"summary_text","text":"reason final"}]},{"type":"function_call","id":"fc0","call_id":"c0","name":"first","arguments":"{\"a\":1}"},{"type":"function_call","id":"fc1","call_id":"c1","name":"second","arguments":"{\"b\":2}"}],"usage":{"input_tokens":9,"output_tokens":5,"output_tokens_details":{"reasoning_tokens":2}}});
    let events = vec![
        json!({"type":"response.created","response":{"id":"r1","status":"in_progress"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m0","content":[]}}),
        json!({"type":"response.content_part.added","output_index":0,"item_id":"m0","content_index":0,"part":{"type":"output_text","text":""}}),
        json!({"type":"response.output_text.delta","output_index":0,"item_id":"m0","content_index":0,"delta":"draft"}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"type":"reasoning","id":"reason","summary":[]}}),
        json!({"type":"response.reasoning_summary_part.added","output_index":1,"item_id":"reason","summary_index":0,"part":{"type":"summary_text","text":""}}),
        json!({"type":"response.reasoning_summary_text.delta","output_index":1,"item_id":"reason","summary_index":0,"delta":"reason draft"}),
        json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc0","call_id":"c0","name":"first","arguments":""}}),
        json!({"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","id":"fc1","call_id":"c1","name":"second","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","output_index":3,"item_id":"fc1","delta":"{\"b\":"}),
        json!({"type":"response.function_call_arguments.delta","output_index":2,"item_id":"fc0","delta":"{\"a\":1}"}),
        json!({"type":"response.function_call_arguments.delta","output_index":3,"item_id":"fc1","delta":"2}"}),
        json!({"type":"response.function_call_arguments.done","output_index":2,"item_id":"fc0","arguments":"{\"a\":1}"}),
        json!({"type":"response.output_item.done","output_index":2,"item":response["output"][2]}),
        json!({"type":"response.completed","response":response}),
    ];
    (events, response)
}

#[test]
fn responses_authoritative_snapshots_and_tool_ordinals_match_full_response() {
    let (events, response) = responses_fixture();
    let expected = parse_openai_responses_output(&response).unwrap();
    let deltas = paired(Protocol::OpenAiResponses, &wire(&events), &expected);
    assert!(
        deltas
            .iter()
            .any(|d| matches!(d,StreamDelta::TextDelta{delta} if delta=="draft"))
    );
    assert!(deltas.iter().any(|d| matches!(d,StreamDelta::ToolCallDelta{call_index:1,provider_call_id:Some(id),..} if id=="c1")));
}

#[test]
fn responses_parts_and_calls_that_open_without_their_empty_string_decode_the_same() {
    // Observed on a compatible server: an opening part carries no `text` and
    // an opening function call no `arguments`, where the reference wire
    // sends an empty string. Both mean "nothing yet".
    let (mut events, response) = responses_fixture();
    for event in &mut events {
        match event["type"].as_str() {
            Some("response.content_part.added" | "response.reasoning_summary_part.added") => {
                event["part"]
                    .as_object_mut()
                    .unwrap()
                    .remove("text")
                    .unwrap();
            }
            Some("response.output_item.added") if event["item"]["type"] == "function_call" => {
                event["item"]
                    .as_object_mut()
                    .unwrap()
                    .remove("arguments")
                    .unwrap();
            }
            _ => {}
        }
    }
    let expected = parse_openai_responses_output(&response).unwrap();
    paired(Protocol::OpenAiResponses, &wire(&events), &expected);

    // An explicit null opens the same way as a missing key.
    let mut nulled = events.clone();
    for event in &mut nulled {
        match event["type"].as_str() {
            Some("response.content_part.added" | "response.reasoning_summary_part.added") => {
                event["part"]["text"] = Value::Null;
            }
            Some("response.output_item.added") if event["item"]["type"] == "function_call" => {
                event["item"]["arguments"] = Value::Null;
            }
            _ => {}
        }
    }
    paired(Protocol::OpenAiResponses, &wire(&nulled), &expected);

    // A refusal part opens without its `refusal` string, or with a null.
    for part in [
        json!({"type":"refusal"}),
        json!({"type":"refusal","refusal":null}),
    ] {
        let refused = json!({"status":"completed","output":[
            {"type":"message","id":"m","content":[{"type":"refusal","refusal":"no"}]},
        ]});
        let stream = [
            json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m","content":[]}}),
            json!({"type":"response.content_part.added","output_index":0,"item_id":"m","content_index":0,"part":part}),
            json!({"type":"response.refusal.delta","output_index":0,"item_id":"m","content_index":0,"delta":"no"}),
            json!({"type":"response.completed","response":refused}),
        ];
        let expected = parse_openai_responses_output(&refused).unwrap();
        assert_eq!(expected.stop_reason, ModelStopReason::Refusal);
        paired(Protocol::OpenAiResponses, &wire(&stream), &expected);
    }

    // A value that is present must still be a string.
    events[2]["part"]["text"] = json!(7);
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
}

fn fail(protocol: Protocol, events: &[Value]) -> ModelInvokeErrorKind {
    collect(protocol, &wire(events), usize::MAX, StreamLimits::default())
        .unwrap_err()
        .kind
}

#[test]
fn explicit_responses_terminal_classifications_are_shared_with_full_parser() {
    {
        let response = json!({"status":"failed","output":[],"error":{"code":"new_error"}});
        let batch = parse_openai_responses_output(&response).unwrap_err().kind;
        let streamed = fail(
            Protocol::OpenAiResponses,
            &[json!({"type":"response.failed","response":response})],
        );
        assert_eq!(batch, streamed);
        assert_eq!(batch, ModelInvokeErrorKind::Permanent);
    }
    // An incomplete response is data even when its reason is unknown: both
    // entry points report Unrecognized and keep the wire reason.
    {
        let response = json!({
            "status":"incomplete",
            "output":[],
            "incomplete_details":{"reason":"new_reason"},
        });
        let expected = parse_openai_responses_output(&response).unwrap();
        assert_eq!(expected.stop_reason, ModelStopReason::Unrecognized);
        assert_eq!(
            expected.raw_stop_reason.as_deref(),
            Some("incomplete.new_reason")
        );
        paired(
            Protocol::OpenAiResponses,
            &wire(&[json!({"type":"response.incomplete","response":response})]),
            &expected,
        );
    }
    for (reason, expected) in [
        ("max_output_tokens", "MaxTokens"),
        // The Chat word, passed through by servers translating a Chat backend.
        ("length", "MaxTokens"),
        ("content_filter", "Refusal"),
    ] {
        let response =
            json!({"status":"incomplete","output":[],"incomplete_details":{"reason":reason}});
        let expected_output = parse_openai_responses_output(&response).unwrap();
        assert_eq!(format!("{:?}", expected_output.stop_reason), expected);
        paired(
            Protocol::OpenAiResponses,
            &wire(&[json!({"type":"response.incomplete","response":response})]),
            &expected_output,
        );
    }
    assert_eq!(
        parse_openai_responses_output(
            &json!({"status":"failed","error":{"code":"rate_limit_exceeded"}})
        )
        .unwrap_err()
        .kind,
        ModelInvokeErrorKind::Transient
    );
}

#[test]
fn malformed_order_ids_and_arguments_fail_as_undecodable() {
    let (mut events, _) = responses_fixture();
    events[3]["item_id"] = json!("other");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
    let (mut events, _) = responses_fixture();
    events.last_mut().unwrap()["response"]["output"][2]["call_id"] = json!("other");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
    let (mut events, _) = responses_fixture();
    events.last_mut().unwrap()["response"]["output"][3]["arguments"] = json!("{");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
    let (mut events, _) = responses_fixture();
    events[13]["item"]["name"] = json!("changed");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
    assert_eq!(
        fail(
            Protocol::AnthropicMessages,
            &[json!({"type":"message_stop"})]
        ),
        ModelInvokeErrorKind::Undecodable
    );
    assert_eq!(
        fail(
            Protocol::OpenAiChatCompletions,
            &[json!({"choices":[{"index":1,"delta":{}}]})]
        ),
        ModelInvokeErrorKind::Undecodable
    );
}

#[test]
fn eof_unknown_events_and_errors_fuse() {
    assert_eq!(
        fail(
            Protocol::OpenAiResponses,
            &[json!({"type":"unknown_event"})]
        ),
        ModelInvokeErrorKind::UnknownOutcome
    );
    let mut decoder =
        StreamDecoder::new(Protocol::OpenAiChatCompletions, StreamLimits::default()).unwrap();
    assert_eq!(
        decoder
            .decode(&mut b"data: [DONE]\n\n".as_slice())
            .unwrap_err()
            .kind,
        ModelInvokeErrorKind::Undecodable
    );
    assert!(
        decoder
            .decode(&mut b"data: [DONE]\n\n".as_slice())
            .unwrap()
            .is_none()
    );
    assert!(!decoder.is_terminal());
    let mut decoder =
        StreamDecoder::new(Protocol::OpenAiResponses, StreamLimits::default()).unwrap();
    assert!(decoder.decode(&mut b"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[]}}".as_slice()).unwrap().is_none());
    assert_eq!(
        decoder.finish().unwrap_err().kind,
        ModelInvokeErrorKind::UnknownOutcome
    );
}

#[test]
fn all_resource_boundaries_and_sparse_indices_fail() {
    let (events, _) = responses_fixture();
    for limits in [
        StreamLimits {
            max_output_bytes: 20,
            ..StreamLimits::default()
        },
        StreamLimits {
            max_items: 2,
            ..StreamLimits::default()
        },
        StreamLimits {
            max_pending_deltas: 1,
            ..StreamLimits::default()
        },
        StreamLimits {
            max_total_bytes: 20,
            ..StreamLimits::default()
        },
    ] {
        assert_eq!(
            collect(
                Protocol::OpenAiResponses,
                &wire(&events),
                usize::MAX,
                limits
            )
            .unwrap_err()
            .kind,
            ModelInvokeErrorKind::Undecodable
        );
    }
    let events = [
        json!({"type":"response.output_item.added","output_index":u64::MAX,"item":{"type":"message","id":"m0"}}),
    ];
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
    let comments = b": heartbeat\n\n: heartbeat\n\n";
    assert_eq!(
        collect(
            Protocol::OpenAiResponses,
            comments,
            usize::MAX,
            StreamLimits {
                max_total_bytes: 10,
                ..StreamLimits::default()
            }
        )
        .unwrap_err()
        .kind,
        ModelInvokeErrorKind::Undecodable
    );
}

#[test]
fn streaming_request_options_preserve_existing_fields() {
    let mut body = json!({"model":"m","stream_options":{"existing":true}});
    configure_stream_request(Protocol::OpenAiChatCompletions, &mut body).unwrap();
    assert_eq!(
        body,
        json!({"model":"m","stream":true,"stream_options":{"existing":true,"include_usage":true}})
    );
    // On a stream call both settings are the adapter's, whatever the body
    // already says.
    for given in [
        json!({"include_usage": false}),
        json!("usage"),
        json!(null),
        json!([]),
    ] {
        let mut body = json!({"model":"m","stream":false,"stream_options":given});
        configure_stream_request(Protocol::OpenAiChatCompletions, &mut body).unwrap();
        assert_eq!(
            body,
            json!({"model":"m","stream":true,"stream_options":{"include_usage":true}})
        );
    }
    for protocol in [Protocol::OpenAiResponses, Protocol::AnthropicMessages] {
        let mut body = json!({"model":"m","stream":false,"stream_options":"kept"});
        configure_stream_request(protocol, &mut body).unwrap();
        assert_eq!(
            body,
            json!({"model":"m","stream":true,"stream_options":"kept"})
        );
    }
    assert!(
        StreamDecoder::new(
            Protocol::OpenAiResponses,
            StreamLimits {
                max_items: 0,
                ..StreamLimits::default()
            }
        )
        .is_err()
    );
}

/// A reasoning item that streams its raw reasoning as `reasoning_text`
/// content parts, as DeepSeek's Responses API does, then a function call.
fn raw_reasoning_fixture() -> (Vec<Value>, Value) {
    let reasoning = json!({"type":"reasoning","id":"rs","status":"completed","content":[{"type":"reasoning_text","text":"6 * 7 = 42."}],"summary":[],"encrypted_content":"opaque"});
    let call = json!({"type":"function_call","id":"fc","call_id":"c0","name":"unlock","arguments":"{\"code\":42}","status":"completed"});
    let response = json!({"id":"r1","status":"completed","output":[reasoning,call],"usage":{"input_tokens":9,"output_tokens":5}});
    let at = |kind: &str, rest: Value| {
        let mut event = json!({"type":kind,"output_index":0,"item_id":"rs","content_index":0});
        event
            .as_object_mut()
            .unwrap()
            .extend(rest.as_object().unwrap().clone());
        event
    };
    let events = vec![
        json!({"type":"response.created","response":{"id":"r1","status":"in_progress"}}),
        json!({"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs","status":"in_progress","content":[],"summary":[],"encrypted_content":"opaque"}}),
        at(
            "response.content_part.added",
            json!({"part":{"type":"reasoning_text","text":""}}),
        ),
        at("response.reasoning_text.delta", json!({"delta":"6 * 7"})),
        at("response.reasoning_text.delta", json!({"delta":" = 42."})),
        at(
            "response.reasoning_text.done",
            json!({"text":"6 * 7 = 42."}),
        ),
        at(
            "response.content_part.done",
            json!({"part":{"type":"reasoning_text","text":"6 * 7 = 42."}}),
        ),
        json!({"type":"response.output_item.done","output_index":0,"item":reasoning}),
        json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc","call_id":"c0","name":"unlock","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc","delta":"{\"code\":42}"}),
        json!({"type":"response.function_call_arguments.done","output_index":1,"item_id":"fc","arguments":"{\"code\":42}"}),
        json!({"type":"response.output_item.done","output_index":1,"item":call}),
        json!({"type":"response.completed","response":response}),
    ];
    (events, response)
}

#[test]
fn responses_raw_reasoning_text_parts_match_full_response() {
    let (events, response) = raw_reasoning_fixture();
    let expected = parse_openai_responses_output(&response).unwrap();
    // The batch form shows the raw text when there is no summary, and keeps
    // the whole reasoning item sealed for replay.
    assert_eq!(expected.reasoning.as_ref().unwrap().text, "6 * 7 = 42.");
    let sealed: Vec<_> = expected
        .response
        .items
        .iter()
        .filter_map(|item| match &item.content {
            causa_kernel::ModelItemContent::Parts(parts) => Some(parts),
            _ => None,
        })
        .flatten()
        .filter_map(|part| match part {
            causa_kernel::ContentPart::Sealed(sealed) => Some(&sealed.item),
            _ => None,
        })
        .collect();
    assert_eq!(sealed, [&response["output"][0]]);
    let deltas = paired(Protocol::OpenAiResponses, &wire(&events), &expected);
    let reasoning: Vec<_> = deltas
        .iter()
        .filter_map(|delta| match delta {
            StreamDelta::ReasoningDelta { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, ["6 * 7", " = 42."]);
    assert_eq!(call_names(output(&deltas)), ["unlock"]);
}

#[test]
fn reasoning_text_parts_keep_the_part_rules() {
    let undecodable = |events: &[Value]| {
        assert_eq!(
            fail(Protocol::OpenAiResponses, events),
            ModelInvokeErrorKind::Undecodable,
            "{events:#?}"
        );
    };
    // A reasoning_text part on a message.
    let (mut events, _) = responses_fixture();
    events[2]["part"]["type"] = json!("reasoning_text");
    undecodable(&events);
    // Any other part type on a reasoning item.
    let (mut events, _) = raw_reasoning_fixture();
    events[2]["part"]["type"] = json!("output_text");
    undecodable(&events);
    // Any content part on a function call.
    let (mut events, _) = raw_reasoning_fixture();
    events.insert(
        9,
        json!({"type":"response.content_part.added","output_index":1,"item_id":"fc","content_index":0,"part":{"type":"reasoning_text","text":""}}),
    );
    undecodable(&events);
    // A delta before its part opens.
    let (mut events, _) = raw_reasoning_fixture();
    events.swap(2, 3);
    undecodable(&events);
    // A delta after its part is done.
    let (mut events, _) = raw_reasoning_fixture();
    let late = events[4].clone();
    events.insert(6, late);
    undecodable(&events);
    // A second done for the same part.
    let (mut events, _) = raw_reasoning_fixture();
    let again = events[5].clone();
    events.insert(6, again);
    undecodable(&events);
    // A part out of content_index order.
    let (mut events, _) = raw_reasoning_fixture();
    events[2]["content_index"] = json!(1);
    undecodable(&events);
    // A completed part whose type changed.
    let (mut events, _) = raw_reasoning_fixture();
    events[6]["part"]["type"] = json!("summary_text");
    undecodable(&events);
}

#[test]
fn duplicate_parts_and_malformed_known_initial_shapes_fail() {
    let (mut events, _) = responses_fixture();
    let completed = json!({"type":"response.content_part.done","output_index":0,"item_id":"m0","content_index":0,"part":{"type":"output_text","text":"draft"}});
    events.insert(4, completed.clone());
    events.insert(5, completed);
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
    let (mut events, _) = responses_fixture();
    events[2]["part"]["text"] = json!({"wrong":"type"});
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
    let (mut events, _) = responses_fixture();
    events[1]["item"]["content"] = json!("wrong");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Undecodable
    );
}

#[test]
fn usage_partial_nested_snapshots_preserve_counts_and_ignore_extensions() {
    let mut bytes = wire(&[
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":4},"extension":"x".repeat(1000)}}),
        json!({"choices":[],"usage":{"completion_tokens":5,"prompt_tokens_details":{}}}),
    ]);
    bytes.extend_from_slice(b"data: [DONE]\n\n");
    let deltas = collect(
        Protocol::OpenAiChatCompletions,
        &bytes,
        17,
        StreamLimits {
            max_output_bytes: 200,
            ..StreamLimits::default()
        },
    )
    .unwrap();
    let usage = output(&deltas).usage.as_ref().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens
        ),
        (10, 5, Some(4))
    );
}

#[test]
fn unknown_anthropic_block_extensions_are_retained_as_sealed_items() {
    let events = [
        json!({"type":"message_start","message":{"content":[]}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"opaque","data":"secret".repeat(1000)}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"opaque","data":"secret".repeat(1000)}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ];
    // Unknown assistant blocks are sealed material now: with a sufficient
    // budget the block survives whole, in place, and the Anthropic renderer
    // writes it back byte-identically (the stored item is the block itself).
    let deltas = collect(
        Protocol::AnthropicMessages,
        &wire(&events),
        usize::MAX,
        StreamLimits::default(),
    )
    .unwrap();
    let out = output(&deltas);
    assert_eq!(out.response.text(), "");
    let sealed: Vec<Value> = out
        .response
        .items
        .iter()
        .flat_map(|item| match &item.content {
            causa_kernel::ModelItemContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    causa_kernel::ContentPart::Sealed(sealed) => Some(sealed.item.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    assert_eq!(
        sealed,
        vec![
            json!({"type": "opaque", "data": "secret".repeat(1000)}),
            json!({"type": "opaque", "data": "secret".repeat(1000)}),
        ]
    );
    let block = causa_kernel::ContextBlock::new(
        causa_kernel::BlockId::new(uuid::Uuid::from_u128(7)),
        causa_kernel::BlockContent::Parts(
            sealed
                .iter()
                .map(|item| {
                    causa_kernel::ContentPart::Sealed(causa_kernel::SealedPart::new(
                        Protocol::AnthropicMessages.as_str(),
                        item.clone(),
                    ))
                })
                .collect(),
        ),
        causa_kernel::BlockMeta::default(),
    );
    let rendered = crate::translation::anthropic::render_anthropic_messages(
        &causa_kernel::ContextFrame {
            blocks: vec![
                causa_kernel::ContextBlock::new(
                    causa_kernel::BlockId::new(uuid::Uuid::from_u128(1)),
                    causa_kernel::BlockContent::Parts(vec![causa_kernel::ContentPart::Text(
                        causa_kernel::TextPayload::new("hi"),
                    )]),
                    causa_kernel::BlockMeta::from_source("user"),
                ),
                block,
            ],
        },
        &crate::translation::media::MediaSet::new(),
        &causa_kernel::GenerationOptions::default().with_max_tokens(64),
        &causa_kernel::ModelRef::new("m"),
        causa_kernel::CacheDirective::None,
        &crate::translation::render_options::AnthropicRenderOptions::default(),
    )
    .unwrap();
    assert_eq!(
        rendered["messages"][1]["content"],
        json!([
            {"type": "opaque", "data": "secret".repeat(1000)},
            {"type": "opaque", "data": "secret".repeat(1000)},
        ])
    );

    // Retention is budgeted, not exempt: the same stream under a 20-byte
    // output budget still fails on the resource boundary.
    let error = collect(
        Protocol::AnthropicMessages,
        &wire(&events),
        usize::MAX,
        StreamLimits {
            max_output_bytes: 20,
            ..StreamLimits::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::Undecodable);
}

#[test]
fn completed_arguments_require_explicit_json_in_full_and_streamed_calls() {
    for raw in [Value::Null, json!(""), json!(" "), json!({}), json!("{")] {
        let chat = json!({"choices":[{"message":{"tool_calls":[{"id":"c","function":{"name":"f","arguments":raw}}]},"finish_reason":"tool_calls"}]});
        assert_eq!(
            parse_openai_chat_response(&chat).unwrap_err().kind,
            ModelInvokeErrorKind::Undecodable
        );
        let responses = json!({"status":"completed","output":[{"type":"function_call","id":"f","call_id":"c","name":"f","arguments":raw}]});
        assert_eq!(
            parse_openai_responses_output(&responses).unwrap_err().kind,
            ModelInvokeErrorKind::Undecodable
        );
    }
    let mut bytes = wire(&[
        json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":""}}]}}]}),
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
    ]);
    bytes.extend_from_slice(b"data: [DONE]\n\n");
    assert_eq!(
        collect(
            Protocol::OpenAiChatCompletions,
            &bytes,
            usize::MAX,
            StreamLimits::default()
        )
        .unwrap_err()
        .kind,
        ModelInvokeErrorKind::Undecodable
    );
    let response = json!({"status":"completed","output":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]});
    assert_eq!(
        parse_openai_responses_output(&response)
            .unwrap()
            .response
            .tool_calls()
            .next()
            .unwrap()
            .arguments,
        json!({})
    );
}

#[test]
fn provider_errors_carry_the_envelope_message_and_code_and_nothing_else() {
    let decode = |protocol, event: Value| {
        let mut decoder = StreamDecoder::new(protocol, StreamLimits::default()).unwrap();
        let bytes = wire(&[event]);
        decoder.decode(&mut bytes.as_slice()).unwrap_err()
    };
    // The same envelope on each wire yields the same kind, code and text, and
    // a provider's own error is not given an event location.
    for (protocol, event) in [
        (
            Protocol::OpenAiResponses,
            json!({"type":"error","code":"rate_limit_exceeded","message":"slow down"}),
        ),
        (
            Protocol::OpenAiChatCompletions,
            json!({"error":{"code":"rate_limit_exceeded","message":"slow down"}}),
        ),
        (
            Protocol::AnthropicMessages,
            json!({"type":"error","error":{"type":"rate_limit_exceeded","message":"slow down"}}),
        ),
    ] {
        let error = decode(protocol, event);
        assert_eq!(error.kind, ModelInvokeErrorKind::Transient, "{protocol:?}");
        assert_eq!(error.code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(error.status, None);
        assert_eq!(
            error.message,
            "provider error (rate_limit_exceeded): slow down"
        );
    }
    let error = decode(
        Protocol::AnthropicMessages,
        json!({"type":"error","error":{"type":"authentication_error","message":"bad key"}}),
    );
    assert_eq!(error.kind, ModelInvokeErrorKind::Permanent);
    assert_eq!(error.code.as_deref(), Some("authentication_error"));

    // A code that is not a short identifier is dropped, the message is cut
    // and made one line, and no other field of the event is echoed.
    let long = "echoed prompt ".repeat(40);
    let error = decode(
        Protocol::OpenAiResponses,
        json!({
            "type":"error",
            "code":"not an identifier",
            "message":format!("line one\r\nline\u{1b} two {long}"),
            "param":"hidden_param",
            "sequence_number":7
        }),
    );
    assert_eq!(error.kind, ModelInvokeErrorKind::Permanent);
    assert_eq!(error.code, None);
    assert!(
        error
            .message
            .starts_with("provider error (unknown): line one line two echoed prompt"),
        "{}",
        error.message
    );
    assert!(error.message.ends_with('…'));
    assert_eq!(
        error.message.chars().count(),
        "provider error (unknown): ".chars().count() + 201
    );
    assert!(!error.message.chars().any(char::is_control));
    assert!(!error.message.contains("hidden_param"));

    // An event with no message keeps the label alone.
    let error = decode(Protocol::OpenAiResponses, json!({"type":"error"}));
    assert_eq!(error.message, "provider error (unknown)");
}

#[test]
fn large_tool_identity_is_not_copied_into_every_argument_fragment() {
    let call_id = "id".repeat(32 * 1024);
    let fragments: Vec<&str> = std::iter::once("{")
        .chain(std::iter::repeat_n(" ", 128))
        .chain(std::iter::once("}"))
        .collect();
    let arguments = fragments.concat();
    for protocol in [
        Protocol::OpenAiChatCompletions,
        Protocol::AnthropicMessages,
        Protocol::OpenAiResponses,
    ] {
        let mut events = match protocol {
            Protocol::OpenAiChatCompletions => vec![
                json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":call_id,"function":{"name":"f","arguments":""}}]}}]}),
            ],
            Protocol::AnthropicMessages => vec![
                json!({"type":"message_start","message":{"content":[]}}),
                json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":call_id,"name":"f","input":{}}}),
            ],
            Protocol::OpenAiResponses => vec![
                json!({"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc","call_id":call_id,"name":"f","arguments":""}}),
            ],
        };
        for fragment in &fragments {
            events.push(match protocol {
                Protocol::OpenAiChatCompletions => json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":fragment}}]}}]}),
                Protocol::AnthropicMessages => json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":fragment}}),
                Protocol::OpenAiResponses => json!({"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc","delta":fragment}),
            });
        }
        events.extend(match protocol {
            Protocol::OpenAiChatCompletions => vec![json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]})],
            Protocol::AnthropicMessages => vec![json!({"type":"content_block_stop","index":0}),json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}),json!({"type":"message_stop"})],
            Protocol::OpenAiResponses => vec![json!({"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","id":"fc","call_id":call_id,"name":"f","arguments":arguments}]}})],
        });
        let mut bytes = wire(&events);
        if protocol == Protocol::OpenAiChatCompletions {
            bytes.extend_from_slice(b"data: [DONE]\n\n");
        }
        let deltas = collect(protocol, &bytes, usize::MAX, StreamLimits::default()).unwrap();
        let identities: Vec<_> = deltas
            .iter()
            .filter_map(|delta| match delta {
                StreamDelta::ToolCallDelta {
                    provider_call_id: Some(id),
                    ..
                } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(identities.len(), 1, "{protocol}");
        assert_eq!(identities[0], &call_id);
        assert_eq!(
            output(&deltas)
                .response
                .tool_calls()
                .next()
                .unwrap()
                .provider_call_id
                .as_ref(),
            Some(&call_id)
        );
        assert_eq!(
            output(&deltas)
                .response
                .tool_calls()
                .next()
                .unwrap()
                .arguments,
            json!({})
        );
    }
}

#[test]
fn unknown_responses_incomplete_reason_and_status_are_safe_diagnostics() {
    let secret = "echoed prompt and tool parameters";
    // An unknown incomplete reason is reported as data: the raw value reaches
    // the host through raw_stop_reason, not through a diagnostic message.
    let response =
        json!({"status":"incomplete","incomplete_details":{"reason":secret},"output":[]});
    let parsed = parse_openai_responses_output(&response).unwrap();
    assert_eq!(parsed.stop_reason, ModelStopReason::Unrecognized);
    assert_eq!(
        parsed.raw_stop_reason.as_deref(),
        Some(format!("incomplete.{secret}").as_str())
    );
    let streamed = collect(
        Protocol::OpenAiResponses,
        &wire(&[json!({"type":"response.incomplete","response":response})]),
        usize::MAX,
        StreamLimits::default(),
    )
    .unwrap();
    let streamed = output(&streamed);
    assert_eq!(streamed.stop_reason, ModelStopReason::Unrecognized);
    assert_eq!(
        streamed.raw_stop_reason.as_deref(),
        Some(format!("incomplete.{secret}").as_str())
    );
    // An unknown status is still a safe diagnostic that never echoes the value.
    let error = parse_openai_responses_output(&json!({"status":secret,"output":[]})).unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::UnknownOutcome);
    assert!(!error.message.contains(secret));
    // The terminal event kind must match the status, so the probe carries the
    // same value in both slots.
    let error = collect(
        Protocol::OpenAiResponses,
        &wire(&[json!({
            "type": format!("response.{secret}"),
            "response": {"status": secret, "output": []},
        })]),
        usize::MAX,
        StreamLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::UnknownOutcome);
    assert!(!error.message.contains(secret));
}

fn anthropic_block_stream(start: Value, block_deltas: &[Value]) -> Vec<Value> {
    let mut events = vec![
        json!({"type":"message_start","message":{"content":[]}}),
        json!({"type":"content_block_start","index":0,"content_block":start}),
    ];
    events.extend(
        block_deltas
            .iter()
            .map(|delta| json!({"type":"content_block_delta","index":0,"delta":delta})),
    );
    events.extend([
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ]);
    events
}

#[test]
fn anthropic_non_client_tool_input_is_assembled_like_the_batch_form() {
    let whole = json!({
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "causa"},
    });
    let batch =
        parse_anthropic_response(&json!({"content": [whole], "stop_reason": "end_turn"})).unwrap();
    let events = anthropic_block_stream(
        json!({"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}),
        &[
            json!({"type":"input_json_delta","partial_json":"{\"query\":"}),
            json!({"type":"input_json_delta","partial_json":"\"causa\"}"}),
        ],
    );
    let deltas = paired(Protocol::AnthropicMessages, &wire(&events), &batch);
    // The block is sealed material, not a client tool call.
    assert!(
        !deltas
            .iter()
            .any(|delta| matches!(delta, StreamDelta::ToolCallDelta { .. }))
    );
    assert_eq!(output(&deltas).response.tool_calls().count(), 0);
}

#[test]
fn anthropic_delta_without_a_rule_fails_instead_of_truncating_the_block() {
    let secret = "wire-controlled-value";
    let unknown_delta = json!({"type": secret, "data": "abc"});
    for start in [
        // An unknown block type and a known one: neither may lose content.
        json!({"type":"future_block","data":""}),
        json!({"type":"text","text":""}),
    ] {
        let events = anthropic_block_stream(start, std::slice::from_ref(&unknown_delta));
        let error = collect(
            Protocol::AnthropicMessages,
            &wire(&events),
            usize::MAX,
            StreamLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind, ModelInvokeErrorKind::Undecodable);
        assert!(!error.message.contains(secret), "{}", error.message);
    }
    // `input_json_delta` still conflicts with a block that declared no input.
    let events = anthropic_block_stream(
        json!({"type":"future_block","data":""}),
        &[json!({"type":"input_json_delta","partial_json":"{}"})],
    );
    let error = collect(
        Protocol::AnthropicMessages,
        &wire(&events),
        usize::MAX,
        StreamLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::Undecodable);
}

#[test]
fn decode_failures_name_the_event_position_and_a_plain_event_type() {
    let message = |protocol, events: &[Value]| {
        collect(protocol, &wire(events), usize::MAX, StreamLimits::default())
            .unwrap_err()
            .message
    };
    assert_eq!(
        message(
            Protocol::OpenAiResponses,
            &[
                json!({"type":"response.created","response":{"id":"r","status":"in_progress"}}),
                json!({"type":"response.output_item.added","item":{"id":"i","type":"message","content":[]}}),
            ],
        ),
        "stream: missing or invalid output_index (event 2: response.output_item.added)"
    );
    // Chat chunks carry no type, so only the position is reported.
    assert_eq!(
        message(Protocol::OpenAiChatCompletions, &[json!({"id":"c"})]),
        "Chat missing choices array (event 1)"
    );
    // A type that is not a short identifier is provider text and is not echoed.
    for kind in ["echoed prompt text".to_owned(), "x".repeat(65)] {
        let bytes = format!("event: ping\ndata: {}\n\n", json!({"type": kind}));
        let text = collect(
            Protocol::AnthropicMessages,
            bytes.as_bytes(),
            usize::MAX,
            StreamLimits::default(),
        )
        .unwrap_err()
        .message;
        assert_eq!(text, "Anthropic SSE event/type mismatch (event 1)");
    }
}

/// A response in which a complete call `c0` is followed by a call `c1` whose
/// arguments stop at `{"b":`, ending with the given stop value. Returns the
/// stream and the batch body that carries the same content.
fn cut_off_call(protocol: Protocol, stop: &str) -> (Vec<u8>, Option<Value>) {
    match protocol {
        Protocol::OpenAiChatCompletions => {
            let mut bytes = wire(&[
                json!({"choices":[{"index":0,"delta":{"content":"hi","tool_calls":[{"index":0,"id":"c0","function":{"name":"first","arguments":"{\"a\":1}"}}]}}]}),
                json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c1","function":{"name":"second","arguments":"{\"b\":"}}]}}]}),
                json!({"choices":[{"index":0,"delta":{},"finish_reason":stop}]}),
            ]);
            bytes.extend_from_slice(b"data: [DONE]\n\n");
            let batch = json!({"choices":[{"message":{"role":"assistant","content":"hi","tool_calls":[
                {"id":"c0","type":"function","function":{"name":"first","arguments":"{\"a\":1}"}},
                {"id":"c1","type":"function","function":{"name":"second","arguments":"{\"b\":"}},
            ]},"finish_reason":stop}]});
            (bytes, Some(batch))
        }
        Protocol::AnthropicMessages => {
            let bytes = wire(&[
                json!({"type":"message_start","message":{"content":[]}}),
                json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hi"}}),
                json!({"type":"content_block_stop","index":0}),
                json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"c0","name":"first","input":{}}}),
                json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}),
                json!({"type":"content_block_stop","index":1}),
                json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"c1","name":"second","input":{}}}),
                json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"b\":"}}),
                json!({"type":"content_block_stop","index":2}),
                json!({"type":"message_delta","delta":{"stop_reason":stop}}),
                json!({"type":"message_stop"}),
            ]);
            // A batch body carries `input` as a value, so it cannot be cut.
            (bytes, None)
        }
        Protocol::OpenAiResponses => {
            let (status, details) = match stop {
                "completed" => ("completed", Value::Null),
                reason => ("incomplete", json!({"reason": reason})),
            };
            let response = json!({"status":status,"incomplete_details":details,"output":[
                {"type":"message","id":"m","content":[{"type":"output_text","text":"hi"}]},
                {"type":"function_call","id":"f0","call_id":"c0","name":"first","arguments":"{\"a\":1}"},
                {"type":"function_call","id":"f1","call_id":"c1","name":"second","arguments":"{\"b\":"},
            ]});
            let bytes = wire(&[
                json!({"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m","content":[]}}),
                json!({"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"f0","call_id":"c0","name":"first","arguments":"{\"a\":1}"}}),
                json!({"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"f1","call_id":"c1","name":"second"}}),
                json!({"type":"response.function_call_arguments.delta","output_index":2,"item_id":"f1","delta":"{\"b\":"}),
                json!({"type":"response.function_call_arguments.done","output_index":2,"item_id":"f1","arguments":"{\"b\":"}),
                json!({"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"f1","call_id":"c1","name":"second","arguments":"{\"b\":"}}),
                json!({"type":format!("response.{status}"),"response":response}),
            ]);
            (bytes, Some(response))
        }
    }
}

fn parse_batch(protocol: Protocol, body: &Value) -> Result<ModelOutput, ModelInvokeError> {
    match protocol {
        Protocol::OpenAiChatCompletions => parse_openai_chat_response(body),
        Protocol::AnthropicMessages => parse_anthropic_response(body),
        Protocol::OpenAiResponses => parse_openai_responses_output(body),
    }
}

fn call_names(output: &ModelOutput) -> Vec<&str> {
    output
        .response
        .items
        .iter()
        .filter_map(|item| match &item.content {
            causa_kernel::ModelItemContent::ToolCall(call) => Some(call.tool_name.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_tool_call_cut_off_by_the_token_limit_is_left_out_and_the_stop_is_max_tokens() {
    for (protocol, stop) in [
        (Protocol::OpenAiChatCompletions, "length"),
        (Protocol::AnthropicMessages, "max_tokens"),
        (Protocol::OpenAiResponses, "max_output_tokens"),
    ] {
        let (bytes, batch) = cut_off_call(protocol, stop);
        let deltas = collect(protocol, &bytes, usize::MAX, StreamLimits::default())
            .unwrap_or_else(|error| panic!("{protocol:?} stream: {error}"));
        let streamed = output(&deltas);
        assert_eq!(
            streamed.stop_reason,
            ModelStopReason::MaxTokens,
            "{protocol:?}"
        );
        assert_eq!(call_names(streamed), ["first"], "{protocol:?}");
        if let Some(batch) = batch {
            let parsed = parse_batch(protocol, &batch)
                .unwrap_or_else(|error| panic!("{protocol:?} batch: {error}"));
            assert_eq!(
                serde_json::to_value(&parsed).unwrap(),
                serde_json::to_value(streamed).unwrap(),
                "{protocol:?}"
            );
        }
    }
}

#[test]
fn a_call_the_server_claims_complete_with_bad_arguments_is_still_undecodable() {
    for (protocol, stop) in [
        (Protocol::OpenAiChatCompletions, "tool_calls"),
        (Protocol::AnthropicMessages, "tool_use"),
        (Protocol::OpenAiResponses, "completed"),
    ] {
        let (bytes, batch) = cut_off_call(protocol, stop);
        let error = collect(protocol, &bytes, usize::MAX, StreamLimits::default()).unwrap_err();
        assert_eq!(
            error.kind,
            ModelInvokeErrorKind::Undecodable,
            "{protocol:?}"
        );
        if let Some(batch) = batch {
            let error = parse_batch(protocol, &batch).unwrap_err();
            assert_eq!(
                error.kind,
                ModelInvokeErrorKind::Undecodable,
                "{protocol:?}"
            );
        }
    }
}

#[test]
fn only_the_last_call_may_be_cut_off() {
    // The incomplete call is first; a complete one follows it.
    let batch = json!({"choices":[{"message":{"role":"assistant","tool_calls":[
        {"id":"c0","type":"function","function":{"name":"first","arguments":"{\"a\":"}},
        {"id":"c1","type":"function","function":{"name":"second","arguments":"{}"}},
    ]},"finish_reason":"length"}]});
    let error = parse_openai_chat_response(&batch).unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::Undecodable);

    let bytes = wire(&[
        json!({"type":"message_start","message":{"content":[]}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"c0","name":"first","input":{}}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":"after"}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"max_tokens"}}),
        json!({"type":"message_stop"}),
    ]);
    let error = collect(
        Protocol::AnthropicMessages,
        &bytes,
        usize::MAX,
        StreamLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::Undecodable);
}

#[test]
fn a_chat_batch_response_with_two_choices_is_an_error() {
    let choice = json!({"message":{"role":"assistant","content":"a"},"finish_reason":"stop"});
    let error = parse_openai_chat_response(&json!({"choices":[choice, choice]})).unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::Undecodable);
    assert_eq!(error.message, "more than one choice; one is supported");
}

#[test]
fn a_null_error_field_is_not_an_error() {
    let mut bytes = wire(&[
        json!({"error":null,"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}),
    ]);
    bytes.extend_from_slice(b"data: [DONE]\n\n");
    let deltas = collect(
        Protocol::OpenAiChatCompletions,
        &bytes,
        usize::MAX,
        StreamLimits::default(),
    )
    .unwrap();
    assert_eq!(output(&deltas).stop_reason, ModelStopReason::EndTurn);

    // Beside a flat Responses error event, a null `error` does not hide the
    // event's own code and message.
    let error = collect(
        Protocol::OpenAiResponses,
        &wire(&[json!({"type":"error","error":null,"code":"server_error","message":"down"})]),
        usize::MAX,
        StreamLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::Transient);
    assert_eq!(error.message, "provider error (server_error): down");
}
