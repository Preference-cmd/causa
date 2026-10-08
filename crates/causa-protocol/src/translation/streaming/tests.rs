use causa_kernel::{ModelInvokeError, ModelInvokeErrorKind, ModelOutput, StreamDelta};
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

fn fail(protocol: Protocol, events: &[Value]) -> ModelInvokeErrorKind {
    collect(protocol, &wire(events), usize::MAX, StreamLimits::default())
        .unwrap_err()
        .kind
}

#[test]
fn explicit_responses_terminal_classifications_are_shared_with_full_parser() {
    for (status, reason, kind) in [
        ("failed", None, ModelInvokeErrorKind::Permanent),
        (
            "incomplete",
            Some("new_reason"),
            ModelInvokeErrorKind::UnknownOutcome,
        ),
    ] {
        let response = json!({"status":status,"output":[],"error":{"code":"new_error"},"incomplete_details":{"reason":reason}});
        assert_eq!(
            parse_openai_responses_output(&response).unwrap_err().kind,
            kind
        );
        assert_eq!(
            fail(
                Protocol::OpenAiResponses,
                &[json!({"type":format!("response.{status}"),"response":response})]
            ),
            kind
        );
    }
    for (reason, expected) in [
        ("max_output_tokens", "MaxTokens"),
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
fn malformed_order_ids_and_arguments_fail_permanently() {
    let (mut events, _) = responses_fixture();
    events[3]["item_id"] = json!("other");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Permanent
    );
    let (mut events, _) = responses_fixture();
    events.last_mut().unwrap()["response"]["output"][2]["call_id"] = json!("other");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Permanent
    );
    let (mut events, _) = responses_fixture();
    events.last_mut().unwrap()["response"]["output"][3]["arguments"] = json!("{");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Permanent
    );
    let (mut events, _) = responses_fixture();
    events[13]["item"]["name"] = json!("changed");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Permanent
    );
    assert_eq!(
        fail(
            Protocol::AnthropicMessages,
            &[json!({"type":"message_stop"})]
        ),
        ModelInvokeErrorKind::Permanent
    );
    assert_eq!(
        fail(
            Protocol::OpenAiChatCompletions,
            &[json!({"choices":[{"index":1,"delta":{}}]})]
        ),
        ModelInvokeErrorKind::Permanent
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
        ModelInvokeErrorKind::Permanent
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
            ModelInvokeErrorKind::Permanent
        );
    }
    let events = [
        json!({"type":"response.output_item.added","output_index":u64::MAX,"item":{"type":"message","id":"m0"}}),
    ];
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Permanent
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
        ModelInvokeErrorKind::Permanent
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

#[test]
fn duplicate_parts_and_malformed_known_initial_shapes_fail() {
    let (mut events, _) = responses_fixture();
    let completed = json!({"type":"response.content_part.done","output_index":0,"item_id":"m0","content_index":0,"part":{"type":"output_text","text":"draft"}});
    events.insert(4, completed.clone());
    events.insert(5, completed);
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Permanent
    );
    let (mut events, _) = responses_fixture();
    events[2]["part"]["text"] = json!({"wrong":"type"});
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Permanent
    );
    let (mut events, _) = responses_fixture();
    events[1]["item"]["content"] = json!("wrong");
    assert_eq!(
        fail(Protocol::OpenAiResponses, &events),
        ModelInvokeErrorKind::Permanent
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
fn unknown_anthropic_block_extensions_are_not_retained() {
    let events = [
        json!({"type":"message_start","message":{"content":[]}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"opaque","data":"secret".repeat(1000)}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"opaque","data":"secret".repeat(1000)}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"}}),
        json!({"type":"message_stop"}),
    ];
    let deltas = collect(
        Protocol::AnthropicMessages,
        &wire(&events),
        usize::MAX,
        StreamLimits {
            max_output_bytes: 20,
            ..StreamLimits::default()
        },
    )
    .unwrap();
    assert_eq!(output(&deltas).response.text(), "");
}

#[test]
fn completed_arguments_require_explicit_json_in_full_and_streamed_calls() {
    for raw in [Value::Null, json!(""), json!(" "), json!({}), json!("{")] {
        let chat = json!({"choices":[{"message":{"tool_calls":[{"id":"c","function":{"name":"f","arguments":raw}}]},"finish_reason":"tool_calls"}]});
        assert_eq!(
            parse_openai_chat_response(&chat).unwrap_err().kind,
            ModelInvokeErrorKind::Permanent
        );
        let responses = json!({"status":"completed","output":[{"type":"function_call","id":"f","call_id":"c","name":"f","arguments":raw}]});
        assert_eq!(
            parse_openai_responses_output(&responses).unwrap_err().kind,
            ModelInvokeErrorKind::Permanent
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
        ModelInvokeErrorKind::Permanent
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
fn error_diagnostics_exclude_echoed_arguments_and_unknown_codes() {
    let secret = "echoed_prompt_and_tool_args";
    let mut decoder =
        StreamDecoder::new(Protocol::OpenAiResponses, StreamLimits::default()).unwrap();
    let bytes = wire(&[json!({"type":"error","code":secret,"message":secret})]);
    let error = decoder.decode(&mut bytes.as_slice()).unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::Permanent);
    assert!(!error.message.contains(secret));
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
    let response =
        json!({"status":"incomplete","incomplete_details":{"reason":secret},"output":[]});
    let error = parse_openai_responses_output(&response).unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::UnknownOutcome);
    assert!(!error.message.contains(secret));
    let error = parse_openai_responses_output(&json!({"status":secret,"output":[]})).unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::UnknownOutcome);
    assert!(!error.message.contains(secret));
    let error = collect(
        Protocol::OpenAiResponses,
        &wire(&[json!({"type":"response.incomplete","response":response})]),
        usize::MAX,
        StreamLimits::default(),
    )
    .unwrap_err();
    assert_eq!(error.kind, ModelInvokeErrorKind::UnknownOutcome);
    assert!(!error.message.contains(secret));
}
