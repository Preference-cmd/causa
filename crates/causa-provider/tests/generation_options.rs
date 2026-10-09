//! Generation options reach actual batch/SSE HTTP bodies or fail before send.
use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CacheDirective, CallControl, CancellationToken, ContentPart,
    ContextBlock, ContextFrame, GenerationOptions, InvocationId, ModelGateway,
    ModelInvokeErrorKind, ModelRef, ModelRequest, ReasoningEffort, RoundId, StreamDelta,
    TextPayload, ToolSurface, TurnId,
};
use causa_provider::{
    AnthropicMessagesGateway, OpenAiChatCompletionsGateway, OpenAiResponsesGateway,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, method, path},
};

fn request(generation: GenerationOptions) -> ModelRequest {
    ModelRequest {
        invocation_id: InvocationId {
            turn_id: TurnId::new("generation"),
            round_id: RoundId(0),
        },
        frame: ContextFrame {
            blocks: vec![ContextBlock::new(
                BlockId::new(uuid::Uuid::from_u128(1)),
                BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("hi"))]),
                BlockMeta {
                    source: Some("user".into()),
                    ..Default::default()
                },
            )],
        },
        model: ModelRef::new("configured-model"),
        tool_surface: ToolSurface::empty(),
        generation,
        cache: CacheDirective::None,
    }
}
fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}
fn gateway(protocol: usize, server: &MockServer) -> Box<dyn ModelGateway> {
    match protocol {
        0 => Box::new(
            OpenAiChatCompletionsGateway::new("test").with_base_url(format!("{}/v1", server.uri())),
        ),
        1 => Box::new(
            OpenAiResponsesGateway::new("test").with_base_url(format!("{}/v1", server.uri())),
        ),
        _ => Box::new(
            AnthropicMessagesGateway::new("test").with_base_url(format!("{}/v1", server.uri())),
        ),
    }
}
fn event(name: &str, body: Value) -> String {
    format!("event: {name}\ndata: {body}\n\n")
}
fn fixture(protocol: usize) -> (&'static str, Value, String) {
    match protocol {
        0 => ("/v1/chat/completions", json!({"choices":[{"message":{"content":""},"finish_reason":"stop"}]}),
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into()),
        1 => ("/v1/responses", json!({"status":"completed","output":[]}),
            event("response.completed", json!({"type":"response.completed","response":{"status":"completed","output":[]}}))),
        _ => ("/v1/messages", json!({"content":[],"stop_reason":"end_turn"}),
            event("message_start", json!({"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"configured-model","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}))
            + &event("message_delta", json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}))
            + &event("message_stop", json!({"type":"message_stop"}))),
    }
}

#[tokio::test]
async fn configured_options_survive_batch_and_stream_request_preparation() {
    for protocol in [0, 1, 2] {
        let choices = if protocol == 2 {
            // Anthropic renders the effort table: unset, disabled, adaptive.
            vec![
                None,
                Some(ReasoningEffort::None),
                Some(ReasoningEffort::High),
            ]
        } else {
            vec![
                Some(ReasoningEffort::High),
                Some(ReasoningEffort::None),
                None,
            ]
        };
        for effort in choices {
            let server = MockServer::start().await;
            let gateway = gateway(protocol, &server);
            let generation = GenerationOptions {
                reasoning_effort: effort,
                max_tokens: Some(8192),
                temperature: (effort != Some(ReasoningEffort::High)).then_some(0.5),
                ..Default::default()
            };
            let mut expected = json!({"model":"configured-model"});
            expected[match protocol {
                0 => "max_completion_tokens",
                1 => "max_output_tokens",
                _ => "max_tokens",
            }] = json!(8192);
            if let Some(effort) = effort {
                match protocol {
                    0 => expected["reasoning_effort"] = json!(effort),
                    1 => expected["reasoning"] = json!({"effort": effort}),
                    // Anthropic: `None` disables thinking; every other level
                    // renders adaptive thinking plus its effort level.
                    _ if effort == ReasoningEffort::None => {
                        expected["thinking"] = json!({"type": "disabled"});
                    }
                    _ => {
                        expected["thinking"] = json!({"type": "adaptive"});
                        expected["output_config"] = json!({"effort": effort});
                    }
                }
            }
            if let Some(temperature) = generation.temperature {
                expected["temperature"] = json!(temperature);
            }
            let request = request(generation);
            let (endpoint, batch_response, stream_response) = fixture(protocol);
            for streaming in [false, true] {
                server.reset().await;
                let response = if streaming {
                    ResponseTemplate::new(200)
                        .set_body_raw(stream_response.as_bytes().to_vec(), "text/event-stream")
                } else {
                    ResponseTemplate::new(200).set_body_json(&batch_response)
                };
                Mock::given(method("POST"))
                    .and(path(endpoint))
                    .and(body_partial_json(&expected))
                    .respond_with(response)
                    .expect(1)
                    .mount(&server)
                    .await;
                if streaming {
                    let mut stream = gateway.stream(&request, &control()).await.unwrap();
                    let mut completed = false;
                    while let Some(delta) = stream.next().await {
                        if matches!(delta, StreamDelta::Done { .. }) {
                            completed = true;
                            break;
                        }
                        assert!(!matches!(delta, StreamDelta::Error { .. }), "{delta:?}");
                    }
                    assert!(completed, "protocol {protocol} did not complete");
                } else {
                    gateway.invoke(&request, &control()).await.unwrap();
                }
                let requests = server.received_requests().await.unwrap();
                assert_eq!(requests.len(), 1);
                let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
                assert_eq!(
                    body.get("stream"),
                    streaming.then_some(json!(true)).as_ref()
                );
                if protocol != 2 {
                    assert!(body.get("max_tokens").is_none());
                }
                if effort.is_none() {
                    assert!(body.get("reasoning").is_none());
                    assert!(body.get("reasoning_effort").is_none());
                }
                if effort == Some(ReasoningEffort::High) {
                    assert!(body.get("temperature").is_none());
                }
            }
        }
    }
}

#[tokio::test]
async fn invalid_generation_options_never_send_batch_or_stream_http() {
    let server = MockServer::start().await;
    for protocol in [0, 1, 2] {
        let gateway = gateway(protocol, &server);
        // A limit is supplied wherever the case is not about the limit: on
        // Anthropic a missing `max_tokens` is its own InvalidRequest.
        let mut cases = vec![
            GenerationOptions {
                max_tokens: Some(0),
                ..Default::default()
            },
            GenerationOptions {
                max_tokens: Some(64),
                temperature: Some(f32::NAN),
                ..Default::default()
            },
            GenerationOptions {
                max_tokens: Some(64),
                temperature: Some(-0.1),
                ..Default::default()
            },
            GenerationOptions {
                max_tokens: Some(64),
                temperature: Some(2.1),
                ..Default::default()
            },
            GenerationOptions {
                max_tokens: Some(64),
                reasoning_effort: Some(ReasoningEffort::High),
                temperature: Some(1.0),
                ..Default::default()
            },
        ];
        if protocol == 2 {
            // `Minimal` has no Anthropic wire level; the others render.
            cases.push(GenerationOptions {
                max_tokens: Some(64),
                reasoning_effort: Some(ReasoningEffort::Minimal),
                ..Default::default()
            });
            // The required output limit is never defaulted.
            cases.push(GenerationOptions {
                max_tokens: None,
                ..Default::default()
            });
            cases.push(GenerationOptions {
                max_tokens: Some(64),
                output_schema: Some(json!({"type":"object"})),
                ..Default::default()
            });
            cases.push(GenerationOptions {
                max_tokens: Some(64),
                temperature: Some(1.1),
                ..Default::default()
            });
        }
        for generation in cases {
            let request = request(generation);
            assert_eq!(
                gateway.invoke(&request, &control()).await.unwrap_err().kind,
                ModelInvokeErrorKind::InvalidRequest
            );
            match gateway.stream(&request, &control()).await {
                Ok(_) => panic!("invalid options unexpectedly produced a stream"),
                Err(error) => assert_eq!(error.kind, ModelInvokeErrorKind::InvalidRequest),
            }
        }
    }
    assert!(server.received_requests().await.unwrap().is_empty());
}
