//! Wiremock-driven integration tests for `ReqwestBackend`.
//!
//! These tests stand up a local `wiremock` server and point the
//! reqwest-backed client at it via the `base_url` config. They assert
//! the request shape (URL, method, auth header, body) and the
//! response translation back into `AgentResponse` / `Vec<ModelInfo>`.

use std::sync::Arc;

use reimagine_agent_harness::{
    AgentRequest, AgentToolDefinition, ContentBlock, FileContentBlock, Message, ModelCapability,
    ModelName, ProviderName,
};
use reimagine_agent_provider::{
    AnthropicMessagesConfig, CompletionBackend, OpenAiChatCompletionsConfig, OpenAiResponsesConfig,
    ProviderAdapterError, ReqwestBackend,
};
use serde_json::{Value, json};
use wiremock::matchers::BodyExactMatcher;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn body_exact_json(body: serde_json::Value) -> BodyExactMatcher {
    BodyExactMatcher::json(body)
}

const OPENAI_KEY: &str = "sk-test-openai";
const ANTHROPIC_KEY: &str = "sk-test-anthropic";
const RESPONSES_KEY: &str = "sk-test-responses";

fn openai_cfg_for(server: &MockServer) -> OpenAiChatCompletionsConfig {
    OpenAiChatCompletionsConfig::new(format!("{}/v1", server.uri()), OPENAI_KEY, "gpt-4o-mini")
}

fn anthropic_cfg_for(server: &MockServer) -> AnthropicMessagesConfig {
    AnthropicMessagesConfig::new(ANTHROPIC_KEY, "claude-3-5-sonnet-latest")
        .with_base_url(server.uri())
}

fn responses_cfg_for(server: &MockServer) -> OpenAiResponsesConfig {
    OpenAiResponsesConfig::new(format!("{}/v1", server.uri()), RESPONSES_KEY, "gpt-5-mini")
}

fn build_request(model: &str) -> AgentRequest {
    AgentRequest::new(ModelName::new(model), vec![Message::user("hi")]).with_tools(vec![
        AgentToolDefinition::new(
            "echo",
            "echo a string",
            json!({"type": "object", "properties": {"x": {"type": "number"}}}),
        ),
    ])
}

fn build_request_with_options(model: &str, options: Value) -> AgentRequest {
    build_request(model).with_options(options)
}

fn openai_completion_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "cmpl-2",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    }))
}

fn anthropic_completion_response() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": "claude-3-5-sonnet-latest",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }))
}

#[tokio::test]
async fn openai_complete_returns_translated_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {OPENAI_KEY}")))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "echo a string",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}}
                    }
                }
            }],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello back",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"x\": 42}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();

    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let resp = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect("upstream returned 2xx");
    assert_eq!(resp.message().content(), "hello back");
    let calls = resp.message().tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id().as_str(), "call_1");
    assert_eq!(calls[0].name(), "echo");
    assert_eq!(calls[0].arguments()["x"], 42);
    assert_eq!(resp.stop_reason(), Some("tool_calls"));
    let usage = resp.usage().expect("usage present");
    assert_eq!(usage.input_tokens(), Some(7));
    assert_eq!(usage.output_tokens(), Some(4));
}

#[tokio::test]
async fn openai_complete_maps_non_2xx_to_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid api key"))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));
    let err = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect_err("expected non-2xx response to surface as an error");
    match err {
        ProviderAdapterError::Api { code, message, .. } => {
            assert_eq!(code, "401");
            assert!(message.contains("invalid api key"));
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn openai_list_models_returns_translated_listing() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", format!("Bearer {OPENAI_KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                { "id": "gpt-4o-mini", "object": "model" },
                { "id": "gpt-4o",       "object": "model" }
            ]
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));
    let models = backend.list_models().await.expect("list ok");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name().as_str(), "gpt-4o-mini");
    assert_eq!(models[1].name().as_str(), "gpt-4o");
    for m in &models {
        assert!(m.capabilities().contains(&ModelCapability::Chat));
        assert!(m.capabilities().contains(&ModelCapability::ToolUse));
        assert_eq!(m.provider().map(|p| p.as_str()), Some("openai-test"));
    }
}

#[tokio::test]
async fn anthropic_complete_returns_translated_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", ANTHROPIC_KEY))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello from claude"}],
            "model": "claude-3-5-sonnet-latest",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );

    let resp = backend
        .complete(build_request("claude-3-5-sonnet-latest"))
        .await
        .expect("upstream returned 2xx");
    assert_eq!(resp.message().content(), "hello from claude");
    assert_eq!(resp.stop_reason(), Some("end_turn"));
    let usage = resp.usage().expect("usage present");
    assert_eq!(usage.input_tokens(), Some(10));
    assert_eq!(usage.output_tokens(), Some(20));
}

#[tokio::test]
async fn anthropic_complete_maps_non_2xx_to_api_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid key"}
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );

    let err = backend
        .complete(build_request("claude-3-5-sonnet-latest"))
        .await
        .expect_err("expected non-2xx");
    match err {
        ProviderAdapterError::Api { code, message, .. } => {
            assert_eq!(code, "401");
            assert!(message.contains("authentication_error"));
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn anthropic_list_models_returns_translated_listing() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("x-api-key", ANTHROPIC_KEY))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "id": "claude-3-5-sonnet-latest", "type": "model" },
                { "id": "claude-3-haiku-latest", "type": "model" }
            ]
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );

    let models = backend.list_models().await.expect("list ok");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name().as_str(), "claude-3-5-sonnet-latest");
    assert_eq!(models[1].name().as_str(), "claude-3-haiku-latest");
    for m in &models {
        assert!(m.capabilities().contains(&ModelCapability::Chat));
        assert!(m.capabilities().contains(&ModelCapability::ToolUse));
        assert_eq!(m.provider().map(|p| p.as_str()), Some("anthropic-test"));
    }
}

fn responses_request() -> AgentRequest {
    AgentRequest::new(
        ModelName::new("gpt-5-mini"),
        vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant_with_tool_calls(
                "",
                vec![reimagine_agent_harness::ToolCall::new(
                    reimagine_agent_harness::ToolCallId::new("c1"),
                    "echo",
                    json!({"x": 1}),
                )],
            ),
            Message::tool_result(reimagine_agent_harness::ToolCallId::new("c1"), "ok"),
        ],
    )
    .with_tools(vec![AgentToolDefinition::new(
        "echo",
        "echo a string",
        json!({"type": "object", "properties": {"x": {"type": "number"}}}),
    )])
}

#[tokio::test]
async fn responses_complete_assembles_full_input_array_and_prompt_cache_key() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .and(body_partial_json(json!({
            "model": "gpt-5-mini",
            "instructions": "sys",
            "input": [
                { "role": "user", "content": [{ "type": "input_text", "text": "hi" }] },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "echo",
                    "arguments": "{\"x\":1}"
                },
                { "type": "function_call_output", "call_id": "c1", "output": "ok" }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "echo a string",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}}
                    }
                }
            }],
            "prompt_cache_key": "session-42",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "hello back" }]
            }],
            "usage": { "input_tokens": 9, "output_tokens": 3, "total_tokens": 12 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));

    let request = responses_request().with_options(json!({ "prompt_cache_key": "session-42" }));
    let resp = backend
        .complete(request)
        .await
        .expect("upstream returned 2xx");
    assert_eq!(resp.message().content(), "hello back");
    assert_eq!(resp.message().tool_calls().len(), 0);
    let usage = resp.usage().expect("usage present");
    assert_eq!(usage.input_tokens(), Some(9));
    assert_eq!(usage.output_tokens(), Some(3));
}

#[tokio::test]
async fn responses_complete_omits_prompt_cache_key_when_absent() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .and(body_exact_json(json!({
            "model": "gpt-5-mini",
            "instructions": "sys",
            "input": [
                { "role": "user", "content": [{ "type": "input_text", "text": "hi" }] },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "echo",
                    "arguments": "{\"x\":1}"
                },
                { "type": "function_call_output", "call_id": "c1", "output": "ok" }
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "echo a string",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}}
                    }
                }
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_2",
            "output": [],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));

    let resp = backend
        .complete(responses_request())
        .await
        .expect("upstream returned 2xx");
    assert!(resp.usage().is_some());
}

#[tokio::test]
async fn responses_complete_parses_function_call_items() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_3",
            "output": [{
                "type": "function_call",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{\"x\": 42}"
            }],
            "usage": { "input_tokens": 5, "output_tokens": 8, "total_tokens": 13 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));

    let resp = backend
        .complete(responses_request())
        .await
        .expect("upstream returned 2xx");
    let calls = resp.message().tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id().as_str(), "call_1");
    assert_eq!(calls[0].name(), "echo");
    assert_eq!(calls[0].arguments(), &json!({"x": 42}));
}

#[tokio::test]
async fn responses_list_models_reuses_models_path() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                { "id": "gpt-5-mini", "object": "model" },
                { "id": "gpt-5",       "object": "model" }
            ]
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));
    let models = backend.list_models().await.expect("list ok");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name().as_str(), "gpt-5-mini");
    assert_eq!(models[1].name().as_str(), "gpt-5");
    for m in &models {
        assert_eq!(m.provider().map(|p| p.as_str()), Some("responses-test"));
    }
}

#[tokio::test]
async fn responses_stream_emits_text_deltas_and_done_with_usage() {
    let server = MockServer::start().await;

    let sse_body = [
        "event: response.output_text.delta",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\",\"item_id\":\"msg_1\",\"output_index\":0}",
        "",
        "event: response.output_text.delta",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo!\",\"item_id\":\"msg_1\",\"output_index\":0}",
        "",
        "event: response.completed",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":7,\"output_tokens\":2,\"total_tokens\":9}}}",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .and(body_partial_json(json!({
            "model": "gpt-5-mini",
            "input": [{ "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }],
            "stream": true,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(responses_request())
        .await
        .expect("stream should be supported");

    let mut text = String::new();
    let mut saw_usage = false;
    let mut saw_done = false;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ContentDelta(delta) => text.push_str(&delta),
            reimagine_agent_harness::AgentStreamEvent::Usage(usage) => {
                saw_usage = true;
                assert_eq!(usage.input_tokens(), Some(7));
                assert_eq!(usage.output_tokens(), Some(2));
            }
            reimagine_agent_harness::AgentStreamEvent::Done { .. } => saw_done = true,
            _ => {}
        }
    }
    assert_eq!(text, "Hello!");
    assert!(saw_usage, "usage event should be emitted");
    assert!(saw_done, "done event should be emitted");
}

#[tokio::test]
async fn responses_stream_forwards_server_compaction_event() {
    // PV-01b reserved channel, consumed in CM-V2e: the provider
    // forwards `response.compaction` as an informational
    // `Compacted` event.
    let server = MockServer::start().await;

    let sse_body = [
        "event: response.compaction",
        "data: {\"type\":\"response.compaction\",\"item_id\":\"fc_compacted_1\",\"compacted_text\":\"opaque\"}",
        "",
        "event: response.output_text.delta",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\",\"item_id\":\"msg_1\",\"output_index\":0}",
        "",
        "event: response.completed",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(responses_request())
        .await
        .expect("stream should be supported");

    let mut saw_compacted: Option<String> = None;
    let mut text = String::new();
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::Compacted { item_id } => {
                saw_compacted = Some(item_id);
            }
            reimagine_agent_harness::AgentStreamEvent::ContentDelta(delta) => text.push_str(&delta),
            _ => {}
        }
    }
    assert_eq!(saw_compacted.as_deref(), Some("fc_compacted_1"));
    assert_eq!(text, "done");
}

#[tokio::test]
async fn responses_stream_decodes_base64_arguments_deltas_and_emits_tool_call() {
    let server = MockServer::start().await;

    // "{\"x\":"  base64 = eyJ4Ijo=
    // " 42}"     base64 = ICA0Mn0=
    let sse_body = [
        "event: response.function_call_arguments.delta",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"eyJ4Ijo=\",\"item_id\":\"fc_1\",\"output_index\":1}",
        "",
        "event: response.function_call_arguments.delta",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"ICA0Mn0=\",\"item_id\":\"fc_1\",\"output_index\":1}",
        "",
        "event: response.output_item.done",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"echo\",\"arguments\":\"{\\\"x\\\":42}\"}}",
        "",
        "event: response.completed",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"status\":\"completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":6,\"total_tokens\":10}}}",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", format!("Bearer {RESPONSES_KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(responses_request())
        .await
        .expect("stream should be supported");

    let mut collected = Vec::new();
    while let Some(event) = stream.next_event().await {
        collected.push(event);
    }

    let tool_calls: Vec<_> = collected
        .iter()
        .filter_map(|e| match e {
            reimagine_agent_harness::AgentStreamEvent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(tool_calls.len(), 1, "one complete tool call expected");
    assert_eq!(tool_calls[0].id().as_str(), "call_1");
    assert_eq!(tool_calls[0].name(), "echo");
    assert_eq!(tool_calls[0].arguments(), &json!({"x": 42}));
    assert!(
        collected
            .iter()
            .any(|e| matches!(e, reimagine_agent_harness::AgentStreamEvent::Done { .. }))
    );
}

#[tokio::test]
async fn openai_complete_forwards_sampling_params_and_unknown_keys() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["end"],
            "seed": 42,
            "user": "u1",
            "frequency_penalty": 0.2,
        })))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));
    let req = build_request_with_options(
        "gpt-4o-mini",
        json!({
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": "end",
            "seed": 42,
            "user": "u1",
            "frequency_penalty": 0.2,
        }),
    );
    let resp = backend.complete(req).await.expect("complete ok");
    assert_eq!(resp.message().content(), "ok");
}

#[tokio::test]
async fn openai_complete_with_reasoning_strips_temperature_and_top_p() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));
    let req = build_request_with_options(
        "gpt-4o-mini",
        json!({
            "reasoning_effort": "high",
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": "end",
            "seed": 42,
        }),
    );
    backend.complete(req).await.expect("complete ok");

    let requests = server.received_requests().await.expect("requests captured");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("body json");
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["stop"], json!(["end"]));
    assert_eq!(body["seed"], 42);
    assert!(body.get("temperature").is_none());
    assert!(body.get("top_p").is_none());
}

#[tokio::test]
async fn openai_stream_forwards_sampling_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "stream": true,
            "max_tokens": 256,
            "temperature": 0.2,
            "seed": 7,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string("data: [DONE]\n\n"))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));
    let req = build_request_with_options(
        "gpt-4o-mini",
        json!({"max_tokens": 256, "temperature": 0.2, "seed": 7}),
    );
    backend.stream(req).await.expect("stream starts");
}

#[tokio::test]
async fn anthropic_complete_forwards_sampling_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 1024,
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 32,
            "stop_sequences": ["a", "b"],
        })))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "claude-3-5-sonnet-latest",
        json!({
            "max_tokens": 1024,
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 32,
            "stop": ["a", "b"],
        }),
    );
    let resp = backend.complete(req).await.expect("complete ok");
    assert_eq!(resp.message().content(), "ok");
}

#[tokio::test]
async fn anthropic_complete_with_reasoning_strips_inapplicable_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "claude-3-5-sonnet-latest",
        json!({
            "reasoning_effort": "high",
            "temperature": 0.4,
            "top_p": 0.8,
            "top_k": 32,
            "stop": ["a"],
            "seed": 9,
            "user": "u1",
        }),
    );
    backend.complete(req).await.expect("complete ok");

    let requests = server.received_requests().await.expect("requests captured");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("body json");
    assert_eq!(body["max_tokens"], 4096);
    assert_eq!(body["top_k"], 32);
    assert_eq!(body["stop_sequences"], json!(["a"]));
    assert!(body.get("temperature").is_none());
    assert!(body.get("top_p").is_none());
    assert!(body.get("reasoning_effort").is_none());
    assert!(body.get("seed").is_none());
    assert!(body.get("user").is_none());
}

#[tokio::test]
async fn anthropic_stream_forwards_sampling_params() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "model": "claude-3-5-sonnet-latest",
            "stream": true,
            "max_tokens": 2048,
            "temperature": 0.4,
            "top_k": 32,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(""))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "claude-3-5-sonnet-latest",
        json!({"max_tokens": 2048, "temperature": 0.4, "top_k": 32}),
    );
    backend.stream(req).await.expect("stream starts");
}

#[tokio::test]
async fn responses_complete_forwards_sampling_params_with_max_output_tokens() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_partial_json(json!({
            "model": "gpt-5-mini",
            "max_output_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "user": "u1",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_4",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "ok" }]
            }],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));
    let req = build_request_with_options(
        "gpt-5-mini",
        json!({"max_tokens": 512, "temperature": 0.7, "top_p": 0.9, "seed": 42, "user": "u1"}),
    );
    backend.complete(req).await.expect("complete ok");
}

#[tokio::test]
async fn anthropic_complete_with_reasoning_enables_extended_thinking() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "thinking": { "type": "enabled", "budget_tokens": 8192 }
        })))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "claude-sonnet-4-5",
        json!({"reasoning": true, "reasoning_budget_tokens": 8192}),
    );
    backend.complete(req).await.expect("complete ok");
}

#[tokio::test]
async fn anthropic_complete_without_reasoning_omits_thinking_block() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request("claude-sonnet-4-5");
    backend.complete(req).await.expect("complete ok");

    let requests = server.received_requests().await.expect("requests captured");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("body json");
    assert!(body.get("thinking").is_none());
}

#[tokio::test]
async fn responses_complete_with_reasoning_requests_summary_text() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_partial_json(json!({
            "include": ["reasoning.summary_text"],
            "reasoning": { "effort": "high" }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_5",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "ok" }]
            }],
            "usage": {
                "input_tokens": 4,
                "output_tokens": 6,
                "total_tokens": 10,
                "output_tokens_details": { "reasoning_tokens": 2 }
            }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));
    let req = build_request_with_options(
        "gpt-5-mini",
        json!({"reasoning": true, "reasoning_effort": "high"}),
    );
    let resp = backend.complete(req).await.expect("complete ok");
    let usage = resp.usage().expect("usage reported");
    assert_eq!(usage.input_tokens(), Some(4));
    assert_eq!(usage.output_tokens(), Some(6));
    assert_eq!(usage.reasoning_tokens(), Some(2));
}

#[tokio::test]
async fn responses_complete_without_reasoning_omits_include() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_6",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "ok" }]
            }],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));
    let req = build_request("gpt-5-mini");
    backend.complete(req).await.expect("complete ok");

    let requests = server.received_requests().await.expect("requests captured");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("body json");
    assert!(body.get("include").is_none());
}

#[tokio::test]
async fn responses_stream_emits_reasoning_summary_delta_and_usage() {
    let server = MockServer::start().await;

    let body = concat!(
        "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_7\"}}\n\n",
        "event: response.reasoning_summary_text.delta\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"r1\",\"delta\":\"Weighing the tradeoffs...\"}\n\n",
        "event: response.reasoning_summary_text.delta\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"item_id\":\"r1\",\"delta\":\" between A and B.\"}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"m1\",\"delta\":\"Answer\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_7\",\"usage\":{\"input_tokens\":9,\"output_tokens\":11,\"total_tokens\":20,\"output_tokens_details\":{\"reasoning_tokens\":5}}}}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));
    let req = build_request_with_options("gpt-5-mini", json!({"reasoning": true}));
    let mut stream = backend.stream(req).await.expect("stream starts");

    let mut reasoning = String::new();
    let mut usage = None;
    let mut done = false;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ReasoningDelta(t) => reasoning.push_str(&t),
            reimagine_agent_harness::AgentStreamEvent::Usage(u) => usage = Some(u),
            reimagine_agent_harness::AgentStreamEvent::Done { .. } => done = true,
            _ => {}
        }
    }
    assert_eq!(reasoning, "Weighing the tradeoffs... between A and B.");
    let usage = usage.expect("usage emitted");
    assert_eq!(usage.input_tokens(), Some(9));
    assert_eq!(usage.output_tokens(), Some(11));
    assert_eq!(usage.reasoning_tokens(), Some(5));
    assert!(done);
}

const PERSON_SCHEMA: &str = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#;

#[tokio::test]
async fn openai_complete_with_structured_output_sends_response_format() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": serde_json::from_str::<Value>(PERSON_SCHEMA).unwrap(),
                    "strict": true,
                }
            }
        })))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::openai_chat_completions_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "gpt-4o-mini",
        json!({
            "output_schema": serde_json::from_str::<Value>(PERSON_SCHEMA).unwrap(),
            "output_schema_name": "person",
        }),
    );
    backend.complete(req).await.expect("complete ok");
}

#[tokio::test]
async fn anthropic_complete_with_structured_output_sends_output_config() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": serde_json::from_str::<Value>(PERSON_SCHEMA).unwrap(),
                }
            }
        })))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "claude-sonnet-4-5",
        json!({ "output_schema": serde_json::from_str::<Value>(PERSON_SCHEMA).unwrap() }),
    );
    backend.complete(req).await.expect("complete ok");
}

#[tokio::test]
async fn responses_complete_with_structured_output_sends_text_format() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_partial_json(json!({
            "text": {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person",
                        "schema": serde_json::from_str::<Value>(PERSON_SCHEMA).unwrap(),
                        "strict": true,
                    }
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_8",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "{\"name\":\"Ada\"}" }]
            }],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_responses_with_http_client(
            ProviderName::new("responses-test"),
            responses_cfg_for(&server),
            http,
        ));
    let req = build_request_with_options(
        "gpt-5-mini",
        json!({
            "output_schema": serde_json::from_str::<Value>(PERSON_SCHEMA).unwrap(),
            "output_schema_name": "person",
        }),
    );
    backend.complete(req).await.expect("complete ok");
}

#[tokio::test]
async fn structured_output_stream_paths_share_the_request_injection() {
    // Streaming must carry the same structured-output payloads; verify
    // the OpenAI stream request body.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "stream": true,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": serde_json::from_str::<Value>(PERSON_SCHEMA).unwrap(),
                    "strict": true,
                }
            }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_string(""))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::openai_chat_completions_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&server),
        http,
    );
    let req = build_request_with_options(
        "gpt-4o-mini",
        json!({ "output_schema": serde_json::from_str::<Value>(PERSON_SCHEMA).unwrap() }),
    );
    backend.stream(req).await.expect("stream starts");
}

#[tokio::test]
async fn anthropic_complete_places_cache_control_at_all_three_breakpoints() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "system": [{
                "type": "text",
                "text": "You are a helpful assistant.",
                "cache_control": { "type": "ephemeral" }
            }],
            "tools": [
                {
                    "name": "echo",
                    "description": "echoes",
                    "input_schema": { "type": "object", "properties": {} }
                },
                {
                    "name": "second",
                    "description": "second tool",
                    "input_schema": { "type": "object", "properties": {} },
                    "cache_control": { "type": "ephemeral" }
                }
            ],
        })))
        .and(body_exact_json(json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "hi",
                        "cache_control": { "type": "ephemeral" }
                    }]
                }
            ],
            "tools": [
                {
                    "name": "echo",
                    "description": "echoes",
                    "input_schema": { "type": "object", "properties": {} }
                },
                {
                    "name": "second",
                    "description": "second tool",
                    "input_schema": { "type": "object", "properties": {} },
                    "cache_control": { "type": "ephemeral" }
                }
            ],
            "max_tokens": 4096,
            "system": [{
                "type": "text",
                "text": "You are a helpful assistant.",
                "cache_control": { "type": "ephemeral" }
            }]
        })))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = AgentRequest::new(
        ModelName::new("claude-sonnet-4-5"),
        vec![
            Message::system("You are a helpful assistant."),
            Message::user("hi"),
        ],
    )
    .with_tools(vec![
        AgentToolDefinition::new(
            "echo",
            "echoes",
            json!({"type": "object", "properties": {}}),
        ),
        AgentToolDefinition::new(
            "second",
            "second tool",
            json!({"type": "object", "properties": {}}),
        ),
    ]);
    backend.complete(req).await.expect("complete ok");
}

#[tokio::test]
async fn anthropic_cache_control_can_be_disabled_via_options() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_exact_json(json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "echo",
                "description": "echo a string",
                "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}}
            }],
            "max_tokens": 4096
        })))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request_with_options("claude-sonnet-4-5", json!({"cache_control": false}));
    backend.complete(req).await.expect("complete ok");
}

#[tokio::test]
async fn anthropic_complete_reports_cache_usage_in_total() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_3",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 200
            }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        http,
    );
    let req = build_request("claude-sonnet-4-5");
    let resp = backend.complete(req).await.expect("complete ok");
    let usage = resp.usage().expect("usage reported");
    assert_eq!(usage.cache_creation_input_tokens(), Some(100));
    assert_eq!(usage.cache_read_input_tokens(), Some(200));
    assert_eq!(usage.total(), Some(315));
}

#[tokio::test]
async fn openai_complete_reports_cached_tokens_as_cache_read() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl-3",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 7,
                "total_tokens": 57,
                "prompt_tokens_details": { "cached_tokens": 40 }
            }
        })))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend = ReqwestBackend::openai_chat_completions_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&server),
        http,
    );
    let req = build_request("gpt-4o-mini");
    let resp = backend.complete(req).await.expect("complete ok");
    let usage = resp.usage().expect("usage reported");
    assert_eq!(usage.cache_read_input_tokens(), Some(40));
    // OpenAI has no cache_creation slot; total = input + cache_read + output.
    assert_eq!(usage.total(), Some(97));
}

// ----- PV-03b: workspace url file-block resolution -----

fn image_url_request(model: &str, block: FileContentBlock) -> AgentRequest {
    AgentRequest::new(
        ModelName::new(model),
        vec![Message::user_with_blocks(vec![
            ContentBlock::Text("describe".into()),
            ContentBlock::File(block),
        ])],
    )
}

#[tokio::test]
async fn openai_complete_resolves_workspace_url_file_block_to_data_url() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(json!({
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
                ]
            }],
        })))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let workspace = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir_all(workspace.path().join("refs")).expect("create refs");
    std::fs::write(workspace.path().join("refs/pic.png"), b"hello").expect("write file");

    let backend = ReqwestBackend::openai_chat_completions_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&server),
        reqwest::Client::new(),
    )
    .with_workspace_dir(workspace.path());

    let resp = backend
        .complete(image_url_request(
            "gpt-4o-mini",
            FileContentBlock::url("image/png", "refs/pic.png"),
        ))
        .await
        .expect("complete ok");
    assert_eq!(resp.message().content(), "ok");
}

#[tokio::test]
async fn anthropic_complete_resolves_workspace_url_file_block_to_image_block() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "model": "claude-3-5-sonnet-latest",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8="
                        }
                    }
                ]
            }],
        })))
        .respond_with(anthropic_completion_response())
        .mount(&server)
        .await;

    let workspace = tempfile::tempdir().expect("tempdir");
    std::fs::write(workspace.path().join("pic.png"), b"hello").expect("write file");

    let backend = ReqwestBackend::anthropic_messages_with_http_client(
        ProviderName::new("anthropic-test"),
        anthropic_cfg_for(&server),
        reqwest::Client::new(),
    )
    .with_workspace_dir(workspace.path());

    let resp = backend
        .complete(image_url_request(
            "claude-3-5-sonnet-latest",
            FileContentBlock::url("image/png", "pic.png"),
        ))
        .await
        .expect("complete ok");
    assert_eq!(resp.message().content(), "ok");
}

#[tokio::test]
async fn openai_complete_missing_workspace_file_is_configuration_error() {
    let workspace = tempfile::tempdir().expect("tempdir");
    let backend = ReqwestBackend::openai_chat_completions_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&MockServer::start().await),
        reqwest::Client::new(),
    )
    .with_workspace_dir(workspace.path());

    let err = backend
        .complete(image_url_request(
            "gpt-4o-mini",
            FileContentBlock::url("image/png", "no-such.png"),
        ))
        .await
        .expect_err("must fail before any HTTP call");
    assert!(matches!(err, ProviderAdapterError::Configuration(_)));
    assert!(
        err.to_string().contains("failed to read workspace file"),
        "{err}"
    );
}

#[tokio::test]
async fn openai_complete_url_file_block_without_workspace_dir_is_configuration_error() {
    let backend = ReqwestBackend::openai_chat_completions_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&MockServer::start().await),
        reqwest::Client::new(),
    );

    let err = backend
        .complete(image_url_request(
            "gpt-4o-mini",
            FileContentBlock::url("image/png", "refs/pic.png"),
        ))
        .await
        .expect_err("must fail before any HTTP call");
    assert!(matches!(err, ProviderAdapterError::Configuration(_)));
    assert!(err.to_string().contains("without one"), "{err}");
}

#[tokio::test]
async fn openai_complete_rejects_remote_url_file_block() {
    let workspace = tempfile::tempdir().expect("tempdir");
    let backend = ReqwestBackend::openai_chat_completions_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&MockServer::start().await),
        reqwest::Client::new(),
    )
    .with_workspace_dir(workspace.path());

    let err = backend
        .complete(image_url_request(
            "gpt-4o-mini",
            FileContentBlock::url("image/png", "https://example.com/pic.png"),
        ))
        .await
        .expect_err("remote downloads must be rejected");
    assert!(matches!(err, ProviderAdapterError::Configuration(_)));
    assert!(
        err.to_string()
            .contains("remote URLs are not supported in V2"),
        "{err}"
    );
}

#[tokio::test]
async fn openai_complete_rejects_non_image_file_block() {
    let workspace = tempfile::tempdir().expect("tempdir");
    std::fs::write(workspace.path().join("clip.mp3"), b"audio").expect("write file");

    let backend = ReqwestBackend::openai_chat_completions_with_http_client(
        ProviderName::new("openai-test"),
        openai_cfg_for(&MockServer::start().await),
        reqwest::Client::new(),
    )
    .with_workspace_dir(workspace.path());

    // Inline non-image data block: rejected by the translation layer.
    let err = backend
        .complete(image_url_request(
            "gpt-4o-mini",
            FileContentBlock::data("audio/mpeg", "QUFBQQ=="),
        ))
        .await
        .expect_err("non-image file blocks must be rejected");
    assert!(matches!(err, ProviderAdapterError::Configuration(_)));
    assert!(err.to_string().contains("audio/mpeg"), "{err}");

    // Workspace-resolved non-image file block: rejected the same way
    // after resolution.
    let err = backend
        .complete(image_url_request(
            "gpt-4o-mini",
            FileContentBlock::url("audio/mpeg", "clip.mp3"),
        ))
        .await
        .expect_err("non-image file blocks must be rejected");
    assert!(err.to_string().contains("audio/mpeg"), "{err}");
}

#[tokio::test]
async fn openai_stream_emits_terminal_done_with_finish_reason() {
    // AC-01: the OpenAI chat-completions stream must surface its
    // finish_reason on the terminal Done so the loop can distinguish
    // truncation ("length") from a clean "stop".
    let server = MockServer::start().await;

    let sse_body = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"index\":0}]}",
        "",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\",\"index\":0}]}",
        "",
        "data: [DONE]",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(build_request("gpt-4o-mini"))
        .await
        .expect("stream starts");

    let mut text = String::new();
    let mut done_reason: Option<String> = None;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ContentDelta(delta) => text.push_str(&delta),
            reimagine_agent_harness::AgentStreamEvent::Done { stop_reason } => {
                done_reason = stop_reason;
            }
            _ => {}
        }
    }
    assert_eq!(text, "partial");
    assert_eq!(done_reason.as_deref(), Some("length"));
}

#[tokio::test]
async fn openai_stream_eof_without_done_still_emits_done() {
    // AC-01/AC-06: a stream that ends at EOF without the [DONE] marker
    // still terminates with a Done carrying the last finish_reason.
    let server = MockServer::start().await;

    let sse_body = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"index\":0}]}",
        "",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(build_request("gpt-4o-mini"))
        .await
        .expect("stream starts");

    let mut text = String::new();
    let mut done_reason: Option<String> = None;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ContentDelta(delta) => text.push_str(&delta),
            reimagine_agent_harness::AgentStreamEvent::Done { stop_reason } => {
                done_reason = stop_reason;
            }
            _ => {}
        }
    }
    assert_eq!(text, "done");
    assert_eq!(done_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn anthropic_stream_emits_done_with_stop_reason() {
    // AC-01: Anthropic's stop_reason from message_delta must reach the
    // terminal Done instead of being discarded.
    let server = MockServer::start().await;

    let sse_body = [
        "event: message_start",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":7}}}",
        "",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}",
        "",
        "event: message_delta",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":5}}",
        "",
        "event: message_stop",
        "data: {\"type\":\"message_stop\"}",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::anthropic_messages_with_http_client(
            ProviderName::new("anthropic-test"),
            anthropic_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(build_request("claude-3-5-sonnet-latest"))
        .await
        .expect("stream starts");

    let mut text = String::new();
    let mut done_reason: Option<String> = None;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ContentDelta(delta) => text.push_str(&delta),
            reimagine_agent_harness::AgentStreamEvent::Done { stop_reason } => {
                done_reason = stop_reason;
            }
            _ => {}
        }
    }
    assert_eq!(text, "partial");
    assert_eq!(done_reason.as_deref(), Some("max_tokens"));
}

#[tokio::test]
async fn openai_empty_stream_yields_no_terminal_done() {
    // AC-06: a zero-event stream must not fabricate a Done; the loop
    // reports it as EMPTY_STREAM.
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(""))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(build_request("gpt-4o-mini"))
        .await
        .expect("stream starts");

    let mut any_event = false;
    while let Some(_event) = stream.next_event().await {
        any_event = true;
    }
    assert!(!any_event, "empty stream yields no events at all");
}

// ------------------------------------------------------------------
// AC-13: retry semantics (complete / list_models only; stream never
// retries). Request-count assertions are the lever: jitter makes exact
// sleep durations nondeterministic, so we assert how many times the
// wiremock was hit (1 initial attempt + up to MAX_RETRIES retries).
// ------------------------------------------------------------------

#[tokio::test]
async fn complete_retries_after_429_then_succeeds() {
    let server = MockServer::start().await;

    // Priority 1 (highest) + `up_to_n_times(1)`: answers the first
    // request with 429, then stops matching so the success mock below
    // handles the retry.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {OPENAI_KEY}")))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let resp = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect("429 is retryable; the retry succeeds");
    assert_eq!(resp.message().content(), "ok");

    let requests = server.received_requests().await.expect("requests captured");
    assert_eq!(requests.len(), 2, "1 initial attempt + 1 retry");
}

#[tokio::test]
async fn complete_retries_after_500_then_succeeds() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {OPENAI_KEY}")))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let resp = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect("5xx is retryable; the retry succeeds");
    assert_eq!(resp.message().content(), "ok");

    let requests = server.received_requests().await.expect("requests captured");
    assert_eq!(requests.len(), 2, "1 initial attempt + 1 retry");
}

#[tokio::test]
async fn complete_retries_exhaust_after_three_retries() {
    let server = MockServer::start().await;

    // No `up_to_n_times`: the mock 429s every request, so all retries
    // are consumed and the last error surfaces.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("still rate limited"))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let err = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect_err("all retries exhausted; the last error surfaces");
    match err {
        ProviderAdapterError::Api { code, .. } => assert_eq!(code, "429"),
        other => panic!("expected Api error, got {other:?}"),
    }

    let requests = server.received_requests().await.expect("requests captured");
    assert_eq!(
        requests.len(),
        4,
        "1 initial attempt + 3 retries (MAX_RETRIES = 3)"
    );
}

#[tokio::test]
async fn complete_honors_retry_after_header() {
    let server = MockServer::start().await;

    // `Retry-After: 0` parses to a zero delay; the loop still sleeps
    // its local backoff (`max(0, backoff)`) and must not kill the
    // retry. The request count proves the header was parsed and the
    // retry fired.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "0")
                .set_body_string("slow down"),
        )
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {OPENAI_KEY}")))
        .respond_with(openai_completion_response())
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let resp = backend
        .complete(build_request("gpt-4o-mini"))
        .await
        .expect("Retry-After: 0 must not prevent the retry");
    assert_eq!(resp.message().content(), "ok");

    let requests = server.received_requests().await.expect("requests captured");
    assert_eq!(requests.len(), 2, "1 initial attempt + 1 retry");
}

#[tokio::test]
async fn stream_does_not_retry_on_429() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let err = match backend.stream(build_request("gpt-4o-mini")).await {
        Ok(_stream) => panic!("expected the 429 to surface as an error"),
        Err(err) => err,
    };
    match err {
        ProviderAdapterError::Api { code, .. } => assert_eq!(code, "429"),
        other => panic!("expected Api error, got {other:?}"),
    }

    let requests = server.received_requests().await.expect("requests captured");
    assert_eq!(requests.len(), 1, "stream must never retry");
}

// ------------------------------------------------------------------
// AC-14: OpenAI / Anthropic SSE decode end-to-end through wiremock
// (the unit-level accumulators are covered in tests/stream_openai.rs
// and tests/stream_anthropic.rs; these exercise the full
// `ReqwestSseStream` pipeline).
// ------------------------------------------------------------------

#[tokio::test]
async fn openai_stream_assembles_tool_call_from_fragmented_deltas() {
    // id + name in the first fragment, arguments split across two
    // chunks; `finish_reason: "tool_calls"` flushes the assembled call
    // and `[DONE]` terminates with that stop reason.
    let server = MockServer::start().await;

    let sse_body = [
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_9\",\"function\":{\"name\":\"echo\",\"arguments\":\"{\\\"x\\\":\"}}]},\"index\":0}]}",
        "",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" 42}\"}}]},\"index\":0}]}",
        "",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\",\"index\":0}]}",
        "",
        "data: [DONE]",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(build_request("gpt-4o-mini"))
        .await
        .expect("stream starts");

    let mut tool_calls = Vec::new();
    let mut done_reason: Option<String> = None;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ToolCall(call) => tool_calls.push(call),
            reimagine_agent_harness::AgentStreamEvent::Done { stop_reason } => {
                done_reason = stop_reason;
            }
            _ => {}
        }
    }
    assert_eq!(tool_calls.len(), 1, "one complete tool call expected");
    assert_eq!(tool_calls[0].id().as_str(), "call_9");
    assert_eq!(tool_calls[0].name(), "echo");
    assert_eq!(tool_calls[0].arguments(), &json!({"x": 42}));
    assert_eq!(done_reason.as_deref(), Some("tool_calls"));
}

#[tokio::test]
async fn anthropic_stream_merges_usage_from_message_start_and_message_delta() {
    // message_start carries input-side counts (input_tokens, cache
    // fields); message_delta carries output_tokens. The accumulator
    // must merge them into a single Usage event.
    let server = MockServer::start().await;

    let sse_body = [
        "event: message_start",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":12,\"cache_creation_input_tokens\":100,\"cache_read_input_tokens\":50}}}",
        "",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}",
        "",
        "event: message_delta",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}",
        "",
        "event: message_stop",
        "data: {\"type\":\"message_stop\"}",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::anthropic_messages_with_http_client(
            ProviderName::new("anthropic-test"),
            anthropic_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(build_request("claude-3-5-sonnet-latest"))
        .await
        .expect("stream starts");

    let mut text = String::new();
    let mut usages = Vec::new();
    let mut done = false;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ContentDelta(delta) => text.push_str(&delta),
            reimagine_agent_harness::AgentStreamEvent::Usage(usage) => usages.push(usage),
            reimagine_agent_harness::AgentStreamEvent::Done { .. } => done = true,
            _ => {}
        }
    }
    assert_eq!(text, "hi");
    assert_eq!(
        usages.len(),
        1,
        "message_start + message_delta counts merge into a single Usage"
    );
    let usage = &usages[0];
    assert_eq!(usage.input_tokens(), Some(12));
    assert_eq!(usage.output_tokens(), Some(7));
    assert_eq!(usage.cache_creation_input_tokens(), Some(100));
    assert_eq!(usage.cache_read_input_tokens(), Some(50));
    assert_eq!(
        usage.total(),
        Some(169),
        "input + cache_creation + cache_read + output"
    );
    assert!(done, "message_stop terminates the stream");
}

#[tokio::test]
async fn openai_stream_skips_malformed_sse_event_without_breaking() {
    // A data line that is not valid JSON is skipped (decode is
    // permissive); surrounding content and the terminal Done still
    // arrive.
    let server = MockServer::start().await;

    let sse_body = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"be\"},\"index\":0}]}",
        "",
        "data: {this is not valid json",
        "",
        "data: {\"choices\":[{\"delta\":{\"content\":\"fore\"},\"index\":0}]}",
        "",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}",
        "",
        "data: [DONE]",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(build_request("gpt-4o-mini"))
        .await
        .expect("stream starts");

    let mut text = String::new();
    let mut done_reason: Option<String> = None;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ContentDelta(delta) => text.push_str(&delta),
            reimagine_agent_harness::AgentStreamEvent::Done { stop_reason } => {
                done_reason = stop_reason;
            }
            _ => {}
        }
    }
    assert_eq!(
        text, "before",
        "content on both sides of the bad event arrives"
    );
    assert_eq!(done_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn openai_stream_eof_with_pending_tool_call_fragments_drops_partial_call() {
    // A stream that ends at EOF with a partially assembled tool call
    // (no `finish_reason: "tool_calls"`, no [DONE]) terminates with a
    // `Done` carrying no stop reason and never emits the partial
    // `ToolCall` (D-5). The drop is no longer silent: the transport
    // emits a host-visible `Warning` before the terminal `Done`.
    let server = MockServer::start().await;

    let sse_body = [
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_9\",\"function\":{\"name\":\"echo\",\"arguments\":\"{\\\"x\\\":\"}}]},\"index\":0}]}",
        "",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" 42}\"}}]},\"index\":0}]}",
        "",
    ]
    .join("\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let backend: Arc<dyn CompletionBackend> =
        Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
            ProviderName::new("openai-test"),
            openai_cfg_for(&server),
            http,
        ));

    let mut stream = backend
        .stream(build_request("gpt-4o-mini"))
        .await
        .expect("stream starts");

    let mut tool_calls = Vec::new();
    let mut done_count = 0;
    let mut done_reason: Option<String> = None;
    let mut warnings = Vec::new();
    let mut seen_done = false;
    while let Some(event) = stream.next_event().await {
        match event {
            reimagine_agent_harness::AgentStreamEvent::ToolCall(call) => tool_calls.push(call),
            reimagine_agent_harness::AgentStreamEvent::Warning(message) => {
                assert!(
                    !seen_done,
                    "the Warning must precede the terminal Done (Warning -> Done ordering)"
                );
                warnings.push(message);
            }
            reimagine_agent_harness::AgentStreamEvent::Done { stop_reason } => {
                seen_done = true;
                done_count += 1;
                done_reason = stop_reason;
            }
            _ => {}
        }
    }
    assert!(
        tool_calls.is_empty(),
        "partial tool call is not flushed at EOF without a finish_reason"
    );
    assert_eq!(done_count, 1, "stream still terminates with a single Done");
    assert_eq!(done_reason, None, "no finish_reason was seen before EOF");
    assert_eq!(
        warnings,
        vec!["stream ended with incomplete tool call(s)"],
        "the dropped partial tool call is surfaced as a Warning (D-5)"
    );
}
