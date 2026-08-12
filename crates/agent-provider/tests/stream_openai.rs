use reimagine_agent_harness::{
    AgentProvider, AgentRequest, AgentResponse, AgentStreamEvent, Message, ModelName, ProviderName,
};
use reimagine_ai_protocol::translation::streaming::OpenAiStreamAccumulator;
use reimagine_agent_provider::{
    CompletionBackend, FakeCompletionBackend, OpenAiChatCompletionsConfig,
    OpenAiChatCompletionsProvider, ScriptedBackendStep,
};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn openai_adapter_complete_returns_response_and_maps_error() {
    let backend: Arc<dyn CompletionBackend> = Arc::new(FakeCompletionBackend::new(vec![
        ScriptedBackendStep::Complete(Ok(
            AgentResponse::new(Message::assistant("hi back")).with_stop_reason("stop")
        )),
    ]));
    let provider = OpenAiChatCompletionsProvider::with_backend(
        ProviderName::new("openai"),
        OpenAiChatCompletionsConfig::new("https://api.example.com/v1", "sk", "gpt-4o-mini"),
        backend,
    );
    let req = AgentRequest::new(ModelName::new("gpt-4o-mini"), vec![Message::user("hi")]);
    let resp = provider.complete(req).await.expect("complete ok");
    assert_eq!(resp.message().content(), "hi back");
    assert_eq!(resp.stop_reason(), Some("stop"));
}

#[tokio::test]
async fn openai_adapter_complete_maps_backend_error_to_provider_error() {
    let backend: Arc<dyn CompletionBackend> = Arc::new(FakeCompletionBackend::new(vec![
        ScriptedBackendStep::Complete(Err(
            reimagine_ai_protocol::ProviderAdapterError::transport("connection refused"),
        )),
    ]));
    let provider = OpenAiChatCompletionsProvider::with_backend(
        ProviderName::new("openai"),
        OpenAiChatCompletionsConfig::new("https://api.example.com/v1", "sk", "gpt-4o-mini"),
        backend,
    );
    let req = AgentRequest::new(ModelName::new("gpt-4o-mini"), vec![Message::user("hi")]);
    let err = provider
        .complete(req)
        .await
        .expect_err("provider error expected");
    assert_eq!(err.code(), "TRANSPORT");
    assert!(err.message().contains("connection refused"));
    assert_eq!(err.provider().map(|p| p.as_str()), Some("openai"));
}

#[tokio::test]
async fn openai_adapter_stream_emits_deltas_complete_tool_call_and_done() {
    // Simulate the OpenAI chunk shape across three chunks.
    let chunks = vec![
        json!({
            "choices": [{
                "delta": { "role": "assistant", "content": "He" }
            }]
        }),
        json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "c1",
                        "function": { "name": "echo", "arguments": "{\"x\":" }
                    }]
                }
            }]
        }),
        json!({
            "choices": [{
                "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": "1}" } }] },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 4 }
        }),
    ];
    let mut acc = OpenAiStreamAccumulator::new();
    let mut events = Vec::new();
    for chunk in &chunks {
        events.extend(acc.ingest_chunk(chunk).unwrap());
    }
    let complete_calls = acc.flush_complete_tool_calls();
    events.extend(complete_calls);
    events.push(acc.finalize());

    // Expect: ContentDelta("He"), ToolCallDelta id, ToolCallDelta name,
    // ToolCallDelta args (×2), Usage, ToolCall, Done.
    let mut kinds: Vec<&'static str> = Vec::new();
    for e in &events {
        match e {
            AgentStreamEvent::ContentDelta(_) => kinds.push("content"),
            AgentStreamEvent::ToolCallDelta { .. } => kinds.push("delta"),
            AgentStreamEvent::ToolCall(_) => kinds.push("complete"),
            AgentStreamEvent::Usage(_) => kinds.push("usage"),
            AgentStreamEvent::ReasoningDelta(_) => kinds.push("reasoning"),
            AgentStreamEvent::Done { .. } => kinds.push("done"),
            AgentStreamEvent::Compacted { .. } => kinds.push("compacted"),
        }
    }
    assert!(kinds.contains(&"content"));
    assert!(kinds.contains(&"delta"));
    assert!(kinds.contains(&"complete"));
    assert!(kinds.contains(&"usage"));
    assert_eq!(kinds.last(), Some(&"done"));

    // Find the complete tool call and check it.
    let complete = events
        .iter()
        .find_map(|e| match e {
            AgentStreamEvent::ToolCall(c) => Some(c),
            _ => None,
        })
        .expect("complete tool call emitted");
    assert_eq!(complete.id().as_str(), "c1");
    assert_eq!(complete.name(), "echo");
    assert_eq!(complete.arguments(), &json!({"x": 1}));
}

#[tokio::test]
async fn openai_reasoning_content_emits_reasoning_delta_and_usage_reasoning_tokens() {
    let chunks = vec![
        json!({
            "choices": [{
                "delta": { "role": "assistant", "reasoning_content": "Let me think..." }
            }]
        }),
        json!({
            "choices": [{
                "delta": { "content": "Answer" }
            }]
        }),
        json!({
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "completion_tokens_details": { "reasoning_tokens": 3 }
            }
        }),
    ];
    let mut acc = OpenAiStreamAccumulator::new();
    let mut events = Vec::new();
    for chunk in &chunks {
        events.extend(acc.ingest_chunk(chunk).unwrap());
    }
    events.push(acc.finalize());

    let reasoning: String = events
        .iter()
        .filter_map(|e| match e {
            AgentStreamEvent::ReasoningDelta(t) => Some(t.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "Let me think...");

    let usage = events
        .iter()
        .find_map(|e| match e {
            AgentStreamEvent::Usage(u) => Some(u.clone()),
            _ => None,
        })
        .expect("usage emitted");
    assert_eq!(usage.input_tokens(), Some(5));
    assert_eq!(usage.output_tokens(), Some(7));
    assert_eq!(usage.reasoning_tokens(), Some(3));
}

#[tokio::test]
async fn openai_reasoning_tokens_fallback_to_top_level_field() {
    let chunk = json!({
        "choices": [{
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 2, "reasoning_tokens": 2 }
    });
    let mut acc = OpenAiStreamAccumulator::new();
    let events = acc.ingest_chunk(&chunk).unwrap();
    let usage = events
        .iter()
        .find_map(|e| match e {
            AgentStreamEvent::Usage(u) => Some(u.clone()),
            _ => None,
        })
        .expect("usage emitted");
    assert_eq!(usage.reasoning_tokens(), Some(2));
}
