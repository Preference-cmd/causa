use reimagine_agent_harness::{AgentRequest, AgentStreamEvent, Message, ModelInfo};
use reimagine_agent_provider::{CompletionBackend, FakeCompletionBackend, ScriptedBackendStep};

// The plan's test file had a bare `impl CompletionBackend for FakeCompletionBackend {}`
// as a compile-time trait-in-scope check, but Rust's orphan rules forbid
// implementing a foreign trait for a foreign type. The actual impl lives in
// `backend.rs`; the import above is enough to confirm the trait is reachable.
//
// The plan also called `futures::stream::StreamExt::next`, but
// `reimagine_agent_harness::AgentStream` is not a `futures::Stream` — it only exposes
// `next_event`. The test loops on `next_event` directly.

#[tokio::test]
async fn fake_backend_replays_scripted_complete_steps() {
    let backend = FakeCompletionBackend::new(vec![ScriptedBackendStep::Complete(Ok(
        reimagine_agent_harness::AgentResponse::new(Message::assistant("hello"))
            .with_stop_reason("end_turn"),
    ))]);
    let req = AgentRequest::new(
        reimagine_agent_harness::ModelName::new("gpt-test"),
        vec![Message::user("hi")],
    );
    let resp = backend.complete(req).await.expect("complete ok");
    assert_eq!(resp.message().content(), "hello");
}

#[tokio::test]
async fn fake_backend_replays_scripted_complete_error() {
    let backend = FakeCompletionBackend::new(vec![ScriptedBackendStep::Complete(Err(
        reimagine_ai_protocol::ProviderAdapterError::transport("connection refused"),
    ))]);
    let req = AgentRequest::new(
        reimagine_agent_harness::ModelName::new("gpt-test"),
        vec![Message::user("hi")],
    );
    let resp = backend.complete(req).await.expect_err("err");
    assert_eq!(
        resp,
        reimagine_ai_protocol::ProviderAdapterError::transport("connection refused")
    );
}

#[tokio::test]
async fn fake_backend_replays_scripted_stream_steps() {
    let backend = FakeCompletionBackend::new(vec![ScriptedBackendStep::Stream(vec![
        Ok(AgentStreamEvent::ContentDelta("hel".into())),
        Ok(AgentStreamEvent::ContentDelta("lo".into())),
        Ok(AgentStreamEvent::Done {
            stop_reason: Some("end_turn".into()),
        }),
    ])]);
    let req = AgentRequest::new(
        reimagine_agent_harness::ModelName::new("gpt-test"),
        vec![Message::user("hi")],
    );
    let mut stream = backend.stream(req).await.expect("stream ok");
    let mut collected = Vec::new();
    while let Some(ev) = stream.next_event().await {
        collected.push(ev);
    }
    assert_eq!(collected.len(), 3);
    assert!(matches!(&collected[0], AgentStreamEvent::ContentDelta(s) if s == "hel"));
    assert!(matches!(&collected[1], AgentStreamEvent::ContentDelta(s) if s == "lo"));
    assert!(collected[2].is_done());
}

#[tokio::test]
async fn fake_backend_list_models_returns_static_set() {
    let backend = FakeCompletionBackend::new(vec![]).with_models(vec![
        ModelInfo::new(reimagine_agent_harness::ModelName::new("gpt-4o-mini"))
            .with_provider(reimagine_agent_harness::ProviderName::new("openai"))
            .with_capability(reimagine_agent_harness::ModelCapability::Chat)
            .with_capability(reimagine_agent_harness::ModelCapability::ToolUse),
    ]);
    let models = backend.list_models().await.expect("list ok");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name().as_str(), "gpt-4o-mini");
}
