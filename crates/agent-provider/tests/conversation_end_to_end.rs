//! Slice 3 Phase D — scripted end-to-end: the kernel's
//! `run_in_conversation` drives the real `AnthropicMessagesGateway`
//! against a local wiremock double through two tool round trips.
//!
//! This is the full Slice 3 path in one test: ContextFrame rendering →
//! HTTP → response parsing → driver rounds → tool execution → fact
//! commit → next-round frame.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use reimagine_agent_provider::AnthropicMessagesGateway;
use reimagine_context_kernel::{
    CallControl, CancellationToken, ConversationId, ConversationState, ModelGateway, ModelRef,
    RunControl, TextPayload, Tool, ToolCallContext, ToolDefinition, ToolExecutionOutcome,
    ToolExecutor, ToolOutput, ToolResultPayload, ToolResultStatus, ToolSurface, TurnId,
    TurnInvocation, TurnResult, TurnRunOptions, TurnRunner,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const KEY: &str = "sk-test-anthropic";

/// Pops scripted responses in order: wiremock mocks keep matching after
/// `expect` is met, so sequencing lives here, not in mock exhaustion.
struct QueuedResponder(Arc<Mutex<VecDeque<ResponseTemplate>>>);
impl Respond for QueuedResponder {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        self.0
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| ResponseTemplate::new(500).set_body_string("script exhausted"))
    }
}

struct ReadTool;
#[async_trait::async_trait]
impl Tool for ReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read".into(),
            description: "read a file".into(),
            parameters: json!({"type": "object"}),
        }
    }
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolExecutionOutcome {
        ToolExecutionOutcome::new(ToolResultPayload {
            call_id: ctx.call_id.clone(),
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("file-a")),
        })
    }
}

fn round_response(text: &str, tool_id: Option<&str>) -> Value {
    let mut content = vec![json!({"type": "text", "text": text})];
    if let Some(tool_id) = tool_id {
        content.push(json!({
            "type": "tool_use", "id": tool_id, "name": "read", "input": {"path": "a"},
        }));
    }
    json!({
        "content": content,
        "stop_reason": if tool_id.is_some() { "tool_use" } else { "end_turn" },
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
}

#[tokio::test]
async fn run_in_conversation_completes_two_tool_round_trips_over_http() {
    let server = MockServer::start().await;
    // Rounds 0 and 1 each request the read tool; round 2 ends the turn.
    let script = Arc::new(Mutex::new(VecDeque::from([
        ResponseTemplate::new(200).set_body_json(round_response("reading a", Some("toolu_1"))),
        ResponseTemplate::new(200).set_body_json(round_response("reading b", Some("toolu_2"))),
        ResponseTemplate::new(200).set_body_json(round_response("done", None)),
    ])));
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(QueuedResponder(script))
        .mount(&server)
        .await;

    let gateway: Arc<dyn ModelGateway> =
        Arc::new(AnthropicMessagesGateway::new(KEY).with_base_url(server.uri()));
    let runner = TurnRunner::new(
        gateway,
        Arc::new(ToolExecutor::from_vec(vec![Arc::new(ReadTool)])),
    );

    let mut state = ConversationState::new(ConversationId("c1".into()));
    state
        .begin_turn(TurnId::new("t1"))
        .unwrap()
        .append_input(TextPayload::new("find files"), "user")
        .unwrap();

    let options = TurnRunOptions {
        invocation: TurnInvocation {
            model: ModelRef::new("claude-test"),
            tool_surface: ToolSurface::from_definitions(vec![ReadTool.definition()]),
            generation: Default::default(),
        },
        ..Default::default()
    };

    let outcome = runner
        .run_in_conversation(
            state,
            options,
            RunControl::new(CancellationToken::new(), None),
        )
        .await
        .unwrap();

    let final_output = match outcome.result {
        TurnResult::Completed { final_output } => final_output,
        other => panic!("expected completion, got {other:?}"),
    };
    assert_eq!(final_output.response.text.0, "done");
    // three model rounds; the first two each executed one tool call
    assert_eq!(outcome.trace.rounds.len(), 3);
    for round in &outcome.trace.rounds[..2] {
        let batch = round.tool_batch.as_ref().expect("tool batch");
        assert_eq!(batch.calls.len(), 1);
        assert_eq!(batch.calls[0].status, ToolResultStatus::Succeeded);
    }
    assert!(outcome.trace.rounds[2].tool_batch.is_none());
    // the runner sealed the active turn as a completed fact
    assert!(
        outcome
            .state
            .active_turn()
            .expect("active turn")
            .is_sealed()
    );

    // P1-5: the scripted responder ignores request bodies, so the pairing
    // round trip is pinned by inspecting what actually went over the wire.
    // Each round's frame must carry the PRIOR round's tool result with the
    // matching provider id — a broken pairing map would fail here.
    let requests = server.received_requests().await.expect("requests captured");
    assert_eq!(requests.len(), 3, "one request per model round");
    let bodies: Vec<Value> = requests
        .iter()
        .map(|r| serde_json::from_slice(&r.body).expect("request body is JSON"))
        .collect();

    // Round 0: the bare input frame — no tool results yet.
    let messages = bodies[0]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");

    // Round 1: carries round 0's tool result, paired to toolu_1.
    let messages = bodies[1]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(
        messages[2]["content"][0],
        json!({"type": "tool_result", "tool_use_id": "toolu_1", "content": "file-a"}),
    );

    // Round 2: carries round 1's tool result, paired to toolu_2.
    let messages = bodies[2]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 5);
    assert_eq!(
        messages[4]["content"][0],
        json!({"type": "tool_result", "tool_use_id": "toolu_2", "content": "file-a"}),
    );
}
