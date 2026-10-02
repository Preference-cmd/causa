//! Scripted end-to-end: direct `TurnRunner::run` drives the real
//! `AnthropicMessagesGateway` against a local wiremock double through two
//! tool round trips.
//!
//! The full path in one test: ContextFrame rendering → HTTP → response
//! parsing → driver rounds → tool execution → fact commit → next-round
//! frame.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use causa_kernel::{
    BlockContent, BlockMeta, CallControl, CancellationToken, ContentPart, Context, ContextBlock,
    ModelGateway, ModelRef, TextPayload, Tool, ToolCallContext, ToolDefinition, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnId,
};
use causa_provider::AnthropicMessagesGateway;
use causa_runtime::{
    RunControl, ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner,
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
    async fn execute(&self, ctx: &ToolCallContext, _c: &CallControl) -> ToolResultPayload {
        ToolResultPayload {
            call_block_id: ctx.call_block_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("file-a")),
            media: Vec::new(),
            notes: Vec::new(),
        }
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
async fn direct_runner_completes_two_tool_round_trips_over_http() {
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
        Arc::new(
            ToolExecutor::new(vec![Arc::new(ReadTool)], ToolExecutorOptions::default()).unwrap(),
        ),
    );

    let context = Context::from_blocks(vec![ContextBlock::new(
        causa_runtime::new_block_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("find files"))]),
        BlockMeta {
            source: Some("user".into()),
            ..Default::default()
        },
    )])
    .unwrap();
    let options = TurnRunOptions::new(ModelRef::new("claude-test"));
    let outcome = runner
        .run(
            TurnId::new("t1"),
            context,
            options,
            RunControl::new(CancellationToken::new(), None),
        )
        .await;

    let final_output = match outcome.result {
        TurnResult::Completed { final_output } => final_output,
        other => panic!("expected completion, got {other:?}"),
    };
    assert_eq!(final_output.response.text.0, "done");
    let results: Vec<_> = outcome
        .context
        .blocks()
        .iter()
        .filter_map(|block| match block.content() {
            BlockContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|result| result.status == ToolResultStatus::Succeeded)
    );
    assert!(outcome.uncommitted_tool_batch.is_none());

    // The scripted responder ignores request bodies, so the pairing
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
