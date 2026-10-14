//! Native tool-change rendering through the real gateways: each switch changes
//! the body it names, and the Anthropic one sends its beta value next to the
//! host's.

use causa_kernel::{
    BlockContent, BlockId, BlockMeta, CallControl, CancellationToken, ContentPart, ContextBlock,
    ContextFrame, GenerationOptions, InvocationId, ModelGateway, ModelRef, ModelRequest, RoundId,
    TextPayload, ToolChange, ToolDefinition, TurnId,
};
use causa_protocol::translation::anthropic::INLINE_TOOLS_BETA;
use causa_provider::{
    AnthropicMessagesGateway, AnthropicRenderOptions, OpenAiResponsesGateway,
    ResponsesRenderOptions,
};
use serde_json::{Value, json};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY: &str = "sk-test";

fn block(id: u128, content: BlockContent, meta: BlockMeta) -> ContextBlock {
    ContextBlock::new(BlockId::new(uuid::Uuid::from_u128(id)), content, meta)
}

fn added(names: &[&str]) -> BlockContent {
    BlockContent::ToolChange(ToolChange::new(
        names
            .iter()
            .map(|name| ToolDefinition::new(*name, *name, json!({"type": "object"})))
            .collect(),
        vec![],
    ))
}

/// A declaration of `read`, a user turn, then `write` offered after it.
fn request() -> ModelRequest {
    let frame = ContextFrame {
        blocks: vec![
            block(1, added(&["read"]), BlockMeta::default()),
            block(
                2,
                BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("hi"))]),
                BlockMeta::from_source("user"),
            ),
            block(3, added(&["write"]), BlockMeta::default()),
        ],
    };
    let mut request = ModelRequest::new(
        InvocationId {
            turn_id: TurnId::new("tools"),
            round_id: RoundId(0),
        },
        frame,
        ModelRef::new("model-test"),
    );
    request.generation = GenerationOptions::default().with_max_tokens(1024);
    request
}

fn control() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

fn tool_names(tools: &Value) -> Vec<&str> {
    tools
        .as_array()
        .expect("tools")
        .iter()
        .map(|tool| tool["name"].as_str().expect("name"))
        .collect()
}

async fn answering(body: Value) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(&server)
        .await;
    server
}

async fn only_request(server: &MockServer) -> wiremock::Request {
    let mut received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    received.remove(0)
}

fn beta(request: &wiremock::Request) -> Option<&str> {
    request
        .headers
        .get("anthropic-beta")
        .and_then(|value| value.to_str().ok())
}

#[tokio::test]
async fn anthropic_native_changes_keep_tools_and_add_the_beta_after_host_values() {
    let answer = json!({"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"});

    let server = answering(answer.clone()).await;
    AnthropicMessagesGateway::new(KEY)
        .with_base_url(format!("{}/v1", server.uri()))
        .invoke(&request(), &control())
        .await
        .unwrap();
    let folded = only_request(&server).await;
    assert_eq!(beta(&folded), None, "no header unless switched on");
    let body: Value = folded.body_json().unwrap();
    assert_eq!(tool_names(&body["tools"]), ["read", "write"]);

    for host in [vec!["context-1m-2025-08-07"], vec![INLINE_TOOLS_BETA]] {
        let server = answering(answer.clone()).await;
        let mut gateway = AnthropicMessagesGateway::new(KEY)
            .with_base_url(format!("{}/v1", server.uri()))
            .with_native_tool_changes(true);
        for value in &host {
            gateway = gateway.with_beta(*value);
        }
        gateway.invoke(&request(), &control()).await.unwrap();
        let native = only_request(&server).await;
        let expected = if host[0] == INLINE_TOOLS_BETA {
            INLINE_TOOLS_BETA.to_owned()
        } else {
            format!("{},{INLINE_TOOLS_BETA}", host[0])
        };
        assert_eq!(beta(&native), Some(expected.as_str()));
        let body: Value = native.body_json().unwrap();
        assert_eq!(
            tool_names(&body["tools"]),
            ["read", "__causa_deferred_placeholder__"]
        );
        let last = body["messages"].as_array().unwrap().last().unwrap().clone();
        assert_eq!(last["role"], "system");
        assert_eq!(last["content"][0]["type"], "tool_addition");
    }
}

#[tokio::test]
async fn responses_additional_tools_keep_tools_and_add_an_input_item() {
    let answer = json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]});
    for additional in [false, true] {
        let server = answering(answer.clone()).await;
        OpenAiResponsesGateway::new(KEY)
            .with_base_url(format!("{}/v1", server.uri()))
            .with_additional_tools(additional)
            .invoke(&request(), &control())
            .await
            .unwrap();
        let body: Value = only_request(&server).await.body_json().unwrap();
        let items: Vec<&Value> = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["type"] == "additional_tools")
            .collect();
        if additional {
            assert_eq!(tool_names(&body["tools"]), ["read"]);
            assert_eq!(items.len(), 1);
            assert_eq!(items[0]["role"], "developer");
            assert_eq!(tool_names(&items[0]["tools"]), ["write"]);
        } else {
            assert_eq!(tool_names(&body["tools"]), ["read", "write"]);
            assert!(items.is_empty());
        }
    }
}

#[tokio::test]
async fn anthropic_with_render_options_native_changes_sends_native_shape_and_beta() {
    let answer = json!({"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"});
    let server = answering(answer).await;

    let mut options = AnthropicRenderOptions::default();
    options.native_tool_changes = true;

    AnthropicMessagesGateway::new(KEY)
        .with_base_url(format!("{}/v1", server.uri()))
        .with_render_options(options)
        .invoke(&request(), &control())
        .await
        .unwrap();

    let req = only_request(&server).await;
    assert_eq!(beta(&req), Some(INLINE_TOOLS_BETA));

    let body: Value = req.body_json().unwrap();
    assert_eq!(
        tool_names(&body["tools"]),
        ["read", "__causa_deferred_placeholder__"]
    );
    let last = body["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["role"], "system");
    assert_eq!(last["content"][0]["type"], "tool_addition");
}

#[tokio::test]
async fn responses_with_render_options_additional_tools_sends_input_item() {
    let answer = json!({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]});
    let server = answering(answer).await;

    let mut options = ResponsesRenderOptions::default();
    options.additional_tools = true;

    OpenAiResponsesGateway::new(KEY)
        .with_base_url(format!("{}/v1", server.uri()))
        .with_render_options(options)
        .invoke(&request(), &control())
        .await
        .unwrap();

    let body: Value = only_request(&server).await.body_json().unwrap();
    let items: Vec<&Value> = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "additional_tools")
        .collect();

    assert_eq!(tool_names(&body["tools"]), ["read"]);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["role"], "developer");
    assert_eq!(tool_names(&items[0]["tools"]), ["write"]);
}
