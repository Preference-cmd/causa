//! Slice 3 Phase D — cross-protocol semantic equivalence.
//!
//! One kernel scenario (two conversation turns with a tool round trip)
//! rendered through the Anthropic Messages, OpenAI Chat Completions, and
//! OpenAI Responses renderers. Each body is normalized to a shared
//! semantic timeline; the three timelines must be identical and must
//! match the expected literal — covering role sequence, system position,
//! and tool pairing (`call_id → provider id → tool_use_id` round trip).

use reimagine_ai_protocol::translation::anthropic::render_anthropic_messages;
use reimagine_ai_protocol::translation::openai_chat::render_openai_chat_messages;
use reimagine_ai_protocol::translation::openai_responses::render_openai_responses_input;
use reimagine_context_kernel::{
    ContextFrame, ConversationId, ConversationState, GenerationOptions, InvocationId, ModelRef,
    ModelResponse, ModelStopReason, RoundId, SealedResult, TextPayload, ToolCallDraft, ToolOutput,
    ToolResultPayload, ToolResultStatus, ToolSurface, TurnContext, TurnId,
};
use serde_json::{Value, json};

fn invocation(turn: &str, round: u32) -> InvocationId {
    InvocationId {
        turn_id: TurnId::new(turn),
        round_id: RoundId(round),
    }
}

/// Build the shared scenario: a system preamble, a user request, a model
/// turn that calls `read`, its tool result, then a second turn where the
/// model finishes.
fn scenario_frame() -> ContextFrame {
    let mut state = ConversationState::new(ConversationId("c1".into()));

    let active: &mut TurnContext = state.begin_turn(TurnId::new("t1")).unwrap();
    active
        .append_input(TextPayload::new("be terse"), "system")
        .unwrap();
    active
        .append_input(TextPayload::new("find the file"), "user")
        .unwrap();
    let applied = active
        .append_model_output(
            invocation("t1", 0),
            &ModelResponse {
                text: TextPayload::new("reading"),
                tool_calls: vec![ToolCallDraft {
                    tool_name: "read".into(),
                    arguments: json!({"path": "a"}),
                    provider_call_id: Some("toolu_1".into()),
                }],
            },
            ModelStopReason::ToolUse,
        )
        .unwrap();
    let call_id = applied.tool_calls[0].call_id.clone();
    active
        .append_tool_results(vec![ToolResultPayload {
            call_id,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!("file-a")),
        }])
        .unwrap();
    state
        .seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    state.commit(TurnId::new("t1")).unwrap();

    let active = state.begin_turn(TurnId::new("t2")).unwrap();
    active
        .append_input(TextPayload::new("and now?"), "user")
        .unwrap();
    active
        .append_model_output(
            invocation("t2", 0),
            &ModelResponse {
                text: TextPayload::new("done"),
                tool_calls: vec![],
            },
            ModelStopReason::EndTurn,
        )
        .unwrap();
    state
        .seal_turn(TurnId::new("t2"), SealedResult::Completed)
        .unwrap();
    // t2 stays active (sealed): the conversation frame is the merged view
    // over committed history + the active turn.
    state.frame(RoundId(0)).unwrap()
}

fn render(frame: &ContextFrame) -> (Value, Value, Value) {
    let model = ModelRef::new("test-model");
    let surface = ToolSurface::empty();
    let generation = GenerationOptions::default();
    (
        render_anthropic_messages(frame, &surface, &generation, &model).unwrap(),
        render_openai_chat_messages(frame, &surface, &generation, &model).unwrap(),
        render_openai_responses_input(frame, &surface, &generation, &model).unwrap(),
    )
}

// --- shared semantic timeline ------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Step {
    System(String),
    UserText(String),
    AssistantText(String),
    Assistant {
        text: String,
        calls: Vec<(String, String, Value)>,
    },
    AssistantCall {
        wire_id: String,
        name: String,
        arguments: Value,
    },
    ToolResult {
        wire_id: String,
        content: String,
    },
}

/// Merge adjacent user texts / assistant texts+calls into whole turns —
/// the protocols differ on where message boundaries fall, so equivalence
/// is asserted at this granularity.
fn merge_steps(steps: Vec<Step>) -> Vec<Step> {
    let mut merged: Vec<Step> = Vec::new();
    for step in steps {
        match (&step, merged.last_mut()) {
            (Step::UserText(t), Some(Step::UserText(last))) => {
                last.push('\n');
                last.push_str(t);
            }
            (Step::AssistantText(t), Some(Step::Assistant { text, .. })) => {
                text.push('\n');
                text.push_str(t);
            }
            (
                Step::AssistantCall {
                    wire_id,
                    name,
                    arguments,
                },
                Some(Step::Assistant { calls, .. }),
            ) => {
                calls.push((wire_id.clone(), name.clone(), arguments.clone()));
            }
            (Step::AssistantText(t), _) => merged.push(Step::Assistant {
                text: t.clone(),
                calls: vec![],
            }),
            (Step::AssistantCall { .. }, _) => merged.push(Step::Assistant {
                text: String::new(),
                calls: vec![],
            }),
            _ => merged.push(step),
        }
    }
    merged
}

fn expected_timeline() -> Vec<Step> {
    vec![
        Step::System("be terse".into()),
        Step::UserText("find the file".into()),
        Step::Assistant {
            text: "reading".into(),
            calls: vec![("toolu_1".into(), "read".into(), json!({"path": "a"}))],
        },
        Step::ToolResult {
            wire_id: "toolu_1".into(),
            content: "file-a".into(),
        },
        Step::UserText("and now?".into()),
        Step::Assistant {
            text: "done".into(),
            calls: vec![],
        },
    ]
}

fn anthropic_steps(body: &Value) -> Vec<Step> {
    let mut steps = Vec::new();
    if let Some(system) = body.get("system").and_then(Value::as_str) {
        steps.push(Step::System(system.to_string()));
    }
    for message in body["messages"].as_array().unwrap() {
        let role = message["role"].as_str().unwrap();
        for block in message["content"].as_array().unwrap() {
            match (role, block["type"].as_str().unwrap()) {
                (_, "text") => {
                    let text = block["text"].as_str().unwrap().to_string();
                    if role == "assistant" {
                        steps.push(Step::AssistantText(text));
                    } else {
                        steps.push(Step::UserText(text));
                    }
                }
                ("assistant", "tool_use") => steps.push(Step::AssistantCall {
                    wire_id: block["id"].as_str().unwrap().into(),
                    name: block["name"].as_str().unwrap().into(),
                    arguments: block["input"].clone(),
                }),
                (_, "tool_result") => steps.push(Step::ToolResult {
                    wire_id: block["tool_use_id"].as_str().unwrap().into(),
                    content: block["content"].as_str().unwrap().into(),
                }),
                other => panic!("unexpected anthropic block: {other:?}"),
            }
        }
    }
    merge_steps(steps)
}

fn chat_steps(body: &Value) -> Vec<Step> {
    let mut steps = Vec::new();
    for message in body["messages"].as_array().unwrap() {
        match message["role"].as_str().unwrap() {
            "system" => steps.push(Step::System(message["content"].as_str().unwrap().into())),
            "user" => steps.push(Step::UserText(message["content"].as_str().unwrap().into())),
            "assistant" => {
                if let Some(text) = message.get("content").and_then(Value::as_str) {
                    steps.push(Step::AssistantText(text.into()));
                }
                for call in message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .map_or(&[] as &[Value], |v| v)
                {
                    let arguments: Value =
                        serde_json::from_str(call["function"]["arguments"].as_str().unwrap())
                            .unwrap();
                    steps.push(Step::AssistantCall {
                        wire_id: call["id"].as_str().unwrap().into(),
                        name: call["function"]["name"].as_str().unwrap().into(),
                        arguments,
                    });
                }
            }
            "tool" => steps.push(Step::ToolResult {
                wire_id: message["tool_call_id"].as_str().unwrap().into(),
                content: message["content"].as_str().unwrap().into(),
            }),
            other => panic!("unexpected chat role: {other:?}"),
        }
    }
    merge_steps(steps)
}

fn responses_steps(body: &Value) -> Vec<Step> {
    let mut steps = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        steps.push(Step::System(instructions.to_string()));
    }
    for item in body["input"].as_array().unwrap() {
        match item["type"].as_str().unwrap_or("message") {
            "message" => {
                let role = item["role"].as_str().unwrap();
                for part in item["content"].as_array().unwrap() {
                    let text = part["text"].as_str().unwrap().to_string();
                    if role == "assistant" {
                        steps.push(Step::AssistantText(text));
                    } else {
                        steps.push(Step::UserText(text));
                    }
                }
            }
            "function_call" => {
                let arguments: Value =
                    serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap();
                steps.push(Step::AssistantCall {
                    wire_id: item["call_id"].as_str().unwrap().into(),
                    name: item["name"].as_str().unwrap().into(),
                    arguments,
                });
            }
            "function_call_output" => steps.push(Step::ToolResult {
                wire_id: item["call_id"].as_str().unwrap().into(),
                content: item["output"].as_str().unwrap().into(),
            }),
            other => panic!("unexpected responses item: {other:?}"),
        }
    }
    merge_steps(steps)
}

#[test]
fn all_three_protocols_produce_the_same_semantic_timeline() {
    let (anthropic, chat, responses) = render(&scenario_frame());
    let expected = expected_timeline();

    assert_eq!(anthropic_steps(&anthropic), expected, "anthropic");
    assert_eq!(chat_steps(&chat), expected, "openai chat");
    assert_eq!(responses_steps(&responses), expected, "openai responses");
}

#[test]
fn system_instruction_leads_every_protocol_body() {
    let (anthropic, chat, responses) = render(&scenario_frame());
    // Anthropic: top-level parameter; chat: first message; responses:
    // top-level parameter ahead of every input item.
    assert_eq!(anthropic["system"], json!("be terse"));
    assert_eq!(chat["messages"][0]["role"], json!("system"));
    assert_eq!(chat["messages"][0]["content"], json!("be terse"));
    assert_eq!(responses["instructions"], json!("be terse"));
    assert_eq!(
        responses["input"][0]["role"],
        json!("user"),
        "first input item is content, not a leaked system item"
    );
}

#[test]
fn tool_result_ids_pair_with_their_calls_in_every_protocol() {
    let (anthropic, chat, responses) = render(&scenario_frame());

    // anthropic: [0] user, [1] assistant(text + tool_use), [2] user(tool_result + text)
    let anthropic_call_id = anthropic["messages"][1]["content"][1]["id"]
        .as_str()
        .unwrap();
    let anthropic_result_id = anthropic["messages"][2]["content"][0]["tool_use_id"]
        .as_str()
        .unwrap();
    assert_eq!(anthropic_call_id, anthropic_result_id);

    // chat: [0] system, [1] user, [2] assistant(tool_calls), [3] tool
    let chat_call_id = chat["messages"][2]["tool_calls"][0]["id"].as_str().unwrap();
    let chat_result_id = chat["messages"][3]["tool_call_id"].as_str().unwrap();
    assert_eq!(chat_call_id, chat_result_id);

    // responses: [0] user, [1] assistant, [2] function_call, [3] function_call_output
    let responses_call_id = responses["input"][2]["call_id"].as_str().unwrap();
    let responses_result_id = responses["input"][3]["call_id"].as_str().unwrap();
    assert_eq!(responses_call_id, responses_result_id);
}

#[test]
fn content_shapes_survive_all_three_renderers() {
    // A non-string tool observation stringifies identically everywhere.
    let mut state = ConversationState::new(ConversationId("c1".into()));
    let active = state.begin_turn(TurnId::new("t1")).unwrap();
    active.append_input(TextPayload::new("go"), "user").unwrap();
    let applied = active
        .append_model_output(
            invocation("t1", 0),
            &ModelResponse {
                text: TextPayload::new("listing"),
                tool_calls: vec![ToolCallDraft {
                    tool_name: "list".into(),
                    arguments: json!({"dir": "/"}),
                    provider_call_id: None,
                }],
            },
            ModelStopReason::ToolUse,
        )
        .unwrap();
    active
        .append_tool_results(vec![ToolResultPayload {
            call_id: applied.tool_calls[0].call_id.clone(),
            status: ToolResultStatus::Failed,
            output: ToolOutput::new(json!({"error": "denied"})),
        }])
        .unwrap();
    state
        .seal_turn(TurnId::new("t1"), SealedResult::Completed)
        .unwrap();
    // active (sealed) turn still exposes the merged conversation frame
    let frame = state.frame(RoundId(0)).unwrap();

    let (anthropic, chat, responses) = render(&frame);
    let expected_payload = json!({"error": "denied"}).to_string();
    // anthropic: [0] user, [1] assistant, [2] user(tool_result)
    assert_eq!(
        anthropic["messages"][2]["content"][0]["content"],
        json!(expected_payload)
    );
    assert_eq!(
        anthropic["messages"][2]["content"][0]["is_error"],
        json!(true)
    );
    // chat: [0] user, [1] assistant, [2] tool
    assert_eq!(chat["messages"][2]["content"], json!(expected_payload));
    // responses: [0] user, [1] assistant, [2] function_call, [3] function_call_output
    assert_eq!(responses["input"][3]["output"], json!(expected_payload));
}
