//! Cross-protocol semantic equivalence.
//!
//! One kernel scenario (two conversation turns with a tool round trip)
//! rendered through the Anthropic Messages, OpenAI Chat Completions, and
//! OpenAI Responses renderers. Each body is normalized to a shared
//! semantic timeline; the three timelines must be identical and must
//! match the expected literal — covering role sequence, system position,
//! and tool pairing (`call_id → provider id → tool_use_id` round trip).
//!
//! `ConversationState` (the session aggregate) lives in `causa-runtime`;
//! this test tree stays kernel-only and materializes the merged projection
//! directly through `merged_frame` over sealed turn snapshots — the same
//! projection the session aggregate produces.

use causa_kernel::{
    BlockId, ContextFrame, ConversationId, GenerationOptions, InvocationId, MediaRef, ModelRef,
    ModelResponse, ModelStopReason, RoundId, TextPayload, ToolCallDraft, ToolOutput,
    ToolResultPayload, ToolResultStatus, ToolSurface, TurnContext, TurnId, TurnSnapshot,
    merged_frame,
};
use causa_protocol::translation::anthropic::render_anthropic_messages;
use causa_protocol::translation::openai_chat::render_openai_chat_messages;
use causa_protocol::translation::openai_responses::render_openai_responses_input;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_BLOCK_ID: AtomicUsize = AtomicUsize::new(1);

fn block_id() -> BlockId {
    BlockId::new(uuid::Uuid::from_u128(
        NEXT_BLOCK_ID.fetch_add(1, Ordering::Relaxed) as u128,
    ))
}

fn invocation(turn: &str, round: u32) -> InvocationId {
    InvocationId {
        turn_id: TurnId::new(turn),
        round_id: RoundId(round),
    }
}

/// Build a turn, seal it, and project it into a history-ready snapshot —
/// what a session's `commit` would admit into history.
fn sealed_turn<F>(turn: &str, build: F) -> TurnSnapshot
where
    F: FnOnce(&mut TurnContext),
{
    let mut ctx = TurnContext::new(TurnId::new(turn));
    build(&mut ctx);
    ctx.seal();
    ctx.snapshot()
}

/// The merged frame over a history of sealed snapshots plus an active
/// (possibly sealed) turn — the lossless conversation projection.
fn session_frame(
    conversation_id: &str,
    history: Vec<TurnSnapshot>,
    active: TurnContext,
) -> ContextFrame {
    merged_frame(
        &ConversationId(conversation_id.into()),
        &history,
        &active,
        RoundId(0),
    )
}

/// Build the shared scenario: a system preamble, a user request, a model
/// turn that calls `read`, its tool result, then a second turn where the
/// model finishes.
fn scenario_frame() -> ContextFrame {
    let history = vec![sealed_turn("t1", |active| {
        active
            .append_input(block_id(), TextPayload::new("be terse"), "system")
            .unwrap();
        active
            .append_input(block_id(), TextPayload::new("find the file"), "user")
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
                vec![block_id(), block_id()],
            )
            .unwrap();
        let call_block_id = applied.tool_calls[0].0;
        active
            .append_tool_results(vec![(
                block_id(),
                ToolResultPayload {
                    call_block_id,
                    status: ToolResultStatus::Succeeded,
                    output: ToolOutput::new(json!("file-a")),
                    media: Vec::new(),
                    notes: Vec::new(),
                },
            )])
            .unwrap();
    })];

    let mut active = TurnContext::new(TurnId::new("t2"));
    active
        .append_input(block_id(), TextPayload::new("and now?"), "user")
        .unwrap();
    active
        .append_model_output(
            invocation("t2", 0),
            &ModelResponse {
                text: TextPayload::new("done"),
                tool_calls: vec![],
            },
            ModelStopReason::EndTurn,
            vec![block_id()],
        )
        .unwrap();
    active.seal();
    session_frame("c1", history, active)
}

fn render(frame: &ContextFrame) -> (Value, Value, Value) {
    let model = ModelRef::new("test-model");
    let surface = ToolSurface::empty();
    let generation = GenerationOptions::default();
    let media = causa_protocol::translation::media::MediaSet::new();
    (
        render_anthropic_messages(
            frame,
            &media,
            &surface,
            &generation,
            &model,
            causa_kernel::CacheDirective::None,
        )
        .unwrap(),
        render_openai_chat_messages(
            frame,
            &media,
            &surface,
            &generation,
            &model,
            causa_kernel::CacheDirective::None,
        )
        .unwrap(),
        render_openai_responses_input(
            frame,
            &media,
            &surface,
            &generation,
            &model,
            causa_kernel::CacheDirective::None,
        )
        .unwrap(),
    )
}

#[test]
fn tool_result_notes_and_media_render_with_their_own_result() {
    use causa_protocol::translation::media::{MediaPayload, MediaSet};

    let mut turn = TurnContext::new(TurnId::new("notes"));
    turn.append_input(block_id(), TextPayload::new("search"), "user")
        .unwrap();
    let applied = turn
        .append_model_output(
            invocation("notes", 0),
            &ModelResponse {
                text: TextPayload::new(""),
                tool_calls: vec![ToolCallDraft {
                    tool_name: "search".into(),
                    arguments: json!({"limit": 20}),
                    provider_call_id: Some("external-call-7".into()),
                }],
            },
            ModelStopReason::ToolUse,
            vec![block_id()],
        )
        .unwrap();
    let media_ref = MediaRef::new("image/png", "search-image");
    turn.append_tool_results(vec![(
        block_id(),
        ToolResultPayload {
            call_block_id: applied.tool_calls[0].0,
            status: ToolResultStatus::Succeeded,
            output: ToolOutput::new(json!({"found": 1})),
            media: vec![media_ref.clone()],
            notes: vec![
                TextPayload::new("The limit was reduced to 5."),
                TextPayload::new("The result includes one image."),
            ],
        },
    )])
    .unwrap();
    turn.seal();
    let frame = session_frame("notes", vec![], turn);
    let mut media = MediaSet::new();
    media.insert("search-image", MediaPayload::new("image/png", "aGVsbG8="));
    let model = ModelRef::new("test-model");
    let surface = ToolSurface::empty();
    let generation = GenerationOptions::default();
    let anthropic = render_anthropic_messages(
        &frame,
        &media,
        &surface,
        &generation,
        &model,
        causa_kernel::CacheDirective::None,
    )
    .unwrap();
    let chat = render_openai_chat_messages(
        &frame,
        &media,
        &surface,
        &generation,
        &model,
        causa_kernel::CacheDirective::None,
    )
    .unwrap();
    let responses = render_openai_responses_input(
        &frame,
        &media,
        &surface,
        &generation,
        &model,
        causa_kernel::CacheDirective::None,
    )
    .unwrap();

    let expected_notes =
        "{\"found\":1}\n\nNotes:\n- The limit was reduced to 5.\n- The result includes one image.";
    let anthropic_result = &anthropic["messages"][2]["content"][0];
    assert_eq!(anthropic_result["type"], "tool_result");
    let anthropic_parts = anthropic_result["content"].as_array().unwrap();
    assert_eq!(anthropic_parts[0]["text"], "{\"found\":1}");
    assert_eq!(anthropic_parts[1]["text"], "Notes:");
    assert_eq!(anthropic_parts[2]["text"], "- The limit was reduced to 5.");
    assert_eq!(
        anthropic_parts[3]["text"],
        "- The result includes one image."
    );
    assert_eq!(anthropic_parts[4]["type"], "image");

    let chat_result = &chat["messages"][2];
    assert_eq!(chat_result["role"], "tool");
    assert_eq!(chat_result["tool_call_id"], "external-call-7");
    assert_eq!(chat_result["content"], expected_notes);
    assert_eq!(chat["messages"][3]["content"][1]["type"], "image_url");

    let response_result = &responses["input"][2];
    assert_eq!(response_result["type"], "function_call_output");
    assert_eq!(response_result["call_id"], "external-call-7");
    assert_eq!(response_result["output"], expected_notes);
    assert_eq!(responses["input"][3]["content"][1]["type"], "input_image");

    let stored_result = frame
        .model_context
        .blocks
        .iter()
        .find_map(|block| match block.content() {
            causa_kernel::BlockContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .unwrap();
    assert_eq!(stored_result.output.content, json!({"found": 1}));
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
    assert!(anthropic["system"].as_str().unwrap().contains("be terse"));
    assert_eq!(chat["messages"][0]["role"], json!("system"));
    assert_eq!(chat["messages"][0]["content"], json!("be terse"));
    assert_eq!(responses["instructions"], json!("be terse"));
}

#[test]
fn tool_pairing_round_trips_through_every_protocol() {
    let (anthropic, chat, responses) = render(&scenario_frame());

    // anthropic: tool_use.id == tool_result.tool_use_id
    let messages = anthropic["messages"].as_array().unwrap();
    let use_id = messages[1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .map(|b| b["id"].as_str().unwrap())
        .unwrap();
    let result_id = messages[2]["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_result")
        .map(|b| b["tool_use_id"].as_str().unwrap())
        .unwrap();
    assert_eq!(use_id, result_id);

    // chat: assistant tool_calls[0].id == tool message tool_call_id
    let chat_messages = chat["messages"].as_array().unwrap();
    let call_id = chat_messages
        .iter()
        .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
        .map(|m| m["tool_calls"][0]["id"].as_str().unwrap())
        .unwrap();
    let result_id = chat_messages
        .iter()
        .find(|m| m["role"] == "tool")
        .map(|m| m["tool_call_id"].as_str().unwrap())
        .unwrap();
    assert_eq!(call_id, result_id);

    // responses: [0] user, [1] assistant, [2] function_call, [3] function_call_output
    let responses_call_id = responses["input"][2]["call_id"].as_str().unwrap();
    let responses_result_id = responses["input"][3]["call_id"].as_str().unwrap();
    assert_eq!(responses_call_id, responses_result_id);
}

#[test]
fn content_shapes_survive_all_three_renderers() {
    // A non-string tool observation stringifies identically everywhere:
    // one turn (input + failed tool round trip) held as the active slot.
    let history = vec![];
    let mut active = TurnContext::new(TurnId::new("t1"));
    active
        .append_input(block_id(), TextPayload::new("go"), "user")
        .unwrap();
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
            vec![block_id(), block_id()],
        )
        .unwrap();
    active
        .append_tool_results(vec![(
            block_id(),
            ToolResultPayload {
                call_block_id: applied.tool_calls[0].0,
                status: ToolResultStatus::Failed,
                output: ToolOutput::new(json!({"error": "denied"})),
                media: Vec::new(),
                notes: Vec::new(),
            },
        )])
        .unwrap();
    active.seal();
    let frame = session_frame("c1", history, active);

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

/// Identical call contents in distinct turns keep distinct declaration
/// identities and their respective external provider IDs.
#[test]
fn tool_result_ids_stay_scoped_to_their_own_turn() {
    let mut history = Vec::new();
    for (turn_name, wire_id) in [("t1", "provider_first"), ("t2", "provider_second")] {
        history.push(sealed_turn(turn_name, |turn| {
            turn.append_input(block_id(), TextPayload::new("read again"), "user")
                .unwrap();
            let applied = turn
                .append_model_output(
                    invocation(turn_name, 0),
                    &ModelResponse {
                        text: TextPayload::new(""),
                        tool_calls: vec![ToolCallDraft {
                            tool_name: "read".into(),
                            arguments: json!({"path": "same"}),
                            provider_call_id: Some(wire_id.into()),
                        }],
                    },
                    ModelStopReason::ToolUse,
                    vec![block_id()],
                )
                .unwrap();
            turn.append_tool_results(vec![(
                block_id(),
                ToolResultPayload {
                    call_block_id: applied.tool_calls[0].0,
                    status: ToolResultStatus::Succeeded,
                    output: ToolOutput::new(json!(turn_name)),
                    media: Vec::new(),
                    notes: Vec::new(),
                },
            )])
            .unwrap();
        }));
    }
    let mut active = TurnContext::new(TurnId::new("t3"));
    active
        .append_input(block_id(), TextPayload::new("next"), "user")
        .unwrap();
    let frame = session_frame("repeat", history, active);

    let (anthropic, chat, responses) = render(&frame);

    let anthropic_call_ids: Vec<&str> = anthropic["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().unwrap())
        .filter(|b| b["type"] == "tool_use")
        .map(|b| b["id"].as_str().unwrap())
        .collect();
    let chat_call_ids: Vec<&str> = chat["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["tool_calls"].as_array().into_iter().flatten())
        .map(|call| call["id"].as_str().unwrap())
        .collect();
    let responses_call_ids: Vec<&str> = responses["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| item["call_id"].as_str().unwrap())
        .collect();
    let anthropic_ids: Vec<&str> = anthropic["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().unwrap())
        .filter(|b| b["type"] == "tool_result")
        .map(|b| b["tool_use_id"].as_str().unwrap())
        .collect();
    let chat_ids: Vec<&str> = chat["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "tool")
        .map(|m| m["tool_call_id"].as_str().unwrap())
        .collect();
    let responses_ids: Vec<&str> = responses["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b["type"] == "function_call_output")
        .map(|b| b["call_id"].as_str().unwrap())
        .collect();

    let expected_ids = vec!["provider_first", "provider_second"];
    assert_eq!(anthropic_call_ids, expected_ids);
    assert_eq!(chat_call_ids, expected_ids);
    assert_eq!(responses_call_ids, expected_ids);
    assert_eq!(anthropic_ids, expected_ids);
    assert_eq!(chat_ids, expected_ids);
    assert_eq!(responses_ids, expected_ids);

    let anthropic_results: Vec<String> = anthropic["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().unwrap())
        .filter(|b| b["type"] == "tool_result")
        .map(|b| {
            b["content"]
                .as_str()
                .or_else(|| b["content"][0]["text"].as_str())
                .unwrap()
                .to_owned()
        })
        .collect();
    let chat_results: Vec<&str> = chat["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "tool")
        .map(|m| m["content"].as_str().unwrap())
        .collect();
    let responses_results: Vec<&str> = responses["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call_output")
        .map(|item| item["output"].as_str().unwrap())
        .collect();
    assert_eq!(anthropic_results, vec!["t1", "t2"]);
    assert_eq!(chat_results, vec!["t1", "t2"]);
    assert_eq!(responses_results, vec!["t1", "t2"]);
}
