use reimagine_agent_harness::AgentStreamEvent;
use reimagine_ai_protocol::translation::streaming::AnthropicStreamAccumulator;
use serde_json::{Value, json};

/// Feed an SSE event (its `event:` field name plus parsed JSON payload)
/// through the accumulator, the same way the transport layer routes
/// parsed SSE events.
fn feed(
    acc: &mut AnthropicStreamAccumulator,
    event_type: &str,
    data: Value,
) -> Vec<AgentStreamEvent> {
    acc.ingest_event(Some(event_type), &data)
}

#[tokio::test]
async fn anthropic_accumulator_emits_text_deltas_complete_tool_call_and_done() {
    let mut acc = AnthropicStreamAccumulator::new();
    let mut collected = Vec::new();
    for e in [
        feed(
            &mut acc,
            "message_start",
            json!({ "type": "message_start" }),
        ),
        feed(
            &mut acc,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "Hello" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 0 }),
        ),
        feed(
            &mut acc,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "echo",
                    "input": {}
                }
            }),
        ),
        feed(
            &mut acc,
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": { "type": "input_json_delta", "partial_json": "{\"x\":" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": { "type": "input_json_delta", "partial_json": "1}" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 1 }),
        ),
        feed(
            &mut acc,
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "tool_use" },
                "usage": { "input_tokens": 1, "output_tokens": 2 }
            }),
        ),
        feed(&mut acc, "message_stop", json!({ "type": "message_stop" })),
    ] {
        collected.extend(e);
    }

    let mut kinds: Vec<&'static str> = Vec::new();
    for e in &collected {
        match e {
            AgentStreamEvent::ContentDelta(_) => kinds.push("content"),
            AgentStreamEvent::ToolCallDelta { .. } => kinds.push("delta"),
            AgentStreamEvent::ToolCall(_) => kinds.push("complete"),
            AgentStreamEvent::Usage(_) => kinds.push("usage"),
            AgentStreamEvent::ReasoningDelta(_) => kinds.push("reasoning"),
            AgentStreamEvent::Done { .. } => kinds.push("done"),
            AgentStreamEvent::Compacted { .. } => kinds.push("compacted"),
            AgentStreamEvent::Error(_) => kinds.push("error"),
            AgentStreamEvent::Warning(_) => kinds.push("warning"),
        }
    }
    assert!(kinds.contains(&"content"));
    assert!(kinds.contains(&"usage"));
    assert!(kinds.contains(&"complete"));
    assert!(
        !kinds.contains(&"delta"),
        "live decoder emits no ToolCallDelta"
    );
    assert_eq!(*kinds.last().unwrap(), "done");

    // The terminal Done carries the stop_reason captured from
    // `message_delta` (AC-01).
    let done = collected.last().unwrap();
    assert_eq!(
        done,
        &AgentStreamEvent::Done {
            stop_reason: Some("tool_use".into())
        }
    );

    // Find the complete tool call and check it.
    let complete = collected
        .iter()
        .find_map(|e| match e {
            AgentStreamEvent::ToolCall(c) => Some(c),
            _ => None,
        })
        .expect("complete tool call emitted");
    assert_eq!(complete.id().as_str(), "c1");
    assert_eq!(complete.name(), "echo");
    assert_eq!(complete.arguments(), &json!({"x": 1}));
    assert!(acc.has_content());
}

#[tokio::test]
async fn anthropic_event_missing_type_is_ignored_silently() {
    // The live decoder ignores events without an SSE `event:` field
    // rather than failing the stream (a provider blip must not abort a
    // turn); the dead accumulator's Serialization error is not revived.
    let mut acc = AnthropicStreamAccumulator::new();
    assert!(acc.ingest_event(None, &json!({})).is_empty());
    assert!(
        acc.ingest_event(Some("unknown_event"), &json!({ "type": "unknown_event" }))
            .is_empty()
    );
    assert!(!acc.has_content());
}

#[tokio::test]
async fn anthropic_thinking_delta_emits_reasoning_delta_before_content() {
    let mut acc = AnthropicStreamAccumulator::new();
    let mut collected = Vec::new();
    for e in [
        feed(
            &mut acc,
            "message_start",
            json!({ "type": "message_start" }),
        ),
        feed(
            &mut acc,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "thinking", "thinking": "" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "thinking_delta", "thinking": "First I check the" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "thinking_delta", "thinking": " inputs." }
            }),
        ),
        feed(
            &mut acc,
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 0 }),
        ),
        feed(
            &mut acc,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": { "type": "text", "text": "" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": { "type": "text_delta", "text": "Answer" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 1 }),
        ),
        feed(&mut acc, "message_stop", json!({ "type": "message_stop" })),
    ] {
        collected.extend(e);
    }

    let reasoning: String = collected
        .iter()
        .filter_map(|e| match e {
            AgentStreamEvent::ReasoningDelta(t) => Some(t.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "First I check the inputs.");

    // Reasoning never feeds the content stream: content deltas carry
    // only the assistant text.
    let content: String = collected
        .iter()
        .filter_map(|e| match e {
            AgentStreamEvent::ContentDelta(t) => Some(t.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(content, "Answer");
}

#[tokio::test]
async fn anthropic_stream_merges_message_start_cache_usage_with_delta_output() {
    let mut acc = AnthropicStreamAccumulator::new();
    let mut collected = Vec::new();
    for e in [
        feed(
            &mut acc,
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 100,
                        "cache_read_input_tokens": 200
                    }
                }
            }),
        ),
        feed(
            &mut acc,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "Answer" }
            }),
        ),
        feed(
            &mut acc,
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 0 }),
        ),
        feed(
            &mut acc,
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 5 }
            }),
        ),
        feed(&mut acc, "message_stop", json!({ "type": "message_stop" })),
    ] {
        collected.extend(e);
    }
    let usage = collected
        .iter()
        .find_map(|e| match e {
            AgentStreamEvent::Usage(u) => Some(u.clone()),
            _ => None,
        })
        .expect("usage emitted");
    assert_eq!(usage.input_tokens(), Some(10));
    assert_eq!(usage.output_tokens(), Some(5));
    assert_eq!(usage.cache_creation_input_tokens(), Some(100));
    assert_eq!(usage.cache_read_input_tokens(), Some(200));
    assert_eq!(usage.total(), Some(315));
}
