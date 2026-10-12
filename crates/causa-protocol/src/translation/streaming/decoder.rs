use std::collections::VecDeque;

use causa_kernel::{ModelInvokeError, ModelInvokeErrorKind, StreamDelta};
use serde_json::Value;

use super::{
    anthropic::Anthropic,
    chat::Chat,
    common::{located, misused, undecodable},
    responses::Responses,
};
use crate::{Protocol, translation::sse_parser::SseParser};

/// Limits for retained protocol state and all consumed wire bytes.
/// Defaults allow 32 MiB lines/events, 64 MiB total wire bytes, 16 MiB
/// cumulative output plus terminal snapshot, 1024 items and 4096 queued deltas.
/// Every limit must be positive; a zero limit is rejected when the decoder is
/// built. A stream that exceeds a limit fails as `Undecodable`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct StreamLimits {
    /// Maximum encoded bytes in one SSE line.
    pub max_line_bytes: usize,
    /// Maximum encoded bytes in one SSE event.
    pub max_event_bytes: usize,
    /// Maximum consumed wire bytes, including comments and unknown events.
    pub max_total_bytes: usize,
    /// Maximum cumulative output/metadata bytes plus authoritative snapshot.
    pub max_output_bytes: usize,
    /// Maximum output items or tool declarations; sparse indexes allocate no gaps.
    pub max_items: usize,
    /// Maximum deltas awaiting delivery from one decoded event.
    pub max_pending_deltas: usize,
}

impl Default for StreamLimits {
    fn default() -> Self {
        Self {
            max_line_bytes: 32 * 1024 * 1024,
            max_event_bytes: 32 * 1024 * 1024,
            max_total_bytes: 64 * 1024 * 1024,
            max_output_bytes: 16 * 1024 * 1024,
            max_items: 1024,
            max_pending_deltas: 4096,
        }
    }
}

enum Assembly {
    Chat(Chat),
    Responses(Responses),
    Anthropic(Anthropic),
}

/// Incremental HTTP SSE decoder. Feed bytes until `decode` returns `None`,
/// then provide the next chunk. Drain pending deltas even with empty input.
/// Errors fuse the decoder; successful terminals ignore unread wire tails.
pub struct StreamDecoder {
    parser: SseParser,
    assembly: Assembly,
    pending: VecDeque<StreamDelta>,
    limits: StreamLimits,
    total: usize,
    events: usize,
    terminal: bool,
    failed: bool,
}

impl StreamDecoder {
    /// Creates a decoder for a supported protocol with explicit resource limits.
    pub fn new(protocol: Protocol, limits: StreamLimits) -> Result<Self, ModelInvokeError> {
        if [
            limits.max_line_bytes,
            limits.max_event_bytes,
            limits.max_total_bytes,
            limits.max_output_bytes,
            limits.max_items,
            limits.max_pending_deltas,
        ]
        .contains(&0)
        {
            return Err(misused("stream limits must be positive"));
        }
        let assembly = match protocol {
            Protocol::OpenAiChatCompletions => Assembly::Chat(Chat::new(limits)),
            Protocol::OpenAiResponses => Assembly::Responses(Responses::new(limits)),
            Protocol::AnthropicMessages => Assembly::Anthropic(Anthropic::new(limits)),
        };
        Ok(Self {
            parser: SseParser::new().with_limits(limits.max_line_bytes, limits.max_event_bytes),
            assembly,
            pending: VecDeque::new(),
            limits,
            total: 0,
            events: 0,
            terminal: false,
            failed: false,
        })
    }

    /// Consumes a prefix until one delta is ready or input is exhausted.
    /// Call again with the remaining slice; a terminal leaves its tail unread.
    pub fn decode(&mut self, input: &mut &[u8]) -> Result<Option<StreamDelta>, ModelInvokeError> {
        if self.failed {
            return Ok(None);
        }
        let result = self.decode_inner(input);
        if result.is_err() {
            self.failed = true;
            self.pending.clear();
        }
        result
    }

    fn decode_inner(&mut self, input: &mut &[u8]) -> Result<Option<StreamDelta>, ModelInvokeError> {
        if let Some(delta) = self.pending.pop_front() {
            return Ok(Some(delta));
        }
        if self.terminal {
            return Ok(None);
        }
        loop {
            let before = input.len();
            // Limit the parser's view so a huge merged network chunk cannot
            // exceed the total limit before the next event boundary.
            let remaining = self.limits.max_total_bytes - self.total;
            if remaining == 0 && !input.is_empty() {
                return Err(undecodable("stream exceeds max_total_bytes"));
            }
            let mut bounded = &input[..input.len().min(remaining)];
            let bounded_len = bounded.len();
            let parsed = self.parser.decode(&mut bounded);
            let consumed = bounded_len - bounded.len();
            *input = &input[consumed..];
            self.total += consumed;
            let Some(event) = parsed? else {
                if self.total == self.limits.max_total_bytes && !input.is_empty() {
                    return Err(undecodable("stream exceeds max_total_bytes"));
                }
                return Ok(None);
            };
            self.events += 1;
            let deltas = match &mut self.assembly {
                Assembly::Chat(state) => state.event(&event),
                Assembly::Responses(state) => state.event(&event),
                Assembly::Anthropic(state) => state.event(&event),
            }
            .map_err(|error| located(error, self.events, &event))?;
            if deltas.len() > self.limits.max_pending_deltas {
                return Err(undecodable("stream exceeds max_pending_deltas"));
            }
            self.terminal = deltas.iter().any(|d| matches!(d, StreamDelta::Done { .. }));
            self.pending.extend(deltas);
            if let Some(delta) = self.pending.pop_front() {
                return Ok(Some(delta));
            }
            if input.is_empty() || input.len() == before {
                return Ok(None);
            }
        }
    }

    /// Ends input explicitly. Malformed framing/encoding is undecodable;
    /// a valid stream prefix without a provider terminal is unknown outcome.
    pub fn finish(&mut self) -> Result<(), ModelInvokeError> {
        if self.failed {
            return Err(misused("stream decoder is fused after an error"));
        }
        if self.terminal {
            return Ok(());
        }
        self.failed = true;
        self.parser.finish()?;
        Err(ModelInvokeError::new(
            ModelInvokeErrorKind::UnknownOutcome,
            "stream ended without a provider terminal",
        ))
    }

    /// Whether an authoritative provider terminal has been accepted.
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }
}

/// Adds protocol-native streaming options to an already rendered request.
/// Chat Completions additionally requests the trailing usage snapshot. Both
/// are the adapter's values on a stream call: a `stream` already in the body
/// is replaced, and so is a `stream_options` that is not an object.
pub fn configure_stream_request(
    protocol: Protocol,
    body: &mut Value,
) -> Result<(), ModelInvokeError> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| misused("stream request must be an object"))?;
    object.insert("stream".into(), Value::Bool(true));
    if protocol == Protocol::OpenAiChatCompletions {
        let options = object
            .entry("stream_options")
            .or_insert_with(|| serde_json::json!({}));
        if !options.is_object() {
            *options = serde_json::json!({});
        }
        options
            .as_object_mut()
            .expect("stream_options is an object")
            .insert("include_usage".into(), Value::Bool(true));
    }
    Ok(())
}
