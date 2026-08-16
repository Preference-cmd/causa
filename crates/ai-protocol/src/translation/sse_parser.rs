//! SSE (Server-Sent Events) parser for LLM API streaming.
//!
//! Simplified from `pi_agent_rust/src/sse.rs` (MIT license).
//! Handles OpenAI, Anthropic, and Vertex SSE formats.
//!
//! Push complete UTF-8 text chunks via [`SseParser::feed`]; get back
//! complete SSE events delimited by blank lines. Call [`SseParser::flush`]
//! when the stream ends to emit any final buffered event.

use std::mem;

/// Default per-event data size cap (1 MB). Protects against unbounded
/// memory growth from a malicious or stuck provider. Doubles as the
/// cap for the line buffer (R2-02): a stream with no newline must not
/// grow `buffer` without bound either.
const DEFAULT_MAX_EVENT_DATA_BYTES: usize = 1024 * 1024;

/// Extra buffer headroom beyond the data cap (R2-02): room for the
/// `data: ` prefix, the line terminator, and a multi-line event's
/// inter-line data before the per-event truncation path takes over.
const LINE_HEADROOM: usize = 128;

/// A parsed SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field value, if present. Omitting it defaults to
    /// `"message"` per the SSE spec.
    pub event: Option<String>,
    /// Concatenated `data:` lines, joined by `\n`.
    pub data: String,
}

/// Streaming SSE parser. Feed UTF-8 text chunks; get complete events back.
///
/// # Usage
///
/// ```ignore
/// let mut parser = SseParser::new();
/// let mut events = parser.feed("data: {\"x\":1}\n\n");
/// assert_eq!(events[0].data, "{\"x\":1}");
/// ```
#[derive(Debug)]
pub struct SseParser {
    buffer: String,
    current_event: Option<String>,
    current_data: String,
    has_data: bool,
    bom_checked: bool,
    max_event_data_bytes: usize,
    /// Set when the line buffer grew past the cap (R2-02): the stream
    /// is malformed (e.g. no newline ever arrives). Callers must check
    /// [`SseParser::overflowed`] after each `feed`/`flush` and treat the
    /// stream as failed.
    overflowed: bool,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    /// Create a new parser with default settings.
    pub fn new() -> Self {
        Self {
            buffer: String::with_capacity(4096),
            current_event: None,
            current_data: String::new(),
            has_data: false,
            bom_checked: false,
            max_event_data_bytes: DEFAULT_MAX_EVENT_DATA_BYTES,
            overflowed: false,
        }
    }

    /// Create a parser with a custom per-event data size cap. Also caps
    /// the unterminated-line buffer (R2-02).
    pub fn with_max_event_data_bytes(mut self, limit: usize) -> Self {
        self.max_event_data_bytes = limit;
        self
    }

    /// Whether the input exceeded the buffer cap (R2-02). Once set, the
    /// parser ignores further input; the transport should fail the
    /// stream rather than synthesize events from truncated data.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Feed a chunk of UTF-8 text to the parser.
    ///
    /// Returns any complete SSE events extracted from this chunk.
    /// Events are delimited by blank lines (`\n\n` or `\r\n\r\n`).
    pub fn feed(&mut self, data: &str) -> Vec<SseEvent> {
        if self.overflowed {
            return Vec::new();
        }
        // Strip UTF-8 BOM on first feed.
        let data = if !self.bom_checked {
            self.bom_checked = true;
            data.strip_prefix('\u{FEFF}').unwrap_or(data)
        } else {
            data
        };

        self.buffer.push_str(data);
        // R2-02: a stream that never delivers a newline must not grow
        // the buffer without bound. The threshold leaves headroom for
        // one full `data:` line (prefix + newline) so a legitimate
        // event up to the data cap is handled by the per-event
        // truncation path below instead of failing the stream.
        // Fail-closed: mark the stream overflowed and drop the buffer
        // (the transport terminates).
        if self.buffer.len() > self.max_event_data_bytes + LINE_HEADROOM {
            self.overflowed = true;
            self.buffer.clear();
            return Vec::new();
        }
        let mut events = Vec::new();

        while let Some(newline_pos) = self.buffer.find('\n') {
            // Extract the line content before mutating the buffer.
            let raw = &self.buffer[..newline_pos];
            let line = raw.trim_end_matches('\r').to_owned();
            // Drain including the \n character.
            self.buffer.drain(..=newline_pos);

            if line.is_empty() {
                // Blank line = end of current event.
                if self.has_data {
                    events.push(self.build_event());
                }
                self.reset_current();
            } else if line.starts_with(':') {
                // Comment line — ignore.
            } else if let Some(value) = line.strip_prefix("event:") {
                self.current_event = Some(value.trim().to_string());
            } else if let Some(value) = line.strip_prefix("data:") {
                let value = value.strip_prefix(' ').unwrap_or(value);
                if self.current_data.is_empty() {
                    self.current_data = value.to_string();
                } else {
                    self.current_data.push('\n');
                    self.current_data.push_str(value);
                }
                self.has_data = true;

                // DoS cap check.
                if self.current_data.len() > self.max_event_data_bytes {
                    // Emit what we have so far and reset.
                    events.push(self.build_event());
                    self.reset_current();
                }
            }
            // Ignore unrecognized fields (id:, retry:, etc.)
        }

        events
    }

    /// Flush any pending event. Call when the stream ends.
    pub fn flush(&mut self) -> Option<SseEvent> {
        if self.overflowed {
            return None;
        }
        // A buffer that still holds lines never saw the terminating
        // blank line — e.g. `"data: a\ndata: b"` where `feed` consumed
        // line 1 and left line 2 pending (R2-02). Re-feed the tail as
        // if the stream ended with a blank line so every line is
        // consumed; the last `data` line must not be dropped.
        if !self.buffer.is_empty() {
            let mut remaining = mem::take(&mut self.buffer);
            if !remaining.ends_with('\n') {
                remaining.push('\n');
            }
            remaining.push('\n');
            let events = self.feed(&remaining);
            if let Some(event) = events.into_iter().next() {
                return Some(event);
            }
        }

        if self.has_data {
            let event = self.build_event();
            self.reset_current();
            Some(event)
        } else {
            None
        }
    }

    fn build_event(&mut self) -> SseEvent {
        SseEvent {
            event: self.current_event.take(),
            data: mem::take(&mut self.current_data),
        }
    }

    fn reset_current(&mut self) {
        self.current_event = None;
        self.current_data.clear();
        self.has_data = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_complete_event() {
        let mut p = SseParser::new();
        let evs = p.feed("data: {\"x\":1}\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "{\"x\":1}");
        assert_eq!(evs[0].event, None);
    }

    #[test]
    fn event_with_event_field() {
        let mut p = SseParser::new();
        let evs = p.feed("event: message\ndata: hello\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].event.as_deref(), Some("message"));
        assert_eq!(evs[0].data, "hello");
    }

    #[test]
    fn multi_line_data_concatenation() {
        let mut p = SseParser::new();
        let evs = p.feed("data: line1\ndata: line2\ndata: line3\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "line1\nline2\nline3");
    }

    #[test]
    fn comment_lines_ignored() {
        let mut p = SseParser::new();
        let evs = p.feed(": this is a comment\ndata: real\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "real");
    }

    #[test]
    fn crlf_line_endings() {
        let mut p = SseParser::new();
        let evs = p.feed("data: crlf\r\n\r\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "crlf");
    }

    #[test]
    fn partial_chunk_then_flush() {
        let mut p = SseParser::new();
        let evs1 = p.feed("data: partial");
        assert!(evs1.is_empty());
        let evs2 = p.feed("\n\n");
        assert_eq!(evs2.len(), 1);
        assert_eq!(evs2[0].data, "partial");
    }

    #[test]
    fn flush_on_stream_end() {
        let mut p = SseParser::new();
        p.feed("data: incomplete");
        let ev = p.flush();
        assert!(ev.is_some());
        assert_eq!(ev.unwrap().data, "incomplete");
    }

    #[test]
    fn flush_keeps_trailing_lines_without_blank_line() {
        // R2-02: "data: a\ndata: b" without a terminating blank line
        // must emit both lines as one event, not drop the second.
        let mut p = SseParser::new();
        let evs = p.feed("data: a\ndata: b");
        assert!(evs.is_empty(), "no blank line yet");
        let ev = p.flush();
        assert!(ev.is_some());
        assert_eq!(ev.unwrap().data, "a\nb");
    }

    #[test]
    fn flush_keeps_complete_event_plus_trailing_line() {
        // A complete event followed by an unterminated line: both
        // survive the flush.
        let mut p = SseParser::new();
        let evs = p.feed("data: first\n\ndata: second");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "first");
        let ev = p.flush();
        assert!(ev.is_some());
        assert_eq!(ev.unwrap().data, "second");
    }

    #[test]
    fn unterminated_line_buffer_overflow_fails_closed() {
        // R2-02: a stream that never delivers a newline must not grow
        // the buffer without bound; the parser flags overflow and
        // refuses further input. The threshold is data cap + line
        // headroom, so the payload must exceed both.
        let mut p = SseParser::new().with_max_event_data_bytes(16);
        p.feed(&format!("data: {}", "x".repeat(200)));
        assert!(p.overflowed(), "no-newline stream exceeds the cap");
        assert!(p.flush().is_none(), "overflowed stream yields no events");
        // Further input is ignored.
        assert!(p.feed("data: x\n\n").is_empty());
    }

    #[test]
    fn flush_empty_buffer() {
        let mut p = SseParser::new();
        assert!(p.flush().is_none());
    }

    #[test]
    fn multiple_events_in_one_chunk() {
        let mut p = SseParser::new();
        let evs = p.feed("data: first\n\ndata: second\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "first");
        assert_eq!(evs[1].data, "second");
    }

    #[test]
    fn empty_data_field() {
        let mut p = SseParser::new();
        let evs = p.feed("data:\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "");
    }

    #[test]
    fn bom_stripped_on_first_feed() {
        let mut p = SseParser::new();
        let evs = p.feed("\u{FEFF}data: no-bom\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "no-bom");
    }

    #[test]
    fn dos_cap_truncates_event() {
        let mut p = SseParser::new().with_max_event_data_bytes(10);
        // 20 bytes of data exceeds the 10-byte cap.
        let evs = p.feed("data: 12345678901234567890\n\n");
        // The cap triggers when data exceeds limit, emitting the event
        // mid-stream. The second feed (the empty line) has no more data.
        assert!(!evs.is_empty());
    }

    #[test]
    fn utf8_content() {
        let mut p = SseParser::new();
        let evs = p.feed("data: 你好世界\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "你好世界");
    }

    #[test]
    fn data_with_space_after_colon() {
        let mut p = SseParser::new();
        // Some SSE implementations put a space after "data: "
        let evs = p.feed("data: {\"x\":1}\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "{\"x\":1}");
    }

    #[test]
    fn chunked_delivery() {
        let mut p = SseParser::new();
        // Simulate TCP chunks splitting mid-event
        let evs1 = p.feed("data: {\"cho");
        assert!(evs1.is_empty());
        let evs2 = p.feed("ices\":[]}");
        assert!(evs2.is_empty());
        let evs3 = p.feed("\n\ndata: done\n\n");
        assert_eq!(evs3.len(), 2);
        assert_eq!(evs3[0].data, "{\"choices\":[]}");
        assert_eq!(evs3[1].data, "done");
    }

    #[test]
    fn openai_done_signal() {
        let mut p = SseParser::new();
        // OpenAI sends "data: [DONE]" as the final event
        let evs = p.feed("data: [DONE]\n\n");
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "[DONE]");
    }
}
