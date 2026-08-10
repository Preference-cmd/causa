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
/// memory growth from a malicious or stuck provider.
const DEFAULT_MAX_EVENT_DATA_BYTES: usize = 1024 * 1024;

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
        }
    }

    /// Create a parser with a custom per-event data size cap.
    pub fn with_max_event_data_bytes(mut self, limit: usize) -> Self {
        self.max_event_data_bytes = limit;
        self
    }

    /// Feed a chunk of UTF-8 text to the parser.
    ///
    /// Returns any complete SSE events extracted from this chunk.
    /// Events are delimited by blank lines (`\n\n` or `\r\n\r\n`).
    pub fn feed(&mut self, data: &str) -> Vec<SseEvent> {
        // Strip UTF-8 BOM on first feed.
        let data = if !self.bom_checked {
            self.bom_checked = true;
            data.strip_prefix('\u{FEFF}').unwrap_or(data)
        } else {
            data
        };

        self.buffer.push_str(data);
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
        // If there's unterminated data in the buffer (no trailing \n\n),
        // extract it and emit as the final event.
        if !self.buffer.is_empty() && !self.has_data {
            let trimmed = self.buffer.trim_end_matches('\r');
            if let Some(value) = trimmed.strip_prefix("data:") {
                let value = value.strip_prefix(' ').unwrap_or(value);
                self.current_data = value.to_string();
                self.has_data = true;
            }
        }

        if self.has_data {
            let event = self.build_event();
            self.reset_current();
            self.buffer.clear();
            Some(event)
        } else {
            self.buffer.clear();
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
