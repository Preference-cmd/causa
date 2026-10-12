use causa_kernel::{ModelInvokeError, ModelInvokeErrorKind};

const DEFAULT_LIMIT: usize = 32 * 1024 * 1024;

/// A complete SSE event, delimited by an empty line.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SseEvent {
    /// Explicit event name; absent or empty means the SSE default `message`.
    pub event: Option<String>,
    /// Data fields joined with one newline, including empty data fields.
    pub data: String,
}
impl SseEvent {
    /// An event with an optional name and its data.
    pub fn new(event: Option<String>, data: impl Into<String>) -> Self {
        Self {
            event,
            data: data.into(),
        }
    }
}

/// A bounded byte decoder with no event queue or input history.
///
/// UTF-8 is validated as lines become complete, so a chunk may end inside a
/// character. LF, CRLF and CR terminate lines; a leading UTF-8 BOM is ignored.
/// Comments and unknown fields are ignored but count against the event limit.
/// Errors latch: after an error, discard the parser. There is no lossy decode,
/// oversized-event truncation, or implicit EOF dispatch.
///
/// ```
/// use causa_protocol::SseParser;
/// let mut parser = SseParser::new();
/// let mut bytes = b"data: hello\n\ndata: world\n\n".as_slice();
/// assert_eq!(parser.decode(&mut bytes)?.unwrap().data, "hello");
/// assert_eq!(parser.decode(&mut bytes)?.unwrap().data, "world");
/// assert!(parser.decode(&mut bytes)?.is_none());
/// parser.finish()?;
/// # Ok::<(), causa_kernel::ModelInvokeError>(())
/// ```
#[derive(Debug)]
pub struct SseParser {
    line: Vec<u8>,
    event_name: Option<String>,
    data: String,
    has_data: bool,
    first_line: bool,
    after_cr: bool,
    event_bytes: usize,
    max_line_bytes: usize,
    max_event_bytes: usize,
    failed: bool,
    finished: bool,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    /// Create a decoder with 32 MiB line and event limits.
    pub fn new() -> Self {
        Self {
            line: Vec::new(),
            event_name: None,
            data: String::new(),
            has_data: false,
            first_line: true,
            after_cr: false,
            event_bytes: 0,
            max_line_bytes: DEFAULT_LIMIT,
            max_event_bytes: DEFAULT_LIMIT,
            failed: false,
            finished: false,
        }
    }

    /// Set raw line and event byte limits before decoding.
    ///
    /// A line excludes its terminator. An event includes its lines, ignored
    /// fields and delimiters. A CRLF counts as CR plus a skipped LF; the LF
    /// does not belong to the next event. Zero limits reject nonempty input.
    pub fn with_limits(mut self, max_line_bytes: usize, max_event_bytes: usize) -> Self {
        self.max_line_bytes = max_line_bytes;
        self.max_event_bytes = max_event_bytes;
        self
    }

    /// Consume bytes through at most one complete event.
    ///
    /// The caller retains the remaining slice and should call again, including
    /// when multiple events arrived in one HTTP chunk. `None` means all supplied
    /// bytes were consumed without completing a data event. Completed comments
    /// and unknown fields do not emit events. Calls after successful `finish`
    /// reject nonempty input.
    pub fn decode(&mut self, input: &mut &[u8]) -> Result<Option<SseEvent>, ModelInvokeError> {
        if self.failed {
            return Err(misused("SSE parser is in a failed state"));
        }
        if self.finished {
            return if input.is_empty() {
                Ok(None)
            } else {
                self.fail(misused("SSE input received after EOF"))
            };
        }
        match self.decode_inner(input) {
            Ok(event) => Ok(event),
            Err(error) => self.fail(error),
        }
    }

    /// Validate EOF without emitting an unfinished event.
    ///
    /// Invalid or truncated UTF-8 is `Undecodable`; a valid unterminated line or
    /// event is `UnknownOutcome`. A clean SSE boundary succeeds, but does not
    /// prove that the enclosing model protocol supplied its terminal event.
    pub fn finish(&mut self) -> Result<(), ModelInvokeError> {
        if self.failed {
            return Err(misused("SSE parser is in a failed state"));
        }
        if std::str::from_utf8(&self.line).is_err() {
            return self.fail(undecodable("SSE input ended with invalid UTF-8"));
        }
        if !self.line.is_empty() || self.has_data || self.event_name.is_some() {
            return self.fail(ModelInvokeError::new(
                ModelInvokeErrorKind::UnknownOutcome,
                "SSE input ended before an event delimiter",
            ));
        }
        self.finished = true;
        Ok(())
    }

    fn decode_inner(&mut self, input: &mut &[u8]) -> Result<Option<SseEvent>, ModelInvokeError> {
        while !input.is_empty() {
            if self.after_cr {
                self.after_cr = false;
                if input[0] == b'\n' {
                    *input = &input[1..];
                    // The CR has already ended the line (possibly the event).
                    continue;
                }
            }
            let length = input
                .iter()
                .position(|byte| matches!(byte, b'\r' | b'\n'))
                .unwrap_or(input.len());
            if self.line.len().saturating_add(length) > self.max_line_bytes {
                return Err(undecodable("SSE line exceeds its byte limit"));
            }
            self.count_bytes(length)?;
            self.line.extend_from_slice(&input[..length]);
            *input = &input[length..];
            if input.is_empty() {
                return Ok(None);
            }
            self.count_bytes(1)?;
            self.after_cr = input[0] == b'\r';
            *input = &input[1..];
            if let Some(event) = self.complete_line()? {
                return Ok(Some(event));
            }
        }
        Ok(None)
    }

    fn count_bytes(&mut self, count: usize) -> Result<(), ModelInvokeError> {
        self.event_bytes = self
            .event_bytes
            .checked_add(count)
            .filter(|bytes| *bytes <= self.max_event_bytes)
            .ok_or_else(|| undecodable("SSE event exceeds its byte limit"))?;
        Ok(())
    }

    fn complete_line(&mut self) -> Result<Option<SseEvent>, ModelInvokeError> {
        let bytes = std::mem::take(&mut self.line);
        let raw = std::str::from_utf8(&bytes)
            .map_err(|_| undecodable("SSE line contains invalid UTF-8"))?;
        let line = if self.first_line {
            self.first_line = false;
            raw.strip_prefix('\u{feff}').unwrap_or(raw)
        } else {
            raw
        };
        if line.is_empty() {
            let event = self.has_data.then(|| SseEvent {
                event: self.event_name.take().filter(|name| !name.is_empty()),
                data: std::mem::take(&mut self.data),
            });
            self.event_name = None;
            self.data.clear();
            self.has_data = false;
            self.event_bytes = 0;
            return Ok(event);
        }
        if line.starts_with(':') {
            return Ok(None);
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            "event" => self.event_name = Some(value.to_owned()),
            _ => {}
        }
        Ok(None)
    }

    fn fail<T>(&mut self, error: ModelInvokeError) -> Result<T, ModelInvokeError> {
        self.failed = true;
        self.line = Vec::new();
        self.event_name = None;
        self.data = String::new();
        Err(error)
    }
}

fn undecodable(message: &str) -> ModelInvokeError {
    ModelInvokeError::new(ModelInvokeErrorKind::Undecodable, message)
}

/// The caller broke the parser's contract; nothing the provider sent is at fault.
fn misused(message: &str) -> ModelInvokeError {
    ModelInvokeError::new(ModelInvokeErrorKind::Permanent, message)
}
