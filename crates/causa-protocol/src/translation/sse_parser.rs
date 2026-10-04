//! Incremental, transport-free Server-Sent Events framing over UTF-8 bytes.
//!
//! [`SseParser::decode`] consumes a borrowed prefix and returns at most one
//! complete event. Chunk boundaries may split characters or line endings.
//! [`SseParser::finish`] checks EOF without inventing a delimiter or event.

mod parser;

pub use parser::{SseEvent, SseParser};

#[cfg(test)]
mod tests;
