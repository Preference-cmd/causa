//! Bounded, transport-free assembly of the three supported HTTP SSE protocols.
//! Deltas are observations; only `Done` contains authoritative output. EOF never
//! manufactures a protocol terminal. Tool identities are emitted when declared
//! (or supplied on a Chat event); subsequent argument deltas use `call_index`
//! and may omit `provider_call_id`. `Done` always retains the complete identities.
//!
//! Unknown SSE event types are ignored. An Anthropic content-block delta of a
//! type the decoder has no rule for is not: that protocol has no terminal
//! snapshot, so a block is rebuilt from its start event and its deltas, and a
//! block missing part of its content would be committed and replayed as sealed
//! material. Such a stream fails as `Permanent` and commits nothing.
//! `input_json_delta` is applied to every block that declares an object
//! `input`, so a non-client tool block matches its batch form; only a client
//! `tool_use` yields tool-call deltas.
//! See [`StreamLimits`] for resource bounds.

mod anthropic;
mod chat;
pub(crate) mod common;
mod decoder;
mod responses;

pub use decoder::{StreamDecoder, StreamLimits, configure_stream_request};

#[cfg(test)]
mod tests;
