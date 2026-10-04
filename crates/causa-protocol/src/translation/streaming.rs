//! Bounded, transport-free assembly of the three supported HTTP SSE protocols.
//! Deltas are observations; only `Done` contains authoritative output. EOF never
//! manufactures a protocol terminal. Tool identities are emitted when declared
//! (or supplied on a Chat event); subsequent argument deltas use `call_index`
//! and may omit `provider_call_id`. `Done` always retains the complete identities.
//! See [`StreamLimits`] for resource bounds.

mod anthropic;
mod chat;
pub(crate) mod common;
mod decoder;
mod responses;

pub use decoder::{StreamDecoder, StreamLimits, configure_stream_request};

#[cfg(test)]
mod tests;
