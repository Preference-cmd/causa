//! Render-side media vocabulary: the resolved payload shape and the
//! resolution table the renderers consume. Facts never carry bytes —
//! this is the only shape media takes on the render path, built by the
//! gateway's injected resolver (causa-provider) right before translation.
//!
//! This slice renders `image/*` to the wire (Anthropic `image` blocks,
//! Chat `image_url`, Responses `input_image`); any other media type —
//! and every unresolvable reference — degrades to a deterministic text
//! placeholder, decided once in the shared walk.

use std::collections::HashMap;

use causa_kernel::MediaRef;

/// Inline media bytes for one [`MediaRef`], as the provider wires want
/// them (base64 source). Render-side only: produced by the host's
/// `MediaResolver`, consumed by the renderers, never persisted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaPayload {
    /// IANA media type of the inline bytes (e.g. `image/png`).
    pub media_type: String,
    /// base64-encoded media bytes.
    pub data_base64: String,
}

impl MediaPayload {
    /// Wraps a media type and base64-encoded bytes.
    pub fn new(media_type: impl Into<String>, data_base64: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            data_base64: data_base64.into(),
        }
    }
}

/// Resolution table for one render pass: durable
/// [`MediaRef::reference`](MediaRef) → payload. The gateway builds it
/// from the frame's media references before calling a renderer; the
/// same reference resolves identically in batch and stream rendering.
/// A reference absent here renders as the deterministic placeholder.
#[derive(Debug, Clone, Default)]
pub struct MediaSet {
    by_reference: HashMap<String, MediaPayload>,
}

impl MediaSet {
    /// An empty table — every reference degrades to its placeholder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the payload for a reference. Later inserts win.
    pub fn insert(&mut self, reference: impl Into<String>, payload: MediaPayload) {
        self.by_reference.insert(reference.into(), payload);
    }

    /// The payload registered for this reference's `reference` key, if
    /// the resolver produced one.
    pub fn get(&self, r: &MediaRef) -> Option<&MediaPayload> {
        self.by_reference.get(&r.reference)
    }
}
