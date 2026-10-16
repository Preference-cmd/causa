//! Render-side media vocabulary: the resolved payload shape and the
//! resolution table the renderers consume. Facts never carry bytes or URLs —
//! this is the only shape media takes on the render path, built by the
//! gateway's injected resolver (causa-provider) right before translation.
//!
//! It renders inline `image/*` to the wire (Anthropic `image` blocks, Chat
//! `image_url`, Responses `input_image`); any other media type, remote URLs,
//! and every unresolvable reference degrade to a deterministic text placeholder,
//! decided once in the shared walk.

use std::collections::HashMap;

use causa_kernel::MediaRef;

/// The source data of a resolved media payload: inline base64 bytes or a
/// remote URL.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaSource {
    /// Inline media bytes represented as a base64-encoded string.
    Inline {
        /// Base64-encoded media bytes.
        data_base64: String,
    },
    /// A remote media location addressed by URL.
    ///
    /// The current renderers do not emit remote URLs; a `Url` payload degrades
    /// to the deterministic text placeholder `[media: {type} {reference}]`.
    Url(String),
}

/// Resolved media payload for one [`MediaRef`], carrying either inline
/// base64-encoded bytes or a remote URL. Render-side only: produced by the
/// host's `MediaResolver`, consumed by the renderers, never persisted.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MediaPayload {
    /// IANA media type of the payload (e.g. `image/png`).
    pub media_type: String,
    /// The source data of the media payload.
    pub source: MediaSource,
}

impl MediaPayload {
    /// Wraps a media type and a [`MediaSource`].
    pub fn new(media_type: impl Into<String>, source: MediaSource) -> Self {
        Self {
            media_type: media_type.into(),
            source,
        }
    }

    /// Wraps a media type and inline base64-encoded bytes.
    pub fn inline(media_type: impl Into<String>, data_base64: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            source: MediaSource::Inline {
                data_base64: data_base64.into(),
            },
        }
    }

    /// Wraps a media type and a remote URL.
    pub fn url(media_type: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            source: MediaSource::Url(url.into()),
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
