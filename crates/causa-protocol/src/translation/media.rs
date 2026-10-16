//! Render-side media vocabulary: the resolved payload shape, coarse media
//! kinds, and the resolution table the renderers consume. Facts never carry
//! bytes or URLs — this is the only shape media takes on the render path, built
//! by the gateway's injected resolver (causa-provider) right before translation.
//!
//! Media payloads are classified into [`MediaKind`]s. What renders to the wire
//! depends on each protocol's rendering capabilities:
//!
//! - **Anthropic Messages**: emits images (`image/*`) and documents
//!   (`application/pdf`) in user messages and tool results, either as inline
//!   base64 bytes or as remote `http`/`https` URLs.
//! - **OpenAI Chat Completions** and **Responses**: emit images only (`image/*`),
//!   either as inline base64 bytes or as remote `http`/`https` URLs; documents
//!   never render and degrade to text placeholders.
//!
//! Any unsupported combination, unclassified media type, unsupported or empty
//! URL, empty payload, or unresolvable reference degrades to a deterministic
//! text placeholder (`[media: {type} {reference}]`, or
//! `[media: {type} {name} {reference}]` when named), decided once in the
//! shared walk.

use std::collections::HashMap;

use causa_kernel::MediaRef;

/// The coarse media kind of a payload or reference.
///
/// Exhaustive on purpose: each protocol's rendering capabilities and the
/// gateway are total over it, so a new kind is a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    /// An image payload (e.g. `image/png`, `image/jpeg`).
    Image,
    /// A document payload (e.g. `application/pdf`).
    Document,
}

impl MediaKind {
    /// Classifies an IANA media type into a [`MediaKind`].
    ///
    /// Any parameter after `;` is dropped, whitespace is trimmed, and the
    /// comparison is case-insensitive. Any type with an `image/` prefix and
    /// a non-empty subtype maps to [`MediaKind::Image`], `application/pdf`
    /// maps to [`MediaKind::Document`], and all other types return `None`.
    pub fn of(media_type: &str) -> Option<Self> {
        let base = media_type.split(';').next().unwrap_or("").trim();
        if let Some(prefix) = base.get(..6)
            && prefix.eq_ignore_ascii_case("image/")
            && base.len() > 6
        {
            return Some(Self::Image);
        }
        if base.eq_ignore_ascii_case("application/pdf") {
            return Some(Self::Document);
        }
        None
    }
}

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
    /// When paired with a supported media type and protocol capability, a URL
    /// with an `http` or `https` scheme and non-empty remainder renders
    /// directly to the wire; any other URL, scheme, or unsupported media type
    /// degrades to the deterministic text placeholder.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_kind_classification() {
        // Exact matches
        assert_eq!(MediaKind::of("image/png"), Some(MediaKind::Image));
        assert_eq!(MediaKind::of("image/jpeg"), Some(MediaKind::Image));
        assert_eq!(MediaKind::of("application/pdf"), Some(MediaKind::Document));

        // Case insensitivity
        assert_eq!(MediaKind::of("IMAGE/PNG"), Some(MediaKind::Image));
        assert_eq!(MediaKind::of("APPLICATION/PDF"), Some(MediaKind::Document));
        assert_eq!(MediaKind::of("Application/Pdf"), Some(MediaKind::Document));

        // Parameters after semicolon
        assert_eq!(
            MediaKind::of("application/pdf; x=y"),
            Some(MediaKind::Document)
        );
        assert_eq!(
            MediaKind::of("image/jpeg; charset=utf-8"),
            Some(MediaKind::Image)
        );
        assert_eq!(
            MediaKind::of("application/pdf;version=1.7"),
            Some(MediaKind::Document)
        );

        // Trimming whitespace
        assert_eq!(
            MediaKind::of("  application/pdf  "),
            Some(MediaKind::Document)
        );
        assert_eq!(
            MediaKind::of("  image/png ; foo=bar  "),
            Some(MediaKind::Image)
        );

        // Non-matches: distinct types or invalid prefixes
        assert_eq!(MediaKind::of("application/pdfx"), None);
        assert_eq!(MediaKind::of("application/zip"), None);
        assert_eq!(MediaKind::of("text/plain"), None);
        assert_eq!(MediaKind::of("image"), None);
        assert_eq!(MediaKind::of("image/"), None);
        assert_eq!(MediaKind::of(""), None);
    }
}
