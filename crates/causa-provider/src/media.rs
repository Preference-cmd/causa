//! The media-resolution seam (Slice 6.5): host-injected lookup that turns
//! fact-level [`MediaRef`]s into renderable inline payloads.
//!
//! The resolver lives here — provider-side — because only the render
//! path needs bytes; the kernel stays reference-only and the reference
//! frame policy stays content-neutral. The contract is deliberately **sync and
//! memory-only**: an implementation should answer from an in-memory or
//! cached asset table and return `None` fast for anything it cannot
//! serve. Disk/network fetches are the host's prefetch responsibility
//! (fill the table before the turn), so resolution never blocks the
//! render loop. `None` is the single failure shape — the reference then
//! degrades to the deterministic text placeholder at render time.
//!
//! Enforcement point for the slice's MIME and size range: the gateway
//! drops payloads beyond its inline ceiling (they degrade like misses),
//! and the renderers inline only `image/*`. Hosts should pre-filter
//! their asset table to the supported set (`image/png`, `image/jpeg`,
//! `image/gif`, `image/webp`).

use std::collections::HashMap;

use causa_kernel::MediaRef;
use causa_protocol::translation::media::MediaPayload;

/// Resolves fact-level references into renderable payloads.
/// Sync + host-injected: the gateway has no asset knowledge.
pub trait MediaResolver: Send + Sync {
    /// The payload for this reference, or `None` when it cannot be
    /// served (asset missing, unsupported type, not prefetched) — the
    /// render degrades to a deterministic text placeholder.
    fn resolve(&self, r: &MediaRef) -> Option<MediaPayload>;
}

/// A plain asset table is a resolver: reference → payload, `None` on a
/// miss. The canonical in-memory host wiring (also the test fixture).
impl MediaResolver for HashMap<String, MediaPayload> {
    fn resolve(&self, r: &MediaRef) -> Option<MediaPayload> {
        self.get(&r.reference).cloned()
    }
}
