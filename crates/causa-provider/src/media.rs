//! The media-resolution seam: host-injected lookup that turns
//! fact-level [`MediaRef`]s into renderable payloads.
//!
//! The resolver lives provider-side because only the render path needs
//! bytes; the kernel stays reference-only and the reference frame policy
//! stays content-neutral. The contract is deliberately **sync and
//! memory-only**: an implementation should answer from an in-memory or
//! cached asset table and return `None` fast for anything it cannot
//! serve. Disk/network fetches are the host's prefetch responsibility
//! (fill the table before the turn), so resolution never blocks the
//! render loop. `None` is the single failure shape — the reference then
//! degrades to the deterministic text placeholder at render time.
//!
//! The gateway drops inline payloads beyond its inline ceiling (they degrade
//! like misses), and the renderers emit only `image/*` (inline bytes or
//! http/https URLs). Hosts should pre-filter their asset table to the
//! supported set (`image/png`, `image/jpeg`, `image/gif`, `image/webp`).
//!
//! # Degradation is not an error
//!
//! A reference this resolver cannot serve, a payload of a non-image type,
//! an empty payload, an unsupported URL scheme, or an inline payload over the
//! inline ceiling all render as the deterministic text placeholder
//! `[media: <type> <reference>]`. The request is still sent; nothing is
//! dropped silently and nothing fails.
//!
//! # The answer must be stable while the context lives
//!
//! **Every request re-resolves every reference in the frame, including the
//! ones in old messages.** The wire value of an earlier message is whatever
//! the resolver answers at that moment, so the same reference must keep
//! answering the same payload — the same bytes, or consistently unserved —
//! for as long as any context containing it is sent, across runs and not
//! only within one. The inline ceiling and the resolver itself must not
//! change under a live context either.
//!
//! A changed answer rewrites a prefix the provider already saw. With
//! Anthropic thinking enabled that invalidates the bound thinking blocks and
//! every later one, and it also drops the prompt cache from that message on.
//! The symptom does not name media: it is a provider 400 about thinking
//! blocks, or reasoning that silently stops being replayed. Using a content
//! hash as `MediaRef::reference` rules out one reference mapping to
//! different bytes, which is why it is the recommended form.
//!
//! For remote URLs, the same stability rules apply: the same reference must
//! answer the same URL throughout the lifetime of the context, and the
//! content behind that URL must not change. Causa cannot check the remote
//! content, so using content-addressed URLs is strongly recommended. The
//! address should return bytes directly and not rely on HTTP redirects (not
//! all providers follow redirects). The gateway does not fetch URLs; the
//! provider fetches them on every request. If the provider fails to fetch a
//! URL, that failure is returned as a request error from the provider: Causa
//! does not degrade or retry. The gateway's inline media ceiling constrains
//! only inline payloads; remote URLs are unconstrained by this limit.
//!
//! Deliberately not chosen: failing on an unresolved reference (it turns the
//! legitimate "keep only recent images" use into an error) and a per-gateway
//! switch between degrading and failing (configuration nobody asked for).
//! Detecting a changed answer would need a digest stored in the fact and
//! could still only end in an error.

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
