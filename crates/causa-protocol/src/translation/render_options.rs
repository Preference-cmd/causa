//! Renderer options: the renderer-specific argument each pure renderer takes.
//!
//! The renderers in this module tree are pure functions with a fixed argument
//! list; these values are the one renderer-specific argument they accept. They
//! live here (translation, transport-free) and are hosted by the matching
//! gateway configuration in `causa-provider`, which exposes one `with_*`
//! builder per option next to its existing builders. The configurations keep
//! `Default`, so the documented defaults hold with no setter call.
//!
//! Options are per gateway value, not per request. A host that needs different
//! behavior for one request holds a second gateway value; the configuration is
//! cheap to clone and the HTTP client, base URL and media resolver can be
//! shared, so the cost is one extra value, not a second connection pool.

/// Anthropic Messages render options.
///
/// Settings of the `thinking` object, including `block_binding`, are not
/// here: they are part of the object the host sends in
/// `GenerationOptions::extra_body`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct AnthropicRenderOptions {
    /// Render tool changes in place (default `false`). `tools` stays at the
    /// frame's declaration followed by a deferred placeholder tool, and each
    /// later change renders at its position as a `system` message of
    /// `tool_removal` and `tool_addition` blocks, so a change keeps the cached
    /// prefix. The API needs the `inline-tools-2026-09-15` beta for it, which
    /// the gateway adds. Off, `tools` carries the current tool state and a
    /// change rewrites it.
    pub native_tool_changes: bool,
}

/// OpenAI Responses render options.
///
/// `Default` sends `include: ["reasoning.encrypted_content"]` and
/// `store: false`, as Codex and `pi-ai` do: encrypted reasoning is what makes
/// a reasoning item replayable under `store: false` or zero data retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResponsesRenderOptions {
    /// Send `include: ["reasoning.encrypted_content"]` (default `true`).
    /// Turning it off leaves the field out entirely, so the response carries
    /// no encrypted reasoning and a reasoning item cannot be meaningfully
    /// replayed under `store: false`.
    pub include_encrypted_reasoning: bool,
    /// Value rendered as `store`; `None` omits the field (default
    /// `Some(false)`). The host owns the data-retention choice this
    /// represents.
    pub store: Option<bool>,
    /// Render added tools in place (default `false`). `tools` stays at the
    /// frame's declaration and each later change renders at its position as
    /// an `additional_tools` input item, so an addition keeps the cached
    /// prefix. A frame with a removal or a second definition of one name
    /// renders as if this were off: `tools` carries the current tool state.
    pub additional_tools: bool,
}

impl Default for ResponsesRenderOptions {
    fn default() -> Self {
        Self {
            include_encrypted_reasoning: true,
            store: Some(false),
            additional_tools: false,
        }
    }
}

/// OpenAI Chat Completions render options.
///
/// `Default` replays sealed `reasoning_content`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChatRenderOptions {
    /// Replay the frame's own-format sealed `reasoning_content` message-level
    /// items (default `true`). DeepSeek requires `reasoning_content` back on
    /// every request that carries `tools` and ignores it otherwise; other
    /// compatible providers differ, which is what this option is for. When it
    /// is off, the renderer removes the `reasoning_content` key from its own
    /// sealed items — dropping an item that carries nothing else — and changes
    /// nothing else. Other sealed message fields still replay.
    pub replay_reasoning_content: bool,
}

impl Default for ChatRenderOptions {
    fn default() -> Self {
        Self {
            replay_reasoning_content: true,
        }
    }
}
