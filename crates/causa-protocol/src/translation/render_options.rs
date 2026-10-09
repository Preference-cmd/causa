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

/// Anthropic `thinking.block_binding.prefix_mismatch_behavior`.
///
/// The provider's default is `error`; `drop_block` lets a turn continue when
/// a replayed thinking prefix no longer matches. The value is rendered only
/// when the request also renders a `thinking` object — that is, only when
/// `GenerationOptions::reasoning_effort` is set. Setting it does not add the
/// `anthropic-beta` header the field needs: a host that sets one without the
/// other gets the provider's 400.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefixMismatchBehavior {
    /// `error` — reject a request whose thinking prefix does not match.
    #[default]
    Error,
    /// `drop_block` — drop the mismatched thinking block instead.
    DropBlock,
}

impl PrefixMismatchBehavior {
    /// The wire value rendered into `thinking.block_binding`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::DropBlock => "drop_block",
        }
    }
}

/// Anthropic Messages render options.
///
/// `Default` renders no `block_binding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnthropicRenderOptions {
    /// Rendered as `thinking.block_binding.prefix_mismatch_behavior` whenever
    /// a `thinking` object is rendered (that is, whenever
    /// `GenerationOptions::reasoning_effort` is set). `None` renders
    /// nothing, including when no `thinking` object is rendered.
    pub prefix_mismatch_behavior: Option<PrefixMismatchBehavior>,
}

/// OpenAI Responses render options.
///
/// `Default` sends `include: ["reasoning.encrypted_content"]` and
/// `store: false`, as Codex and `pi-ai` do: encrypted reasoning is what makes
/// a reasoning item replayable under `store: false` or zero data retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl Default for ResponsesRenderOptions {
    fn default() -> Self {
        Self {
            include_encrypted_reasoning: true,
            store: Some(false),
        }
    }
}

/// OpenAI Chat Completions render options.
///
/// `Default` replays sealed `reasoning_content`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
