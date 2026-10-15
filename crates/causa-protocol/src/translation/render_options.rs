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
//!
//! These types represent configuration, not context: they are outside the
//! stable serialized set, and their serialized shape follows the field names
//! and carries no compatibility promise. Which servers and models support an
//! option is for the host to decide; an option a server does not support may
//! be rejected by it or silently ignored.

use serde::{Deserialize, Serialize};

use super::anthropic::INLINE_TOOLS_BETA;

/// Anthropic Messages render options.
///
/// Configuration, not context: see the module docs for what the serialized form
/// promises and who decides support.
///
/// Settings of the `thinking` object, including `block_binding`, are not
/// here: they are part of the object the host sends in
/// `GenerationOptions::extra_body`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct AnthropicRenderOptions {
    /// Render tool changes in place (default `false`). `tools` stays at the
    /// frame's declaration followed by a deferred placeholder tool, and each
    /// later change renders at its position as a `system` message of
    /// `tool_removal` and `tool_addition` blocks, so a change keeps the cached
    /// prefix. The API needs the [`INLINE_TOOLS_BETA`] beta for it (reported by
    /// [`Self::beta_values`]), which the gateway adds. Off, `tools` carries the
    /// current tool state and a change rewrites it. Enable only for a server
    /// known to support it: a server without support may reject the request, or
    /// ignore the change so the model never sees the added tool.
    pub native_tool_changes: bool,
}

impl AnthropicRenderOptions {
    /// Values for the `anthropic-beta` header required by the enabled options.
    ///
    /// A request rendered with these options is only valid when the HTTP
    /// request carries each returned value. The `causa-provider` gateway adds
    /// them to its `anthropic-beta` header; a host that sends requests itself
    /// must add them.
    pub fn beta_values(&self) -> Vec<&'static str> {
        let mut values = Vec::new();
        if self.native_tool_changes {
            values.push(INLINE_TOOLS_BETA);
        }
        values
    }
}

/// OpenAI Responses render options.
///
/// Configuration, not context: see the module docs for what the serialized form
/// promises and who decides support.
///
/// `Default` sends `include: ["reasoning.encrypted_content"]` and
/// `store: false`, as Codex and `pi-ai` do: encrypted reasoning is what makes
/// a reasoning item replayable under `store: false` or zero data retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
    ///
    /// When deserializing, a missing key yields the default `Some(false)`,
    /// while an explicit JSON `null` yields `None` (omits the field).
    /// Formats without `null`, such as TOML, use `with_store(None)` instead.
    pub store: Option<bool>,
    /// Render added tools in place (default `false`). `tools` stays at the
    /// frame's declaration and each later change renders at its position as
    /// an `additional_tools` input item, so an addition keeps the cached
    /// prefix. A frame with a removal or a second definition of one name
    /// renders as if this were off: `tools` carries the current tool state.
    /// Enable only for a server known to support it: a server without support
    /// may reject the request, or ignore the change so the model never sees
    /// the added tool.
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
/// Configuration, not context: see the module docs for what the serialized form
/// promises and who decides support.
///
/// `Default` replays sealed `reasoning_content`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug>(
        value: T,
    ) {
        let json = serde_json::to_string(&value).unwrap();
        let back: T = serde_json::from_str(&json).unwrap();
        assert_eq!(value, back);
    }

    #[test]
    fn empty_json_deserializes_to_default() {
        let anthropic: AnthropicRenderOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(anthropic, AnthropicRenderOptions::default());

        let responses: ResponsesRenderOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(responses, ResponsesRenderOptions::default());

        let chat: ChatRenderOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(chat, ChatRenderOptions::default());
    }

    #[test]
    fn serialize_deserialize_round_trip_is_identity() {
        round_trip(AnthropicRenderOptions::default());
        round_trip(AnthropicRenderOptions {
            native_tool_changes: true,
        });

        round_trip(ResponsesRenderOptions::default());
        round_trip(ResponsesRenderOptions {
            include_encrypted_reasoning: false,
            store: None,
            additional_tools: true,
        });
        round_trip(ResponsesRenderOptions {
            include_encrypted_reasoning: true,
            store: Some(true),
            additional_tools: false,
        });

        round_trip(ChatRenderOptions::default());
        round_trip(ChatRenderOptions {
            replay_reasoning_content: false,
        });
    }

    #[test]
    fn unknown_field_is_rejected() {
        let err =
            serde_json::from_str::<AnthropicRenderOptions>(r#"{"unknown": true}"#).unwrap_err();
        assert!(err.to_string().contains("unknown field `unknown`"));

        let err = serde_json::from_str::<ResponsesRenderOptions>(r#"{"typo": 123}"#).unwrap_err();
        assert!(err.to_string().contains("unknown field `typo`"));

        let err =
            serde_json::from_str::<ChatRenderOptions>(r#"{"unrecognized": false}"#).unwrap_err();
        assert!(err.to_string().contains("unknown field `unrecognized`"));
    }

    #[test]
    fn cross_protocol_field_is_rejected() {
        let err = serde_json::from_str::<AnthropicRenderOptions>(r#"{"additional_tools": true}"#)
            .unwrap_err();
        assert!(err.to_string().contains("unknown field `additional_tools`"));

        let err = serde_json::from_str::<ChatRenderOptions>(r#"{"native_tool_changes": true}"#)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown field `native_tool_changes`")
        );

        let err =
            serde_json::from_str::<ResponsesRenderOptions>(r#"{"replay_reasoning_content": true}"#)
                .unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown field `replay_reasoning_content`")
        );
    }

    #[test]
    fn responses_store_null_gives_none_and_missing_gives_some_false() {
        let missing: ResponsesRenderOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(missing.store, Some(false));

        let null_store: ResponsesRenderOptions =
            serde_json::from_str(r#"{"store": null}"#).unwrap();
        assert_eq!(null_store.store, None);

        let true_store: ResponsesRenderOptions =
            serde_json::from_str(r#"{"store": true}"#).unwrap();
        assert_eq!(true_store.store, Some(true));

        let false_store: ResponsesRenderOptions =
            serde_json::from_str(r#"{"store": false}"#).unwrap();
        assert_eq!(false_store.store, Some(false));
    }

    #[test]
    fn anthropic_beta_values_reflect_enabled_options() {
        let default_options = AnthropicRenderOptions::default();
        assert!(default_options.beta_values().is_empty());

        let native_options = AnthropicRenderOptions {
            native_tool_changes: true,
        };
        assert_eq!(native_options.beta_values(), [INLINE_TOOLS_BETA]);
    }
}
