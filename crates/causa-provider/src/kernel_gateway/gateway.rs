//! Gateway configuration and shared request preparation.

use super::config::{
    AnthropicGatewayConfig, KernelGatewayConfig, OpenAiChatGatewayConfig,
    OpenAiResponsesGatewayConfig,
};
use super::stream::{establish_stream, model_stream};
use crate::gateway_transport::{GatewayCore, check_control, finish_response, send_with_control};
use crate::media::MediaResolver;
use async_trait::async_trait;
use causa_kernel::{
    BlockContent, CallControl, ContentPart, MediaRef, ModelGateway, ModelInvokeError, ModelOutput,
    ModelRequest,
};
use causa_protocol::translation::media::MediaSet;
use causa_protocol::translation::render_options::{
    AnthropicRenderOptions, ChatRenderOptions, ResponsesRenderOptions,
};
use causa_protocol::translation::streaming::{
    StreamDecoder, StreamLimits, configure_stream_request,
};
use reqwest::Client;
use std::sync::Arc;
use tracing::Instrument;

/// Generic kernel gateway: reqwest transport + the pure translation in
/// `causa_protocol::translation` composed with the shared error
/// mapping table and control-plane wiring from `crate::gateway_transport`.
///
/// Media: an optional host-injected [`MediaResolver`] turns the frame's
/// fact-level references into inline payloads right before rendering —
/// batch and stream share one resolution pass. Without a resolver, every
/// reference degrades to its text placeholder.
pub struct KernelHttpGateway<C: KernelGatewayConfig> {
    core: GatewayCore,
    api_key: String,
    config: C,
    media_resolver: Option<Arc<dyn MediaResolver>>,
    /// Inline media ceiling, in decoded bytes (base64 is ~4/3 of this);
    /// payloads beyond it degrade to placeholders like resolver misses.
    /// Default: 5 MiB (the documented per-image cap of the strictest of
    /// the three wires).
    max_inline_media_bytes: usize,
    stream_limits: StreamLimits,
}

impl<C: KernelGatewayConfig> std::fmt::Debug for KernelHttpGateway<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelHttpGateway")
            .field("protocol", &C::PROTOCOL.as_str())
            .field("path", &C::PATH)
            .field("media_resolver", &self.media_resolver.is_some())
            .field("max_inline_media_bytes", &self.max_inline_media_bytes)
            .field("stream_limits", &self.stream_limits)
            .finish_non_exhaustive()
    }
}

impl<C: KernelGatewayConfig> Clone for KernelHttpGateway<C> {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            api_key: self.api_key.clone(),
            config: self.config.clone(),
            media_resolver: self.media_resolver.clone(),
            max_inline_media_bytes: self.max_inline_media_bytes,
            stream_limits: self.stream_limits,
        }
    }
}

impl<C: KernelGatewayConfig> KernelHttpGateway<C> {
    /// Build a gateway with the protocol's default endpoint and configuration.
    /// Media resolution is absent until the host injects a resolver.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            core: GatewayCore::new(C::DEFAULT_ENDPOINT),
            api_key: api_key.into(),
            config: C::default(),
            media_resolver: None,
            max_inline_media_bytes: 5 * 1024 * 1024,
            stream_limits: StreamLimits::default(),
        }
    }

    /// Point at a non-default host (test doubles, gateways). `C::PATH`
    /// is appended to the given base URL, which must carry its own version
    /// segment: `https://api.anthropic.com/v1`, `https://api.openai.com/v1`,
    /// or `https://host/provider/v1` for a compatible provider. A base
    /// without a version segment is used as given.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.core = self.core.with_base_url(base_url, C::PATH);
        self
    }

    /// Full endpoint override (host + path).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.core = self.core.with_endpoint(endpoint);
        self
    }

    /// Replace the HTTP client with one configured by the host.
    ///
    /// The default client disables redirects and retries, and configures a
    /// connect timeout and idle read timeout without a total timeout. A supplied
    /// client's transport policies remain host-owned: disable both for the same
    /// single-request guarantee. Note that a supplied client configured with its
    /// own `.timeout(...)` reintroduces a total request cap that cuts off long
    /// streaming or generation turns; total duration belongs to the
    /// [`CallControl`] deadline instead. Cancellation and deadlines remain
    /// enforced independently of the supplied client's timeout settings.
    pub fn with_http_client(mut self, http: Client) -> Self {
        self.core = self.core.with_http_client(http);
        self
    }

    /// Inject the media resolver: the host's asset table the gateway
    /// consults for every media reference in the frame.
    pub fn with_media_resolver(mut self, resolver: Arc<dyn MediaResolver>) -> Self {
        self.media_resolver = Some(resolver);
        self
    }

    /// Override the inline media ceiling, in decoded bytes.
    pub fn with_max_inline_media_bytes(mut self, max_bytes: usize) -> Self {
        self.max_inline_media_bytes = max_bytes;
        self
    }

    /// Override the pure decoder's framing, total-input, output and item budgets.
    /// Invalid limits are rejected before sending a streaming request.
    pub fn with_stream_limits(mut self, limits: StreamLimits) -> Self {
        self.stream_limits = limits;
        self
    }

    /// Shared media resolution and request validation for both call entrances.
    fn render_request(
        &self,
        request: &ModelRequest,
    ) -> Result<serde_json::Value, ModelInvokeError> {
        self.config.render(request, &self.resolve_media(request))
    }

    /// One resolution pass over the request's frame: every media
    /// reference (Parts blocks and tool results) the injected resolver
    /// serves within the inline ceiling enters the table. Deterministic
    /// — the same frame and asset state produce the same table.
    fn resolve_media(&self, request: &ModelRequest) -> MediaSet {
        let mut set = MediaSet::new();
        let Some(resolver) = &self.media_resolver else {
            return set;
        };
        let mut refs: Vec<MediaRef> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for block in &request.frame.blocks {
            let collect = |r: &MediaRef,
                           refs: &mut Vec<MediaRef>,
                           seen: &mut std::collections::HashSet<String>| {
                if seen.insert(r.reference.clone()) {
                    refs.push(r.clone());
                }
            };
            match block.content() {
                BlockContent::Parts(parts) => {
                    for part in parts {
                        if let ContentPart::Media(r) = part {
                            collect(r, &mut refs, &mut seen);
                        }
                    }
                }
                BlockContent::ToolResult(result) => {
                    for r in &result.media {
                        collect(r, &mut refs, &mut seen);
                    }
                }
                BlockContent::ToolCall(_) | BlockContent::ToolChange(_) => {}
            }
        }
        for r in refs {
            if let Some(payload) = resolver.resolve(&r)
                && payload.data_base64.len() * 3 / 4 <= self.max_inline_media_bytes
            {
                set.insert(r.reference, payload);
            }
        }
        set
    }
}

impl KernelHttpGateway<AnthropicGatewayConfig> {
    /// Override the `anthropic-version` header (default `2023-06-01`).
    pub fn with_anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.config.anthropic_version = version.into();
        self
    }

    /// Append one extra `anthropic-beta` value for this gateway. Values are
    /// sent as one comma-separated `anthropic-beta` header, in the order
    /// listed, followed by the value an enabled option needs; nothing is
    /// sent by default and no model table decides headers.
    pub fn with_beta(mut self, value: impl Into<String>) -> Self {
        self.config.beta.push(value.into());
        self
    }

    /// Replace every render option at once, including any set earlier with
    /// the per-option setters (a later call wins either way). Wire
    /// consequences follow from the options: `native_tool_changes` still adds
    /// `inline-tools-2026-09-15` to `anthropic-beta`. The value usually comes
    /// from host configuration, as which servers and models support an option
    /// is the host's to decide.
    pub fn with_render_options(mut self, options: AnthropicRenderOptions) -> Self {
        self.config.render_options = options;
        self
    }

    /// Render tool changes in place (default `false`).
    ///
    /// `tools` stays at the context's first tool change and each later
    /// change is sent at its position as `tool_removal` / `tool_addition`
    /// blocks, so a changed tool listing keeps the cached prefix and earlier
    /// thinking. The gateway adds the `inline-tools-2026-09-15` beta value
    /// it needs. Off, every request carries the current tools in `tools`.
    /// Enable only for a server known to support it: a server without support
    /// may reject the request, or ignore the change so the model never sees
    /// the added tool.
    pub fn with_native_tool_changes(mut self, native: bool) -> Self {
        self.config.render_options.native_tool_changes = native;
        self
    }
}

impl KernelHttpGateway<OpenAiChatGatewayConfig> {
    /// Replace every render option at once, including any set earlier with
    /// the per-option setters (a later call wins either way). Wire
    /// consequences follow from the options. The value usually comes from host
    /// configuration, as which servers and models support an option is the
    /// host's to decide.
    pub fn with_render_options(mut self, options: ChatRenderOptions) -> Self {
        self.config.render_options = options;
        self
    }

    /// Replay sealed `reasoning_content` message-level items (default `true`).
    ///
    /// DeepSeek requires `reasoning_content` back on every request that
    /// carries `tools`; turning this off removes the key from the renderer's
    /// own sealed items and changes nothing else. The choice is per gateway
    /// value, not per request: a host that needs both holds two gateways.
    pub fn with_reasoning_content_replay(mut self, replay: bool) -> Self {
        self.config.render_options.replay_reasoning_content = replay;
        self
    }
}

impl KernelHttpGateway<OpenAiResponsesGatewayConfig> {
    /// Replace every render option at once, including any set earlier with
    /// the per-option setters (a later call wins either way). Wire
    /// consequences follow from the options. The value usually comes from host
    /// configuration, as which servers and models support an option is the
    /// host's to decide.
    pub fn with_render_options(mut self, options: ResponsesRenderOptions) -> Self {
        self.config.render_options = options;
        self
    }

    /// Send `include: ["reasoning.encrypted_content"]` (default `true`).
    ///
    /// Without it a reasoning item cannot be meaningfully replayed under
    /// `store: false` or zero data retention. The choice is per gateway value,
    /// not per request: a host that needs both holds two gateways.
    pub fn with_encrypted_reasoning(mut self, include: bool) -> Self {
        self.config.render_options.include_encrypted_reasoning = include;
        self
    }

    /// Render `store`; `None` omits the field (default `Some(false)`, as Codex
    /// and `pi-ai` send it). The host owns the data-retention choice. The
    /// choice is per gateway value, not per request.
    pub fn with_store(mut self, store: Option<bool>) -> Self {
        self.config.render_options.store = store;
        self
    }

    /// Send added tools as `additional_tools` input items (default `false`).
    ///
    /// `tools` stays at the context's first tool change and each later
    /// addition is sent at its position, as Codex does, so the cached prefix
    /// survives it. A context in which a later change removes a tool or
    /// redefines one renders with the current tools in `tools`, as with this
    /// off. Enable only for a server known to support it: a server without
    /// support may reject the request, or ignore the change so the model never
    /// sees the added tool.
    pub fn with_additional_tools(mut self, additional_tools: bool) -> Self {
        self.config.render_options.additional_tools = additional_tools;
        self
    }
}

#[async_trait]
impl<C: KernelGatewayConfig> ModelGateway for KernelHttpGateway<C> {
    async fn stream(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<causa_kernel::ModelStream, ModelInvokeError> {
        let span = tracing::debug_span!(
            "agent.http",
            protocol = C::PROTOCOL.as_str(),
            path = C::PATH
        );
        async {
            check_control(control)?;
            let decoder = StreamDecoder::new(C::PROTOCOL, self.stream_limits)?;
            let mut body = self.render_request(request)?;
            configure_stream_request(C::PROTOCOL, &mut body)?;
            check_control(control)?;
            let req = self
                .config
                .decorate_request(self.core.post(), &self.api_key)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .json(&body);
            let response = establish_stream(req, control, C::PROTOCOL).await?;
            Ok(model_stream(response, decoder, control.clone()))
        }
        .instrument(span)
        .await
    }

    async fn invoke(
        &self,
        request: &ModelRequest,
        control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        // Observability baseline: ids and names only — never arguments,
        // message bodies, or API keys. The span is entered per poll via
        // `Instrument` so the boxed future stays `Send`.
        let span = tracing::debug_span!(
            "agent.http",
            protocol = C::PROTOCOL.as_str(),
            path = C::PATH
        );
        async {
            // A frame the renderer rejects never reaches the wire. Media
            // resolves first — the table feeds every renderer branch.
            check_control(control)?;
            let body = self.render_request(request)?;
            check_control(control)?;
            let req = self
                .config
                .decorate_request(self.core.post(), &self.api_key)
                .json(&body);
            let response = send_with_control(req, control).await?;
            finish_response(response, C::PROTOCOL, |v| self.config.parse(v))
        }
        .instrument(span)
        .await
    }
}
