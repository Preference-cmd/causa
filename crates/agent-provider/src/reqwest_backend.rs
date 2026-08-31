//! ⚠️ FROZEN — Slice 3 起冻结；本文件实现 `agent-provider::CompletionBackend`
//!（frozen V1 harness 栈 `AgentProvider`，对外由 `BackendProvider` 包装）的
//! production backend。
//! Slice 3 新路径（`gateway_transport.rs` + `KernelHttpGateway<C>`）接管的是
//! `ModelGateway` 的 non-streaming `complete` 链路；`stream` /
//! `list_models` / `FileSource::Url` → base64 / `AgentRequest`→wire 翻译等
//! 能力仍在冻层，短链路上再迁移前需保留。本文件随 `BackendProvider` /
//! `ReqwestBackend` 在 Slice 5+ 的 harness 迁移后一并删除
//!（`ai-protocol/translation::{request,streaming,files}` 同理）。

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use reimagine_agent_harness::{
    AgentRequest, AgentResponse, AgentStream, AgentStreamEvent, ContentBlock, FileSource, Message,
    ModelInfo, ProviderName,
};
use reimagine_ai_protocol::translation;
use reimagine_ai_protocol::translation::sse_parser::{SseEvent, SseParser};
use reimagine_ai_protocol::{
    AnthropicMessagesConfig, CompletionBackend, OpenAiChatCompletionsConfig, OpenAiResponsesConfig,
    Protocol, ProviderAdapterError,
};
use serde_json::{Value, json};

/// Maximum number of retries for transient HTTP errors.
const MAX_RETRIES: u32 = 3;
/// Base delay for exponential backoff (doubled each retry).
const RETRY_BASE_DELAY: Duration = Duration::from_millis(500);
/// Random jitter added to each retry delay (0..=RETRY_JITTER_MS) so a
/// burst of 429s across hosts does not re-synchronize (AC-05).
const RETRY_JITTER_MS: u64 = 250;
/// Connect timeout for all provider HTTP calls (AC-05).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Total request timeout for non-streaming calls (AC-05).
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Idle (read) timeout between streamed SSE events (AC-05). Generous:
/// reasoning models may produce no delta for minutes.
const STREAM_READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Production backend. `complete` and `list_models` route through
/// direct reqwest HTTP calls.
///
/// `workspace_dir` roots `FileSource::Url` resolution: file blocks
/// referencing workspace-relative paths are read from this directory and
/// inlined as base64 before wire translation. When it is `None`, url
/// sources are rejected at request time.
#[derive(Clone)]
pub struct ReqwestBackend {
    name: ProviderName,
    config: BackendConfig,
    http: reqwest::Client,
    /// Client used for streaming requests. Distinct from `http`:
    /// streaming needs a generous idle (read) timeout instead of a
    /// total request timeout (AC-05).
    stream_http: reqwest::Client,
    workspace_dir: Option<PathBuf>,
}

impl std::fmt::Debug for ReqwestBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let protocol_repr = match &self.config {
            BackendConfig::OpenAiChatCompletions(_) => "OpenAiChatCompletions(<redacted>)",
            BackendConfig::AnthropicMessages(_) => "AnthropicMessages(<redacted>)",
            BackendConfig::OpenAiResponses(_) => "OpenAiResponses(<redacted>)",
        };
        f.debug_struct("ReqwestBackend")
            .field("name", &self.name)
            .field("config", &protocol_repr)
            .field("workspace_dir", &self.workspace_dir)
            .finish_non_exhaustive()
    }
}

/// A protocol paired with its typed config. `protocol()` derives the
/// [`Protocol`](reimagine_ai_protocol::Protocol) discriminator from the variant.
#[derive(Debug, Clone)]
pub enum BackendConfig {
    OpenAiChatCompletions(OpenAiChatCompletionsConfig),
    AnthropicMessages(AnthropicMessagesConfig),
    OpenAiResponses(OpenAiResponsesConfig),
}

impl BackendConfig {
    pub fn protocol(&self) -> Protocol {
        match self {
            Self::OpenAiChatCompletions(_) => Protocol::OpenAiChatCompletions,
            Self::AnthropicMessages(_) => Protocol::AnthropicMessages,
            Self::OpenAiResponses(_) => Protocol::OpenAiResponses,
        }
    }
}

impl ReqwestBackend {
    /// Build the two default clients: `http` for complete/list_models
    /// (connect + total timeout) and `stream_http` for streaming
    /// (connect + generous idle timeout, no total cap so long-lived
    /// streams and minutes-long reasoning gaps are not killed) (AC-05).
    fn default_clients() -> (reqwest::Client, reqwest::Client) {
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .expect("reqwest client build");
        let stream_http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(STREAM_READ_TIMEOUT)
            .build()
            .expect("reqwest client build");
        (http, stream_http)
    }

    /// Construct an OpenAI-compatible backend with a default
    /// `reqwest::Client`.
    pub fn openai_chat_completions(name: ProviderName, cfg: OpenAiChatCompletionsConfig) -> Self {
        let (http, stream_http) = Self::default_clients();
        Self {
            name,
            config: BackendConfig::OpenAiChatCompletions(cfg),
            http,
            stream_http,
            workspace_dir: None,
        }
    }

    /// Construct an OpenAI-compatible backend with an explicit
    /// `reqwest::Client` (used by tests).
    pub fn openai_chat_completions_with_http_client(
        name: ProviderName,
        cfg: OpenAiChatCompletionsConfig,
        http: reqwest::Client,
    ) -> Self {
        Self {
            name,
            config: BackendConfig::OpenAiChatCompletions(cfg),
            stream_http: http.clone(),
            http,
            workspace_dir: None,
        }
    }

    /// Construct an Anthropic backend with a default `reqwest::Client`.
    pub fn anthropic_messages(name: ProviderName, cfg: AnthropicMessagesConfig) -> Self {
        let (http, stream_http) = Self::default_clients();
        Self {
            name,
            config: BackendConfig::AnthropicMessages(cfg),
            http,
            stream_http,
            workspace_dir: None,
        }
    }

    /// Construct an Anthropic backend with an explicit `reqwest::Client`
    /// (used by tests).
    pub fn anthropic_messages_with_http_client(
        name: ProviderName,
        cfg: AnthropicMessagesConfig,
        http: reqwest::Client,
    ) -> Self {
        Self {
            name,
            config: BackendConfig::AnthropicMessages(cfg),
            stream_http: http.clone(),
            http,
            workspace_dir: None,
        }
    }

    /// Construct an OpenAI Responses backend with a default
    /// `reqwest::Client`.
    pub fn openai_responses(name: ProviderName, cfg: OpenAiResponsesConfig) -> Self {
        let (http, stream_http) = Self::default_clients();
        Self {
            name,
            config: BackendConfig::OpenAiResponses(cfg),
            http,
            stream_http,
            workspace_dir: None,
        }
    }

    /// Construct an OpenAI Responses backend with an explicit
    /// `reqwest::Client` (used by tests).
    pub fn openai_responses_with_http_client(
        name: ProviderName,
        cfg: OpenAiResponsesConfig,
        http: reqwest::Client,
    ) -> Self {
        Self {
            name,
            config: BackendConfig::OpenAiResponses(cfg),
            stream_http: http.clone(),
            http,
            workspace_dir: None,
        }
    }

    /// Root `FileSource::Url` resolution at `workspace_dir`: file blocks
    /// referencing workspace-relative paths are read from this directory
    /// and inlined as base64 before wire translation. When unset, url
    /// sources are rejected at request time.
    pub fn with_workspace_dir(mut self, workspace_dir: impl Into<PathBuf>) -> Self {
        self.workspace_dir = Some(workspace_dir.into());
        self
    }

    /// The configured workspace directory, if any.
    pub fn workspace_dir(&self) -> Option<&Path> {
        self.workspace_dir.as_deref()
    }

    fn openai_config(&self) -> Result<&OpenAiChatCompletionsConfig, ProviderAdapterError> {
        match &self.config {
            BackendConfig::OpenAiChatCompletions(cfg) => Ok(cfg),
            BackendConfig::AnthropicMessages(_) => Err(ProviderAdapterError::configuration(
                "expected openai_chat_completions protocol, got anthropic_messages",
            )),
            BackendConfig::OpenAiResponses(_) => Err(ProviderAdapterError::configuration(
                "expected openai_chat_completions protocol, got openai_responses",
            )),
        }
    }

    fn responses_config(&self) -> Result<&OpenAiResponsesConfig, ProviderAdapterError> {
        match &self.config {
            BackendConfig::OpenAiResponses(cfg) => Ok(cfg),
            BackendConfig::OpenAiChatCompletions(_) => Err(ProviderAdapterError::configuration(
                "expected openai_responses protocol, got openai_chat_completions",
            )),
            BackendConfig::AnthropicMessages(_) => Err(ProviderAdapterError::configuration(
                "expected openai_responses protocol, got anthropic_messages",
            )),
        }
    }

    fn anthropic_config(&self) -> Result<&AnthropicMessagesConfig, ProviderAdapterError> {
        match &self.config {
            BackendConfig::AnthropicMessages(cfg) => Ok(cfg),
            BackendConfig::OpenAiChatCompletions(_) => Err(ProviderAdapterError::configuration(
                "expected anthropic_messages protocol, got openai_chat_completions",
            )),
            BackendConfig::OpenAiResponses(_) => Err(ProviderAdapterError::configuration(
                "expected anthropic_messages protocol, got openai_responses",
            )),
        }
    }

    /// Resolve `FileSource::Url` file blocks against the configured
    /// workspace directory, replacing them with inline base64 payloads
    /// (PV-03b). Messages without url-backed file blocks are returned
    /// unchanged without cloning.
    ///
    /// The translation layer stays pure: it only ever sees `Data`
    /// sources. Workspace-relative paths are read via
    /// `translation::files::read_workspace_file` (10MB limit, path
    /// traversal protection); `http(s)://` references are rejected —
    /// remote downloads are not supported in V2.
    fn resolve_file_sources<'a>(
        &self,
        request: &'a AgentRequest,
    ) -> Result<Cow<'a, [Message]>, ProviderAdapterError> {
        let has_url_blocks = request.messages().iter().any(|m| {
            m.blocks()
                .iter()
                .any(|b| matches!(b, ContentBlock::File(f) if f.source().url().is_some()))
        });
        if !has_url_blocks {
            return Ok(Cow::Borrowed(request.messages()));
        }
        let workspace_dir = match &self.workspace_dir {
            Some(dir) => dir.clone(),
            None => {
                return Err(ProviderAdapterError::configuration(
                    "file block url sources require a workspace directory; \
                     the provider backend was constructed without one",
                ));
            }
        };
        let mut rebuilt = request.messages().to_vec();
        for message in rebuilt.iter_mut() {
            if !message
                .blocks()
                .iter()
                .any(|b| matches!(b, ContentBlock::File(f) if f.source().url().is_some()))
            {
                continue;
            }
            let mut blocks = Vec::with_capacity(message.blocks().len());
            for block in message.blocks() {
                match block {
                    ContentBlock::File(file) if file.source().url().is_some() => {
                        let path = file.source().url().unwrap_or_default();
                        let bytes = translation::files::read_workspace_file(&workspace_dir, path)?;
                        let base64 = base64::engine::general_purpose::STANDARD.encode(bytes);
                        blocks.push(ContentBlock::File(
                            file.clone().with_source(FileSource::Data { base64 }),
                        ));
                    }
                    other => blocks.push(other.clone()),
                }
            }
            *message = message.clone().with_blocks(blocks);
        }
        Ok(Cow::Owned(rebuilt))
    }

    /// OpenAI-compatible completion. Builds the request body from
    /// the V1 translation functions, POSTs it via reqwest, and parses
    /// the response via `translation::response`.
    async fn run_openai_complete(
        &self,
        request: &AgentRequest,
    ) -> Result<AgentResponse, ProviderAdapterError> {
        let cfg = self.openai_config()?;
        let url = format!("{}/chat/completions", cfg.base_url().trim_end_matches('/'));

        let resolved = self.resolve_file_sources(request)?;
        let messages = translation::request::to_openai_messages(&resolved)?;
        let tools = translation::tools::to_openai_tools(request.tools());
        let params = sampling_params(request);
        let mut body = serde_json::Map::new();
        body.insert("model".into(), json!(request.model().as_str()));
        body.insert("messages".into(), json!(messages));
        body.insert("tools".into(), json!(tools));
        params.apply_openai(&mut body);

        let resp = self
            .http
            .post(&url)
            .bearer_auth(cfg.api_key())
            .json(&Value::Object(body))
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        translation::response::from_openai_response(&value)
    }

    /// Anthropic completion. Mirrors `run_openai_complete` but
    /// splits `system` out of the messages array and sets a
    /// `max_tokens` default of 4096 (overridable via
    /// `request.options().get("max_tokens")`).
    async fn run_anthropic_complete(
        &self,
        request: &AgentRequest,
    ) -> Result<AgentResponse, ProviderAdapterError> {
        let cfg = self.anthropic_config()?;
        let base = cfg.base_url().unwrap_or("https://api.anthropic.com");
        let url = format!("{}/v1/messages", base.trim_end_matches('/'));

        let cache_control = translation::params::cache_control_enabled(request.options());
        let resolved = self.resolve_file_sources(request)?;
        let (system, messages) =
            translation::request::to_anthropic_messages(&resolved, cache_control)?;
        let tools = translation::tools::to_anthropic_tools(request.tools(), cache_control);
        let params = sampling_params(request);
        let mut body = serde_json::Map::new();
        body.insert("model".into(), json!(request.model().as_str()));
        body.insert("messages".into(), json!(messages));
        body.insert("tools".into(), json!(tools));
        body.insert(
            "max_tokens".into(),
            json!(params.max_tokens.unwrap_or(4096)),
        );
        params.apply_anthropic(&mut body);
        // Extended thinking: enabled when the request opts in, with the
        // budget from options (default 4096).
        if let Some(budget) = translation::params::reasoning_budget(request.options()) {
            body.insert(
                "thinking".into(),
                json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
        if let Some(sys) = system {
            body.insert("system".into(), json!(sys));
        }

        let resp = self
            .http
            .post(&url)
            .header("x-api-key", cfg.api_key())
            .header("anthropic-version", "2023-06-01")
            .json(&Value::Object(body))
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        translation::response::from_anthropic_response(&value)
    }

    /// OpenAI Responses completion. Builds the request body from the V1
    /// translation functions: system text goes into `instructions`,
    /// the rest into the `input` array. `prompt_cache_key` is forwarded
    /// from `request.options()` when present. The full input array is
    /// resent on every request — no `previous_response_id` chaining.
    async fn run_openai_responses_complete(
        &self,
        request: &AgentRequest,
    ) -> Result<AgentResponse, ProviderAdapterError> {
        let cfg = self.responses_config()?;
        let url = format!("{}/responses", cfg.base_url().trim_end_matches('/'));

        let resolved = self.resolve_file_sources(request)?;
        let instructions = translation::request::to_responses_instructions(&resolved);
        let input = translation::request::to_responses_input(&resolved, instructions.as_deref())?;
        let tools = translation::tools::to_responses_tools(request.tools());
        let prompt_cache_key = request
            .options()
            .get("prompt_cache_key")
            .and_then(|v| v.as_str());
        let params = sampling_params(request);
        let mut body = serde_json::json!({
            "model": request.model().as_str(),
            "input": input,
            "tools": tools,
        });
        if let Some(sys) = instructions {
            body["instructions"] = serde_json::json!(sys);
        }
        if let Some(key) = prompt_cache_key {
            body["prompt_cache_key"] = serde_json::json!(key);
        }
        // Request the reasoning summary so the stream can surface
        // thinking progress (PV-04). Encrypted reasoning content is
        // deliberately not requested — V2 displays the summary only and
        // never replays encrypted reasoning.
        if translation::params::reasoning_enabled(request.options()) {
            body["include"] = serde_json::json!(["reasoning.summary_text"]);
        }
        params.apply_responses(body.as_object_mut().expect("body is an object"));

        let resp = self
            .http
            .post(&url)
            .bearer_auth(cfg.api_key())
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        translation::response::from_responses_response(&value)
    }

    /// OpenAI-compatible model listing. Hits `/models` relative to the
    /// configured base URL and stamps the configured provider name on
    /// every entry. Shared by the chat-completions and Responses
    /// protocols.
    async fn run_openai_list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        let (base_url, api_key) = match &self.config {
            BackendConfig::OpenAiChatCompletions(cfg) => (cfg.base_url(), cfg.api_key()),
            BackendConfig::OpenAiResponses(cfg) => (cfg.base_url(), cfg.api_key()),
            BackendConfig::AnthropicMessages(_) => {
                return Err(ProviderAdapterError::configuration(
                    "expected an openai protocol, got anthropic_messages",
                ));
            }
        };
        let url = format!("{}/models", base_url.trim_end_matches('/'));

        let resp = self
            .http
            .get(&url)
            .bearer_auth(api_key)
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        let models = translation::listing::from_openai_models(&value)?;
        Ok(models
            .into_iter()
            .map(|m| m.with_provider(self.name.clone()))
            .collect())
    }

    /// Anthropic model listing. Hits `/v1/models` and stamps the
    /// configured provider name on every entry.
    async fn run_anthropic_list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        let cfg = self.anthropic_config()?;
        let base = cfg.base_url().unwrap_or("https://api.anthropic.com");
        let url = format!("{}/v1/models", base.trim_end_matches('/'));

        let resp = self
            .http
            .get(&url)
            .header("x-api-key", cfg.api_key())
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let value = response_json_or_error(resp).await?;
        let models = translation::listing::from_anthropic_models(&value)?;
        Ok(models
            .into_iter()
            .map(|m| m.with_provider(self.name.clone()))
            .collect())
    }

    /// OpenAI-compatible streaming. Sends `stream: true` and returns a
    /// `ReqwestSseStream` that yields `AgentStreamEvent` values.
    async fn run_openai_stream(
        &self,
        request: &AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        let cfg = self.openai_config()?;
        let url = format!("{}/chat/completions", cfg.base_url().trim_end_matches('/'));

        let resolved = self.resolve_file_sources(request)?;
        let messages = translation::request::to_openai_messages(&resolved)?;
        let tools = translation::tools::to_openai_tools(request.tools());
        let params = sampling_params(request);
        let mut body = serde_json::Map::new();
        body.insert("model".into(), json!(request.model().as_str()));
        body.insert("messages".into(), json!(messages));
        body.insert("tools".into(), json!(tools));
        body.insert("stream".into(), json!(true));
        params.apply_openai(&mut body);

        let resp = self
            .stream_http
            .post(&url)
            .bearer_auth(cfg.api_key())
            .json(&Value::Object(body))
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let resp = check_stream_status(resp).await?;
        Ok(Box::new(ReqwestSseStream::new(resp, StreamKind::OpenAi)))
    }

    /// Anthropic streaming. Sends `stream: true` and returns a
    /// `ReqwestSseStream` that yields `AgentStreamEvent` values.
    async fn run_anthropic_stream(
        &self,
        request: &AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        let cfg = self.anthropic_config()?;
        let base = cfg.base_url().unwrap_or("https://api.anthropic.com");
        let url = format!("{}/v1/messages", base.trim_end_matches('/'));

        let cache_control = translation::params::cache_control_enabled(request.options());
        let resolved = self.resolve_file_sources(request)?;
        let (system, messages) =
            translation::request::to_anthropic_messages(&resolved, cache_control)?;
        let tools = translation::tools::to_anthropic_tools(request.tools(), cache_control);
        let params = sampling_params(request);
        let mut body = serde_json::Map::new();
        body.insert("model".into(), json!(request.model().as_str()));
        body.insert("messages".into(), json!(messages));
        body.insert("tools".into(), json!(tools));
        body.insert(
            "max_tokens".into(),
            json!(params.max_tokens.unwrap_or(4096)),
        );
        body.insert("stream".into(), json!(true));
        params.apply_anthropic(&mut body);
        // Extended thinking: enabled when the request opts in, with the
        // budget from options (default 4096).
        if let Some(budget) = translation::params::reasoning_budget(request.options()) {
            body.insert(
                "thinking".into(),
                json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
        if let Some(sys) = system {
            body.insert("system".into(), json!(sys));
        }

        let resp = self
            .stream_http
            .post(&url)
            .header("x-api-key", cfg.api_key())
            .header("anthropic-version", "2023-06-01")
            .json(&Value::Object(body))
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let resp = check_stream_status(resp).await?;
        Ok(Box::new(ReqwestSseStream::new(resp, StreamKind::Anthropic)))
    }

    /// OpenAI Responses streaming (PV-01b). Sends `stream: true` with
    /// the full input array and returns a `ReqwestSseStream` that yields
    /// `AgentStreamEvent` values. The Responses API resends the full
    /// input array on every request — no `previous_response_id` chaining.
    async fn run_responses_stream(
        &self,
        request: &AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        let cfg = self.responses_config()?;
        let url = format!("{}/responses", cfg.base_url().trim_end_matches('/'));

        let resolved = self.resolve_file_sources(request)?;
        let instructions = translation::request::to_responses_instructions(&resolved);
        let input = translation::request::to_responses_input(&resolved, instructions.as_deref())?;
        let tools = translation::tools::to_responses_tools(request.tools());
        let prompt_cache_key = request
            .options()
            .get("prompt_cache_key")
            .and_then(|v| v.as_str());
        let params = sampling_params(request);
        let mut body = serde_json::json!({
            "model": request.model().as_str(),
            "input": input,
            "tools": tools,
            "stream": true,
        });
        if let Some(sys) = instructions {
            body["instructions"] = serde_json::json!(sys);
        }
        if let Some(key) = prompt_cache_key {
            body["prompt_cache_key"] = serde_json::json!(key);
        }
        // Request the reasoning summary so the stream can surface
        // thinking progress (PV-04). Encrypted reasoning content is
        // deliberately not requested — V2 displays the summary only and
        // never replays encrypted reasoning.
        if translation::params::reasoning_enabled(request.options()) {
            body["include"] = serde_json::json!(["reasoning.summary_text"]);
        }
        params.apply_responses(body.as_object_mut().expect("body is an object"));

        let resp = self
            .stream_http
            .post(&url)
            .bearer_auth(cfg.api_key())
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

        let resp = check_stream_status(resp).await?;
        Ok(Box::new(ReqwestSseStream::new(resp, StreamKind::Responses)))
    }
}

/// Read sampling params from a request, stripping keys that reasoning
/// models reject when reasoning is enabled.
fn sampling_params(request: &AgentRequest) -> translation::params::SamplingParams {
    let (params, stripped) =
        translation::params::from_options_with_applicability(request.options());
    if !stripped.is_empty() {
        tracing::warn!(
            keys = ?stripped,
            "stripped sampling parameters that reasoning models reject"
        );
    }
    params
}

/// Cap on honoring an upstream `Retry-After` hint (R2-03): a
/// misconfigured or hostile proxy can return absurd values (e.g.
/// `99999999` seconds), and sleeping years inside a retry loop is worse
/// than retrying early. The local exponential backoff still applies.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

/// Parse a `Retry-After` response header into a delay. Supports the
/// seconds form (RFC 9110); HTTP-date values are not parsed and fall
/// back to a conservative 10s (AC-05). The delay is capped at
/// [`MAX_RETRY_AFTER`] (R2-03).
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let value = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    if let Ok(secs) = value.parse::<u64>() {
        return Some(Duration::from_secs(secs).min(MAX_RETRY_AFTER));
    }
    Some(Duration::from_secs(10))
}

/// Parse a reqwest response into a `serde_json::Value`, mapping non-2xx
/// status codes to `ProviderAdapterError::Api`.
async fn response_json_or_error(resp: reqwest::Response) -> Result<Value, ProviderAdapterError> {
    let status = resp.status();
    let retry_after = parse_retry_after(resp.headers());
    let text = resp
        .text()
        .await
        .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

    if status.is_success() {
        serde_json::from_str(&text)
            .map_err(|e| ProviderAdapterError::serialization(format!("response json: {e}")))
    } else {
        Err(ProviderAdapterError::api(status.as_u16().to_string(), text)
            .with_retry_after(retry_after))
    }
}

/// Check a streaming response for non-2xx status. Consumes the response
/// on error so the caller gets the error body.
async fn check_stream_status(
    resp: reqwest::Response,
) -> Result<reqwest::Response, ProviderAdapterError> {
    let status = resp.status();
    if status.is_success() {
        Ok(resp)
    } else {
        let text = resp
            .text()
            .await
            .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;
        Err(ProviderAdapterError::api(status.as_u16().to_string(), text))
    }
}

// ------------------------------------------------------------------
// Streaming adapter
// ------------------------------------------------------------------

/// Which provider format the SSE stream carries.
#[derive(Debug, Clone, Copy)]
enum StreamKind {
    OpenAi,
    Anthropic,
    Responses,
}

/// An `AgentStream` backed by a reqwest SSE response.
///
/// Reads chunks from the HTTP response, feeds them through the SSE
/// parser, and routes parsed events through the protocol-layer
/// accumulators in `reimagine_ai_protocol::translation::streaming` —
/// the single implementation of streaming delta translation in the
/// workspace. This struct owns only transport concerns: HTTP reads,
/// incremental UTF-8 decoding (`feed_sse_bytes`, AC-03), and terminal
/// `Done` bookkeeping (`[DONE]` / EOF, AC-01/AC-06).
struct ReqwestSseStream {
    response: reqwest::Response,
    parser: SseParser,
    kind: StreamKind,
    /// Events produced by the current chunk but not yet yielded.
    pending: std::collections::VecDeque<AgentStreamEvent>,
    /// OpenAI Chat Completions delta accumulator.
    openai: translation::streaming::OpenAiStreamAccumulator,
    /// Anthropic Messages delta accumulator.
    anthropic: translation::streaming::AnthropicStreamAccumulator,
    /// OpenAI Responses delta accumulator.
    responses: translation::streaming::ResponsesStreamAccumulator,
    /// Bytes buffered across chunk boundaries while decoding UTF-8
    /// (AC-03): a multi-byte character split across TCP chunks stays
    /// here until the next chunk completes it.
    byte_buf: Vec<u8>,
    /// Whether the stream is done.
    done: bool,
}

impl ReqwestSseStream {
    fn new(response: reqwest::Response, kind: StreamKind) -> Self {
        Self {
            response,
            parser: SseParser::new(),
            kind,
            pending: std::collections::VecDeque::new(),
            openai: translation::streaming::OpenAiStreamAccumulator::new(),
            anthropic: translation::streaming::AnthropicStreamAccumulator::new(),
            responses: translation::streaming::ResponsesStreamAccumulator::new(),
            byte_buf: Vec::new(),
            done: false,
        }
    }

    /// Process one SSE event, routing it through the protocol-layer
    /// accumulator for this stream's kind and pushing any resulting
    /// `AgentStreamEvent` values into `self.pending`.
    fn process_event(&mut self, event: translation::sse_parser::SseEvent) {
        // OpenAI sends "data: [DONE]" as the terminal event.
        if event.data == "[DONE]" {
            self.done = true;
            // Terminal `Done` with the last finish_reason seen (AC-01):
            // truncation ("length") must reach the loop.
            self.pending.push_back(AgentStreamEvent::Done {
                stop_reason: self.openai.take_finish_reason(),
            });
            return;
        }

        let value: Value = match serde_json::from_str(&event.data) {
            Ok(v) => v,
            Err(e) => {
                // R2-02: a provider error event truncated mid-JSON (common
                // under network flakiness) must not vanish silently — the
                // turn would otherwise end on a misleading `Done`.
                tracing::warn!(
                    error = %e,
                    event_type = ?event.event,
                    "dropping SSE event that failed JSON parsing"
                );
                return;
            }
        };

        let events = match self.kind {
            StreamKind::OpenAi => self.openai.ingest_chunk(&value),
            StreamKind::Anthropic => self.anthropic.ingest_event(event.event.as_deref(), &value),
            StreamKind::Responses => self.responses.ingest_event(event.event.as_deref(), &value),
        };
        for e in events {
            // Terminal events (Anthropic `message_stop`, Responses
            // `response.completed`/`response.failed`) stop the stream.
            if e.is_done() {
                self.done = true;
            }
            self.pending.push_back(e);
        }
    }
}

#[async_trait]
impl AgentStream for ReqwestSseStream {
    async fn next_event(&mut self) -> Option<AgentStreamEvent> {
        loop {
            // Return any pending events from the last chunk.
            if let Some(event) = self.pending.pop_front() {
                return Some(event);
            }

            if self.done {
                return None;
            }

            // Read next chunk from the HTTP response.
            let chunk = match self.response.chunk().await {
                Ok(Some(c)) => c,
                Ok(None) => {
                    // Stream ended without an explicit terminal event.
                    // Feed any incomplete UTF-8 tail lossily, then flush
                    // the SSE parser.
                    if !self.byte_buf.is_empty() {
                        let tail = std::mem::take(&mut self.byte_buf);
                        let text = String::from_utf8_lossy(&tail);
                        for event in self.parser.feed(&text) {
                            self.process_event(event);
                        }
                    }
                    if let Some(final_event) = self.parser.flush() {
                        self.process_event(final_event);
                    }
                    self.done = true;
                    if let Some(event) = self.pending.pop_front() {
                        return Some(event);
                    }
                    // Protocols are expected to terminate with an
                    // explicit `Done` ([DONE], message_stop,
                    // response.completed). When they don't but content
                    // was produced, emit a final `Done` with the last
                    // known stop reason so the loop sees proper
                    // termination; a zero-event stream stays Done-less
                    // and the loop reports it as EMPTY_STREAM (AC-06).
                    let stop_reason = match self.kind {
                        StreamKind::OpenAi => self.openai.take_finish_reason(),
                        StreamKind::Anthropic => self.anthropic.take_stop_reason(),
                        StreamKind::Responses => None,
                    };
                    let had_content = self.openai.has_content()
                        || self.anthropic.has_content()
                        || self.responses.has_content();
                    // An EOF that drops partially assembled tool calls
                    // (no `finish_reason: "tool_calls"`, no terminal
                    // event) is surfaced as a host-visible Warning
                    // before the synthesized `Done` (D-5): the loop
                    // must not stop on it, but hosts should know the
                    // stream ended with incomplete tool calls.
                    if self.openai.has_partial_tool_calls()
                        || self.anthropic.has_partial_tool_calls()
                        || self.responses.has_partial_tool_calls()
                    {
                        self.pending.push_back(AgentStreamEvent::Warning(
                            "stream ended with incomplete tool call(s)".to_string(),
                        ));
                    }
                    if stop_reason.is_some() || had_content {
                        self.pending
                            .push_back(AgentStreamEvent::Done { stop_reason });
                        return self.pending.pop_front();
                    }
                    return None;
                }
                Err(e) => {
                    // Surface the failure as a terminal Error event so
                    // the loop can distinguish a dropped connection from
                    // a clean end (AC-05).
                    self.done = true;
                    return Some(AgentStreamEvent::Error(format!(
                        "agent stream read error: {e}"
                    )));
                }
            };

            // Decode incrementally so multi-byte characters split
            // across chunk boundaries are not corrupted (AC-03).
            let events = feed_sse_bytes(&mut self.parser, &mut self.byte_buf, &chunk);
            for event in events {
                self.process_event(event);
            }
        }
    }
}

/// Feed raw bytes into `parser`, decoding UTF-8 incrementally so a
/// multi-byte character split across TCP chunks is not corrupted
/// (AC-03). An incomplete trailing sequence is kept in `byte_buf`
/// until the next chunk completes it; a corrupted leading byte (one
/// that can never become valid UTF-8) is dropped instead of buffering
/// forever. `parser.feed` is called with the longest valid prefix each
/// pass.
fn feed_sse_bytes(parser: &mut SseParser, byte_buf: &mut Vec<u8>, chunk: &[u8]) -> Vec<SseEvent> {
    byte_buf.extend_from_slice(chunk);
    let mut events = Vec::new();
    loop {
        match std::str::from_utf8(byte_buf) {
            Ok(text) => {
                events.extend(parser.feed(text));
                byte_buf.clear();
                return events;
            }
            Err(err) => {
                let valid = err.valid_up_to();
                if valid > 0 {
                    let text = std::str::from_utf8(&byte_buf[..valid]).expect("valid_up_to prefix");
                    events.extend(parser.feed(text));
                    byte_buf.drain(..valid);
                    continue;
                }
                if err.error_len().is_some() {
                    // A truly invalid byte, not an incomplete trailing
                    // sequence: drop it and keep decoding.
                    byte_buf.drain(..1);
                    continue;
                }
                // Incomplete trailing sequence: wait for the next chunk.
                return events;
            }
        }
    }
}

impl crate::backend_provider::ProviderConfig for OpenAiChatCompletionsConfig {
    fn arc_real_backend(name: ProviderName, config: Self) -> Arc<dyn CompletionBackend> {
        Arc::new(ReqwestBackend::openai_chat_completions(name, config))
    }
    fn arc_real_backend_with_workspace_dir(
        name: ProviderName,
        config: Self,
        workspace_dir: PathBuf,
    ) -> Arc<dyn CompletionBackend> {
        Arc::new(
            ReqwestBackend::openai_chat_completions(name, config).with_workspace_dir(workspace_dir),
        )
    }
    fn base_url(&self) -> Option<&str> {
        Some(self.base_url())
    }
    fn default_model(&self) -> &str {
        self.default_model()
    }
}

impl crate::backend_provider::ProviderConfig for AnthropicMessagesConfig {
    fn arc_real_backend(name: ProviderName, config: Self) -> Arc<dyn CompletionBackend> {
        Arc::new(ReqwestBackend::anthropic_messages(name, config))
    }
    fn arc_real_backend_with_workspace_dir(
        name: ProviderName,
        config: Self,
        workspace_dir: PathBuf,
    ) -> Arc<dyn CompletionBackend> {
        Arc::new(ReqwestBackend::anthropic_messages(name, config).with_workspace_dir(workspace_dir))
    }
    fn base_url(&self) -> Option<&str> {
        self.base_url()
    }
    fn default_model(&self) -> &str {
        self.default_model()
    }
}

impl crate::backend_provider::ProviderConfig for OpenAiResponsesConfig {
    fn arc_real_backend(name: ProviderName, config: Self) -> Arc<dyn CompletionBackend> {
        Arc::new(ReqwestBackend::openai_responses(name, config))
    }
    fn arc_real_backend_with_workspace_dir(
        name: ProviderName,
        config: Self,
        workspace_dir: PathBuf,
    ) -> Arc<dyn CompletionBackend> {
        Arc::new(ReqwestBackend::openai_responses(name, config).with_workspace_dir(workspace_dir))
    }
    fn base_url(&self) -> Option<&str> {
        Some(self.base_url())
    }
    fn default_model(&self) -> &str {
        self.default_model()
    }
}

#[async_trait]
impl CompletionBackend for ReqwestBackend {
    async fn complete(&self, request: AgentRequest) -> Result<AgentResponse, ProviderAdapterError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let result = match &self.config {
                BackendConfig::OpenAiChatCompletions(_) => self.run_openai_complete(&request).await,
                BackendConfig::AnthropicMessages(_) => self.run_anthropic_complete(&request).await,
                BackendConfig::OpenAiResponses(_) => {
                    self.run_openai_responses_complete(&request).await
                }
            };

            match result {
                Ok(resp) => return Ok(resp),
                Err(err) => {
                    if err.is_retryable() && attempt <= MAX_RETRIES {
                        // Prefer the upstream's Retry-After hint, but
                        // never below the local backoff (AC-05).
                        let delay = err
                            .retry_after()
                            .map(|d| d.max(retry_delay(attempt)))
                            .unwrap_or_else(|| retry_delay(attempt));
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }

    async fn stream(
        &self,
        request: AgentRequest,
    ) -> Result<Box<dyn AgentStream>, ProviderAdapterError> {
        // Streaming does not retry — the caller owns the stream lifecycle.
        // Retry is the caller's responsibility for streaming.
        match &self.config {
            BackendConfig::OpenAiChatCompletions(_) => self.run_openai_stream(&request).await,
            BackendConfig::AnthropicMessages(_) => self.run_anthropic_stream(&request).await,
            BackendConfig::OpenAiResponses(_) => self.run_responses_stream(&request).await,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, ProviderAdapterError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let result = match &self.config {
                BackendConfig::OpenAiChatCompletions(_) => self.run_openai_list_models().await,
                BackendConfig::AnthropicMessages(_) => self.run_anthropic_list_models().await,
                BackendConfig::OpenAiResponses(_) => self.run_openai_list_models().await,
            };

            match result {
                Ok(models) => return Ok(models),
                Err(err) => {
                    if err.is_retryable() && attempt <= MAX_RETRIES {
                        let delay = err
                            .retry_after()
                            .map(|d| d.max(retry_delay(attempt)))
                            .unwrap_or_else(|| retry_delay(attempt));
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }
}

/// Backoff delay for retry `attempt` (1-based): exponential base
/// (`RETRY_BASE_DELAY * 2^(attempt-1)`) plus a small randomized jitter
/// so a burst of 429s across hosts does not re-synchronize (AC-05).
fn retry_delay(attempt: u32) -> Duration {
    let base = RETRY_BASE_DELAY.saturating_mul(2u32.pow(attempt - 1));
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 % (RETRY_JITTER_MS + 1))
        .unwrap_or(0);
    base + Duration::from_millis(jitter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_sse_bytes_handles_multibyte_chars_split_across_chunks() {
        // "data: 你好\n\ndata: done\n\n" fed one byte at a time (the
        // worst-case TCP fragmentation). 你好 is 6 UTF-8 bytes; no
        // chunk boundary may corrupt it (AC-03).
        let mut parser = SseParser::new();
        let mut byte_buf = Vec::new();
        let mut events = Vec::new();
        for byte in "data: 你好\n\ndata: done\n\n".as_bytes() {
            events.extend(feed_sse_bytes(&mut parser, &mut byte_buf, &[*byte]));
        }
        assert!(byte_buf.is_empty());
        let texts: Vec<String> = events.iter().map(|e| e.data.clone()).collect();
        assert_eq!(texts, vec!["你好".to_string(), "done".to_string()]);
    }

    #[test]
    fn feed_sse_bytes_buffers_incomplete_trailing_sequence() {
        let mut parser = SseParser::new();
        let mut byte_buf = Vec::new();
        // "data: " + the first 2 of "你"'s 3 UTF-8 bytes — an
        // incomplete trailing sequence.
        let mut partial = b"data: ".to_vec();
        partial.extend_from_slice(&"你".as_bytes()[..2]);
        let events = feed_sse_bytes(&mut parser, &mut byte_buf, &partial);
        assert!(events.is_empty(), "incomplete sequence yields nothing");
        assert_eq!(
            byte_buf,
            &"你".as_bytes()[..2],
            "partial bytes stay buffered"
        );
        // Completing the sequence (the missing 3rd byte + the rest)
        // yields the event.
        let mut rest = "你".as_bytes()[2..].to_vec();
        rest.extend_from_slice("好\n\n".as_bytes());
        let events = feed_sse_bytes(&mut parser, &mut byte_buf, &rest);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "你好");
        assert!(byte_buf.is_empty());
    }

    #[test]
    fn feed_sse_bytes_drops_corrupted_leading_byte_instead_of_deadlocking() {
        // A corrupted multi-byte sequence (E4 BD followed by a non
        // continuation byte) can never decode; the buffered head must
        // be dropped, not buffered forever (AC-03).
        let mut parser = SseParser::new();
        let mut byte_buf = Vec::new();
        let mut corrupted = b"data: ".to_vec();
        corrupted.extend_from_slice(&"你".as_bytes()[..2]);
        corrupted.push(b'E'); // not a continuation byte — E4 BD E5 is invalid
        let events = feed_sse_bytes(&mut parser, &mut byte_buf, &corrupted);
        assert!(events.is_empty());
        assert!(
            byte_buf.is_empty(),
            "corrupted bytes dropped, none buffered"
        );
        // The stream continues to decode after the corrupted byte.
        let events = feed_sse_bytes(&mut parser, &mut byte_buf, b"nd\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "End");
    }

    #[test]
    fn parse_retry_after_seconds_and_http_date() {
        let mut headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), None);
        headers.insert(reqwest::header::RETRY_AFTER, "120".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(60)));
        // HTTP-date form is not parsed; fall back to a conservative 10s.
        headers.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(10)));
    }

    #[test]
    fn parse_retry_after_caps_absurd_values() {
        // R2-03: a misconfigured proxy returning a huge Retry-After must
        // not make the client sleep for years inside a retry loop.
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "99999999".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(MAX_RETRY_AFTER));
    }
}
