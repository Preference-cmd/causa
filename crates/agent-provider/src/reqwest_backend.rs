//! Reqwest-backed completion backend.
//!
//! `ReqwestBackend` is the production `CompletionBackend` used by the
//! two concrete adapters. It makes direct HTTP calls via `reqwest::Client`
//! to the OpenAI-compatible and Anthropic APIs. We deliberately do NOT
//! use any agent loop or tool execution framework — Reimagine owns the
//! agent loop, tool execution, and tool policy.
//!
//! The backend is not used in V1 unit tests (which always substitute
//! `FakeCompletionBackend`). It is provided so app-host can construct
//! the production provider with `build_provider(config)` and the
//! adapters will route to a working real backend.

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
use reimagine_ai_protocol::{
    AnthropicMessagesConfig, CompletionBackend, OpenAiChatCompletionsConfig, OpenAiResponsesConfig,
    Protocol, ProviderAdapterError,
};
use reimagine_ai_protocol::translation::sse_parser::SseParser;
use serde_json::{Value, json};

/// Maximum number of retries for transient HTTP errors.
const MAX_RETRIES: u32 = 3;
/// Base delay for exponential backoff (doubled each retry).
const RETRY_BASE_DELAY: Duration = Duration::from_millis(500);

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
    /// Construct an OpenAI-compatible backend with a default
    /// `reqwest::Client`.
    pub fn openai_chat_completions(name: ProviderName, cfg: OpenAiChatCompletionsConfig) -> Self {
        Self::openai_chat_completions_with_http_client(name, cfg, reqwest::Client::new())
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
            http,
            workspace_dir: None,
        }
    }

    /// Construct an Anthropic backend with a default `reqwest::Client`.
    pub fn anthropic_messages(name: ProviderName, cfg: AnthropicMessagesConfig) -> Self {
        Self::anthropic_messages_with_http_client(name, cfg, reqwest::Client::new())
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
            http,
            workspace_dir: None,
        }
    }

    /// Construct an OpenAI Responses backend with a default
    /// `reqwest::Client`.
    pub fn openai_responses(name: ProviderName, cfg: OpenAiResponsesConfig) -> Self {
        Self::openai_responses_with_http_client(name, cfg, reqwest::Client::new())
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
            .http
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
            .http
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
            .http
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

/// Parse a reqwest response into a `serde_json::Value`, mapping non-2xx
/// status codes to `ProviderAdapterError::Api`.
async fn response_json_or_error(resp: reqwest::Response) -> Result<Value, ProviderAdapterError> {
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| ProviderAdapterError::transport(e.to_string()))?;

    if status.is_success() {
        serde_json::from_str(&text)
            .map_err(|e| ProviderAdapterError::serialization(format!("response json: {e}")))
    } else {
        Err(ProviderAdapterError::api(status.as_u16().to_string(), text))
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
/// parser, and yields `AgentStreamEvent` values produced by the
/// provider-specific accumulator.
struct ReqwestSseStream {
    response: reqwest::Response,
    parser: SseParser,
    kind: StreamKind,
    /// Events produced by the current chunk but not yet yielded.
    pending: std::collections::VecDeque<AgentStreamEvent>,
    /// Accumulated text for the OpenAI path.
    openai_text: String,
    /// Accumulated tool calls for the OpenAI path.
    openai_tool_calls: Vec<OpenAiPartialToolCall>,
    /// Accumulated text for the Anthropic path.
    anthropic_text: String,
    /// Accumulated tool calls for the Anthropic path.
    anthropic_tool_calls: Vec<AnthropicPartialToolCall>,
    /// Accumulated tool calls for the OpenAI Responses path, keyed by
    /// `output_index`. Arguments deltas arrive base64-encoded and are
    /// decoded before accumulation.
    responses_tool_calls: Vec<ResponsesPartialToolCall>,
    /// Input-side Anthropic usage captured from `message_start`
    /// (input_tokens + cache fields); merged into the report emitted on
    /// `message_delta` (output_tokens).
    anthropic_input: Option<u64>,
    anthropic_cache_creation: Option<u64>,
    anthropic_cache_read: Option<u64>,
    /// Whether the stream is done.
    done: bool,
}

#[derive(Debug, Default, Clone)]
struct OpenAiPartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Default, Clone)]
struct AnthropicPartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Partial tool call for the OpenAI Responses path. The Responses API
/// streams only base64-encoded argument fragments in
/// `response.function_call_arguments.delta`; id and name arrive with the
/// complete item in `response.output_item.done`.
#[derive(Debug, Default, Clone)]
struct ResponsesPartialToolCall {
    arguments: String,
}

impl ReqwestSseStream {
    fn new(response: reqwest::Response, kind: StreamKind) -> Self {
        Self {
            response,
            parser: SseParser::new(),
            kind,
            pending: std::collections::VecDeque::new(),
            openai_text: String::new(),
            openai_tool_calls: Vec::new(),
            anthropic_text: String::new(),
            anthropic_tool_calls: Vec::new(),
            responses_tool_calls: Vec::new(),
            anthropic_input: None,
            anthropic_cache_creation: None,
            anthropic_cache_read: None,
            done: false,
        }
    }

    /// Process one SSE event, pushing any resulting `AgentStreamEvent`
    /// values into `self.pending`.
    fn process_event(&mut self, event: translation::sse_parser::SseEvent) {
        // OpenAI sends "data: [DONE]" as the terminal event.
        if event.data == "[DONE]" {
            self.done = true;
            return;
        }

        let value: Value = match serde_json::from_str(&event.data) {
            Ok(v) => v,
            Err(_) => return,
        };

        match self.kind {
            StreamKind::OpenAi => self.process_openai_chunk(&value),
            StreamKind::Anthropic => self.process_anthropic_event(&event.event, &value),
            StreamKind::Responses => self.process_responses_event(&event.event, &value),
        }
    }

    fn process_openai_chunk(&mut self, chunk: &Value) {
        // Extract content delta.
        if let Some(delta_content) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|v| v.as_str())
            && !delta_content.is_empty()
        {
            self.openai_text.push_str(delta_content);
            self.pending
                .push_back(AgentStreamEvent::ContentDelta(delta_content.to_string()));
        }

        // Extract reasoning deltas (o-series / DeepSeek-style
        // `reasoning_content`). Display-only: never accumulated into the
        // assistant message.
        if let Some(reasoning) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("reasoning_content"))
            .and_then(|v| v.as_str())
            && !reasoning.is_empty()
        {
            self.pending
                .push_back(AgentStreamEvent::ReasoningDelta(reasoning.to_string()));
        }

        // Extract tool call deltas.
        if let Some(tool_calls) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("tool_calls"))
            .and_then(|v| v.as_array())
        {
            for (i, call) in tool_calls.iter().enumerate() {
                let index = call
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .map(|i| i as usize)
                    .unwrap_or(i);
                while self.openai_tool_calls.len() <= index {
                    self.openai_tool_calls
                        .push(OpenAiPartialToolCall::default());
                }
                let entry = &mut self.openai_tool_calls[index];
                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                    entry.id = Some(id.to_string());
                }
                if let Some(name) = call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                {
                    entry.name = Some(name.to_string());
                }
                if let Some(args) = call
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                {
                    entry.arguments.push_str(args);
                }
            }
        }

        // Extract finish reason — flush complete tool calls.
        if let Some(finish_reason) = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("finish_reason"))
            .and_then(|v| v.as_str())
            && finish_reason == "tool_calls"
        {
            self.flush_openai_tool_calls();
        }

        // Extract usage.
        if let Some(usage) = chunk.get("usage") {
            let input = usage.get("prompt_tokens").and_then(|v| v.as_u64());
            let output = usage.get("completion_tokens").and_then(|v| v.as_u64());
            let reasoning = translation::usage::openai_reasoning_tokens(usage);
            let cached = translation::usage::openai_cached_tokens(usage);
            self.pending.push_back(AgentStreamEvent::Usage(
                reimagine_agent_harness::Usage::new(input, output)
                    .with_reasoning_tokens(reasoning)
                    .with_cache_read(cached),
            ));
        }
    }

    fn flush_openai_tool_calls(&mut self) {
        for partial in self.openai_tool_calls.drain(..) {
            if let (Some(id), Some(name)) = (partial.id, partial.name) {
                let arguments = if partial.arguments.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                };
                self.pending
                    .push_back(AgentStreamEvent::ToolCall(reimagine_agent_harness::ToolCall::new(
                        reimagine_agent_harness::ToolCallId::new(id),
                        name,
                        arguments,
                    )));
            }
        }
    }

    fn process_anthropic_event(&mut self, event_type: &Option<String>, data: &Value) {
        let event_type = match event_type.as_deref() {
            Some(t) => t,
            None => return,
        };

        match event_type {
            "content_block_start" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    while self.anthropic_tool_calls.len() <= index {
                        self.anthropic_tool_calls
                            .push(AnthropicPartialToolCall::default());
                    }
                    if let Some(block) = data.get("content_block")
                        && block.get("type").and_then(|v| v.as_str()) == Some("tool_use")
                    {
                        if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                            self.anthropic_tool_calls[index].id = Some(id.to_string());
                        }
                        if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                            self.anthropic_tool_calls[index].name = Some(name.to_string());
                        }
                    }
                }
            }
            "content_block_delta" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    if let Some(delta) = data.get("delta") {
                        match delta.get("type").and_then(|v| v.as_str()) {
                            Some("text_delta") => {
                                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                    self.anthropic_text.push_str(text);
                                    self.pending.push_back(AgentStreamEvent::ContentDelta(
                                        text.to_string(),
                                    ));
                                }
                            }
                            Some("thinking_delta") => {
                                if let Some(text) = delta.get("thinking").and_then(|v| v.as_str())
                                {
                                    self.pending.push_back(AgentStreamEvent::ReasoningDelta(
                                        text.to_string(),
                                    ));
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(partial) =
                                    delta.get("partial_json").and_then(|v| v.as_str())
                                {
                                    while self.anthropic_tool_calls.len() <= index {
                                        self.anthropic_tool_calls
                                            .push(AnthropicPartialToolCall::default());
                                    }
                                    self.anthropic_tool_calls[index].arguments.push_str(partial);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "content_block_stop" => {
                if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                    let index = index as usize;
                    if let Some(partial) = self.anthropic_tool_calls.get_mut(index)
                        && let (Some(id), Some(name)) = (partial.id.clone(), partial.name.clone())
                    {
                        let arguments = if partial.arguments.is_empty() {
                            Value::Null
                        } else {
                            serde_json::from_str(&partial.arguments).unwrap_or(Value::Null)
                        };
                        *partial = AnthropicPartialToolCall::default();
                        self.pending.push_back(AgentStreamEvent::ToolCall(
                            reimagine_agent_harness::ToolCall::new(
                                reimagine_agent_harness::ToolCallId::new(id),
                                name,
                                arguments,
                            ),
                        ));
                    }
                }
            }
            "message_start" => {
                if let Some(usage) = data.get("message").and_then(|m| m.get("usage")) {
                    self.anthropic_input = usage.get("input_tokens").and_then(|v| v.as_u64());
                    self.anthropic_cache_creation = usage
                        .get("cache_creation_input_tokens")
                        .and_then(|v| v.as_u64());
                    self.anthropic_cache_read = usage
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_u64());
                }
            }
            "message_delta" => {
                if let Some(delta) = data.get("delta")
                    && let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str())
                {
                    // Store stop_reason; will be emitted on message_stop.
                    let _ = reason;
                }
                if let Some(usage) = data.get("usage") {
                    let input = self.anthropic_input.or_else(|| {
                        usage.get("input_tokens").and_then(|v| v.as_u64())
                    });
                    let output = usage.get("output_tokens").and_then(|v| v.as_u64());
                    let cache_creation = self.anthropic_cache_creation.or_else(|| {
                        usage.get("cache_creation_input_tokens").and_then(|v| v.as_u64())
                    });
                    let cache_read = self.anthropic_cache_read.or_else(|| {
                        usage.get("cache_read_input_tokens").and_then(|v| v.as_u64())
                    });
                    self.pending
                        .push_back(AgentStreamEvent::Usage(
                            reimagine_agent_harness::Usage::new(input, output)
                                .with_cache_creation(cache_creation)
                                .with_cache_read(cache_read),
                        ));
                }
            }
            "message_stop" => {
                self.done = true;
                self.pending
                    .push_back(AgentStreamEvent::Done { stop_reason: None });
            }
            _ => {}
        }
    }

    /// Process one OpenAI Responses stream event.
    ///
    /// V1 handles the event families the Agent loop consumes:
    /// `response.output_text.delta` (content), the `response.function_call*`
    /// family (tool calls), `response.completed` (usage + done), and
    /// `response.failed` (terminal done without usage). `response.created`,
    /// `response.in_progress`, reasoning deltas, and item bookkeeping
    /// events are ignored.
    fn process_responses_event(&mut self, event_type: &Option<String>, data: &Value) {
        let event_type = match event_type.as_deref() {
            Some(t) => t,
            None => return,
        };

        match event_type {
            "response.output_text.delta" => {
                if let Some(text) = data.get("delta").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    self.pending
                        .push_back(AgentStreamEvent::ContentDelta(text.to_string()));
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let Some(text) = data.get("delta").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    self.pending
                        .push_back(AgentStreamEvent::ReasoningDelta(text.to_string()));
                }
            }
            "response.function_call_arguments.delta" => {
                let Some(delta) = data.get("delta").and_then(|v| v.as_str()) else {
                    return;
                };
                let index = data
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|i| i as usize)
                    .unwrap_or(0);
                while self.responses_tool_calls.len() <= index {
                    self.responses_tool_calls
                        .push(ResponsesPartialToolCall::default());
                }
                // Arguments deltas are base64-encoded JSON fragments;
                // decode before accumulating. Tolerate plain fragments.
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(delta)
                    .ok()
                    .and_then(|bytes| String::from_utf8(bytes).ok())
                    .unwrap_or_else(|| delta.to_string());
                self.responses_tool_calls[index].arguments.push_str(&decoded);
            }
            "response.function_call_arguments.done" => {
                // The provider may deliver the full arguments here; when
                // the accumulated deltas are empty, use this payload.
                let index = data
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|i| i as usize)
                    .unwrap_or(0);
                if let Some(arguments) = data.get("arguments").and_then(|v| v.as_str())
                    && !arguments.is_empty()
                {
                    while self.responses_tool_calls.len() <= index {
                        self.responses_tool_calls
                            .push(ResponsesPartialToolCall::default());
                    }
                    if self.responses_tool_calls[index].arguments.is_empty() {
                        self.responses_tool_calls[index].arguments = arguments.to_string();
                    }
                }
            }
            "response.output_item.done" => {
                let Some(item) = data.get("item") else {
                    return;
                };
                if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
                    return;
                }
                let Some(name) = item.get("name").and_then(|v| v.as_str()) else {
                    return;
                };
                // The streamed item carries the full arguments; prefer the
                // complete payload over accumulated deltas.
                let arguments = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        let index = data
                            .get("output_index")
                            .and_then(|v| v.as_u64())
                            .map(|i| i as usize)
                            .unwrap_or(0);
                        self.responses_tool_calls
                            .get(index)
                            .map(|partial| partial.arguments.clone())
                    })
                    .unwrap_or_else(|| "{}".to_string());
                // The Responses API uses `call_id` as the stable id that
                // tool-result messages reference; fall back to `id`.
                let id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("id").and_then(|v| v.as_str()))
                    .unwrap_or("");
                let arguments_value =
                    serde_json::from_str(&arguments).unwrap_or(Value::Null);
                self.pending
                    .push_back(AgentStreamEvent::ToolCall(
                        reimagine_agent_harness::ToolCall::new(
                            reimagine_agent_harness::ToolCallId::new(id),
                            name.to_string(),
                            arguments_value,
                        ),
                    ));
            }
            "response.compaction" => {
                // Server-side compaction (PV-01b reserved channel,
                // consumed in CM-V2e): the provider replaced earlier
                // items with an opaque compaction item. Informational
                // for the runtime.
                if let Some(item_id) = data.get("item_id").and_then(|v| v.as_str()) {
                    self.pending.push_back(AgentStreamEvent::Compacted {
                        item_id: item_id.to_string(),
                    });
                }
            }
            "response.completed" => {
                if let Some(usage) = data.get("response").and_then(|r| r.get("usage")) {
                    let input = usage.get("input_tokens").and_then(|v| v.as_u64());
                    let output = usage.get("output_tokens").and_then(|v| v.as_u64());
                    let reasoning = usage
                        .get("output_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                        .and_then(|v| v.as_u64());
                    let cached = translation::usage::openai_cached_tokens(usage);
                    self.pending.push_back(AgentStreamEvent::Usage(
                        reimagine_agent_harness::Usage::new(input, output)
                            .with_reasoning_tokens(reasoning)
                            .with_cache_read(cached),
                    ));
                }
                self.done = true;
                self.pending
                    .push_back(AgentStreamEvent::Done { stop_reason: None });
            }
            "response.failed" => {
                self.done = true;
                self.pending
                    .push_back(AgentStreamEvent::Done { stop_reason: None });
            }
            _ => {}
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
                    // Stream ended. Flush any remaining SSE data.
                    if let Some(final_event) = self.parser.flush() {
                        self.process_event(final_event);
                        if let Some(event) = self.pending.pop_front() {
                            return Some(event);
                        }
                    }
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    eprintln!("agent stream read error: {e}");
                    self.done = true;
                    return None;
                }
            };

            // Convert bytes to string and feed to SSE parser.
            let text = String::from_utf8_lossy(&chunk);
            let events = self.parser.feed(&text);

            for event in events {
                self.process_event(event);
            }
        }
    }
}

pub fn arc_real_openai_chat_completions_backend(
    name: ProviderName,
    cfg: OpenAiChatCompletionsConfig,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::openai_chat_completions(name, cfg))
}

pub fn arc_real_openai_chat_completions_backend_with_workspace_dir(
    name: ProviderName,
    cfg: OpenAiChatCompletionsConfig,
    workspace_dir: PathBuf,
) -> Arc<dyn CompletionBackend> {
    Arc::new(
        ReqwestBackend::openai_chat_completions(name, cfg).with_workspace_dir(workspace_dir),
    )
}

pub fn arc_real_openai_chat_completions_backend_with_http_client(
    name: ProviderName,
    cfg: OpenAiChatCompletionsConfig,
    http: reqwest::Client,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::openai_chat_completions_with_http_client(
        name, cfg, http,
    ))
}

pub fn arc_real_anthropic_messages_backend(
    name: ProviderName,
    cfg: AnthropicMessagesConfig,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::anthropic_messages(name, cfg))
}

pub fn arc_real_anthropic_messages_backend_with_workspace_dir(
    name: ProviderName,
    cfg: AnthropicMessagesConfig,
    workspace_dir: PathBuf,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::anthropic_messages(name, cfg).with_workspace_dir(workspace_dir))
}

pub fn arc_real_anthropic_messages_backend_with_http_client(
    name: ProviderName,
    cfg: AnthropicMessagesConfig,
    http: reqwest::Client,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::anthropic_messages_with_http_client(
        name, cfg, http,
    ))
}

pub fn arc_real_openai_responses_backend(
    name: ProviderName,
    cfg: OpenAiResponsesConfig,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::openai_responses(name, cfg))
}

pub fn arc_real_openai_responses_backend_with_workspace_dir(
    name: ProviderName,
    cfg: OpenAiResponsesConfig,
    workspace_dir: PathBuf,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::openai_responses(name, cfg).with_workspace_dir(workspace_dir))
}

pub fn arc_real_openai_responses_backend_with_http_client(
    name: ProviderName,
    cfg: OpenAiResponsesConfig,
    http: reqwest::Client,
) -> Arc<dyn CompletionBackend> {
    Arc::new(ReqwestBackend::openai_responses_with_http_client(
        name, cfg, http,
    ))
}

#[async_trait]
impl CompletionBackend for ReqwestBackend {
    async fn complete(&self, request: AgentRequest) -> Result<AgentResponse, ProviderAdapterError> {
        let mut last_err: Option<ProviderAdapterError> = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = RETRY_BASE_DELAY * 2u32.pow(attempt - 1);
                tokio::time::sleep(delay).await;
            }

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
                    if err.is_retryable() && attempt < MAX_RETRIES {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
            }
        }

        Err(last_err
            .unwrap_or_else(|| ProviderAdapterError::transport("retry exhausted".to_string())))
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
        let mut last_err: Option<ProviderAdapterError> = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = RETRY_BASE_DELAY * 2u32.pow(attempt - 1);
                tokio::time::sleep(delay).await;
            }

            let result = match &self.config {
                BackendConfig::OpenAiChatCompletions(_) => self.run_openai_list_models().await,
                BackendConfig::AnthropicMessages(_) => self.run_anthropic_list_models().await,
                BackendConfig::OpenAiResponses(_) => self.run_openai_list_models().await,
            };

            match result {
                Ok(models) => return Ok(models),
                Err(err) => {
                    if err.is_retryable() && attempt < MAX_RETRIES {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
            }
        }

        Err(last_err
            .unwrap_or_else(|| ProviderAdapterError::transport("retry exhausted".to_string())))
    }
}
