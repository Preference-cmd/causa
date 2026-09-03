//! `mcp` — first-class MCP client for the agent-stack (Slice 10).
//!
//! Wraps the official Rust SDK [`rmcp`](https://docs.rs/rmcp) so external
//! MCP servers' tools enter the kernel's `ToolSurface` next to local Rust
//! tools. The module depends on `causa-kernel` alone (plus
//! rmcp) and exposes only kernel port vocabulary — the same layering as
//! `causa-provider` implementing `ModelGateway`.
//!
//! # Layout
//!
//! - [`McpToolSource`] implements the kernel's `DynamicToolSource` port
//!   over stdio (child process), Streamable HTTP, or an arbitrary
//!   in-process I/O pair.
//! - Tools are namespaced `mcp_{server_id}_{tool}` so identically-named
//!   tools on different servers cannot collide.
//! - `tools/list_changed` notifications bump the source's version, which
//!   the executor-side cache in `causa-runtime` keys on.
//!
//! # Connecting
//!
//! Local MCP server as a child process (stdio):
//!
//! ```ignore
//! use causa_extension::McpToolSource;
//!
//! let mut command = tokio::process::Command::new("uvx");
//! command.args(["mcp-server-fetch"]);
//! let source = McpToolSource::connect_stdio("fetch", command).await?;
//!
//! // With the executor (causa-runtime): the tools enter the turn's
//! // ToolSurface under the `mcp_fetch_*` namespace, cached until the
//! // server notifies `tools/list_changed`.
//! executor.register_dynamic(std::sync::Arc::new(source))?;
//! ```
//!
//! Remote MCP server (Streamable HTTP) with a static bearer token:
//!
//! ```ignore
//! let source = McpToolSource::connect_http(
//!     "remote",
//!     "https://mcp.example.com/mcp",
//!     Some(std::env::var("MCP_TOKEN")?),
//! ).await?;
//! executor.register_dynamic(std::sync::Arc::new(source))?;
//! ```
//!
//! # Namespacing
//!
//! Every tool is exposed as `mcp_{server_id}_{tool}`, so identically-named
//! tools on different servers cannot collide. The executor routes by
//! listing membership (a source is dispatched exactly the names it
//! advertised in the surface); this source de-namespaces the call before
//! invoking and rejects names outside its namespace.
//!
//! # Safety notes
//!
//! External tools are an injection surface: their output is model-visible
//! input and their effects are host-executed. Hosts should keep the Slice
//! 7 approval gate (`TurnInteraction::decide_batch`) in front of
//! dynamic-source batches, exactly as for local tools.
#![deny(unsafe_code)]

use async_trait::async_trait;
use causa_kernel::{
    CallControl, DynamicToolSource, SourceError, ToolCallPayload, ToolDefinition,
    ToolExecutionError, ToolExecutionOutcome, ToolOutput, ToolResultPayload, ToolResultStatus,
};
use rmcp::handler::client::ClientHandler;
use rmcp::model::ContentBlock;
use rmcp::service::{NotificationContext, RoleClient, RunningService, ServiceExt};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Client handler that counts `tools/list_changed` notifications — the
/// change signal the executor-side cache keys on.
#[derive(Debug, Default)]
struct McpClientHandler {
    list_version: Arc<AtomicU64>,
}

impl ClientHandler for McpClientHandler {
    async fn on_tool_list_changed(&self, _context: NotificationContext<RoleClient>) {
        self.list_version.fetch_add(1, Ordering::SeqCst);
    }
}

/// MCP server catalog as a kernel [`DynamicToolSource`]. One instance per
/// server connection, shared behind the executor's `Arc`.
///
/// `server_id` becomes part of every model-facing tool name
/// (`mcp_{server_id}_{tool}`) — keep it non-empty `[A-Za-z0-9_-]` so names
/// stay provider-callable (debug builds assert this).
///
/// Error classification follows the gateway discipline: connection
/// failures and timeouts are `Unavailable`/`TimedOut` (transient);
/// protocol violations are `Protocol` (permanent).
///
/// Listing caches live executor-side (in `causa-runtime`, keyed on
/// [`DynamicToolSource::version`]); this source re-lists on every `list()`
/// call and stays a thin transport adapter.
pub struct McpToolSource {
    server_id: String,
    namespace: String,
    service: RunningService<RoleClient, McpClientHandler>,
    list_version: Arc<AtomicU64>,
}

impl std::fmt::Debug for McpToolSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpToolSource")
            .field("server_id", &self.server_id)
            .field("list_version", &self.list_version.load(Ordering::SeqCst))
            .finish()
    }
}

impl McpToolSource {
    fn build(
        server_id: String,
        service: RunningService<RoleClient, McpClientHandler>,
        list_version: Arc<AtomicU64>,
    ) -> Self {
        debug_assert!(
            !server_id.is_empty()
                && server_id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "server_id must be non-empty [A-Za-z0-9_-], got: {server_id}"
        );
        Self {
            namespace: format!("mcp_{}_", server_id),
            server_id,
            service,
            list_version,
        }
    }

    /// Connect to a local MCP server spawned as a child process (stdio
    /// transport). `command` is the server launch command; stdio frames
    /// the JSON-RPC session.
    pub async fn connect_stdio(
        server_id: impl Into<String>,
        command: tokio::process::Command,
    ) -> Result<Self, SourceError> {
        let server_id = server_id.into();
        let list_version = Arc::new(AtomicU64::new(0));
        let transport = rmcp::transport::TokioChildProcess::new(command)
            .map_err(|e| SourceError::Unavailable(e.to_string()))?;
        let handler = McpClientHandler {
            list_version: list_version.clone(),
        };
        let service = handler
            .serve(transport)
            .await
            .map_err(|e| SourceError::Unavailable(e.to_string()))?;
        Ok(Self::build(server_id, service, list_version))
    }

    /// Connect to a remote MCP server over Streamable HTTP. `auth_token`
    /// (v1 static authorization) is sent as a Bearer header on every
    /// request when present; OAuth flows stay out of scope until the
    /// rmcp `auth` feature is adopted here.
    pub async fn connect_http(
        server_id: impl Into<String>,
        uri: impl Into<String>,
        auth_token: Option<String>,
    ) -> Result<Self, SourceError> {
        let server_id = server_id.into();
        let list_version = Arc::new(AtomicU64::new(0));
        let mut config = StreamableHttpClientTransportConfig::with_uri(uri.into());
        if let Some(token) = auth_token {
            config = config.auth_header(token);
        }
        let transport = StreamableHttpClientTransport::from_config(config);
        let handler = McpClientHandler {
            list_version: list_version.clone(),
        };
        let service = handler
            .serve(transport)
            .await
            .map_err(|e| SourceError::Unavailable(e.to_string()))?;
        Ok(Self::build(server_id, service, list_version))
    }

    /// Connect over an arbitrary byte stream — the in-process server
    /// path (tests, embedded servers). `io` is the client side of a
    /// connected pipe: writes go to the server, reads carry the server's
    /// output.
    pub async fn connect_io<I>(server_id: impl Into<String>, io: I) -> Result<Self, SourceError>
    where
        I: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + 'static,
    {
        let server_id = server_id.into();
        let list_version = Arc::new(AtomicU64::new(0));
        let handler = McpClientHandler {
            list_version: list_version.clone(),
        };
        let service = handler
            .serve(io)
            .await
            .map_err(|e| SourceError::Unavailable(e.to_string()))?;
        Ok(Self::build(server_id, service, list_version))
    }

    /// Shut the underlying session down. Best-effort: the rmcp
    /// `RunningService::cancel` handshake can hang if the transport's
    /// session-cleanup DELETE races with the service being torn down
    /// (notably the axum `StreamableHttpService` test fixture wrapped in
    /// a `capture_auth` middleware — see `tests/integration.rs`). Bound
    /// the wait so callers never hang — the service is dropped on
    /// timeout and the transport closes with it.
    pub async fn close(self) -> Result<(), SourceError> {
        // Move the service so a timeout still drops it (closing the
        // transport) instead of keeping it alive in the caller's stack.
        let server_id = self.server_id.clone();
        let service = self.service;
        match tokio::time::timeout(std::time::Duration::from_secs(2), service.cancel()).await {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(SourceError::Unavailable(e.to_string())),
            Err(_) => {
                tracing::warn!(server_id = %server_id, "MCP close timed out; dropping service without session cleanup");
                Ok(())
            }
        }
    }

    /// The tool name this source answers to for `tool`.
    pub fn namespaced(&self, tool: &str) -> String {
        format!("{}{}", self.namespace, tool)
    }

    /// Strip the namespace prefix; `None` when the call does not belong
    /// to this server.
    fn denamespace<'a>(&self, tool_name: &'a str) -> Option<&'a str> {
        tool_name.strip_prefix(&self.namespace)
    }

    fn map_service_error(e: rmcp::ServiceError) -> SourceError {
        match e {
            rmcp::ServiceError::Timeout { .. }
            | rmcp::ServiceError::TransportSend(_)
            | rmcp::ServiceError::TransportClosed => SourceError::Unavailable(e.to_string()),
            _ => SourceError::Protocol(e.to_string()),
        }
    }

    fn map_invoke_error(e: rmcp::ServiceError) -> ToolExecutionError {
        match e {
            rmcp::ServiceError::Timeout { .. } => ToolExecutionError::TimedOut,
            rmcp::ServiceError::Cancelled { .. } => ToolExecutionError::Cancelled,
            rmcp::ServiceError::TransportSend(_) | rmcp::ServiceError::TransportClosed => {
                ToolExecutionError::Unavailable(e.to_string())
            }
            _ => ToolExecutionError::Protocol(e.to_string()),
        }
    }
}

#[async_trait]
impl DynamicToolSource for McpToolSource {
    fn id(&self) -> &str {
        &self.server_id
    }

    /// Bumped by `tools/list_changed` notifications. The executor
    /// re-lists only when this differs from its cached snapshot.
    fn version(&self) -> u64 {
        self.list_version.load(Ordering::SeqCst)
    }

    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        let listed = self
            .service
            .list_tools(None)
            .await
            .map_err(Self::map_service_error)?;
        Ok(listed
            .tools
            .into_iter()
            .map(|tool| ToolDefinition {
                name: self.namespaced(tool.name.as_ref()),
                description: tool.description.map(|d| d.to_string()).unwrap_or_default(),
                parameters: serde_json::Value::Object((*tool.input_schema).clone()),
            })
            .collect())
    }

    async fn invoke(
        &self,
        call: &ToolCallPayload,
        control: &CallControl,
    ) -> Result<ToolExecutionOutcome, ToolExecutionError> {
        // 1. De-namespace; a foreign call is an executor routing bug.
        let Some(tool_name) = self.denamespace(&call.tool_name) else {
            return Err(ToolExecutionError::UnknownTool(call.tool_name.clone()));
        };
        let arguments = match &call.arguments {
            serde_json::Value::Object(map) => Some(map.clone()),
            serde_json::Value::Null => None,
            other => {
                return Err(ToolExecutionError::Protocol(format!(
                    "MCP tool arguments must be an object, got: {other}"
                )));
            }
        };
        let mut params = rmcp::model::CallToolRequestParams::new(Cow::Owned(tool_name.to_string()));
        if let Some(args) = arguments {
            params = params.with_arguments(args);
        }

        // 2. Race the call against the shared cancellation token and the
        //    effective call deadline — the same control-plane semantics
        //    local tools get.
        let deadline = control.deadline();
        let token = control.cancellation_token().clone();
        let result = tokio::select! {
            biased;
            _ = token.cancelled() => return Err(ToolExecutionError::Cancelled),
            r = async {
                match deadline {
                    Some(deadline) => {
                        tokio::time::timeout_at(deadline.into(), self.service.call_tool(params))
                            .await
                            .map_err(|_| ToolExecutionError::TimedOut)?
                            .map_err(Self::map_invoke_error)
                    }
                    None => self
                        .service
                        .call_tool(params)
                        .await
                        .map_err(Self::map_invoke_error),
                }
            } => r?,
        };

        // 3. Translate the MCP result: text becomes the observation,
        //    images become placeholders until Slice 6.5's media path, and
        //    `is_error` results become Failed outcomes the model reads.
        let mut parts: Vec<String> = Vec::new();
        for block in result.content {
            match block {
                ContentBlock::Text(t) => parts.push(t.text),
                ContentBlock::Image(image) => parts.push(format!(
                    "[image: {} mime, {} base64 chars — media passthrough lands with Slice 6.5]",
                    image.mime_type,
                    image.data.len()
                )),
                ContentBlock::Audio(audio) => {
                    parts.push(format!("[audio: {} mime]", audio.mime_type))
                }
                ContentBlock::Resource(embedded) => match &embedded.resource {
                    rmcp::model::ResourceContents::TextResourceContents {
                        uri,
                        text: inner,
                        ..
                    } => parts.push(format!("[resource {uri}]\n{inner}")),
                    other => parts.push(format!("[resource: {other:?}]")),
                },
                ContentBlock::ResourceLink(link) => parts.push(format!("[link: {}]", link.uri)),
                other => parts.push(format!("[unhandled content: {other:?}]")),
            }
        }
        let text = parts.join("\n");
        let content = match result.structured_content {
            Some(value) => value,
            None => serde_json::Value::String(text),
        };
        let status = if result.is_error.unwrap_or(false) {
            ToolResultStatus::Failed
        } else {
            ToolResultStatus::Succeeded
        };
        Ok(ToolExecutionOutcome::new(ToolResultPayload {
            call_id: call.call_id.clone(),
            status,
            output: ToolOutput::new(content),
        }))
    }
}
