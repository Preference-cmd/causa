//! `mcp` — a first-class MCP client.
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
//! input and their effects are host-executed. Hosts should keep the
//! approval policy in a `ToolBatchProcessor` registered in the executor
//! before-processing chain, for dynamic-source and local tools alike.
#![deny(unsafe_code)]

use async_trait::async_trait;
use causa_kernel::{
    ArtifactHint, ArtifactKind, ArtifactStore, CallControl, DynamicToolSource, MediaRef,
    SourceError, ToolCallContext, ToolDefinition, ToolExecutionError, ToolOutput,
    ToolResultPayload, ToolResultStatus,
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
            .map(|tool| {
                ToolDefinition::new(
                    self.namespaced(tool.name.as_ref()),
                    tool.description.map(|d| d.to_string()).unwrap_or_default(),
                    serde_json::Value::Object((*tool.input_schema).clone()),
                )
            })
            .collect())
    }

    async fn invoke(
        &self,
        call: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        Self::invoke_inner(self, call, control, store).await
    }
}

impl McpToolSource {
    /// The shared invoke body: `store` is the host's artifact store for
    /// media ingest — with one, MCP image, audio and binary resource
    /// content is persisted and returned as [`MediaRef`]s on the result;
    /// without one, each degrades to a placeholder line of text.
    async fn invoke_inner(
        &self,
        call: &ToolCallContext,
        control: &CallControl,
        store: Option<&dyn ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        // 1. De-namespace; a foreign call is an executor routing bug.
        let Some(tool_name) = self.denamespace(&call.input.tool_name) else {
            return Err(ToolExecutionError::UnknownTool(
                call.input.tool_name.clone(),
            ));
        };
        let arguments = match &call.input.arguments {
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

        // 3. Translate the MCP result: text becomes the observation;
        //    images, audio and binary resources persist into the host's
        //    store and return as media references, with one line of text
        //    naming each; `is_error` results become Failed outcomes the
        //    model reads. Bytes never enter the text: without a store, or
        //    when the ingest fails, only a placeholder line remains.
        let mut parts: Vec<String> = Vec::new();
        let mut media: Vec<MediaRef> = Vec::new();
        for block in result.content {
            match block {
                ContentBlock::Text(t) => parts.push(t.text),
                ContentBlock::Image(image) => {
                    let ingest =
                        Ingest::run(&image.data, &image.mime_type, None, store, call).await;
                    parts.push(ingest.describe("image", &image.mime_type));
                    media.extend(ingest.into_media());
                }
                ContentBlock::Audio(audio) => {
                    let ingest =
                        Ingest::run(&audio.data, &audio.mime_type, None, store, call).await;
                    parts.push(ingest.describe("audio", &audio.mime_type));
                    media.extend(ingest.into_media());
                }
                ContentBlock::Resource(embedded) => match embedded.resource {
                    rmcp::model::ResourceContents::TextResourceContents {
                        uri,
                        text: inner,
                        ..
                    } => parts.push(format!("[resource {uri}]\n{inner}")),
                    rmcp::model::ResourceContents::BlobResourceContents {
                        uri,
                        mime_type,
                        blob,
                        ..
                    } => {
                        let mime_type =
                            mime_type.unwrap_or_else(|| "application/octet-stream".to_owned());
                        let name = resource_name_from_uri(&uri);
                        let ingest = Ingest::run(&blob, &mime_type, name, store, call).await;
                        parts.push(ingest.describe("resource", &format!("{uri} ({mime_type})")));
                        media.extend(ingest.into_media());
                    }
                    other => parts.push(format!(
                        "[resource: {} — unsupported contents]",
                        serialized_field(serde_json::to_value(&other), "uri")
                    )),
                },
                ContentBlock::ResourceLink(link) => parts.push(format!("[link: {}]", link.uri)),
                other => parts.push(format!(
                    "[unsupported content: {}]",
                    serialized_field(serde_json::to_value(&other), "type")
                )),
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
        Ok(
            ToolResultPayload::new(call.call_block_id, status, ToolOutput::new(content))
                .with_media(media),
        )
    }
}

/// The result of storing one piece of MCP binary content (image, audio or
/// blob resource) in the host's artifact store.
enum Ingest {
    /// Persisted; the reference is keyed by the store-assigned artifact id.
    Stored(MediaRef),
    /// No store is wired for this call.
    NoStore,
    /// A store is wired, but the base64 is malformed or the persist failed.
    Failed,
}

impl Ingest {
    /// Decode `data` and persist the raw bytes, tagged with the call's
    /// provenance, as an artifact of `media_type`.
    async fn run(
        data: &str,
        media_type: &str,
        name: Option<String>,
        store: Option<&dyn ArtifactStore>,
        call: &ToolCallContext,
    ) -> Self {
        use base64::Engine as _;
        let Some(store) = store else {
            return Self::NoStore;
        };
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data) else {
            return Self::Failed;
        };
        let hint = ArtifactHint::new(
            call.input.tool_name.clone(),
            call.call_block_id,
            ArtifactKind::Binary,
        );
        match store.persist(&bytes, hint).await {
            Ok(artifact) => {
                let mut media_ref = MediaRef::new(media_type, artifact.id);
                if let Some(name) = name {
                    media_ref = media_ref.with_name(name);
                }
                Self::Stored(media_ref)
            }
            Err(_) => Self::Failed,
        }
    }

    /// The line of observation text standing for the content: `kind` is
    /// `image`, `audio` or `resource`, and `label` names it (its MIME type,
    /// plus the uri for a resource).
    fn describe(&self, kind: &str, label: &str) -> String {
        match self {
            Self::Stored(reference) => {
                format!("[{kind} attached: {label} — asset {}]", reference.reference)
            }
            Self::NoStore => format!("[{kind}: {label} — no media store available]"),
            Self::Failed => format!("[{kind}: {label} — could not be stored]"),
        }
    }

    fn into_media(self) -> Option<MediaRef> {
        match self {
            Self::Stored(reference) => Some(reference),
            Self::NoStore | Self::Failed => None,
        }
    }
}

/// One string field of an MCP value this adapter does not translate, read
/// from its serialized form; `unknown` when absent. Never formats the whole
/// value, which may carry base64 bytes.
fn serialized_field(value: serde_json::Result<serde_json::Value>, key: &str) -> String {
    value
        .ok()
        .and_then(|value| value.get(key)?.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

/// Extracts a file name from a resource URI by stripping query and fragment
/// components and taking the trailing path segment after the last slash.
/// Returns `None` if the segment is empty or no slash is present.
fn resource_name_from_uri(uri: &str) -> Option<String> {
    let path = match uri.find(['?', '#']) {
        Some(idx) => &uri[..idx],
        None => uri,
    };
    let idx = path.rfind('/')?;
    let segment = &path[idx + 1..];
    (!segment.is_empty()).then(|| segment.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_name_from_uri_extracts_name_or_none() {
        assert_eq!(
            resource_name_from_uri("file:///memo.pdf").as_deref(),
            Some("memo.pdf")
        );
        assert_eq!(
            resource_name_from_uri("https://h/a/b.pdf?x=1#y").as_deref(),
            Some("b.pdf")
        );
        assert_eq!(resource_name_from_uri("mem://"), None);
        assert_eq!(resource_name_from_uri("file:///memo/"), None);
        assert_eq!(
            resource_name_from_uri("https://example.com/dir/?foo=1"),
            None
        );
    }
}
