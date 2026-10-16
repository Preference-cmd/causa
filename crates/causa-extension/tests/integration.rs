//! Integration tests: an in-process rmcp server fixture served over a
//! duplex pipe, exercised through `McpToolSource`, the executor's dynamic
//! registry, and a full kernel turn (`TurnRunner` untouched).

use async_trait::async_trait;
use causa_extension::McpToolSource;
use causa_kernel::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, BlockContent, BlockId, CallControl,
    CancellationToken, Context, ContextBlock, DynamicToolSource, InvocationId, MediaRef,
    ModelGateway, ModelInvokeError, ModelItem, ModelOutput, ModelRef, ModelRequest, ModelResponse,
    ModelStopReason, Role, RoundId, SourceError, StoreError, TextPayload, ToolBatch,
    ToolCallContext, ToolCallPayload, ToolDefinition, ToolExecutionError, ToolOutput,
    ToolResultPayload, ToolResultStatus, TurnId,
};
use causa_runtime::{
    ToolCatalogError, ToolExecutor, ToolExecutorOptions, TurnLimits, TurnResult, TurnRunOptions,
    TurnRunner, new_block_id,
};

use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
    PaginatedRequestParams, ResourceContents, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer, ServiceExt};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

fn call_context(tool_name: &str, arguments: serde_json::Value) -> ToolCallContext {
    ToolCallContext::new(
        BlockId::new(uuid::Uuid::now_v7()),
        ToolCallPayload::new(tool_name, arguments),
    )
}

// ---- in-process MCP server fixture ---------------------------------------------

#[derive(Clone)]
struct FixtureServer {
    tools: Arc<std::sync::Mutex<Vec<String>>>,
    /// When true, calls return `is_error` results (tool-level failure).
    fail_calls: bool,
    /// When true, calls hang until the client's deadline/cancellation.
    hang_calls: bool,
}

impl FixtureServer {
    fn new(fail_calls: bool, hang_calls: bool) -> Self {
        Self {
            tools: Arc::new(std::sync::Mutex::new(vec!["echo".to_string()])),
            fail_calls,
            hang_calls,
        }
    }
}

impl ServerHandler for FixtureServer {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        let tools = self
            .tools
            .lock()
            .unwrap()
            .iter()
            .map(|name| {
                Tool::new(
                    name.clone(),
                    "fixture tool",
                    Arc::new(
                        serde_json::json!({"type": "object"})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                )
            })
            .collect();
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, rmcp::ErrorData> {
        // Bound the hang so the server-side shutdown handshake can finish
        // once the client's deadline/cancel has fired: rmcp's
        // `RunningService::cancel` waits for in-flight requests, so an
        // unbounded hang here would hang the test teardown too.
        if self.hang_calls {
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        let args = request
            .arguments
            .map(|a| serde_json::Value::Object(a).to_string())
            .unwrap_or_default();
        let known = self
            .tools
            .lock()
            .unwrap()
            .iter()
            .any(|t| t == request.name.as_ref());
        if !known {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "unknown tool: {}",
                request.name
            ))])
            .into());
        }
        let result = if self.fail_calls {
            CallToolResult::error(vec![ContentBlock::text(format!("fixture failure: {args}"))])
        } else {
            CallToolResult::success(vec![ContentBlock::text(format!("echo: {args}"))])
        };
        Ok(result.into())
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }
}

/// Serve the fixture on one side of a duplex pipe; return the source and
/// the running server handle (for `notify_tool_list_changed`).
///
/// `serve().await` only resolves after the initialize handshake, so the
/// server side must run as a spawned task while the client connects.
async fn served(
    fail_calls: bool,
    hang_calls: bool,
) -> (
    McpToolSource,
    rmcp::service::RunningService<RoleServer, FixtureServer>,
) {
    served_as("srv", fail_calls, hang_calls).await
}

async fn served_as(
    server_id: &str,
    fail_calls: bool,
    hang_calls: bool,
) -> (
    McpToolSource,
    rmcp::service::RunningService<RoleServer, FixtureServer>,
) {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let server = FixtureServer {
        tools: Arc::new(std::sync::Mutex::new(vec!["echo".to_string()])),
        fail_calls,
        hang_calls,
    };
    let server_task = tokio::spawn(async move { server.serve(server_io).await });
    let source = McpToolSource::connect_io(server_id, client_io)
        .await
        .expect("client connect");
    let running = server_task
        .await
        .expect("server task")
        .expect("server serve");
    (source, running)
}

fn invocation() -> InvocationId {
    InvocationId {
        turn_id: TurnId::new("mcp-integration"),
        round_id: RoundId(0),
    }
}

async fn process_single(executor: &ToolExecutor, call: ToolCallContext) -> ToolResultPayload {
    let bound = executor.bind(invocation(), ctrl()).await.expect("bind");
    let mut batch = ToolBatch::new(vec![call]).expect("new batch");
    bound.process(&mut batch).await.expect("process");
    batch.results()[0].result().expect("result").1.clone()
}

fn ctrl() -> CallControl {
    CallControl::new(CancellationToken::new(), None)
}

/// rmcp transport logs are the only window into handshake stalls; enable
/// them (RUST_LOG overrides the default filter) for the transport tests.
fn init_tracing() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing_subscriber::fmt()
            .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "rmcp=debug".to_owned()))
            .with_test_writer()
            .try_init()
            .ok();
    });
}

// ---- stdio-shape semantics over in-process transport ----------------------------

#[tokio::test]
async fn lists_tools_under_the_server_namespace() {
    let (source, server) = served(false, false).await;
    let defs = source.list().await.expect("list");
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, "mcp_srv_echo");
    assert_eq!(defs[0].description, "fixture tool");
    assert_eq!(defs[0].parameters, serde_json::json!({"type": "object"}));
    assert_eq!(source.id(), "srv");
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn invoke_executes_and_returns_text_content() {
    let (source, server) = served(false, false).await;
    let mut call = call_context("mcp_srv_echo", serde_json::json!({"a": 1}));
    call.result_notes.push(TextPayload::new("incoming note"));
    let outcome = source.invoke(&call, &ctrl(), None).await.expect("invoke");
    assert_eq!(outcome.status, ToolResultStatus::Succeeded);
    assert_eq!(outcome.output.content, serde_json::json!("echo: {\"a\":1}"));
    assert_eq!(outcome.call_block_id, call.call_block_id);
    assert!(outcome.notes.is_empty(), "the source adds no echoed notes");
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn invoke_rejects_names_outside_the_namespace() {
    let (source, server) = served(false, false).await;
    let call = call_context("other_server_tool", serde_json::json!({}));
    let err = source
        .invoke(&call, &ctrl(), None)
        .await
        .expect_err("unknown");
    assert!(matches!(err, ToolExecutionError::UnknownTool(_)));
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn is_error_results_map_to_failed_outcomes() {
    let (source, server) = served(true, false).await;
    let call = call_context("mcp_srv_echo", serde_json::json!({"x": 1}));
    let outcome = source.invoke(&call, &ctrl(), None).await.expect("invoke");
    assert_eq!(outcome.status, ToolResultStatus::Failed);
    assert!(
        outcome
            .output
            .content
            .to_string()
            .contains("fixture failure"),
        "model-readable copy: {}",
        outcome.output.content
    );
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn call_deadline_maps_to_timed_out() {
    let (source, server) = served(false, true).await;
    let call = call_context("mcp_srv_echo", serde_json::json!({}));
    let control = ctrl().with_timeout(Duration::from_millis(200));
    let err = source
        .invoke(&call, &control, None)
        .await
        .expect_err("deadline must fire");
    assert!(matches!(err, ToolExecutionError::TimedOut));
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn cancellation_maps_to_cancelled() {
    let (source, server) = served(false, true).await;
    let call = call_context("mcp_srv_echo", serde_json::json!({}));
    let token = CancellationToken::new();
    let control = CallControl::new(token.clone(), None).with_timeout(Duration::from_secs(30));
    let task = tokio::spawn(async move { source.invoke(&call, &control, None).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    token.cancel();
    let err = task.await.unwrap().expect_err("cancel must fire");
    assert!(matches!(err, ToolExecutionError::Cancelled));
    server.cancel().await.expect("server stop");
}

// ---- list_changed ---------------------------------------------------------------

#[tokio::test]
async fn list_changed_bumps_the_version_and_refreshes_the_listing() {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let tools = Arc::new(std::sync::Mutex::new(vec!["echo".to_string()]));
    let server = FixtureServer {
        tools: tools.clone(),
        fail_calls: false,
        hang_calls: false,
    };
    let server_task = tokio::spawn(async move { server.serve(server_io).await });
    let source = McpToolSource::connect_io("srv", client_io)
        .await
        .expect("client connect");
    let running = server_task
        .await
        .expect("server task")
        .expect("server serve");

    let v0 = source.version();
    assert_eq!(
        source
            .list()
            .await
            .expect("list")
            .iter()
            .map(|d| d.name.clone())
            .collect::<Vec<_>>(),
        vec!["mcp_srv_echo".to_string()]
    );

    // Server adds a tool and notifies; the source's version signal moves.
    tools.lock().unwrap().push("extra".to_string());
    running.notify_tool_list_changed().await.expect("notify");
    let mut v1 = v0;
    for _ in 0..100 {
        tokio::time::sleep(Duration::from_millis(20)).await;
        v1 = source.version();
        if v1 != v0 {
            break;
        }
    }
    assert_ne!(v1, v0, "list_changed must bump the version");
    let names: Vec<String> = source
        .list()
        .await
        .expect("re-list")
        .iter()
        .map(|d| d.name.clone())
        .collect();
    assert_eq!(
        names,
        vec!["mcp_srv_echo".to_string(), "mcp_srv_extra".to_string()]
    );
    running.cancel().await.expect("server stop");
}

// ---- executor aggregation -------------------------------------------------------

/// A source that always fails to list — for the skip-without-interrupt
/// guarantee.
struct BrokenSource;
#[async_trait]
impl DynamicToolSource for BrokenSource {
    fn id(&self) -> &str {
        "broken"
    }
    async fn list(&self) -> Result<Vec<ToolDefinition>, SourceError> {
        Err(SourceError::Unavailable("fixture outage".into()))
    }
    async fn invoke(
        &self,
        call: &ToolCallContext,
        _control: &CallControl,
        _store: Option<&dyn causa_kernel::ArtifactStore>,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        Err(ToolExecutionError::Unavailable(format!(
            "fixture outage: {}",
            call.input.tool_name
        )))
    }
}

fn echo_static_tool() -> Arc<dyn causa_kernel::Tool> {
    struct Echo;
    #[async_trait]
    impl causa_kernel::Tool for Echo {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("echo", "echo", serde_json::json!({"type": "object"}))
        }
        async fn execute(
            &self,
            ctx: &ToolCallContext,
            _control: &CallControl,
            _store: Option<&dyn causa_kernel::ArtifactStore>,
        ) -> ToolResultPayload {
            ToolResultPayload::new(
                ctx.call_block_id,
                ToolResultStatus::Succeeded,
                ToolOutput::new(serde_json::json!({"echo": ctx.input.arguments})),
            )
        }
    }
    Arc::new(Echo)
}

#[tokio::test]
async fn executor_aggregates_static_and_dynamic_definitions() {
    let (source, server) = served(false, false).await;
    let executor =
        ToolExecutor::new(vec![echo_static_tool()], ToolExecutorOptions::default()).unwrap();
    executor
        .register_dynamic(Arc::new(source))
        .expect("register");

    let bound = executor.bind(invocation(), ctrl()).await.expect("bind");
    let surface = bound.surface();
    let mut names: Vec<String> = surface.definitions.iter().map(|d| d.name.clone()).collect();
    names.sort();
    assert_eq!(names, vec!["echo", "mcp_srv_echo"]);
    assert_eq!(executor.dynamic_ids(), vec!["srv".to_string()]);

    // Duplicate registration is rejected; unknown unregistration too.
    executor
        .register_dynamic(Arc::new(BrokenSource))
        .expect("register broken");
    assert!(executor.register_dynamic(Arc::new(BrokenSource)).is_err());
    assert!(executor.unregister_dynamic("nope").is_err());
    assert!(executor.unregister_dynamic("broken").is_ok());
    assert_eq!(executor.dynamic_ids(), vec!["srv".to_string()]);
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn executor_rejects_a_failing_catalog_source() {
    let (source, server) = served(false, false).await;
    let executor = ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap();
    executor
        .register_dynamic(Arc::new(source))
        .expect("register");
    executor
        .register_dynamic(Arc::new(BrokenSource))
        .expect("register broken");

    assert!(matches!(executor.bind(invocation(), ctrl()).await,
        Err(ToolCatalogError::Source { source_id, .. }) if source_id == "broken"));
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn executor_dispatch_routes_dynamic_calls_and_maps_errors() {
    let (source, server) = served(false, false).await;
    let executor = ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap();
    executor
        .register_dynamic(Arc::new(source))
        .expect("register");

    // A binding exposes its directory and pins its concrete source targets.
    let bound = executor.bind(invocation(), ctrl()).await.expect("bind");
    let surface = bound.surface();
    assert_eq!(surface.definitions.len(), 1);

    // A dynamic call uses the same complete batch path as a local tool.
    let outcome = process_single(
        &executor,
        call_context("mcp_srv_echo", serde_json::json!({"k": "v"})),
    )
    .await;
    assert_eq!(outcome.status, ToolResultStatus::Succeeded);
    assert_eq!(
        outcome.output.content,
        serde_json::json!("echo: {\"k\":\"v\"}")
    );

    // A name the server never advertised is rejected by the executor
    // itself — the model can only call what the surface showed.
    let outcome = process_single(
        &executor,
        call_context("mcp_srv_nothere", serde_json::json!({})),
    )
    .await;
    assert_eq!(outcome.status, ToolResultStatus::Rejected);

    // A name in no listing at all is rejected by the executor too.
    let outcome = process_single(
        &executor,
        call_context("no_such_tool_anywhere", serde_json::json!({})),
    )
    .await;
    assert_eq!(outcome.status, ToolResultStatus::Rejected);

    // A remote tool-level failure (`is_error` result) is a Failed outcome
    // with model-readable copy, from a distinct server namespace.
    let (flaky, server3) = served_as("flaky", true, false).await;
    executor
        .register_dynamic(Arc::new(flaky) as Arc<dyn DynamicToolSource>)
        .expect("register flaky");
    let outcome = process_single(
        &executor,
        call_context("mcp_flaky_echo", serde_json::json!({})),
    )
    .await;
    assert_eq!(outcome.status, ToolResultStatus::Failed);
    assert!(
        outcome
            .output
            .content
            .to_string()
            .contains("fixture failure"),
        "model-readable copy: {}",
        outcome.output.content
    );
    server3.cancel().await.expect("server stop");
    server.cancel().await.expect("server stop");
}

// ---- full kernel turn through the MCP source ------------------------------------

struct ScriptedGateway {
    outputs: std::sync::Mutex<Vec<ModelOutput>>,
}
#[async_trait]
impl ModelGateway for ScriptedGateway {
    async fn invoke(
        &self,
        _req: &ModelRequest,
        _control: &CallControl,
    ) -> Result<ModelOutput, ModelInvokeError> {
        let mut guard = self.outputs.lock().unwrap();
        if guard.is_empty() {
            return Err(ModelInvokeError::new(
                causa_kernel::ModelInvokeErrorKind::Permanent,
                "script exhausted",
            ));
        }
        Ok(guard.remove(0))
    }
}

fn tool_use_output(tool: &str, args: serde_json::Value) -> ModelOutput {
    ModelOutput::new(
        ModelResponse {
            items: vec![
                ModelItem::text("calling"),
                ModelItem::tool_call(causa_kernel::ToolCallDraft::new(tool, args)),
            ],
        },
        ModelStopReason::ToolUse,
    )
}

fn end_turn(text: &str) -> ModelOutput {
    ModelOutput::new(
        ModelResponse {
            items: vec![ModelItem::text(text)],
        },
        ModelStopReason::EndTurn,
    )
}

#[tokio::test]
async fn kernel_turn_completes_through_the_mcp_source() {
    let (source, server) = served(false, false).await;
    let executor = Arc::new(ToolExecutor::new(vec![], ToolExecutorOptions::default()).unwrap());
    executor
        .register_dynamic(Arc::new(source))
        .expect("register");
    let gateway = Arc::new(ScriptedGateway {
        outputs: std::sync::Mutex::new(vec![
            tool_use_output("mcp_srv_echo", serde_json::json!({"q": 42})),
            end_turn("done"),
        ]),
    });
    let runner = TurnRunner::new(gateway, executor);
    let options = TurnRunOptions::new(ModelRef::new("fake")).with_limits(TurnLimits::new(5, 8));
    let ctx =
        Context::from_blocks(vec![ContextBlock::text(new_block_id(), Role::User, "hi")]).unwrap();

    let out = runner
        .run(
            TurnId::new("t-mcp"),
            ctx,
            options,
            CallControl::new(CancellationToken::new(), None),
        )
        .await;
    assert!(
        matches!(out.result, TurnResult::Completed { .. }),
        "expected completion, got {:?}",
        out
    );
    assert_eq!(
        out.context
            .blocks()
            .iter()
            .filter(|block| matches!(block.content(), BlockContent::ToolCall(_)))
            .count(),
        1
    );
    let declaration_block_id = out
        .context
        .blocks()
        .iter()
        .find_map(|block| match block.content() {
            causa_kernel::BlockContent::ToolCall(_) => Some(block.id()),
            _ => None,
        })
        .expect("model declaration block");
    let result_block = out
        .context
        .blocks()
        .iter()
        .find_map(|b| match b.content() {
            causa_kernel::BlockContent::ToolResult(r) => Some(r.clone()),
            _ => None,
        })
        .expect("tool result fact");
    assert_eq!(result_block.status, ToolResultStatus::Succeeded);
    assert_eq!(
        result_block.call_block_id, declaration_block_id,
        "the dynamic source result stays attached to its declaration"
    );
    assert_eq!(
        result_block.output.content,
        serde_json::json!("echo: {\"q\":42}")
    );
    server.cancel().await.expect("server stop");
}

// ---- Streamable HTTP transport --------------------------------------------------

/// Serve the fixture behind a real axum server (rmcp `StreamableHttpService`)
/// on an ephemeral loopback port. Returns the client source and every
/// `Authorization` header the server observed (for the static-token-injection
/// assertion).
async fn served_http(
    auth_token: Option<String>,
) -> (McpToolSource, Arc<std::sync::Mutex<Vec<Option<String>>>>) {
    use axum::extract::State;
    use axum::http::header::AUTHORIZATION;
    use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use rmcp::transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService,
    };

    init_tracing();
    let seen_auth: Arc<std::sync::Mutex<Vec<Option<String>>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));

    async fn capture_auth(
        State(seen): State<Arc<std::sync::Mutex<Vec<Option<String>>>>>,
        request: axum::extract::Request,
        next: axum::middleware::Next,
    ) -> axum::response::Response {
        seen.lock().unwrap().push(
            request
                .headers()
                .get(AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned),
        );
        next.run(request).await
    }

    let http_service = StreamableHttpService::new(
        || Ok(FixtureServer::new(false, false)),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let app = axum::Router::new()
        .nest_service("/mcp", http_service)
        .layer(axum::middleware::from_fn_with_state(
            seen_auth.clone(),
            capture_auth,
        ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move { axum::serve(listener, app).await });

    let source = McpToolSource::connect_http("http", format!("http://{addr}/mcp"), auth_token)
        .await
        .expect("http connect");
    (source, seen_auth)
}

#[tokio::test]
async fn http_transport_lists_and_invokes() {
    let (source, _seen_auth) = served_http(None).await;
    let defs = source.list().await.expect("list over http");
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, "mcp_http_echo");

    let call = call_context("mcp_http_echo", serde_json::json!({"via": "http"}));
    let outcome = source
        .invoke(&call, &ctrl(), None)
        .await
        .expect("invoke over http");
    assert_eq!(outcome.status, ToolResultStatus::Succeeded);
    assert_eq!(
        outcome.output.content,
        serde_json::json!("echo: {\"via\":\"http\"}")
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), source.close())
        .await
        .expect("close must not hang")
        .expect("close");
}

#[tokio::test]
async fn http_transport_injects_the_static_bearer_token() {
    let (source, seen_auth) = served_http(Some("secret-token".into())).await;
    source.list().await.expect("list over http");
    let seen = seen_auth.lock().unwrap().clone();
    assert!(
        seen.iter()
            .any(|h| h.as_deref() == Some("Bearer secret-token")),
        "every request must carry the injected bearer token, saw: {seen:?}"
    );
    drop(seen);
    tokio::time::timeout(std::time::Duration::from_secs(5), source.close())
        .await
        .expect("close must not hang")
        .expect("close");
}

// ---- media ingest ---------------------------------------------------------------

/// Distinctive bytes whose base64 must never appear in observation text.
const PAYLOAD: &[u8] = b"causa-binary-payload";
const PAYLOAD_B64: &str = "Y2F1c2EtYmluYXJ5LXBheWxvYWQ=";

/// A fixture whose tools each return one kind of binary content block.
#[derive(Clone)]
struct MediaServer;

impl ServerHandler for MediaServer {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        let schema = Arc::new(
            serde_json::json!({"type": "object"})
                .as_object()
                .unwrap()
                .clone(),
        );
        Ok(ListToolsResult::with_all_items(
            [
                "render_image",
                "record_audio",
                "export_pdf",
                "export_blob",
                "broken_image",
            ]
            .into_iter()
            .map(|name| Tool::new(name, "returns binary content", schema.clone()))
            .collect(),
        ))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, rmcp::ErrorData> {
        let block = match request.name.as_ref() {
            // "AAAA" is valid base64 (3 bytes); the mime marks it a png
            "render_image" => ContentBlock::image("AAAA", "image/png"),
            "record_audio" => ContentBlock::audio(PAYLOAD_B64, "audio/wav"),
            "export_pdf" => ContentBlock::resource(
                ResourceContents::blob(PAYLOAD_B64, "file:///memo.pdf")
                    .with_mime_type("application/pdf"),
            ),
            "export_blob" => {
                ContentBlock::resource(ResourceContents::blob(PAYLOAD_B64, "file:///data.bin"))
            }
            _ => ContentBlock::image("not base64!", "image/png"),
        };
        Ok(CallToolResult::success(vec![block]).into())
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }
}

async fn served_media() -> (
    McpToolSource,
    rmcp::service::RunningService<RoleServer, MediaServer>,
) {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let server_task = tokio::spawn(async move { MediaServer.serve(server_io).await });
    let source = McpToolSource::connect_io("srv", client_io)
        .await
        .expect("client connect");
    let running = server_task
        .await
        .expect("server task")
        .expect("server serve");
    (source, running)
}

/// One-slot in-memory store: every persist lands as `asset-1`.
#[derive(Default)]
struct MemStore {
    hint: std::sync::Mutex<Option<ArtifactHint>>,
    data: std::sync::Mutex<Option<Vec<u8>>>,
}
#[async_trait::async_trait]
impl ArtifactStore for MemStore {
    async fn persist(&self, data: &[u8], hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        *self.hint.lock().unwrap() = Some(hint);
        *self.data.lock().unwrap() = Some(data.to_vec());
        Ok(ArtifactRef::new(
            "asset-1",
            data.len(),
            ArtifactKind::Binary,
            true,
        ))
    }
    async fn read(
        &self,
        _id: &str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError> {
        Ok(self.data.lock().unwrap().clone().unwrap_or_default())
    }
}

/// Invoke `tool` without a store and with one; return both results and the store.
async fn invoke_both(tool: &str) -> (ToolResultPayload, ToolResultPayload, MemStore) {
    let (source, _server) = served_media().await;
    let call = call_context(&format!("mcp_srv_{tool}"), json!({}));
    let bare = source.invoke(&call, &ctrl(), None).await.unwrap();
    let store = MemStore::default();
    let stored = source
        .invoke(&call, &ctrl(), Some(&store as &dyn ArtifactStore))
        .await
        .unwrap();
    assert_eq!(
        store.hint.lock().unwrap().as_ref().map(|h| h.call_block_id),
        stored.media.first().map(|_| call.call_block_id),
        "artifact provenance is keyed by the declaration BlockId"
    );
    (bare, stored, store)
}

fn text(result: &ToolResultPayload) -> String {
    result.output.content.as_str().unwrap().to_owned()
}

#[tokio::test]
async fn image_results_ingest_into_media_references_when_a_store_is_wired() {
    let (bare, stored, _store) = invoke_both("render_image").await;

    // Without a store: the deterministic placeholder text, no media.
    assert!(bare.media.is_empty());
    assert_eq!(text(&bare), "[image: image/png — no media store available]");

    // With a store: bytes persisted, reference attached, note names the asset.
    assert_eq!(stored.media, vec![MediaRef::new("image/png", "asset-1")]);
    assert_eq!(text(&stored), "[image attached: image/png — asset asset-1]");
}

#[tokio::test]
async fn audio_results_ingest_like_images() {
    let (bare, stored, store) = invoke_both("record_audio").await;

    assert!(bare.media.is_empty());
    assert_eq!(text(&bare), "[audio: audio/wav — no media store available]");

    assert_eq!(stored.media, vec![MediaRef::new("audio/wav", "asset-1")]);
    assert_eq!(text(&stored), "[audio attached: audio/wav — asset asset-1]");
    assert_eq!(store.data.lock().unwrap().as_deref(), Some(PAYLOAD));
}

#[tokio::test]
async fn blob_resources_ingest_and_never_reach_the_text() {
    let (bare, stored, store) = invoke_both("export_pdf").await;

    assert!(bare.media.is_empty());
    assert_eq!(
        text(&bare),
        "[resource: file:///memo.pdf (application/pdf) — no media store available]"
    );
    assert_eq!(
        stored.media,
        vec![MediaRef::new("application/pdf", "asset-1").with_name("memo.pdf")]
    );
    assert_eq!(
        text(&stored),
        "[resource attached: file:///memo.pdf (application/pdf) — asset asset-1]"
    );
    assert_eq!(store.data.lock().unwrap().as_deref(), Some(PAYLOAD));
    for result in [&bare, &stored] {
        assert!(!text(result).contains(PAYLOAD_B64));
    }
}

#[tokio::test]
async fn blob_resources_without_a_mime_type_are_octet_streams() {
    let (bare, stored, _store) = invoke_both("export_blob").await;

    assert_eq!(
        text(&bare),
        "[resource: file:///data.bin (application/octet-stream) — no media store available]"
    );
    assert_eq!(
        stored.media,
        vec![MediaRef::new("application/octet-stream", "asset-1").with_name("data.bin")]
    );
    assert!(!text(&stored).contains(PAYLOAD_B64));
}

#[tokio::test]
async fn malformed_base64_reports_a_failed_store_not_a_missing_one() {
    let (_bare, stored, store) = invoke_both("broken_image").await;

    assert!(stored.media.is_empty());
    assert_eq!(text(&stored), "[image: image/png — could not be stored]");
    assert!(store.data.lock().unwrap().is_none());
}
