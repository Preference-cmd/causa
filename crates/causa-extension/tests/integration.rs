//! Integration tests: an in-process rmcp server fixture served over a
//! duplex pipe, exercised through `McpToolSource`, the executor's dynamic
//! registry, and a full kernel turn (`TurnRunner` untouched).

use async_trait::async_trait;
use causa_extension::McpToolSource;
use causa_kernel::{
    ArtifactHint, ArtifactKind, ArtifactRef, ArtifactStore, AttemptControl, CallControl,
    CancellationToken, DynamicToolSource, MediaRef, ModelGateway, ModelInvokeError, ModelOutput,
    ModelRef, ModelRequest, ModelResponse, ModelStopReason, SourceError, StoreError, TextPayload,
    ToolCallContext, ToolCallId, ToolCallPayload, ToolDefinition, ToolExecutionError, ToolOutput,
    ToolResultPayload, ToolResultStatus, ToolSurface, TurnContext, TurnId,
};
use causa_runtime::{
    RunControl, ToolExecutor, ToolOutputLimits, TurnInvocation, TurnLimits, TurnPolicy, TurnResult,
    TurnRunOptions, TurnRunner,
};
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
    PaginatedRequestParams, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer, ServiceExt};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

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
    let call = ToolCallPayload {
        call_id: causa_kernel::ToolCallId("call-1".into()),
        tool_name: "mcp_srv_echo".into(),
        arguments: serde_json::json!({"a": 1}),
    };
    let outcome = source.invoke(&call, &ctrl()).await.expect("invoke");
    assert_eq!(outcome.status, ToolResultStatus::Succeeded);
    assert_eq!(outcome.output.content, serde_json::json!("echo: {\"a\":1}"));
    assert_eq!(outcome.call_id, call.call_id);
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn invoke_rejects_names_outside_the_namespace() {
    let (source, server) = served(false, false).await;
    let call = ToolCallPayload {
        call_id: causa_kernel::ToolCallId("call-1".into()),
        tool_name: "other_server_tool".into(),
        arguments: serde_json::json!({}),
    };
    let err = source.invoke(&call, &ctrl()).await.expect_err("unknown");
    assert!(matches!(err, ToolExecutionError::UnknownTool(_)));
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn is_error_results_map_to_failed_outcomes() {
    let (source, server) = served(true, false).await;
    let call = ToolCallPayload {
        call_id: causa_kernel::ToolCallId("call-1".into()),
        tool_name: "mcp_srv_echo".into(),
        arguments: serde_json::json!({"x": 1}),
    };
    let outcome = source.invoke(&call, &ctrl()).await.expect("invoke");
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
    let call = ToolCallPayload {
        call_id: causa_kernel::ToolCallId("call-1".into()),
        tool_name: "mcp_srv_echo".into(),
        arguments: serde_json::json!({}),
    };
    let control = CallControl::new(CancellationToken::new(), Some(Duration::from_millis(200)));
    let err = source
        .invoke(&call, &control)
        .await
        .expect_err("deadline must fire");
    assert!(matches!(err, ToolExecutionError::TimedOut));
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn cancellation_maps_to_cancelled() {
    let (source, server) = served(false, true).await;
    let call = ToolCallPayload {
        call_id: causa_kernel::ToolCallId("call-1".into()),
        tool_name: "mcp_srv_echo".into(),
        arguments: serde_json::json!({}),
    };
    let token = CancellationToken::new();
    let control = CallControl::new(token.clone(), Some(Duration::from_secs(30)));
    let task = tokio::spawn(async move { source.invoke(&call, &control).await });
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
        call: &ToolCallPayload,
        _control: &CallControl,
    ) -> Result<ToolResultPayload, ToolExecutionError> {
        Err(ToolExecutionError::Unavailable(format!(
            "fixture outage: {}",
            call.tool_name
        )))
    }
}

fn echo_static_tool() -> Arc<dyn causa_kernel::Tool> {
    struct Echo;
    #[async_trait]
    impl causa_kernel::Tool for Echo {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "echo".into(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }
        async fn execute(
            &self,
            ctx: &ToolCallContext,
            _control: &CallControl,
        ) -> ToolResultPayload {
            ToolResultPayload {
                call_id: ctx.call_id.clone(),
                status: ToolResultStatus::Succeeded,
                output: ToolOutput::new(serde_json::json!({"echo": ctx.arguments})),
                media: Vec::new(),
            }
        }
    }
    Arc::new(Echo)
}

#[tokio::test]
async fn executor_aggregates_static_and_dynamic_definitions() {
    let (source, server) = served(false, false).await;
    let executor = ToolExecutor::from_vec(vec![echo_static_tool()]);
    executor
        .register_dynamic(Arc::new(source))
        .expect("register");

    let surface = executor.tool_surface().await;
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
async fn executor_skips_a_failing_source_without_interrupting() {
    let (source, server) = served(false, false).await;
    let executor = ToolExecutor::from_vec(vec![]);
    executor
        .register_dynamic(Arc::new(source))
        .expect("register");
    executor
        .register_dynamic(Arc::new(BrokenSource))
        .expect("register broken");

    // Surface assembly still succeeds; only the healthy source's tools
    // are present.
    let surface = executor.tool_surface().await;
    assert_eq!(surface.definitions.len(), 1);
    assert_eq!(surface.definitions[0].name, "mcp_srv_echo");
    server.cancel().await.expect("server stop");
}

#[tokio::test]
async fn executor_dispatch_routes_dynamic_calls_and_maps_errors() {
    let (source, server) = served(false, false).await;
    let executor = ToolExecutor::from_vec(vec![]);
    executor
        .register_dynamic(Arc::new(source))
        .expect("register");

    // Dispatch routes by listing membership: the surface assembly (the
    // host's per-turn job) populates the routing cache.
    let surface = executor.tool_surface().await;
    assert_eq!(surface.definitions.len(), 1);

    // A dynamic call executes like a local one (truncation pipeline etc.).
    let outcome = executor
        .execute_with_limits(
            ToolCallPayload {
                call_id: causa_kernel::ToolCallId("call-1".into()),
                tool_name: "mcp_srv_echo".into(),
                arguments: serde_json::json!({"k": "v"}),
            },
            ctrl(),
            None,
            None,
            ToolOutputLimits::default(),
        )
        .await;
    assert_eq!(outcome.status, ToolResultStatus::Succeeded);
    assert_eq!(
        outcome.output.content,
        serde_json::json!("echo: {\"k\":\"v\"}")
    );

    // A name the server never advertised is rejected by the executor
    // itself — the model can only call what the surface showed.
    let outcome = executor
        .execute_with_limits(
            ToolCallPayload {
                call_id: causa_kernel::ToolCallId("call-2".into()),
                tool_name: "mcp_srv_nothere".into(),
                arguments: serde_json::json!({}),
            },
            ctrl(),
            None,
            None,
            ToolOutputLimits::default(),
        )
        .await;
    assert_eq!(outcome.status, ToolResultStatus::Rejected);

    // A name in no listing at all is rejected by the executor too.
    let outcome = executor
        .execute_with_limits(
            ToolCallPayload {
                call_id: causa_kernel::ToolCallId("call-4".into()),
                tool_name: "no_such_tool_anywhere".into(),
                arguments: serde_json::json!({}),
            },
            ctrl(),
            None,
            None,
            ToolOutputLimits::default(),
        )
        .await;
    assert_eq!(outcome.status, ToolResultStatus::Rejected);

    // A remote tool-level failure (`is_error` result) is a Failed outcome
    // with model-readable copy, from a distinct server namespace.
    let (flaky, server3) = served_as("flaky", true, false).await;
    executor
        .register_dynamic(Arc::new(flaky) as Arc<dyn DynamicToolSource>)
        .expect("register flaky");
    executor.tool_surface().await;
    let outcome = executor
        .execute_with_limits(
            ToolCallPayload {
                call_id: causa_kernel::ToolCallId("call-5".into()),
                tool_name: "mcp_flaky_echo".into(),
                arguments: serde_json::json!({}),
            },
            ctrl(),
            None,
            None,
            ToolOutputLimits::default(),
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
        _control: &AttemptControl,
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
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new("calling"),
            tool_calls: vec![causa_kernel::ToolCallDraft {
                tool_name: tool.into(),
                arguments: args,
                provider_call_id: None,
            }],
        },
        usage: None,
        stop_reason: ModelStopReason::ToolUse,
        reasoning: None,
    }
}

fn end_turn(text: &str) -> ModelOutput {
    ModelOutput {
        response: ModelResponse {
            text: TextPayload::new(text),
            tool_calls: vec![],
        },
        usage: None,
        stop_reason: ModelStopReason::EndTurn,
        reasoning: None,
    }
}

#[tokio::test]
async fn kernel_turn_completes_through_the_mcp_source() {
    let (source, server) = served(false, false).await;
    let executor = Arc::new(ToolExecutor::from_vec(vec![]));
    executor
        .register_dynamic(Arc::new(source))
        .expect("register");
    // Surface assembled from the executor (static ∪ dynamic) — the host's
    // only job; the driver is untouched by dynamic sources.
    let surface: ToolSurface = executor.tool_surface().await;
    assert_eq!(surface.definitions.len(), 1);

    let gateway = Arc::new(ScriptedGateway {
        outputs: std::sync::Mutex::new(vec![
            tool_use_output("mcp_srv_echo", serde_json::json!({"q": 42})),
            end_turn("done"),
        ]),
    });
    let runner = TurnRunner::new(gateway, executor);
    let options = TurnRunOptions {
        invocation: TurnInvocation {
            model: ModelRef::new("fake"),
            tool_surface: surface,
            ..Default::default()
        },
        policy: TurnPolicy {
            limits: TurnLimits {
                max_model_rounds: 5,
                max_tool_calls: 8,
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut ctx = TurnContext::new(TurnId::new("t-mcp"));
    ctx.append_input(causa_kernel::TextPayload::new("hi"), "user")
        .unwrap();

    let out = runner
        .run(
            ctx,
            options,
            RunControl::new(CancellationToken::new(), None),
        )
        .await;
    assert!(
        matches!(out.result, TurnResult::Completed { .. }),
        "expected completion, got {:?}",
        out
    );
    assert_eq!(out.trace.tool_calls_total, 1);
    let result_block = out
        .context
        .blocks()
        .iter()
        .find_map(|b| match &b.content {
            causa_kernel::BlockContent::ToolResult(r) => Some(r.clone()),
            _ => None,
        })
        .expect("tool result fact");
    assert_eq!(result_block.status, ToolResultStatus::Succeeded);
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

    let call = ToolCallPayload {
        call_id: causa_kernel::ToolCallId("call-h1".into()),
        tool_name: "mcp_http_echo".into(),
        arguments: serde_json::json!({"via": "http"}),
    };
    let outcome = source
        .invoke(&call, &ctrl())
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

/// A fixture whose one tool returns an image content block.
#[derive(Clone)]
struct ImageServer;

impl ServerHandler for ImageServer {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        Ok(ListToolsResult::with_all_items(vec![Tool::new(
            "render_image",
            "returns a tiny png",
            Arc::new(
                serde_json::json!({"type": "object"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        )]))
    }

    async fn call_tool(
        &self,
        _request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, rmcp::ErrorData> {
        // "AAAA" is valid base64 (3 bytes); the mime marks it a png
        Ok(CallToolResult::success(vec![ContentBlock::image("AAAA", "image/png")]).into())
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }
}

async fn served_image() -> (
    McpToolSource,
    rmcp::service::RunningService<RoleServer, ImageServer>,
) {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let server_task = tokio::spawn(async move { ImageServer.serve(server_io).await });
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
struct MemStore;
#[async_trait::async_trait]
impl ArtifactStore for MemStore {
    async fn persist(&self, _data: &[u8], _hint: ArtifactHint) -> Result<ArtifactRef, StoreError> {
        Ok(ArtifactRef {
            id: "asset-1".into(),
            size_bytes: 3,
            kind: ArtifactKind::Binary,
            persisted: true,
        })
    }
    async fn read(
        &self,
        _id: &str,
        _range: Option<std::ops::Range<u64>>,
    ) -> Result<Vec<u8>, StoreError> {
        Ok(vec![0, 0, 0])
    }
}

#[tokio::test]
async fn image_results_ingest_into_media_references_when_a_store_is_wired() {
    let (source, _server) = served_image().await;
    let call = ToolCallPayload {
        call_id: ToolCallId::new("c1"),
        tool_name: "mcp_srv_render_image".into(),
        arguments: json!({}),
    };

    // Without a store: the deterministic placeholder text, no media.
    let out = source.invoke(&call, &ctrl()).await.unwrap();
    assert!(out.media.is_empty());
    assert!(
        out.output
            .content
            .to_string()
            .contains("[image: image/png mime — no media store available]")
    );

    // With a store: bytes persisted, reference attached, note names the asset.
    let store = MemStore;
    let out = source
        .invoke_with_store(&call, &ctrl(), Some(&store as &dyn ArtifactStore))
        .await
        .unwrap();
    assert_eq!(out.media, vec![MediaRef::new("image/png", "asset-1")]);
    assert!(
        out.output
            .content
            .to_string()
            .contains("[image attached: image/png — asset asset-1]")
    );
}
