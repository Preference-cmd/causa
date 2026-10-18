// The "Connect MCP servers" guide. Runs offline by default: an MCP server
// in this process offers one tool, a scripted model calls it, and the
// program prints the tool list, the recorded result and the model's answer.
//
// To list the tools of a real server instead, pass its launch command or URL:
//   cargo run --bin connect_mcp_servers -- uvx mcp-server-fetch
//   MCP_TOKEN=... cargo run --bin connect_mcp_servers -- https://mcp.example.com/mcp

// ANCHOR: imports
use causa::{
    extension::McpToolSource,
    kernel::{
        BlockContent, CallControl, CancellationToken, Context, ContextBlock, InvocationId,
        ModelRef, Role, RoundId, SourceError, TurnId,
    },
    runtime::{
        ToolExecutor, ToolExecutorOptions, TurnResult, TurnRunOptions, TurnRunner, new_block_id,
    },
};
use std::sync::Arc;
// ANCHOR_END: imports

use causa_site_guides::scripted::{ScriptedModel, call, reply, tool_results};
use serde_json::json;

// ANCHOR: stdio
/// Starts a local server as a child process and talks to it over stdio.
/// Its tools are named `mcp_local_*`.
async fn connect_stdio(program: String, args: Vec<String>) -> Result<McpToolSource, SourceError> {
    let mut command = tokio::process::Command::new(program);
    command.args(args);
    McpToolSource::connect_stdio("local", command).await
}
// ANCHOR_END: stdio

// ANCHOR: http
/// Connects to a remote server over Streamable HTTP, with an optional
/// bearer token. Its tools are named `mcp_remote_*`.
async fn connect_http(url: String) -> Result<McpToolSource, SourceError> {
    let token = std::env::var("MCP_TOKEN").ok();
    McpToolSource::connect_http("remote", url, token).await
}
// ANCHOR_END: http

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (server_id, source) = match args.next() {
        None => ("notes", notes_server::connect("notes").await?),
        Some(url) if url.starts_with("http") => ("remote", connect_http(url).await?),
        Some(program) => ("local", connect_stdio(program, args.collect()).await?),
    };

    // ANCHOR: register
    let source = Arc::new(source);
    let executor = Arc::new(ToolExecutor::new(
        Vec::new(),
        ToolExecutorOptions::default(),
    )?);
    executor.register_dynamic(source.clone())?;
    // ANCHOR_END: register

    // ANCHOR: list
    // Lists the tools the way the runner does before each model request.
    let bound = executor
        .bind(
            InvocationId {
                turn_id: TurnId::new("tool-list"),
                round_id: RoundId(0),
            },
            CallControl::new(CancellationToken::new(), None),
        )
        .await?;
    for definition in &bound.surface().definitions {
        println!("tool: {}", definition.name);
    }
    drop(bound);
    // ANCHOR_END: list

    if server_id == "notes" {
        // Stands in for a provider: calls the server's tool, then answers.
        let gateway = Arc::new(ScriptedModel(|request: &_| {
            match tool_results(request).last() {
                None => call("mcp_notes_read_note", json!({ "name": "todo.txt" })),
                Some(result) => reply(format!(
                    "The note says: {}",
                    result.output.content.as_str().unwrap_or_default()
                )),
            }
        }));
        let runner = TurnRunner::new(gateway, executor.clone());
        let context = Context::from_blocks(vec![ContextBlock::text(
            new_block_id(),
            Role::User,
            "What is on my todo list?",
        )])?;
        let outcome = runner
            .run(
                TurnId::new("notes"),
                context,
                TurnRunOptions::new(ModelRef::new("scripted")),
                CallControl::new(CancellationToken::new(), None),
            )
            .await;
        for block in outcome.context.blocks() {
            if let BlockContent::ToolResult(result) = block.content() {
                println!("result: {:?} {}", result.status, result.output.content);
            }
        }
        match outcome.result {
            TurnResult::Completed { final_output } => {
                println!("{}", final_output.response.text())
            }
            result => return Err(format!("turn did not complete: {result:?}").into()),
        }
    }

    // ANCHOR: close
    // Take the source out of the executor, then end the session.
    drop(executor.unregister_dynamic(server_id)?);
    if let Ok(source) = Arc::try_unwrap(source) {
        source.close().await?;
    }
    // ANCHOR_END: close
    Ok(())
}

/// An MCP server with one tool, `read_note`, served over an in-memory pipe.
mod notes_server {
    use causa::{extension::McpToolSource, kernel::SourceError};
    use rmcp::{
        handler::server::ServerHandler,
        model::{
            CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock,
            InitializeResult, ListToolsResult, PaginatedRequestParams, Tool,
        },
        service::{RequestContext, RoleServer, ServiceExt},
    };
    use serde_json::json;
    use std::sync::Arc;

    struct NotesServer;

    impl ServerHandler for NotesServer {
        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, rmcp::ErrorData> {
            let schema = json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            });
            let schema = schema.as_object().cloned().unwrap_or_default();
            Ok(ListToolsResult::with_all_items(vec![Tool::new(
                "read_note",
                "Read one note by its file name.",
                Arc::new(schema),
            )]))
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, rmcp::ErrorData> {
            let name = request
                .arguments
                .as_ref()
                .and_then(|args| args.get("name"))
                .and_then(|name| name.as_str());
            let result = match name {
                Some("todo.txt") => CallToolResult::success(vec![ContentBlock::text("Buy milk.")]),
                _ => CallToolResult::error(vec![ContentBlock::text("no such note")]),
            };
            Ok(result.into())
        }

        fn get_info(&self) -> InitializeResult {
            InitializeResult::default()
        }
    }

    pub async fn connect(server_id: &str) -> Result<McpToolSource, SourceError> {
        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        // The server answers the handshake while the client connects, so
        // it runs as its own task.
        tokio::spawn(async move {
            if let Ok(running) = NotesServer.serve(server_io).await {
                let _ = running.waiting().await;
            }
        });
        McpToolSource::connect_io(server_id, client_io).await
    }
}
