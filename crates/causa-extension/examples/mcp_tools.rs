//! Wire an MCP server's tool catalog into the executor — stdio (child
//! process) and, optionally, Streamable HTTP.
//!
//! ```text
//! # stdio: any MCP server launch command works; the default below needs `uvx`.
//! cargo run --example mcp_tools -p causa-extension
//! # or: cargo run --example mcp_tools -p causa-extension -- npx -y @modelcontextprotocol/server-everything
//! # http: MCP_HTTP_URL=https://mcp.example.com/mcp [MCP_HTTP_TOKEN=...] cargo run ...
//! ```
//!
//! Tools enter the executor under the `mcp_{server_id}_{tool}`
//! namespace, cached until the server notifies `tools/list_changed`.
//! From here the model-facing flow is identical to local tools: hand
//! `executor.tool_surface()` to a turn's `TurnInvocation` and the driver
//! routes dispatch by listing membership (see
//! `causa-runtime/examples/quickstart.rs` for the model-side half).

use causa_extension::McpToolSource;
use causa_runtime::ToolExecutor;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let executor = ToolExecutor::from_vec(Vec::new());

    // --- stdio: spawn a local MCP server as a child process ----------------
    let mut argv = std::env::args().skip(1);
    let (program, args) = match argv.next() {
        Some(program) => (program, argv.collect()),
        None => ("uvx".to_string(), vec!["mcp-server-fetch".to_string()]),
    };
    let mut command = tokio::process::Command::new(&program);
    command.args(&args);
    println!("connecting over stdio: {program} {}", args.join(" "));

    let stdio_source = Arc::new(McpToolSource::connect_stdio("fetch", command).await?);
    executor.register_dynamic(stdio_source.clone())?;

    // --- Streamable HTTP: connect when MCP_HTTP_URL is set -----------------
    if let Ok(url) = std::env::var("MCP_HTTP_URL") {
        let token = std::env::var("MCP_HTTP_TOKEN").ok();
        println!("connecting over http: {url}");
        let http_source = Arc::new(McpToolSource::connect_http("remote", url, token).await?);
        executor.register_dynamic(http_source)?;
    }

    // --- the merged model-facing surface ------------------------------------
    let surface = executor.tool_surface().await;
    println!("registered sources: {:?}", executor.dynamic_ids());
    println!("model-facing tools:");
    for definition in &surface.definitions {
        println!("  {} — {}", definition.name, definition.description);
    }

    // Sessions end when the sources drop; for a bounded shutdown
    // handshake instead, unregister from the executor and call
    // `McpToolSource::close` on the owned source (see its docs).
    drop(executor);
    println!("done");
    Ok(())
}
