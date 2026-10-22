//! The stdio transport against a real child process, checking where the
//! server's stderr goes.
//!
//! This binary is also the server (`harness = false`). Run by `cargo test`,
//! it starts copies of itself with `CAUSA_MCP_STDIO_FIXTURE` set: `serve`
//! answers MCP on its own stdin and stdout after writing a line to stderr,
//! `fail` writes a reason to stderr and exits before the handshake.

use causa_extension::McpToolSource;
use causa_kernel::{DynamicToolSource, SourceError};
use rmcp::handler::server::ServerHandler;
use rmcp::model::{ListToolsResult, PaginatedRequestParams, ServerInfo, Tool};
use rmcp::service::{RequestContext, RoleServer, ServiceExt};
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

const ROLE: &str = "CAUSA_MCP_STDIO_FIXTURE";
const STARTING: &str = "fixture: starting";
const FAILURE: &str = "fixture: missing configuration";

fn main() {
    match std::env::var(ROLE).as_deref() {
        Ok("serve") => serve(),
        Ok("fail") => {
            eprintln!("{FAILURE}");
            std::process::exit(2);
        }
        _ => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async {
                stderr_reaches_the_hosts_pipe().await;
                failed_start_leaves_the_reason_in_the_pipe().await;
            });
            println!("stdio_stderr: 2 checks passed");
        }
    }
}

/// One tool, `ping`, served on this process's stdin and stdout.
struct PingServer;

impl ServerHandler for PingServer {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::ErrorData> {
        let schema = serde_json::json!({ "type": "object" });
        let schema = schema.as_object().cloned().unwrap_or_default();
        Ok(ListToolsResult::with_all_items(vec![Tool::new(
            "ping",
            "Answers pong.",
            Arc::new(schema),
        )]))
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo::default()
    }
}

fn serve() {
    eprintln!("{STARTING}");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        if let Ok(running) = PingServer
            .serve((tokio::io::stdin(), tokio::io::stdout()))
            .await
        {
            let _ = running.waiting().await;
        }
    });
}

fn fixture(role: &str) -> tokio::process::Command {
    let mut command =
        tokio::process::Command::new(std::env::current_exe().expect("test binary path"));
    command.env(ROLE, role);
    command
}

/// Reads the pipe to end of file on a plain thread. Failing to reach end
/// of file in time means a writing end is still open somewhere.
fn read_to_end(mut reader: std::io::PipeReader) -> String {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut output = String::new();
        let _ = sender.send(reader.read_to_string(&mut output).map(|_| output));
    });
    receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("the reading end did not reach end of file")
        .expect("read stderr pipe")
}

async fn stderr_reaches_the_hosts_pipe() {
    let (reader, writer) = std::io::pipe().expect("pipe");
    let source = McpToolSource::connect_stdio_with_stderr("fx", fixture("serve"), writer)
        .await
        .expect("connect to the stdio fixture");
    let names: Vec<String> = source
        .list()
        .await
        .expect("list tools")
        .into_iter()
        .map(|definition| definition.name)
        .collect();
    assert_eq!(names, ["mcp_fx_ping"]);
    source.close().await.expect("close");

    let output = read_to_end(reader);
    assert!(
        output.contains(STARTING),
        "server stderr not in the host's pipe: {output:?}"
    );
}

async fn failed_start_leaves_the_reason_in_the_pipe() {
    let (reader, writer) = std::io::pipe().expect("pipe");
    let error = McpToolSource::connect_stdio_with_stderr("fx", fixture("fail"), writer)
        .await
        .expect_err("a server that exits before the handshake cannot connect");
    assert!(
        matches!(error, SourceError::Unavailable(_)),
        "unexpected error: {error:?}"
    );

    let output = read_to_end(reader);
    assert!(
        output.contains(FAILURE),
        "failure reason not in the host's pipe: {output:?}"
    );
}
