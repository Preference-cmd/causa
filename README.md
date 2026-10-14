<div align="center">
  <h1>Causa</h1>
  <h3>A minimal context kernel for AI agents</h3>
  <p><em>There is no harness.</em></p>
  <p>
    <a href="https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/getting-started.mdx">Getting started</a> •
    <a href="#examples">Examples</a> •
    <a href="#architecture">Architecture</a> •
    <a href="https://github.com/Preference-cmd/causa/blob/main/CHANGELOG.md">Changelog</a>
  </p>
  <p>
    <a href="https://github.com/Preference-cmd/causa/actions/workflows/ci.yml"><img src="https://github.com/Preference-cmd/causa/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://github.com/Preference-cmd/causa/blob/main/Cargo.toml"><img src="https://img.shields.io/badge/MSRV-1.96-blue" alt="MSRV: Rust 1.96"></a>
    <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green" alt="License: MIT OR Apache-2.0"></a>
  </p>
</div>

Causa keeps messages, tool calls and results as typed facts in a Rust kernel.
Its optional runtime adds turn execution, streaming and tool-processing chains, with Anthropic, OpenAI and MCP adapters available alongside it.
Your application owns the tools, storage and execution policy — facts in the
kernel, behavior in yours.

> **Experimental 0.0.4.** Requires Rust **1.96+**. APIs and stored formats
> may change before 0.1; read the
> [migration notes](https://github.com/Preference-cmd/causa/blob/v0.0.4/CHANGELOG.md#004---2026-10-13) when upgrading.

## Try it offline

Run a conversation, save its context and reload it — no API key or server needed:

```bash
git clone https://github.com/Preference-cmd/causa.git
cd causa
cargo run --example conversation_persistence -p causa-runtime
```

The default checkout follows development; use `git checkout v0.0.4` to run
the examples for this release.

## Use in your application

Depend on **`causa`**; Cargo resolves the underlying crates.

```bash
cargo new causa-hello
cd causa-hello
cargo add causa@0.0.4
cargo add tokio@1 --features macros,rt
```

Put this in `src/main.rs` to run one model turn through the facade:

```rust
use causa::{
    kernel::{BlockContent, BlockMeta, CallControl, CancellationToken, ContentPart, Context, ContextBlock,
             ModelRef, TextPayload, TurnId},
    providers::AnthropicMessagesGateway,
    runtime::{new_block_id, ToolExecutor, ToolExecutorOptions,
              TurnResult, TurnRunOptions, TurnRunner},
};
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gateway = Arc::new(AnthropicMessagesGateway::new(
        std::env::var("ANTHROPIC_API_KEY")?,
    ));
    let executor = Arc::new(ToolExecutor::new(Vec::new(), ToolExecutorOptions::default())?);
    let runner = TurnRunner::new(gateway, executor);
    let context = Context::from_blocks(vec![ContextBlock::new(
        new_block_id(),
        BlockContent::Parts(vec![ContentPart::Text(TextPayload::new("Say hello in one sentence."))]),
        BlockMeta::from_source("user"),
    )])?;
    let options = TurnRunOptions::new(ModelRef::new(std::env::var("ANTHROPIC_MODEL")?));
    let outcome = runner
        .run(TurnId::new("hello"), context, options,
             CallControl::new(CancellationToken::new(), None))
        .await;
    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text()),
        result => return Err(format!("turn did not complete: {result:?}").into()),
    }
    Ok(())
}
```

Set `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` to your key and an available model
ID, then run `cargo run`. This example makes a real network request. To add a
local tool, see the
[provider quickstart](https://github.com/Preference-cmd/causa/blob/main/crates/causa-provider/examples/quickstart.rs).

### Streaming and reasoning

The built-in Anthropic Messages, OpenAI Chat Completions and OpenAI Responses
gateways stream over HTTP SSE. Use `TurnRunner::run_streaming()` with
`TurnRunOptions::observer` to watch a run; deltas are provisional, and only
the complete output becomes context material. The runner makes one gateway
call per round and does not retry; wrap the gateway in
`causa::runtime::RetryingGateway` if you want retries.

`GenerationOptions::reasoning_effort` is sent to the provider as you wrote
it, and settings only one protocol has go in `extra_body`. Provider reasoning
items are kept whole and replayed, so a thinking-enabled tool loop runs on
all three protocols.

The tools offered to the model are facts in the context too: the runner
appends a `ToolChange` block when the bound tools change. Each gateway sends
the current set in `tools` by default; the Anthropic and Responses gateways
can instead keep `tools` fixed and render the change in place
(`with_native_tool_changes`, `with_additional_tools`), so a tool list change
does not miss the prompt cache. Render options can be loaded from host
configuration via serde and applied with `with_render_options`; the host
decides which servers and models receive which options.

See the [streaming guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/streaming.mdx)
for events and limits, and the
[crate guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/crates.mdx)
for generation options and reasoning replay.

### Choose your features

The default includes the kernel, runtime and provider adapters. MCP is opt-in.
Choose the dependencies your application needs:

| Configuration | Included |
|---|---|
| `causa = "0.0.4"` | Kernel + runtime + providers |
| `causa = { version = "0.0.4", features = ["full"] }` | Default stack + MCP extensions |
| `causa = { version = "0.0.4", default-features = false }` | Kernel only |
| `causa = { version = "0.0.4", default-features = false, features = ["runtime"] }` | Kernel + runtime, no network adapters |

See the [crate guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/crates.mdx)
for layer selection.

## Examples

Run these from the repository root using
`cargo run -p <crate> --example <name>`:

| Example | Crate | Demonstrates | Needs |
|---|---|---|---|
| `conversation_persistence` | `causa-runtime` | Save and reload caller-owned context | Offline |
| `streaming_print` | `causa-runtime` | Observe deltas as a turn runs | Offline |
| `approval_processor` | `causa-runtime` | Await approval inside a tool processor | Offline |
| `interrupted_batch` | `causa-runtime` | Close the tool batch an interrupted run hands back | Offline |
| `tool_processors` | `causa-runtime` | Compose tool processors | Offline |
| `media_feedback` | `causa-runtime` | Carry tool-produced media references into the next round | Offline |
| `quickstart` | `causa-provider` | A model turn with a local tool | `ANTHROPIC_API_KEY` |
| `mcp_tools` | `causa-extension` | Connect MCP tool sources | MCP server; default `mcp` feature |

The [example catalog](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/examples.mdx)
links to the source. For MCP server launch commands, see the
[mcp_tools header](https://github.com/Preference-cmd/causa/blob/main/crates/causa-extension/examples/mcp_tools.rs).

## Architecture

The kernel holds facts and contracts, with no I/O, transport or execution
policy. The runtime supplies reference execution components. Your application
provides storage, credentials, tools and the policies it needs.

| Crate | Responsibility |
|---|---|
| `causa` | Facade and feature selection — the default entry point |
| `causa-kernel` | Editable context material and contracts for models, preparation and tools |
| `causa-runtime` | Material-returning turn runner, borrowed observations and bound tool execution |
| `causa-protocol` | Pure request/response and incremental SSE translation for Anthropic and OpenAI |
| `causa-provider` | HTTP adapters implementing complete and streaming model gateway calls |
| `causa-extension` | Dynamic tool-source adapters, currently MCP |

All six crates share one release version and can also be used directly.
Read the [concepts guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/concepts.mdx)
for layer boundaries, the
[context editing guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/context-editing.mdx)
for changing retained blocks, and the
[runtime policy overview](https://github.com/Preference-cmd/causa/blob/main/crates/causa-runtime/src/lib.rs)
for defaults and component contracts.

## Status and boundaries

**0.0.4 is experimental.** During `0.0.x`, patch releases may break the Rust
API and serialized formats, and Cargo does not upgrade `"0.0.3"` to `0.0.4`
on its own; read the
[changelog](https://github.com/Preference-cmd/causa/blob/main/CHANGELOG.md)
before upgrading. From `0.1.0`, a breaking wire-format change bumps the minor
version.

Sessions, history, input admission, persistence and subagent orchestration
belong to your application. An interrupted run returns its uncommitted tool
batch and the cause, and your code decides what to do with it; the
[interrupted runs guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/interrupted-runs.mdx)
shows how to close such a batch. Rebuilding
context does not make an external tool run exactly once.

## Contributing

See [AGENTS.md](https://github.com/Preference-cmd/causa/blob/main/AGENTS.md) for
repository layout, conventions and verification commands, and the
[release guide](https://github.com/Preference-cmd/causa/blob/main/.github/RELEASING.md)
for publishing.

## License

Licensed under either [MIT](https://github.com/Preference-cmd/causa/blob/main/LICENSE-MIT)
or [Apache-2.0](https://github.com/Preference-cmd/causa/blob/main/LICENSE-APACHE),
at your option.
