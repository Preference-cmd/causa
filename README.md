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

> **Experimental 0.0.2.** Requires Rust **1.96+**. This release separates
> editable context material from execution and removes session ownership.
> APIs and stored formats have no backward compatibility guarantee before 0.1;
> review the [migration notes](https://github.com/Preference-cmd/causa/blob/v0.0.2/CHANGELOG.md#002---2026-10-02) when upgrading.

## Try it offline

Run a conversation, save its context and reload it — no API key or server needed:

```bash
git clone https://github.com/Preference-cmd/causa.git
cd causa
cargo run --example conversation_persistence -p causa-runtime
```

Repository examples run from this checkout, even if you have already added
`causa` to another project. The default checkout follows development; use
`git checkout v0.0.2` to run the examples for this release.

## Use in your application

Start with **`causa`**. Cargo resolves the underlying crates. Create an
application using the published release:

```bash
cargo new causa-hello
cd causa-hello
cargo add causa@0.0.2
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

### Streaming on main (unreleased)

The built-in Anthropic Messages, OpenAI Chat Completions and OpenAI Responses
gateways now implement real HTTP SSE through `ModelGateway::stream()`.
Use `TurnRunner::run_streaming()` with `TurnRunOptions::observer` to observe
increments during a run. Deltas are provisional; only the complete terminal
output supplies context material. Cancellation, deadlines and dropping the
stream stop local consumption without an automatic retry: the runner makes one
logical gateway call per round and the first failure ends it. Retries are an
explicit host choice — wrap the gateway in `causa::runtime::RetryingGateway`
with a `RetryPolicy` — and the wrapper never retries once any stream delta has
been yielded.

This provider implementation is not in the published `0.0.2` crates, whose
built-in gateways return a single completed output from the stream entry.
See the [streaming guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/streaming.mdx)
for supported events, resource limits and direct consumption.

### Generation options on main (unreleased)

On unreleased `main`, `GenerationOptions::reasoning_effort` is a string that
is sent as the host wrote it: OpenAI Chat `reasoning_effort`, Responses
`reasoning.effort`, Anthropic `output_config.effort`. Causa does not check or
map the value, and derives no `thinking` object from it. Settings that only
one protocol or model has go in `GenerationOptions::extra_body`, whose keys
are copied to the top level of the request unless the adapter already wrote
that key; an Anthropic `thinking` object is sent this way. Which values a
model accepts is the host's knowledge, and a refused value comes back as the
provider's error. Output limits render as `max_completion_tokens`,
`max_output_tokens` and `max_tokens`, and include reasoning tokens where the
provider counts them. Anthropic requires `max_tokens` (a request without one
is rejected before HTTP rather than defaulted) and has no `output_schema`
mapping.

Reasoning state replays as sealed context material: the Anthropic and
Responses parsers keep provider reasoning items whole, and the Chat renderer
sends sealed `reasoning_content` back, so a thinking-enabled tool loop can run
on any of the three wires. Responses sends
`include: ["reasoning.encrypted_content"]` and `store: false` by default.
Renderer-specific choices live in `causa_protocol::translation::render_options`
and are set on the matching gateway (`with_beta`, `with_encrypted_reasoning`, `with_store`,
`with_reasoning_content_replay`); each applies to the gateway value, not to
one request.

### Choose your features

The default includes the kernel, runtime and provider adapters. MCP is opt-in.
Choose the dependencies your application needs:

| Configuration | Included |
|---|---|
| `causa = "0.0.2"` | Kernel + runtime + providers |
| `causa = { version = "0.0.2", features = ["full"] }` | Default stack + MCP extensions |
| `causa = { version = "0.0.2", default-features = false }` | Kernel only |
| `causa = { version = "0.0.2", default-features = false, features = ["runtime"] }` | Kernel + runtime, no network adapters |

The kernel is always present. `providers` also enables `protocol`; `full`
enables the MCP adapter through `extensions`. See the
[crate guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/crates.mdx)
for layer selection.

## Examples

Run these from the repository root using
`cargo run -p <crate> --example <name>`:

| Example | Crate | Demonstrates | Needs |
|---|---|---|---|
| `conversation_persistence` | `causa-runtime` | Save and reload caller-owned context | Offline |
| `streaming_print` | `causa-runtime` | Observe deltas as a turn runs | Offline |
| `approval_processor` | `causa-runtime` | Await approval inside a tool processor | Offline |
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
The runtime and extension depend on the kernel; provider adapters use the
protocol layer. CI checks the family dependency directions.

Read the [concepts guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/concepts.mdx)
for layer boundaries and the
[runtime policy overview](https://github.com/Preference-cmd/causa/blob/main/crates/causa-runtime/src/lib.rs)
for defaults and component contracts.

To edit the retained blocks explicitly, see the
[context editing guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/context-editing.mdx).

## Status and boundaries

**0.0.2 is an experimental release.** It provides
multimodal material, turn execution and tool-processing contracts. Session management and subagent orchestration belong to applications built
on Causa. This release does not announce a 0.1 stability commitment.

During `0.0.x`, patch releases may break Rust API and serialized-format
compatibility. Cargo does not automatically upgrade `"0.0.1"` to `0.0.2`;
review the [changelog](https://github.com/Preference-cmd/causa/blob/main/CHANGELOG.md)
before upgrading. Starting with `0.1.0`, breaking wire-format changes bump the
minor version. The 0.0.2 API replaces `TurnContext` with editable `Context`
and removes session ownership, implicit traces and built-in budget policies.
Applications own history, input admission, preparation and persistence.

Persistence stays in your application. Processing interruptions return the
uncommitted tool batch and their cause; your code chooses whether to inspect,
save or discard it. Reconstructing context does not guarantee exactly-once
execution of external tools.

## Contributing

See [AGENTS.md](https://github.com/Preference-cmd/causa/blob/main/AGENTS.md) for
repository layout, conventions and verification commands. Public facade consumers
are checked with `cargo test --manifest-path tests/consumers/material-flow/Cargo.toml --locked`,
the same command with `--features runtime`, and
`cargo test --manifest-path tests/consumers/provider-streaming/Cargo.toml --locked`. The
[release guide](https://github.com/Preference-cmd/causa/blob/main/.github/RELEASING.md)
covers the manual publishing workflow.

## License

Licensed under either [MIT](https://github.com/Preference-cmd/causa/blob/main/LICENSE-MIT)
or [Apache-2.0](https://github.com/Preference-cmd/causa/blob/main/LICENSE-APACHE),
at your option.
