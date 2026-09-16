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
Its optional runtime adds turn execution, streaming, approval pauses and session
checkpoints, with Anthropic, OpenAI and MCP adapters available alongside it.
Your application owns the tools, storage and execution policy — facts in the
kernel, behavior in yours.

> **Experimental 0.0.1.** Requires Rust **1.96+**. Rust APIs and serialized
> formats may change between `0.0.x` patch releases.

## Try it offline

Run a conversation, save its history and reload it — no API key or server needed:

```bash
git clone https://github.com/Preference-cmd/causa.git
cd causa
cargo run --example conversation_persistence -p causa-runtime
```

Repository examples run from this checkout, even if you have already added
`causa` to another project. The default checkout follows development; use
`git checkout v0.0.1` to run the examples for this release.

## Use in your application

Start with **`causa`**. Cargo resolves the underlying
crates; you do not need to add all six yourself.

```bash
cargo new causa-hello
cd causa-hello
cargo add causa@0.0.1
cargo add tokio@1 --features macros,rt
```

Put this in `src/main.rs` to run one model turn through the facade:

```rust
use causa::{
    kernel::{CancellationToken, ModelRef, TextPayload, TurnContext, TurnId},
    providers::AnthropicMessagesGateway,
    runtime::{RunControl, ToolExecutor, TurnResult, TurnRunOptions, TurnRunner},
};
use std::sync::Arc;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gateway = Arc::new(AnthropicMessagesGateway::new(
        std::env::var("ANTHROPIC_API_KEY")?,
    ));
    let runner = TurnRunner::new(gateway, Arc::new(ToolExecutor::from_vec(Vec::new())));
    let mut context = TurnContext::new(TurnId::new("hello"));
    context.append_input(TextPayload::new("Say hello in one sentence."), "user")?;
    let mut options = TurnRunOptions::default();
    options.invocation.model = ModelRef::new(std::env::var("ANTHROPIC_MODEL")?);
    let outcome = runner
        .run(context, options, RunControl::new(CancellationToken::new(), None))
        .await;
    match outcome.result {
        TurnResult::Completed { final_output } => println!("{}", final_output.response.text.0),
        result => return Err(format!("turn did not complete: {result:?}").into()),
    }
    Ok(())
}
```

Set `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` to your key and an available model
ID, then run `cargo run`. This example makes a real network request. To add a
local tool, see the
[provider quickstart](https://github.com/Preference-cmd/causa/blob/main/crates/causa-provider/examples/quickstart.rs).

### Choose your features

The default includes the kernel, runtime and provider adapters. MCP is opt-in.
These are alternative dependency configurations:

| Configuration | Included |
|---|---|
| `causa = "0.0.1"` | Kernel + runtime + providers |
| `causa = { version = "0.0.1", features = ["full"] }` | Default stack + MCP extensions |
| `causa = { version = "0.0.1", default-features = false }` | Kernel only |
| `causa = { version = "0.0.1", default-features = false, features = ["runtime"] }` | Kernel + runtime, no network adapters |

The kernel is always present. `providers` also enables `protocol`; `full`
enables the MCP adapter through `extensions`. See the
[crate guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/crates.mdx)
for layer selection.

## Examples

Run these from the repository root using
`cargo run -p <crate> --example <name>`:

| Example | Crate | Demonstrates | Needs |
|---|---|---|---|
| `conversation_persistence` | `causa-runtime` | Save and reload conversation history | Offline |
| `streaming_print` | `causa-runtime` | Observe deltas as a turn runs | Offline |
| `approval_pause_resume` | `causa-runtime` | Pause a tool batch and resume with approval | Offline |
| `allow_deny_filter` | `causa-runtime` | Compose tool-use filters | Offline |
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
| `causa-kernel` | Conversation facts, turn state and contracts for models and tools |
| `causa-runtime` | Turn execution, sessions, pause/resume, checkpoints and policy configuration |
| `causa-protocol` | Pure translation for Anthropic and OpenAI wire formats |
| `causa-provider` | HTTP adapters implementing the model gateway contract |
| `causa-extension` | Dynamic tool-source adapters, currently MCP |

All six crates share one release version and can also be used directly.
The runtime and extension depend on the kernel; provider adapters use the
protocol layer. CI checks the family dependency directions.

Read the [concepts guide](https://github.com/Preference-cmd/causa/blob/main/website/src/content/docs/concepts.mdx)
for layer boundaries and the
[runtime policy overview](https://github.com/Preference-cmd/causa/blob/main/crates/causa-runtime/src/lib.rs)
for defaults and component contracts.

## Status and boundaries

**0.0.1 is an experimental development snapshot.** Multimodal I/O and session
building blocks are implemented. Subagent collaboration is not yet released;
it remains part of the 0.1 functional completeness gate.

During `0.0.x`, patch releases may break Rust API and serialized-format
compatibility. Cargo does not automatically upgrade `"0.0.1"` to `0.0.2`;
review the [changelog](https://github.com/Preference-cmd/causa/blob/main/CHANGELOG.md)
before upgrading. Starting with `0.1.0`, breaking wire-format changes bump the
minor version. Checkpoint schema versions are validated independently.

Persistence stays in your application. Session checkpoints support saving an
idle or paused session; they do not guarantee exactly-once execution across
arbitrary process crashes.

## Contributing

See [AGENTS.md](https://github.com/Preference-cmd/causa/blob/main/AGENTS.md) for
repository layout, conventions and verification commands. The
[release guide](https://github.com/Preference-cmd/causa/blob/main/.github/RELEASING.md)
covers the manual publishing workflow.

## License

Licensed under either [MIT](https://github.com/Preference-cmd/causa/blob/main/LICENSE-MIT)
or [Apache-2.0](https://github.com/Preference-cmd/causa/blob/main/LICENSE-APACHE),
at your option.
