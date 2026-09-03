# Causa

[![CI](https://github.com/Preference-cmd/causa/actions/workflows/ci.yml/badge.svg)](https://github.com/Preference-cmd/causa/actions/workflows/ci.yml)

> derive, don't drift.

**A small, principled agent kernel for Rust** — facts, ports, driver.

Part of [Project inceptae](https://example.invalid/inceptae) — a wider inquiry
into whether AI's productivity gains can really reach the people they displace.

The publish set is five crates:

| crate | role |
|---|---|
| `causa-kernel` | the facts layer — ContextBlock conversation kernel + ports (`ModelGateway`, `ConversationStore`, `Tool`, `DynamicToolSource`, `CallControl`, budget) |
| `causa-protocol` | kernel-native wire-protocol translation (Anthropic / OpenAI Chat / OpenAI Responses) |
| `causa-runtime` | the reference driver — turn loop, tool dispatch, streaming, approval pause/resume |
| `causa-provider` | reqwest adapters for the kernel `ModelGateway` seam |
| `causa-mcp` | first-class MCP client over the `DynamicToolSource` port |

Positioning versus rig (provider-generic layer) and swiftide (RAG pipelines):
**facts / ports / driver layering, a three-entry fact machine, approval
pause with recoverable interruption, first-class MCP support.**

## Quick start

One turn, one local tool, one provider round-trip — the full runnable
version lives at `crates/causa-provider/examples/quickstart.rs`:

```rust
let executor = Arc::new(ToolExecutor::from_vec(vec![Arc::new(WordCount)]));
let gateway = Arc::new(AnthropicMessagesGateway::new(api_key));
let runner = TurnRunner::new(gateway, executor.clone());

let mut context = TurnContext::new(TurnId::new("quickstart"));
context.append_input(TextPayload::new("Count the words in … use the tool."), "user")?;

let options = TurnRunOptions {
    invocation: TurnInvocation {
        model: ModelRef::new("claude-sonnet-4-5"),
        tool_surface: executor.tool_surface().await,
        ..Default::default()
    },
    ..Default::default()
};

let outcome = runner.run(context, options, RunControl::new(Default::default(), None)).await;
if let TurnResult::Completed { final_output } = outcome.result {
    println!("{}", final_output.response.text.0);
}
```

```text
ANTHROPIC_API_KEY=sk-ant-… cargo run --example quickstart -p causa-provider
```

Five examples, each doubling as docs.rs-runnable documentation:

| example | shows | runs offline |
|---|---|---|
| `quickstart` (`-p causa-provider`) | one turn + a local tool + a real provider | needs API key |
| `conversation_persistence` (`-p causa-runtime`) | multi-turn + `FsConversationStore` + reload across restart | ✅ |
| `streaming_print` (`-p causa-runtime`) | streaming deltas via `TurnInteraction::on_delta` | ✅ |
| `approval_pause_resume` (`-p causa-runtime`) | `decide_batch` pause + `resume_turn` with the withheld verdict | ✅ |
| `mcp_tools` (`-p causa-mcp`) | stdio + Streamable HTTP MCP servers into the executor | needs a server |

MSRV: **1.96** (pinned by CI; also declared as the workspace `rust-version`).

## Principles

1. **The kernel carries contracts and facts — nothing else**: no I/O, no
   transport, no policy with a sole opinion.
2. **The driver's defaults are visible and swappable**: every policy is
   either a config object or documented in one place as the driver's
   opinion. No trait seams for single-implementation policies.
3. **Edge crates are interop adapters** (providers, MCP) — capabilities,
   not opinions. This is the publish set's only exception to "no concrete
   implementations": adapters exist to translate, not to decide.
4. Reference implementations of ports (stores, token counters, exporters)
   live in examples or host applications, never in the publish set.

**The kernel never does:** network, filesystem, or process I/O; provider
wire formats; concrete stores, token counters, or exporters; retry,
timeout, or approval policy; any behavior the host did not opt into. The
layering is machine-enforced — CI asserts the dependency directions on
every push (7.1), not left to review.

**Non-goals:** RAG pipelines, orchestration graphs, UIs, telemetry export
(0.2+), agent-substrate features outside the kernel/runtime/driver story.

## Status

Pre-0.1. Crate names finalized as `causa-*`; CI (fmt / clippy / test
matrix + MSRV 1.96 + dependency-direction guard) and the five examples are
in place; the first-class MCP client (`causa-mcp`) shipped with Slice 10.
The 0.1 release gate is functional completeness (multimodal I/O and
subagents, slices 6.5 / 8), tracked in the slice 11 proposal.

The brand name **Causa** is Latin for *cause* — the reason an action is taken,
and what an effect is traced back to. The kernel keeps the causes (facts and
ports) apart from their effects (drivers and adapters); Project inceptae, the
wider inquiry this crate family serves, asks whether the productivity effects
of AI reach the people they displace.

## License

Dual-licensed under `MIT OR Apache-2.0`.
