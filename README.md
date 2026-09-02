# agent-stack (name pending)

**A small, principled agent kernel for Rust** — context facts, ports, driver.

The publish set is five crates:

| crate | role |
|---|---|
| `reimagine-context-kernel` | the facts layer — ContextBlock conversation kernel + ports (`ModelGateway`, `ConversationStore`, `Tool`, `DynamicToolSource`, `CallControl`, budget) |
| `reimagine-ai-protocol` | kernel-native wire-protocol translation (Anthropic / OpenAI Chat / OpenAI Responses) |
| `reimagine-agent-runtime` | the reference driver — turn loop, tool dispatch, streaming, approval pause/resume |
| `reimagine-agent-provider` | reqwest adapters for the kernel `ModelGateway` seam |
| `reimagine-agent-mcp` | first-class MCP client over the `DynamicToolSource` port |

Positioning versus rig (provider-generic layer) and swiftide (RAG pipelines):
**facts / ports / driver layering, a three-entry fact machine, approval
pause with recoverable interruption, first-class MCP support.**

## Principles

1. **The kernel carries contracts and facts — nothing else**: no I/O, no
   transport, no policy with a sole opinion.
2. **The driver's defaults are visible and swappable**: every policy is
   either a config object or documented in one place as the driver's
   opinion. No trait seams for single-implementation policies.
3. **Edge crates are interop adapters** (providers, MCP) — capabilities,
   not opinions.
4. Reference implementations of ports (stores, token counters, exporters)
   live in examples or host applications, never in the publish set.

## Status

Pre-0.1. Branding, CI, and examples are landing (Slice 11). Crate names are
`reimagine-*` pending a brand decision.

## License

Dual-licensed under `MIT OR Apache-2.0`.
