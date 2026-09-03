# Causa

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

## Status

Pre-0.1. Crate names finalized as `causa-*`. CI, examples, and the first-class
MCP client (`causa-mcp`) are landing (Slice 11).

The brand name **Causa** is Latin for *cause* — the reason an action is taken,
and what an effect is traced back to. The kernel keeps the causes (facts and
ports) apart from their effects (drivers and adapters); Project inceptae, the
wider inquiry this crate family serves, asks whether the productivity effects
of AI reach the people they displace.

## License

Dual-licensed under `MIT OR Apache-2.0`.
