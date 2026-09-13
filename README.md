# Causa

[![CI](https://github.com/Preference-cmd/causa/actions/workflows/ci.yml/badge.svg)](https://github.com/Preference-cmd/causa/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.96-blue)](https://github.com/Preference-cmd/causa)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)](https://github.com/Preference-cmd/causa)

> There is no harness.

**A minimal context kernel for Rust agents** — facts in the kernel, behavior in yours.

```bash
cargo add causa
```

## Crates

The kernel is always on; the default features are `runtime` + `providers`,
and `--no-default-features` is kernel-only.

| crate | role |
|---|---|
| `causa` | facade over the family |
| `causa-kernel` | facts + contracts: conversation kernel and ports, no I/O |
| `causa-protocol` | wire-protocol translation (Anthropic / OpenAI Chat / OpenAI Responses) |
| `causa-runtime` | reference driver: turn loop, tool dispatch, streaming, pause/resume, session coordination and checkpoint, unknown-outcome & output-retention config |
| `causa-provider` | reqwest adapters for the `ModelGateway` port |
| `causa-extension` | extension adapters (`mcp` feature: MCP client) |

| selection | gets |
|---|---|
| `causa = "0.1"` (default) | kernel + runtime + providers |
| `features = ["full"]` | + extensions (MCP) |
| `default-features = false` | kernel only (offline audit / minimal embed) |
| `default-features = false, features = ["runtime"]` | kernel + offline driver |

A provider renders through the protocol crate, so `providers` implies
`protocol` — you never enable a transport without its translation.
Extensions stay opt-in because `rmcp` is a heavier dependency that offline
hosts should not pay for.

## Examples

Each crate ships runnable examples under `crates/*/examples/`. Start with
`quickstart` (`causa-provider`, needs `ANTHROPIC_API_KEY`); the
`causa-runtime` examples run offline against scripted stubs, and
`mcp_tools` (`causa-extension`, `mcp` feature) needs an MCP server.

## Concepts

- **The kernel carries facts and contracts — nothing else**: no I/O, no
  transport, no policy. Anything the host didn't opt into doesn't happen;
  CI enforces the layering on every push.
- **The driver's defaults are visible and swappable**: every policy is a
  config object or a documented opinion, listed in one table in the
  `causa-runtime` crate docs.
- **Edge crates are adapters, not opinions**: providers and extensions
  translate; they don't decide. Port implementations (stores, counters,
  exporters) live in examples or hosts, never in the publish set.

## Status

Pre-0.1 — **the API is unstable; read the examples and the crate docs for
the current shape rather than relying on this file for signatures.**
Breaking changes may land without notice.

The 0.1 gate is functional completeness: **multimodal I/O (delivered) and
subagents (not yet — session-level building blocks are in `causa-runtime`;
the collaboration surface is not released).** During the `0.x` series,
breaking changes to the wire serde shapes bump the minor version and are
flagged at the top of the CHANGELOG.

## Contributing

See [AGENTS.md](./AGENTS.md) for layout, commands, layering rules, and
workflow.

## License

Dual-licensed under `MIT OR Apache-2.0`.
