# Causa

[![CI](https://github.com/Preference-cmd/causa/actions/workflows/ci.yml/badge.svg)](https://github.com/Preference-cmd/causa/actions/workflows/ci.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.96-blue)](https://github.com/Preference-cmd/causa)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)](https://github.com/Preference-cmd/causa)

> derive, don't drift.

**A small, principled agent kernel for Rust** — facts, ports, driver.

Part of [Project inceptae](https://example.invalid/inceptae) — a wider inquiry
into whether AI's productivity gains can really reach the people they displace.

## Crates

`cargo add causa` is the default entry; the family crates stay published
for fine-grained use. The kernel is always on; default features are
`runtime` + `providers`, `full` adds extensions, `--no-default-features`
is kernel-only.

| crate | role |
|---|---|
| `causa` | facade over the family |
| `causa-kernel` | facts + contracts: conversation kernel and ports, no I/O |
| `causa-protocol` | wire-protocol translation (Anthropic / OpenAI Chat / OpenAI Responses) |
| `causa-runtime` | reference driver: turn loop, tool dispatch, streaming, pause/resume, unknown-outcome & output-retention config |
| `causa-provider` | reqwest adapters for the `ModelGateway` port |
| `causa-extension` | extension adapters (`mcp` feature: MCP client) |

## Examples

> The API is still unstable — check the in-tree examples for the current
> shape rather than relying on this file for signatures.

| example | shows | runs offline |
|---|---|---|
| `quickstart` (`-p causa-provider`) | one turn + a local tool + a real provider | needs API key |
| `conversation_persistence` (`-p causa-runtime`) | multi-turn + `FsConversationStore` + reload across restart | ✅ |
| `streaming_print` (`-p causa-runtime`) | streaming deltas via `TurnInteraction::on_delta` | ✅ |
| `approval_pause_resume` (`-p causa-runtime`) | `decide_batch` pause + `resume_turn` with the withheld verdict | ✅ |
| `mcp_tools` (`-p causa-extension`) | stdio + Streamable HTTP MCP servers into the executor | needs a server |

## Concepts

- **The kernel carries facts and contracts — nothing else**: no I/O, no
  transport, no policy. Anything the host didn't opt into doesn't happen;
  CI enforces the layering on every push.
- **The driver's defaults are visible and swappable**: every policy is a
  config object or a documented opinion in one place.
- **Edge crates are adapters, not opinions**: providers and extensions
  translate; they don't decide. Port implementations (stores, counters,
  exporters) live in examples or hosts, never in the publish set.

## Status

Pre-0.1 — expect breaking changes without notice. The 0.1 gate is
functional completeness: multimodal I/O and subagents.

## Contributing

See [AGENTS.md](./AGENTS.md) for layout, commands, layering rules, and workflow.

## License

Dual-licensed under `MIT OR Apache-2.0`.
