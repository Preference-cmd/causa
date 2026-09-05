# Changelog

All notable changes to the Causa crate family are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
During the 0.x series, breaking changes to the wire serde shapes
(`ContextEvent` / `TurnResult` / `TurnTrace` family) bump the **minor**
version and are flagged at the top of the release notes — those shapes are a
load-bearing external contract pinned by `causa-runtime` serialization tests.

## [Unreleased]

Planned as **0.1.0** — the release gate is functional completeness
(slices 6.5 multimodal I/O and 8 subagents), per the slice 11 proposal.

### Added

- **`causa-kernel`** — the facts layer: ContextBlock conversation fact
  machine, turn state machine with deterministic projections, and the
  self-contained behavior ports (`ModelGateway`, `ConversationStore`,
  `Tool`, `DynamicToolSource`, `CallControl`, budget seams). Zero I/O;
  `#![deny(missing_docs)]`.
- **`causa-protocol`** — kernel-native wire translation for Anthropic
  Messages, OpenAI Chat Completions, and OpenAI Responses.
- **`causa-runtime`** — the reference driver: bounded model retry with
  exponential backoff, tool batch dispatch (dedup / allow / deny filter
  chain), streaming, approval pause with recoverable interruption, and
  `ContextEvent` projections for UI / observability consumers.
- **`causa-provider`** — reqwest `ModelGateway` adapters for the three
  protocols, with transport timeouts and classified error mapping.
- **`causa-mcp`** — MCP client (`McpToolSource`) over stdio, Streamable
  HTTP, and in-process I/O; tools namespaced `mcp_{server}_{tool}` into
  the executor with `tools/list_changed` cache invalidation.
- Kernel-face prompt caching: `CacheDirective` rides every model
  request — the Anthropic translation renders three-anchor
  `cache_control` breakpoints (tool surface, system prefix, latest
  stable conversation message); OpenAI-family renderers accept the
  directive as a documented no-op (server-side automatic caching).
- Structured output: `GenerationOptions::output_schema` maps to Chat
  `response_format` and Responses `text.format`; schema validation and
  corrective retry stay host-side.
- Tracing baseline: `agent.turn` / `agent.round` / `agent.attempt` /
  `agent.tool` / `agent.http` spans across driver, executor, and
  gateway — ids and names only, never message payloads.
- CI: fmt / clippy / test matrix (ubuntu + macos), MSRV 1.96 job, and the
  dependency-direction guard; a manual-trigger publish workflow with a
  full dry-run pass.
- Five examples doubling as runnable documentation; this CHANGELOG.

### Changed

- Brand: **Causa** (formerly Archy) — every crate renamed to `causa-*`;
  project pages live under the Project inceptae domain.
