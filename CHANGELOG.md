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
- **Multimodal vocabulary (Slice 6.5)**: the block content vocabulary is
  frozen as **Parts** — `BlockContent::Text(TextPayload)` is replaced by
  `BlockContent::Parts(Vec<ContentPart>)` with `ContentPart = Text |
  Media(MediaRef)`; one logical message's mixed content commits as one
  fact block through the new `TurnContext::append_parts` door
  (`append_input` is now its single-text-part sugar). Media travels as
  cheap durable references: `ToolResultPayload.media` (serde-additive)
  carries `MediaRef`s whose bytes live only in the host's asset store;
  snapshots stay proportional to reference count. On the render path,
  the provider's host-injected `MediaResolver` (+ the gateway's
  `HashMap` asset-table impl) resolves references into a `MediaSet`
  consumed by all three renderers: Anthropic `image` blocks (including
  native embedding inside `tool_result` content), OpenAI Chat
  `image_url` and Responses `input_image` (tool-result media hoists
  into a provenance-labeled user message right after its tool message);
  missing, non-image, oversized, or non-user-position media degrades to
  a deterministic `[media: …]` placeholder, decided once in the shared
  walk. MCP tool images persist through `DynamicToolSource::
  invoke_with_store` (additive default method) into the host store and
  return as references; the executor's textual truncation never touches
  media references. Offline closed loop: `cargo run -p causa-runtime
  --example media_feedback`.
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

### Fixed

- Protocol translation resolves tool result ids through a
  `(turn_id, call_id)` map instead of a bare `call_id` map: two turns
  calling the same tool with the same arguments keep their own provider
  ids instead of the later turn's overwriting the earlier turn's.
- Tool-output truncation sizes the retained head+tail against the
  declared token budget — notice and JSON-string wrapping measured, with
  re-estimation until the output fits — so truncated content actually
  shrinks and lands at or under `max_tokens`; a budget smaller than the
  notice itself leaves the notice as the defined floor.
- Batch completeness is enforced by the runner on every dispatch path:
  the withheld decision must cover the emitted batch exactly (no silent
  drops, duplicates, or foreign calls), and an approval resume must match
  the turn's unanswered tool calls — violations interrupt as
  `RunnerInvariantViolation` before anything executes.
- The driver re-snapshots the executor's tool surface at every round
  boundary (first model phase keeps the host-declared baseline), so
  dynamic-catalog changes made during a turn reach the next model
  request; retries within a round reuse the round's snapshot.
