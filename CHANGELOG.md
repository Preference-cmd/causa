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
  self-contained behavior ports (`ModelGateway`, `Tool`,
  `DynamicToolSource`, `CallControl`, budget seams). Zero I/O;
  `#![deny(missing_docs)]`.
- **Context / harness separation, first batch (Slice 6.5 Phase D–E)**:
  the session aggregate moved from the kernel to `causa-runtime` —
  `ConversationState` (single active slot, completed-only history,
  commit-time ordering), `SealedResult`, `TurnSequence`,
  `ConversationVersion`, and the `ConversationStore` archive port are
  runtime vocabulary now; the kernel keeps the facts
  (`TurnContext` / `TurnSnapshot`), the validated recovery entries
  (`from_validated_blocks` / `validate_blocks` are public), and the
  shared `merged_frame` projection. `TurnSnapshot` describes the record
  itself only: `turn_sequence` left the snapshot and lives in the
  runtime's `HistoryEntry { sequence, snapshot }` (the store port saves
  entries via `save_entry` / `load_entries`). A paused turn carries a
  single serialization-ready **`Continuation`** (`PausePoint`,
  `accounted_tool_calls`, hook `PreparedApproval` with awaiting calls and
  saved rejections, `queued_inputs`) — the duplicated snapshot is gone,
  and trace trimming no longer affects resume position or quotas. Both
  resume entries consume the complete paused outcome plus a
  `ResumeRequest { decision, inject }`; validation runs before any
  execution and rejections return the untouched paused material
  (`ResumeRejection`). Lower limits stop the turn before external
  execution. Pre-6.5 wire payloads (snapshots with `turn_sequence`,
  pause outcomes with `snapshot` + `reason`) are migrated by explicit
  extraction or rejected, never silently converted.
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
- **Reference budget & interaction ownership (Slice 13)**: `FramePolicy`,
  `WindowBudget`, `FrameError`, `Compaction`, `CompactionInput`,
  `CompactionOutput`, `CompactionError`, `TokenCounter`,
  `TurnInteraction`, and `BatchDecision` are the reference harness's
  opinions, not cross-harness kernel contracts — they moved from
  `causa-kernel` to `causa-runtime` (new `budget` / `interaction`
  modules with explicit root re-exports; no kernel re-export or reverse
  alias remains). `FramePolicy::materialize` now composes the public
  lossless `TurnContext::frame` and replaces only the projected block
  list, so the deterministic frame identity is unchanged. The runtime's
  crate docs state the four optional usage paths (reference execution,
  standalone tool execution via `ToolExecutor::execute_with_limits`,
  reference session, observation).
- **Tool results split from continuation actions and host limits (Slice
  13)**: a shared `Tool::execute` / `execute_with_store` now returns the
  recorded `ToolResultPayload` only, and `DynamicToolSource::invoke` /
  `invoke_with_store` return `Result<ToolResultPayload,
  ToolExecutionError>` — what an `UnknownOutcome` result does next and
  how much output to retain are reference-harness configuration, not
  declarations on the capability. The action lives in
  `TurnPolicy.unknown_outcome` (`UnknownOutcomeConfig`: a `Stop` default
  plus per-executed-name overrides — the name after hook / rewrite /
  resume decisions, full namespace for dynamic tools, never the draft's
  name or a value read out of a result body); results always commit
  first, `Stop` interrupts afterwards, `Continue` only allows the next
  round and never re-runs a call, rewrites it into a success, or cancels
  siblings. Retention lives in `ExecutionOptions`
  (`tool_output_limits` fallback plus `tool_output_limits_overrides` —
  the explicit override is the chosen limit even when larger than the
  fallback; the executor no longer reads tool declarations).
  `HookOutcome.rejected`, `BatchDecision::Reject.results`, and
  `PreparedApproval.rejected` carry `ToolResultPayload`s plus per-call
  `UnknownDecision` entries for `UnknownOutcome` results: new host
  inputs may omit entries (resolved by configuration), while a
  checkpoint fixes exactly one action per saved `UnknownOutcome` and
  resume validation refuses foreign, duplicate, or non-covering
  decision sets before anything executes. The continuation's serialized
  shape is unchanged — a private runtime DTO writes and reads the old
  `{result, policy}` entries (fixed action for unknown entries,
  canonical `Stop` otherwise), so existing pause material round-trips.

### Removed

- `IsolationLevel` and `Tool::isolation_level` (Slice 13): no executor
  ever read the declaration, and panic capture plus a call deadline are
  not process isolation — the "declaration the driver obeys" claim was
  unfounded. No replacement enum or subprocess framework is provided.
- `causa_runtime::defaults::{NoopTokenCounter, NoopCompaction}` (Slice
  13): no consumers, and a zero counter is behaviorally distinct from no
  counter — the executor's chars/4 fallback estimates real sizes and can
  trigger truncation, while a zero counter never does. Hosts that need a
  zero estimate keep their own `TokenCounter` implementation. The
  `defaults` module is private now; the chars/4 fallback keeps its
  algorithm unchanged as a crate-internal function, and the unused
  `placeholder_token_estimate` blocks wrapper is gone.
- Tool-side action and retention declarations (Slice 13): the kernel's
  `ToolExecutionOutcome` envelope, `UnknownOutcomePolicy`, and
  `ToolOutputLimits` leave the kernel's public interface, and
  `Tool::unknown_outcome_policy` / `Tool::output_limits` are gone —
  hosts migrating a tool that declared `Continue` place an entry in
  `TurnPolicy.unknown_outcome.overrides`, and a tool that declared
  output limits gets an `ExecutionOptions.tool_output_limits_overrides`
  entry (same truncation algorithm and artifact spill as before; the
  default remains no truncation). `UnknownOutcomePolicy`,
  `UnknownOutcomeConfig`, `ToolOutputLimits`, and `UnknownDecision` are
  root-exported by `causa-runtime`.

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
