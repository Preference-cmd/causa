# Changelog

All notable changes to the Causa crate family are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
During the experimental `0.0.x` series, patch releases may break Rust API
and wire-serde compatibility. Breaking changes and migration notes appear
at the top of each release. Starting with `0.1.0`, breaking wire changes
(serialized context and tool material) bump the **minor** version.
Before 0.1, obsolete APIs and formats may be removed without compatibility
adapters or data converters.

## [Unreleased]

Tool changes are facts in the context. The runner records which tools the
model is offered, and from where, as `ToolChange` blocks; every renderer
reads the tools from the frame, and Anthropic Messages and OpenAI Responses
can keep the request's `tools` unchanged while the set changes.

### Breaking

- New block shape `BlockContent::ToolChange(ToolChange)`, serialized as
  `"shape": "tool_change"`. `ToolChange` and `ToolDefinition` join the
  stable serialized set; `ToolDefinition` moves into `causa_kernel::context::tools`
  (the root path `causa_kernel::ToolDefinition` is unchanged).

  Migration: an exhaustive `match` over `BlockContent` adds a
  `ToolChange` arm. Stored contexts load unchanged.

- `ModelRequest::tool_surface` is removed. The frame is the only source of
  model-facing tools: the tool state is the fold of the frame's
  `ToolChange` blocks (`causa_kernel::current_tools`).

  Migration: a host that builds requests itself puts a `ToolChange` holding
  its tools before the conversation, for example
  `ToolChange::between(&[], &definitions)`. Code that read
  `request.tool_surface` reads `current_tools(&request.frame.blocks)`.

- The `render` functions of `causa-protocol` (`anthropic`, `openai_chat`,
  `openai_responses`) no longer take a tool surface argument.

- The runner appends a `ToolChange` block after binding whenever the bound
  tools differ from the context's tool state, and reports it in
  `RunEvent::BlocksCommitted`. A context that starts without tool facts
  gains the declaration in its first round. A preparer whose frame does not
  add up to the bound tools stops the round with `PrepareFailed`.

  Migration: a preparer that replaces or filters blocks keeps the
  `ToolChange` blocks. Code that counted the blocks of a run, or the
  `BlocksCommitted` events, expects the extra tool facts. A failed append
  is reported as the new `TurnInterruption::ToolChangeCommitFailed`.

### Added

- `ToolChange::new`, `ToolChange::between`, `ToolChange::apply_to` and
  `current_tools`.
- Native tool changes, off by default:
  `AnthropicMessagesGateway::with_native_tool_changes(true)` keeps `tools`
  at the declaration plus a deferred placeholder tool, renders each later
  change in place as a `system` message of `tool_removal` /
  `tool_addition` blocks, and adds `inline-tools-2026-09-15` to
  `anthropic-beta` after the host's values.
  `OpenAiResponsesGateway::with_additional_tools(true)` keeps `tools` at
  the declaration and renders each later change as an `additional_tools`
  input item; a change that removes or redefines a tool falls back to
  rewriting `tools`. Chat Completions always sends the current set.
- Every renderer rejects a tool change between a tool call and its result,
  and a change that names a tool twice or with an empty name.

### Changed

- CI runs on Linux only. The workspace has no platform-specific code, and
  the macOS jobs repeated the Linux results while waiting on scarce runners,
  which blocked a release upload.
- Workflows name the runner image `ubuntu-26.04` instead of `ubuntu-latest`,
  so the image changes only when the workflow does.

### Fixed

- The OpenAI Responses stream decoder accepts a reasoning item that streams
  its raw reasoning as `reasoning_text` content parts
  (`response.content_part.added`, `response.reasoning_text.delta` /
  `.done`), as DeepSeek's Responses API does on every call. Such a stream
  failed as `Undecodable` before; its deltas are now `ReasoningDelta`, under
  the same order and identity checks as a message's parts. The reasoning
  item is still sealed whole for replay.
- A Responses reasoning item without summary text contributes its
  `reasoning_text` content to the output's reasoning observation, in the
  batch and streamed forms alike.

## [0.0.4] - 2026-10-13

Preparation of the public surface for `0.1`: struct-like variants of
library-built enums are `#[non_exhaustive]`, the serialized forms held stable
from `0.1.0` are named, the tool port states its concurrency requirements,
and interrupted runs are documented. This release breaks the `0.0.3` Rust
API for code that matches those variants.

### Breaking

- The struct-like variants of nine enums that only the library builds are
  `#[non_exhaustive]`: `RunEvent`, `TurnInterruption`, `ToolProcessingError`,
  `ToolCatalogError`, `ToolRegistryError`, `BatchError`, `EditError`,
  `ToolResultError` and `ModelBlockError`. A pattern over one of these
  variants must end in `..`, and the variant can no longer be written as a
  value outside its crate.

  Migration: add `..` to each such pattern, for example
  `TurnInterruption::MaxModelRounds { limit, .. }`. A test that compared an
  error with a constructed variant uses `matches!` instead.

- `ToolOutput` and `ToolOutputMeta` leave an unset optional field out of
  the serialized form instead of writing `null`, like the rest of the
  context vocabulary. Reading accepts both, so stored contexts still load;
  a consumer that compares serialized text sees the difference.

### Changed

- The `Tool` port now states its requirements: `execute` must not block the
  thread and must return promptly once its control is stopped. Calls of one
  batch are concurrent, not parallel, and unbounded. Dispatch itself is
  unchanged; a tool that blocks the thread was never stopped by cancellation
  or a timeout, and tests now pin that.

- The serialized forms held stable from `0.1.0` are now named: `Context` and
  every type it contains, listed in the `causa-kernel` crate docs. Adding an
  optional field is not a break. Other serializable types (model output,
  request options, identifiers, error kinds) carry no promise, and their
  docs say so. No serialized form changes.

### Added

- An interrupted runs guide on the website and the offline
  `interrupted_batch` example: what batch each interruption returns and how
  a host answers the pending calls and commits it.
- `ToolBatch::validate_complete` checks the batch invariants and that every
  entry has a result, returning `BatchError::Incomplete` otherwise.

## [0.0.3] - 2026-10-12

Real provider streaming, reasoning replay and host-written reasoning
settings, an optional retrying gateway, classified model-call failures, and
one evolution rule per public type. This release breaks the `0.0.2` Rust API
and the serialized form of tool results.

### Breaking

- Four kernel enums serialize in snake_case, like the rest of the context
  vocabulary. Two of them are stored in context material, so a context saved
  by an earlier build that holds a tool result no longer reads; no reader
  for the old form is provided. Migration: rewrite the strings in stored
  data, or discard it.

  | type | before | now |
  |---|---|---|
  | `ToolResultStatus` (in `ToolResultPayload.status`) | `Succeeded`, `Failed`, `Rejected`, `Cancelled`, `TimedOut`, `UnknownOutcome` | `succeeded`, `failed`, `rejected`, `cancelled`, `timed_out`, `unknown_outcome` |
  | `ArtifactKind` (in `ArtifactRef.kind`) | `FullOutput`, `PipeCache`, `Binary` | `full_output`, `pipe_cache`, `binary` |
  | `ModelItemContent` | `{"Parts": …}`, `{"ToolCall": …}` | `{"parts": …}`, `{"tool_call": …}` |
  | `ModelInvokeErrorKind` | `Transient`, `TimedOut`, `Cancelled`, `Permanent`, `Undecodable`, `InvalidRequest`, `UnknownOutcome` | `transient`, `timed_out`, `cancelled`, `permanent`, `undecodable`, `invalid_request`, `unknown_outcome` |
- Messages no longer repeat text that is available as a value.
  `TurnInterruption::ModelStopUnrecognized` prints a fixed sentence; the
  provider's stop value, which may be any length and hold control
  characters, is read from `output.raw_stop_reason`. `EditFailure` prints
  `context edit failed` and reports the `EditError` as its source, so a
  printed error chain shows the reason once. Migration: read the field, or
  print the source chain, where the code relied on the message.
- `ArtifactKind` is `#[non_exhaustive]`; a `match` on it outside the crate
  needs a wildcard arm.
- `GenerationOptions` leaves `temperature` and `max_tokens` out of its
  serialized form when they are unset, as it already did for its other
  fields. The earlier form wrote them as `null` and still reads back.
- A tool batch that finished processing is committed before a stop is
  reported. When cancellation or a deadline is first seen after every tool
  returned and every after-processor ran (or none is configured), the run
  still returns `Cancelled`
  or `DeadlineExceeded` and starts no further model round, but the results
  are in the returned context and `uncommitted_tool_batch` is `None`. Before,
  the complete batch was handed back and a host that saved the context
  without committing it lost the results. `BoundTools::process` likewise
  returns `Ok` for such a batch instead of a control error. A stop while
  tools run, or before the last after-processor finishes, still returns the
  batch uncommitted. Migration: a host that committed such a batch by hand
  no longer receives it.
- `Tool` and `DynamicToolSource` each have one entry method. `Tool::execute`
  and `DynamicToolSource::invoke` take the host's artifact store as a third
  parameter, `store: Option<&dyn ArtifactStore>`; `execute_with_store` and
  `invoke_with_store` are removed. Migration: add the parameter to every
  implementation (a tool that stores nothing ignores it), rename an
  overridden `_with_store` method to the plain name, and pass `None` or the
  store when calling a tool directly. A direct call to `invoke` used to drop
  the store without saying so; it can no longer be left out.
- Reasoning settings are sent as the host wrote them; Causa no longer checks,
  maps or derives them.
  - `ReasoningEffort` is removed and `GenerationOptions::reasoning_effort` is
    `Option<String>`, written to Chat `reasoning_effort`, Responses
    `reasoning.effort` and Anthropic `output_config.effort`. Migration:
    replace `Some(ReasoningEffort::High)` with `Some("high".into())`; the
    serialized form of the seven old values is unchanged. What a stored
    value sends can still differ: a persisted `"none"` now sends
    `output_config.effort: "none"` on Anthropic, where it sent
    `thinking: {"type": "disabled"}` and no effort. A host that stored
    `"none"` to turn thinking off should clear the effort and put the
    disabled `thinking` object in `extra_body`.
  - New `GenerationOptions::extra_body`: a JSON object merged into the
    request body. An object joins the object the adapter wrote under the same
    key (`{"reasoning": {"summary": "auto"}}` beside the effort); any other
    value the adapter wrote is kept and arrays are never merged. The rule
    goes by key name: Chat `max_tokens` is sent beside the adapter's
    `max_completion_tokens`. `output_schema` is kept whole: a key beside the
    schema (`description`) is merged, a key inside it is not, and `strict`
    stays `false`; a host that needs another `strict` leaves `output_schema`
    unset and sends the whole format object. Responses `include` and `store`
    are written by the default render options and so win over the same keys
    here. A `null` is sent as given and does not unset a key the adapter
    wrote. On a `stream` call the adapter sets `stream: true` and, on Chat,
    `stream_options.include_usage: true` whatever `extra_body` says; a
    `stream_options` that is not an object is replaced. On `invoke` neither
    key is written, so both are sent as given. A serialized
    `"extra_body": null` reads as empty.
  - Anthropic: an effort no longer adds `thinking: {"type": "adaptive"}`, and
    `none` no longer sends `thinking: {"type": "disabled"}`. Migration: put
    the `thinking` object you want in `extra_body`. A host that relied on the
    derived object and sends none now gets the provider's default.
  - Removed checks: Anthropic no longer refuses the effort `minimal`, and no
    adapter refuses a temperature beside an effort. The provider decides,
    and its refusal arrives as its error.
  - `PrefixMismatchBehavior`, `AnthropicRenderOptions::prefix_mismatch_behavior`
    and `KernelHttpGateway::with_prefix_mismatch_behavior` are removed.
    Migration: add `"block_binding": {"prefix_mismatch_behavior": ...}` to
    the `thinking` object in `extra_body`; the beta header is still listed
    with `with_beta`.
- Structs that may gain a field are `#[non_exhaustive]`. Outside the defining
  crate a struct literal no longer compiles, with or without
  `..Default::default()`, and a pattern needs `..`. Fields stay public.
  Migration: build the value with the constructor below, then assign any
  other field.

  | type | construct with |
  |---|---|
  | `GenerationOptions`, `ToolOutputMeta`, `ToolExecutorOptions`, `StreamLimits`, `AnthropicRenderOptions`, `ResponsesRenderOptions`, `ChatRenderOptions` | `Default::default()` |
  | `BlockMeta` | `Default::default()` or `BlockMeta::from_source(source)` |
  | `ModelOutput` | `ModelOutput::new(response, stop_reason)` |
  | `ModelRequest` | `ModelRequest::new(invocation_id, frame, model)` |
  | `ModelUsage` | `ModelUsage::new(input_tokens, output_tokens)` |
  | `ReasoningPayload` | `ReasoningPayload::new(text)` |
  | `ToolResultPayload` | `ToolResultPayload::new(call_block_id, status, output)` |
  | `ToolOutput` | `ToolOutput::new(content)` |
  | `ToolDefinition` | `ToolDefinition::new(name, description, parameters)` |
  | `SealedPart`, `WireMeta` | `new(format, item_or_fields)` |
  | `ToolCallPayload`, `ToolCallDraft` | `new(tool_name, arguments)` |
  | `ArtifactRef` | `ArtifactRef::new(id, size_bytes, kind, persisted)` |
  | `ArtifactHint` | `ArtifactHint::new(tool_name, call_block_id, kind)` |
  | `ModelItem` | `ModelItem::new(content)`, `text`, `tool_call` |
  | `ToolCallContext` | `ToolCallContext::new(call_block_id, input)`, `from_declaration` |
  | `PrepareError` | `PrepareError::new(message)` |
  | `RoundInfo`, `ProcessorContext` | `new(...)`, each field in declaration order |
  | `TurnLimits` | `TurnLimits::new(max_model_rounds, max_tool_calls)` or `Default::default()` |
  | `SseEvent` | `SseEvent::new(event, data)` |
  | `ToolSurface`, `MediaRef`, `MediaPayload`, `TurnRunOptions`, `RetryPolicy` | their existing constructors |

  `ModelInvokeError`, `TurnOutcome`, `RejectedToolResult`, `EditFailure` and
  `ErrorDetail` are also `#[non_exhaustive]`; callers read them and do not
  build them, except `ModelInvokeError::new`.
  `ToolOutputMeta` now implements `Default`.
- Enums a caller only reads are `#[non_exhaustive]`, so a later variant is not
  a breaking change: `ModelInvokeErrorKind`, `StreamDelta`, `RunEvent`,
  `TurnInterruption`, and the error enums `ModelBlockError`,
  `ToolResultError`, `ProcessorError`, `BatchError`, `EditError`,
  `SourceError`, `StoreError`, `ToolRegistryError`, `ToolCatalogError` and
  `ToolProcessingError`. Migration: add a wildcard arm to each `match` on one
  of them. Treat an unknown `ModelInvokeErrorKind` as not retryable.
- `ModelInvokeError::kind()` is removed; read the public `kind` field.
- `Display` for `ModelInvokeError` prints the message alone (it was
  `"{kind:?}: {message}"`), and `Display` for `TurnInterruption` prints one
  sentence per cause (it forwarded `Debug`). Both now implement
  `std::error::Error` through `thiserror`. Code that parsed either output
  must read the fields instead.
- `ModelInvokeErrorKind` gains `Undecodable`: the provider answered and the
  adapter could not use the answer (a malformed 2xx body or stream, a stream
  that breaks the protocol's rules, a response over a size limit). These
  failures were `Permanent`, which now means that another attempt will not
  help: the provider refused (the error carries a status, or a code when it
  arrived inside a successful response), or the transport could not build
  the request (neither is set). Neither kind is retried. Migration: add
  an `Undecodable` arm to exhaustive matches on `ModelInvokeErrorKind`; code
  that treated `Permanent` as "check the credential, model or endpoint"
  should give `Undecodable` a different message, since those settings are not
  the cause.
- `ModelInvokeError` gains two public fields, `status: Option<u16>` (the HTTP
  status of a non-2xx response) and `code: Option<String>` (the provider's
  error `code` or `type`, kept only when it is a short identifier), with the
  builders `with_status` and `with_code`. Migration: construct errors with
  `ModelInvokeError::new` and the builders in place of a struct literal. A
  host can now tell 401 from 404 without parsing the message.
- An error event inside a successful stream now carries the provider's
  message, read by the rule for an HTTP error body: `error.message`, at most
  200 characters. The text is `"provider error ({code}): {message}"`, also
  for a failed Responses call that was not streamed.
  Before, the message was dropped and a code outside a fixed list was shown
  as `unknown`; now any code that is a short identifier is shown. Hosts that
  log these errors will see provider text they did not see before.
- Provider text in an error message is one line: each run of control
  characters becomes a space, on HTTP errors and stream error events alike.
  The Unicode line and paragraph separators and the bidirectional
  formatting characters are replaced the same way.
- A response the token limit stopped inside a tool call is returned, not
  failed. When the stop maps to `MaxTokens` and the arguments of the last
  call are not complete JSON, that call is left out of the output and the
  rest is returned with `MaxTokens`; before, all three protocols failed the
  call as `Undecodable`, batch and stream. A stream has already delivered
  `ToolCallDelta` values for the call that is left out, so a host that
  builds calls from deltas must take the call list from `Done`. Incomplete
  arguments on any other call, or under any other stop, are still
  `Undecodable`.
- A transport failure before any response arrived, such as a connection the
  peer closed, is `Transient` where it was `Permanent`, so a retry policy
  now retries it. The request may have reached the server; a host that
  cannot tolerate a repeated call should not retry `Transient`. Only a
  request the client could not build is still a transport `Permanent`.
- An error inside a successful response is classified by its `code` and its
  `type` together: `code` is looked up first, `type` second. Before, `type`
  was ignored when a `code` was present, so `invalid_request_error` with
  `context_length_exceeded` was `Permanent` in a stream and `InvalidRequest`
  over HTTP 400, and `server_error` with a specific code was not retried. A
  `code` that is not a short identifier no longer hides the `type`.
  `timeout_error` is `Transient`.
- A 2xx response whose body is an error envelope returns the provider's
  error with its code and message, on `invoke` and on a `stream` call that
  was answered without an SSE content type. Before, both were `Undecodable`
  with none of the provider's text. A null `error` in a body or a stream
  event is not an error.
- A Chat response with more than one choice is `Undecodable` on `invoke`, as
  it already was on `stream`. Before, `invoke` returned the first choice and
  dropped the rest while the usage counted all of them. Sending `n` above 1
  through `extra_body` is therefore refused when the response arrives.
- Using a decoder against its contract (a zero stream limit, a stream
  request body that is not an object, input after the end) is
  `InvalidRequest`, the kind the renderers already use for a request refused
  before sending; it was `Permanent`. `InvalidRequest` therefore means the
  request is malformed, whether the provider or the adapter refused it.
- `RunControl` and the `causa_runtime::control` module are removed.
  `TurnRunner::run` and `run_streaming` take a kernel `CallControl` by value.
  Migration: replace `RunControl::new(token, deadline)` with
  `CallControl::new(token, deadline)`, and `control.call_control()` with
  `control.clone()`. `should_stop` and `remaining_turn_time` have no
  replacement; use `CallControl::check` and `CallControl::deadline`.
- `ToolBatch::resolve_at` returns `Result<(), RejectedToolResult>`. A
  refused result is handed back with its `BatchError`, so callers no longer
  clone a payload to keep it. Migration: read `rejected.error` where the code
  matched on `BatchError`, and `rejected.result` to recover the payload.
- Gateway errors for non-2xx responses follow one rule on `invoke` and
  `stream`: `"{provider} error {status}: {error.message}"` when the body is
  the documented JSON error envelope (message truncated to 200 characters),
  and `"{provider} error {status}"` otherwise. `invoke` no longer falls back
  to the raw body, and `stream` now includes the envelope message.
- `ToolSurface` is defined in the kernel's tool port. The public path
  `causa_kernel::ToolSurface` is unchanged.
- `ReasoningPayload` no longer has a `signature` field. A provider signature
  now lives only inside the sealed item it belongs to
  (`ContentPart::Sealed`): Anthropic `thinking`, Responses `reasoning`, Chat
  `reasoning_content`. Payloads serialized before this change still
  deserialize; the field is ignored.
- `causa_protocol::translation::anthropic::DEFAULT_MAX_TOKENS` is removed, and
  the Anthropic renderer rejects a missing `GenerationOptions::max_tokens`
  with `InvalidRequest` before HTTP instead of sending 4096. Set an explicit
  limit for Anthropic requests.
- The three renderers take a new renderer-options argument
  (`AnthropicRenderOptions`, `ResponsesRenderOptions`, `ChatRenderOptions`
  in `causa_protocol::translation::render_options`), and the provider gateway
  configs host them; `OpenAiChatGatewayConfig` and
  `OpenAiResponsesGatewayConfig` are no longer unit structs.
- The Anthropic and Responses renderers no longer hoist a `system` block that
  sits after the conversation content: only the leading system run renders into
  the top-level `system` parameter (Anthropic) or `instructions` (Responses),
  and an appended system block renders at its position as a `role: "system"`
  message or input item. A system block appended mid-conversation therefore no
  longer rewrites the top-level parameter, and one that lands between a
  `tool_use` and its `tool_result` now fails `InvalidRequest` before HTTP
  instead of being hoisted out of the way. The Anthropic top-level `system` is
  always the block-array form — one `{"type": "text", "text": ...}` block per
  leading system part, no `"\n"` join — where it used to be a joined string
  without a cache directive and a one-element array with one. With
  `CacheDirective::StablePrefix` the system marker lands on the last block of
  that array, and the message marker lands on the last content block that is
  not a sealed item, searching backward to an earlier message when the anchor
  holds only sealed items; a sealed item is never modified.
- The Anthropic and Chat renderers omit a whitespace-only assistant text part
  from the request (as `pi-ai` does); the part stays in the `Context` and the
  Responses renderer keeps it. This is a listed exception to the round-trip
  property, and with thinking enabled it is a framework-made prefix change for
  a turn that contains such a part. A Chat assistant message left with
  neither `content` nor `tool_calls` by that omission is not rendered.
- Responses requests now send `include: ["reasoning.encrypted_content"]` and
  `store: false` by default, which changes the body of every existing
  Responses request. `with_encrypted_reasoning(false)` and `with_store(None)`
  restore the previous body.
- Anthropic streaming fails as `Undecodable` on a content-block delta of a type
  the decoder has no rule for, where it used to drop the delta. A block is
  rebuilt from its start event and its deltas, and a block missing part of its
  content must not be committed and replayed as sealed material; the turn
  commits nothing and the batch entry point is unaffected. `input_json_delta`
  is now applied to every block that declares an object `input`, so a
  non-client tool block streams to the same sealed item the batch parser
  keeps instead of failing with a type conflict. Unknown SSE event types are
  still ignored.
- `StreamDelta::Error` now carries the gateway's own `ModelInvokeError` as
  `Error { error }` instead of the `{ kind, message }` pair, so an in-stream
  failure keeps its classification and any server-requested retry delay.
  `ModelInvokeError` gained `Clone` and a public
  `retry_delay: Option<Duration>` filled from the HTTP `retry-after-ms` /
  `retry-after` headers; `ModelInvokeError::new` leaves it `None` and
  `with_retry_delay` sets it. The type still has no serde derive. Match sites
  and constructions that name the old fields must update.
- HTTP 409 now classifies as `Transient` in the built-in provider gateways,
  where it was `Permanent`. An `x-should-retry: false` response header makes
  any other status `Permanent` (400/422 stay `InvalidRequest`), and
  `x-should-retry: true` makes any status `Transient`.
- Stop-reason classification is explicit and shared by batch and streaming.
  `ModelStopReason` gained `ContextWindowExceeded` and `Unrecognized`; a
  value the kernel has no meaning for is no longer degraded to `EndTurn` by
  `parse_anthropic_response` or `parse_openai_chat_response` (the streaming
  decoders used to reject it as a permanent error instead — the two entry
  points now agree). Chat additionally accepts `end` as `EndTurn` and
  `function_call` as `ToolUse`. `ModelOutput` gained
  `raw_stop_reason: Option<String>` (`stop_reason`, `finish_reason`, or the
  Responses `status` / `status.reason` form such as
  `incomplete.max_output_tokens`), and an unknown Responses
  incomplete reason is reported as `Unrecognized` with the output intact
  instead of failing as `UnknownOutcome`. The runner commits and continues
  only for `EndTurn` and `ToolUse`; `MaxTokens`, `Refusal`,
  `ContextWindowExceeded` and `Unrecognized` return an interruption carrying
  the complete uncommitted output — the new
  `TurnInterruption::ModelContextWindowExceeded` and
  `TurnInterruption::ModelStopUnrecognized` join `ModelMaxTokens` and
  `ModelRefusal`. `StreamDelta::Done` no longer carries `stop_reason`; the
  authoritative value is on its `final_output`. Hosts that list every
  `ModelStopReason` variant, match it exhaustively, or construct
  `ModelOutput` must update.
- `ModelResponse` is now an ordered list of items:
  `ModelResponse { items: Vec<ModelItem> }`, where a `ModelItem` holds
  `ModelItemContent::Parts(Vec<ContentPart>)` or
  `ModelItemContent::ToolCall(ToolCallDraft)` plus an optional
  `wire: Option<WireMeta>`. The `text` and `tool_calls` fields are gone; use
  `ModelItem::text`, `ModelItem::tool_call`, `ModelResponse::text()` (all text
  parts concatenated in order, no separator) and `ModelResponse::tool_calls()`.
- One model item becomes one block, in item order (`to_blocks`/`block_count`).
  A response with text, a call and more text is three blocks, not a text block
  followed by call blocks. Only zero-length text is dropped; whitespace-only
  model text is now kept verbatim, where it used to be omitted as blank.
- `BlockMeta` gained `wire: Option<WireMeta>` (`format`, `fields`), the
  residue of the provider item that the block's facts do not hold, and
  `ContentPart` gained `Sealed(SealedPart)` (`format`, `item`), a model-issued
  wire item the kernel carries and never interprets. Hosts that list every
  `BlockMeta` field or match `ContentPart` exhaustively must update. The
  stored format gained these fields; blocks written before this change still
  load (`wire` defaults to absent and is not written when absent).
- Model output the host has not edited is rendered back value-equal to what
  the provider returned, item for item and in order, on the format it came
  from. OpenAI Responses `message` items keep `id`, `phase`, `annotations`
  and their boundaries, and `function_call` items keep `id` and the original
  `arguments` string (sent unchanged while it parses to the fact's value).
  Anthropic `text` and `tool_use` blocks keep every other field, such as
  `citations` and `caller`. Chat Completions assistant messages keep
  `reasoning_content` and every other message-level key, which fixes the
  400 DeepSeek returns on the second request of a thinking-mode tool loop.
  Fields kept from one protocol are used only by renderers of that protocol;
  blocks carrying another format's `wire`, or edited blocks, render from facts.
  Unknown fields survive without a code change.
- Renderers no longer join text across block seams, for any role: adjacent
  same-role blocks render as separate content blocks (Anthropic), messages
  (Chat user text) or items (Responses). The Chat assistant message joins its
  texts with no separator where it used to insert `\n`.
- A `ContentPart::Sealed` part in a user- or system-position block is rejected
  with `InvalidRequest` before any HTTP; one of another protocol's format in
  assistant position is skipped. At that change reasoning replay was not
  included and those items fed only the reasoning observation; replay landed
  later in the same unreleased series — see the entries above.
- Streaming assembles the same material as batch parsing: Chat streaming
  carries `reasoning_content` into the assembled message, and Anthropic
  streaming keeps the extra fields of `text` and `tool_use` blocks, including
  `citations_delta`. Fields a stream cannot deliver (a Chat `content: null`,
  per-call `index`) are not invented.
- `with_base_url` no longer appends the `/v1` version segment. The base URL
  must carry it: pass `https://host/v1` where `https://host` worked before.
  The appended paths are now `/messages`, `/chat/completions` and
  `/responses`, so a provider base that already ends in `/v1` no longer
  resolves to `/v1/v1/...`. Default endpoints and `with_endpoint` are
  unchanged. The `path` field of the `agent.http` span reports the same
  shortened protocol path.
- `GenerationOptions` adds optional `reasoning_effort`; explicit struct literals
  must include it or use `..Default::default()`. OpenAI Chat output limits now
  render as `max_completion_tokens`, including visible and reasoning tokens.
  Generation rendering rejects invalid temperatures, zero output limits and an
  Anthropic output schema before HTTP instead of dropping or coercing them.
- OpenAI-family function arguments must be an encoded, complete JSON string.
  Missing, null, blank or inline non-string arguments now fail rather than
  becoming `{}` or passing through. Zero-parameter calls must supply `"{}"`.
- The default HTTP client no longer retries protocol failures or follows
  redirects. A host-injected client retains its own retry/redirect settings.
- `KernelGatewayConfig` implementations now declare their `PROTOCOL` so the
  HTTP adapter can select the matching stream decoder. The three built-in
  configs supply it; hosts implementing a config must add it explicitly.
- `SseParser` now consumes borrowed byte slices through `decode` and validates
  EOF through `finish`. Replace `feed`/`flush`/`overflowed`; errors are returned
  explicitly, oversized events fail closed, and EOF no longer invents a final
  event delimiter. Split UTF-8 characters and LF/CRLF/CR line endings are supported.
- OpenAI Responses parsing rejects failed responses instead of accepting them
  as `EndTurn`. The known incomplete reasons `max_output_tokens` and
  `content_filter` map to `MaxTokens` and `Refusal`; an unknown incomplete
  reason is `Unrecognized` (see the stop-reason entry above).

### Added

- Chainable `with_*` setters for the optional and defaulted fields of the six
  most-built types (`GenerationOptions`, `BlockMeta`, `ToolResultPayload`,
  `ToolCallDraft`, `ToolExecutorOptions`, `TurnRunOptions`), allowing values to
  be constructed in a single expression while leaving public fields and
  constructors unchanged.
- Reasoning replay: the Anthropic parser keeps `thinking`,
  `redacted_thinking` and unknown content blocks as sealed items and the
  renderer replays them verbatim; the Responses parser keeps a `reasoning`
  item whole (and any unknown output item) and the renderer replays it; the
  Chat renderer replays sealed `reasoning_content` by default.
- `GenerationOptions::reasoning_effort` renders on Anthropic as
  `output_config.effort`; unset sends no effort.
- Render options and their gateway setters: `with_beta` (extra
  `anthropic-beta` values, one comma-separated header, empty by default),
  `with_encrypted_reasoning`, `with_store`,
  `with_reasoning_content_replay`.
- Optional `causa_runtime::RetryingGateway<G>` and `RetryPolicy`: an
  ordinary `ModelGateway` wrapper that retries `Transient` and `TimedOut`
  failures with exponential backoff and fixed 25% jitter, honors a
  server-requested `retry_delay` up to the policy's ceiling, and never retries
  once any stream delta has been yielded. It is off by default — the runner
  still makes one logical call per invocation — and the policy has no
  `Default`. The pre-0.1 runtime `RetryPolicy` removed in the
  execution-stack rewrite was a different type.
- `causa-runtime` depends on `fastrand` for that jitter and
  `causa-provider` on `httpdate` for the HTTP-date form of
  `retry-after`; both versions were already resolved in `Cargo.lock`, so
  no new package version, source or checksum is introduced.
- Optional `GenerationOptions::reasoning_effort`, a string. OpenAI Chat
  writes `reasoning_effort`; Responses writes `reasoning.effort` in batch and
  streaming requests. Omission preserves the provider default. Hosts validate
  model-specific supported values, sampling defaults and tool compatibility.
  *(At that change no reasoning-state replay existed and the Anthropic adapter
  rejected effort; both landed later in this same unreleased series — see the
  replay and effort entries above.)*
- Real HTTP SSE for all built-in Anthropic Messages, OpenAI Chat Completions
  and OpenAI Responses gateways. `stream()` yields incremental observations
  and one authoritative complete output, with cancellation/deadline control,
  bounded parsing and prompt local resource release. There is no automatic
  reconnect, hidden retry or non-streaming fallback.
- Transport-free `StreamDecoder` and configurable `StreamLimits`, exposed by
  the provider gateway through `with_stream_limits`. Text, supported reasoning,
  function-tool fragments and cumulative usage preserve complete-output semantics.
- The reqwest minimum is now 0.12.28 to configure its retry policy explicitly;
  provider uses the existing workspace futures-util library for a pull-driven
  stream with no background reader task.
- Gated local HTTP tests and an isolated public facade consumer prove that an
  increment is visible before the server sends its terminal response. Runtime
  integration checks prevent partial model output or tool execution on failure.

### Changed

- A Responses `incomplete` status whose reason is `length` now maps to
  `ModelStopReason::MaxTokens`, the same as `max_output_tokens`. A
  compatible server was observed to pass the Chat word through, which made
  a truncated output an unrecognized stop. The raw value is still reported
  as `incomplete.length`.
- A stream decode failure now ends its message with where it happened: the
  1-based position of the event and, when the event names one, its type, as
  in `stream: missing or invalid text (event 7:
  response.reasoning_summary_part.added)`. The type is provider text and is
  echoed only when it is at most 64 characters of ASCII letters, digits,
  `.`, `_` and `-`. Code matching on the exact message of a stream error
  must allow for the suffix.
- The Responses stream decoder accepts an opening part without `text` and
  an opening function call without `arguments`, reading both as empty. A
  compatible server was observed to omit the empty string, which failed the
  whole stream as `Permanent`. A present value must still be a string.
- Gateway contract purity (`causa-kernel`): doc comments no longer reference
  specific provider names, wire field names, or adapter behaviors; wire translation
  details are documented in `causa-protocol`.
- `causa-kernel` no longer depends directly on `tokio` or `futures-util`; its
  direct normal dependencies are `async-trait`, `futures-core`, `serde`,
  `serde_json`, `thiserror`, `tokio-util` and `uuid`. `tokio` remains in the
  transitive graph through `tokio-util`'s cancellation token.
- `causa-protocol` no longer declares the unused `async-trait` and `tracing`
  dependencies.
- `causa-runtime` has one stop-wait helper shared by the driver and the
  executor; `causa-provider` keeps its own because it cannot depend on the runtime.

## [0.0.2] - 2026-10-02

An experimental release separating editable context material from execution
across all six crates. Update dependencies explicitly from `0.0.1` to `0.0.2`;
this release breaks the Rust API and stored material format. Applications own
sessions, history, work management and subagent orchestration.

### Breaking

- Replace `TurnContext` with `Context`: it owns only ordered blocks, stays
  editable after a run and serializes as `{ "blocks": [...] }`. Remove turn
  lifecycle, sealing, snapshots, `merged_frame` and automatic input/output
  append methods. Construct blocks explicitly, use `apply` / `edit`, and supply
  the `TurnId` separately to `TurnRunner::run` or `run_streaming`.
- `ContextFrame` contains only an owned block vector. `ModelRequest` owns the
  invocation identity, explicit model, generation, cache and bound tool surface.
  `ModelGateway` receives `CallControl`; gateway-owned retries reuse the logical
  request. `AttemptControl`, attempt identities and implicit round traces are removed.
- `TurnRunOptions::new(model)` replaces default model selection. Optional
  `ContextPreparer` edits retained context and selects the next request frame;
  already committed preparation edits survive failure or cancellation.
- Replace executor constructors with `ToolExecutor::new(tools, options)`.
  `bind(invocation_id, control)` captures both definitions and actual targets;
  `BoundTools::process` consumes that binding and accepts a fresh `ToolBatch`.
  Duplicate static or dynamic names and failed catalog listings are errors.
  Cache ownership is per registration, including re-registration of the same Arc.
- Before/after processors live in `ToolExecutorOptions`. Session ownership,
  conversation storage, implicit traces, budget/interaction policies and built-in
  deduplication/output processors are removed. Implement these policies in the
  host; the examples demonstrate approval, deduplication and persistence.
- `TurnOutcome` returns context and a typed `TurnResult`, plus the complete
  current batch if results have not committed. Conversion and commit failures
  retain their submitted inputs. Six borrowed `RunEvent` variants replace stored
  event history. Observers copy only what they need to retain.
- Unknown tool outcomes are committed and stop the run. A module failure never
  cancels the caller's parent token. Controlled tool interruption collects started
  calls before returning; dropping the whole future bypasses material return.
- Protocol adapters reject incomplete or invalid tool exchanges locally before
  HTTP. OpenAI Chat and Anthropic additionally enforce their message ordering.
  Generic context import/edit still permits partial material. No synthetic
  provider ID is invented for orphan results.
- Block identity/content/metadata remain private and UUID IDs are caller supplied.
  Atomic edits preserve reused identity and return all submitted material on
  failure. The prior context/session formats have no compatibility adapters.

### Verification

- Independent facade consumers exercise kernel-only and runtime selections,
  custom preparation and budgeting, logical retry ownership, standalone tool
  binding, borrowed observations and material reuse across runs.
- Offline protocol/provider tests cover invalid histories and zero HTTP requests
  for rejected frames, including streaming entry points.

## [0.0.1] - 2026-09-16

The first experimental release of all six crates, with `causa` as the default
entry. This is a working development snapshot with unstable APIs and serialized
formats. **0.1.0** retains the functional completeness gate (multimodal I/O and
subagents).

### Added

- **Session resume receipt lookup**: `SessionHandle::resume_receipt` reads
  an accepted resume receipt, reports conflicting arguments, or returns
  `None` without starting work. Hosts can resolve retries before applying
  capacity limits while Session remains the sole owner of the replay table.
- **`causa-kernel`** — the facts layer: ContextBlock conversation fact
  machine, turn state machine with deterministic projections, and the
  self-contained behavior ports (`ModelGateway`, `Tool`,
  `DynamicToolSource`, `CallControl`, `ArtifactStore`). Budget and interaction
  components live in `causa-runtime`. Zero I/O;
  `#![deny(missing_docs)]`.
- **Context / component separation**: the session aggregate moved from the
  kernel to `causa-runtime` —
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
  execution. Legacy wire payloads (snapshots with `turn_sequence`,
  pause outcomes with `snapshot` + `reason`) are migrated by explicit
  extraction or rejected, never silently converted.
- **Multimodal vocabulary**: the block content vocabulary is
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
- **`causa-runtime`** — the execution stack: bounded model retry with
  exponential backoff, tool batch dispatch (dedup / allow / deny filter
  chain), streaming, approval pause with recoverable interruption, and
  `ContextEvent` projections for UI / observability consumers.
- **`causa-runtime` session coordination**: the new `session` module adds
  one `Session` owner per conversation plus cloneable `SessionHandle`s
  (`id` / `submit` / `observe` / `wait`) over an assembled `TurnRunner` +
  `TurnRunOptions`. `Session::new` accepts only an idle `ConversationState`
  and never calls the model; a rejected construction returns every by-value
  input (state, runner, options, config) in `SessionBuildRejection`. One work
  is active at a time (no queue, no steering, no implicit approval): `submit`
  on a busy session is `SessionError::Busy` naming the active `WorkRef`. Each
  work gets a fresh `TurnId` at acceptance that is never reused, plus a local
  `request_key` dedup table (same key + parts → the original `WorkReceipt`;
  same key, different parts → `Conflict`; rejected submits do not consume the
  key), and an explicit `SessionConfig::retained_work_capacity` (default 256)
  bounding the registry; an optional `SessionConfig::work_deadline` is measured
  from acceptance and wired into the work's `RunControl`. Completed turns
  commit into history; interrupted ones retain their real aborted facts (a
  `TurnSnapshot`) and cause outside the completed-only history; paused ones
  retain the complete `ConversationOutcome` for a later explicit resume. A
  worker panic or runner rejection is published as `Faulted` and the session
  refuses new work, while dropping the owner closes acceptance and fires the
  active work's stop token. Execution is one `tokio::spawn` worker per
  accepted work holding an `Arc` of a private session core — no `JoinHandle`,
  no reference cycle, and exactly one writable `ConversationState`.
- **`causa-runtime` session resume / cancel / shutdown**: `SessionHandle::resume`
  continues a paused work after validating its current paused revision and the
  request against the continuation — a rejected request executes nothing and
  leaves the paused material untouched, a terminal work is `NotPaused`, a
  wrong revision is `StaleRevision`, a rejected request is `InvalidResume`, and
  a same-key repeat returns the original `WorkReceipt` without re-executing,
  while the same key with a different decision or injection is `Conflict`.
  `SessionHandle::cancel` fires the target work's own control token (each work
  has an independent token, so a completion that races the cancel keeps its
  result) or terminates a paused work in place, retaining its
  committed facts and continuation (`FinishedKind::Interrupted.continuation`);
  a terminal work reports `AlreadyTerminal` and its result is not rewritten,
  and a same-key retry is stable. `Session::shutdown` stops acceptance, winds
  the running work down, and keeps retrievable results and paused material
  readable through surviving handles. The optional
  `SessionConfig::work_deadline` is stored per work and preserved across a
  pause, so a resume continues under the same absolute bound rather than a
  fresh one. A cancel that fires while the host's approval gate is deciding
  wins over the pause — the runner hands back a paused outcome, but the work's
  own token has fired, so the session ends it interrupted with the
  continuation retained instead of publishing `Paused` — and an accepted
  `request_key` still replays its original receipt after shutdown.
- **Fact-payload equality**: `causa-kernel`'s `ToolCallPayload`,
  `ToolOutputMeta`, `ToolOutput`, `ToolResultPayload`, `ArtifactRef`, and
  `ArtifactKind`, plus the runtime's `HookOutcome` and `ResumeRequest`,
  derive `PartialEq` (`ArtifactRef` / `ArtifactKind` also `Eq`) so a host can
  compare recorded calls, results, and resume requests without re-serializing
  them.
- **Session checkpoint envelope (slice 8 Phase C)**: the versioned
  `SessionCheckpoint` save envelope on `causa-runtime::session` — a
  `CheckpointPhase` carrying the idle state or the complete paused outcome
  (the active state is saved once, inside the outcome), per-work `SavedWork`
  views with the deadline as an absolute UTC expiry, the three request-key
  replay tables (`SavedSubmitKey` / `SavedResumeKey` / `SavedCancelKey`), the
  TurnId allocation progress, and the comparable `SessionConfigDescription`
  (`SessionConfigDescription::of`). Persistence stays host-side — the runtime
  ships no store. Supporting serialization is additive only:
  `ResumeRequest`, `HookOutcome`, `UnknownDecision`, the session value
  vocabulary (`WorkRef`, `WorkState`, `CancelOutcome`, `FinishedKind`,
  `WorkReceipt`, `CancelReceipt`, `SubmitRequest`, `SessionConfig`) and the
  config axis types (`RetryPolicy`, `TurnLimits`, `UnknownOutcomeConfig`,
  `ToolOutputLimits`, `WindowBudget`) derive serde; `ConversationState`,
  `ConversationOutcome`, and the kernel's `TurnContext` derive `Clone`;
  `ToolSurface`, `ToolDefinition`, and `GenerationOptions` derive
  `PartialEq`, and the runtime's `RetryPolicy`, `TurnLimits`,
  `UnknownOutcomeConfig`, and `WindowBudget` derive `PartialEq` (`Eq` for
  `WindowBudget`) so the configuration description can be compared by
  value. Existing wire shapes are unchanged.
- **Session checkpoint / restore (slice 8 Phase C)**: `Session::checkpoint`
  exports the versioned `SessionCheckpoint` — only from an idle or paused
  session: an accepted / running work (a cancel in progress included)
  reports `Busy`, a faulted session refuses (no complete outcome exists),
  and a closed session still exports with `closed` recorded. Record order
  is deterministic; two exports differ only in each work's re-anchored
  absolute deadline, so byte equality holds only for deadline-free states.
  `SessionCheckpoint::restore` validates the envelope version, the assembled
  configuration description (a swapped model, surface, policy, or limits is
  refused, never silently adopted), and the material's internal consistency
  before registering — a completed work must carry its committed history
  entry (an interrupted one must not), work identities must follow the
  conversation's allocation sequence with the saved progress past them, the
  paused outcome must carry the `Paused` stamp, and submit receipts their
  zero acceptance revision; a rejection hands back the original checkpoint
  and the by-value inputs, leaving nothing behind. Restore calls no model
  and no
  tool: it rebuilds the request-key replay tables (same key replays its
  original receipt, different arguments still conflict), continues the
  TurnId allocation from the saved progress, keeps every retained result
  observable, and re-derives each work's remaining deadline from the saved
  absolute UTC expiry — never a fresh full duration. A restored pause still
  needs an explicit `resume` and continues its original continuation.
  Restoring a checkpoint stays distinct from opening a conversation from
  history, which rebuilds no dedup table. A checkpoint supports resuming a
  deliberately paused, quiescent session — it promises no exactly-once
  across arbitrary process crashes: unsaved acceptance records, unknown
  external results, and a re-loaded old checkpoint gain nothing
  automatically, and a host needing crash recovery wires that itself.
  Persistence (whether and when to save) stays host-side; the runtime ships
  no store.
- **`causa-provider`** — reqwest `ModelGateway` adapters for the three
  protocols, with transport timeouts and classified error mapping.
- **`causa-extension`** — `DynamicToolSource` adapters; the default `mcp`
  feature provides the MCP client (`McpToolSource`) over stdio, Streamable
  HTTP, and in-process I/O, namespacing tools `mcp_{server}_{tool}` into
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
- Seven examples across runtime, provider and extension doubling as runnable
  documentation; this CHANGELOG.

### Changed

- **Session pause observations**: `WorkObservation` gains a required
  `paused: Option<PausePoint>` field in the unreleased Rust API (update
  struct literals). `observe` and `wait` return the hook-prepared approval
  calls and pause description with the same revision, so hosts can build
  resume requests without retaining a separate callback-side copy. Mutating
  the observation does not edit the session's retained pause. Existing wire
  shapes are unchanged.
- **`causa-runtime` session handle API**: the session work filed under Added
  above changed the shapes Phase A introduced, still inside the unreleased
  first release — `SessionHandle::submit` and `observe` are synchronous now (drop the
  `.await`), `FinishedKind::Interrupted` gained the required `continuation`
  field, and `SessionError` gained the `StaleRevision` / `NotPaused` /
  `InvalidResume` variants. The `resume` / `cancel` operations,
  `CancelOutcome` / `CancelReceipt`, and `Session::shutdown` are new.
- **`causa-runtime` session checkpoint surface**: `SessionError` gained the
  `ConfigMismatch` and `InvalidCheckpoint` variants (breaking for exhaustive
  matches) — restore refuses a checkpoint whose configuration description
  does not match the freshly assembled options, or whose material is
  inconsistent (unsupported version, an unregistered paused work, a receipt
  referencing an unknown work, an impossible state). The rejection carries
  the original checkpoint and the by-value inputs back.
- Brand: **Causa** (formerly Archy) — every crate renamed to `causa-*`;
  project pages live under the Project inceptae domain.
- **Reference budget & interaction ownership**: `FramePolicy`,
  `WindowBudget`, `FrameError`, `Compaction`, `CompactionInput`,
  `CompactionOutput`, `CompactionError`, `TokenCounter`,
  `TurnInteraction`, and `BatchDecision` are runtime-component opinions,
  not cross-host kernel contracts — they moved from
  `causa-kernel` to `causa-runtime` (new `budget` / `interaction`
  modules with explicit root re-exports; no kernel re-export or reverse
  alias remains). `FramePolicy::materialize` now composes the public
  lossless `TurnContext::frame` and replaces only the projected block
  list, so the deterministic frame identity is unchanged. The runtime's
  crate docs state the four optional usage paths (reference execution,
  standalone tool execution via `ToolExecutor::execute_with_limits`,
  reference session, observation).
- **Tool results split from continuation actions and host limits**: a
  shared `Tool::execute` / `execute_with_store` now returns the
  recorded `ToolResultPayload` only, and `DynamicToolSource::invoke` /
  `invoke_with_store` return `Result<ToolResultPayload,
  ToolExecutionError>` — what an `UnknownOutcome` result does next and
  how much output to retain are runtime-component configuration, not
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

- `IsolationLevel` and `Tool::isolation_level`: no executor
  ever read the declaration, and panic capture plus a call deadline are
  not process isolation — the "declaration the driver obeys" claim was
  unfounded. No replacement enum or subprocess framework is provided.
- `causa_runtime::defaults::{NoopTokenCounter, NoopCompaction}`: no
  consumers, and a zero counter is behaviorally distinct from no
  counter — the executor's chars/4 fallback estimates real sizes and can
  trigger truncation, while a zero counter never does. Hosts that need a
  zero estimate keep their own `TokenCounter` implementation. The
  `defaults` module is private now; the chars/4 fallback keeps its
  algorithm unchanged as a crate-internal function, and the unused
  `placeholder_token_estimate` blocks wrapper is gone.
- Tool-side action and retention declarations: the kernel's
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

- Public documentation now distinguishes consumer setup from repository examples,
  reflects current API ownership and dependency directions, and includes a complete
  facade quickstart. All six libraries enforce public API documentation; CI checks
  library module layout and the runtime-to-protocol dependency boundary.
- Experimental release packaging includes both license texts in every crate.
  Publishing uses a workspace preflight and explicit version/tag checks.
- The two resume entry points document their intentional by-value error
  payloads and locally allow `clippy::result_large_err`, preserving the
  complete-material return contract on newer toolchains.
- **Session checkpoint consistency (slice 8 Phase C)**: restore rejects
  duplicate keys within each request table, cancel receipts whose work
  differs from their target, paused continuations inconsistent with the
  active facts, and interrupted snapshots with mismatched turn identities
  or invalid blocks. Rejections return the original material before any
  registration; continuation/fact checks are shared with resume.
- **Coordinator capacity and replay (slice 8 Phase D harness)**: resuming
  a later work updates the reference used for capacity accounting without
  changing the original creation relation. Accepted resume receipts replay
  before the capacity gate, and replaying an older work does not replace
  the currently counted work.
- Session workers now publish `Faulted` if the hosting runtime drops their
  task, including an accepted submission or resume not yet polled. Retained
  handles no longer report a permanently running work, and `shutdown` can
  complete after the original runtime has exited.
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
- Retry backoff now respects the turn deadline: the inter-attempt wait
  is capped at the remaining deadline, the stop guard is re-checked
  before each further attempt, and a retry loop that gives up after the
  deadline has passed ends the turn as `TurnDeadlineExceeded` (matching
  the round-boundary check) instead of the last attempt's error kind.
- `ConversationState` reload validates the whole payload, not only the
  active slot: committed history entries go through the same closed set
  as `from_history` (strictly increasing turn sequences, distinct turn
  ids, kernel block validation), and the active turn's id must not
  duplicate a committed one. The wire shape is unchanged.
