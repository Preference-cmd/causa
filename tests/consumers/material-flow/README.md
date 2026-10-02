# Public material-flow consumers

This is an independent, unpublished workspace. Its only Causa-family dependency
is the public `causa` facade with default features disabled. The `runtime`
feature enables only `causa/runtime`; providers and extensions are unnecessary.

```sh
cargo test --manifest-path tests/consumers/material-flow/Cargo.toml --locked
cargo test --manifest-path tests/consumers/material-flow/Cargo.toml --features runtime --locked
cargo tree --manifest-path tests/consumers/material-flow/Cargo.toml -e normal
cargo clippy --manifest-path tests/consumers/material-flow/Cargo.toml --features runtime --all-targets -- -D warnings
```

- `kernel_contract.rs` (C1): direct material import/edit/serde, pure conversion,
  work derived only from new blocks, scope validation and atomic result commit;
  conversion/edit failures preserve caller inputs and the batch.
- `logical_retry.rs` (C6): a finite caller-owned gateway wrapper over real
  scripted failures. All request fields remain unchanged. Retrying is possible
  before yielding; text, reasoning, tool and usage visibility prevents further
  transparent attempts. Parent cancellation/deadlines bound backoff, stream
  establishment and waiting for the first delta. Local expiry leaves parent
  cancellation unchanged.
- `preparation.rs` (C3/E3 and C6/E4, runtime): a source explicitly records read,
  commit and confirmation, retains input on commit failure and returns committed
  edits on a later preparation failure. Observation refreshes once per logical
  invocation and is not repeated by gateway retries. A concrete budget consumer
  counts serialized blocks/tool definitions, a media surcharge and reserved
  output. Schema growth alone crosses its threshold with identical blocks;
  the admitted actual request is captured and the rejected request never reaches
  the gateway. These are estimated cost units, not exact token counts.

- `runtime_basic.rs` (C2/R2-R4): T1/T2 input admission and reuse, plus an
  actual tool round whose declarations/results are removed by preparation.
  Later requests contain the summary and current input, while saved frames,
  contexts and copied commit observations retain the original material.
- `tool_consumer.rs` (C4): an independent host binds tools, processes a fresh
  batch and explicitly decides when to validate and commit returned results.
- `observation.rs` (C5): a host copies selected borrowed observations; refusal
  output and usage remain available without observation and survive callback
  cancellation once the terminal cause is selected.
The tests are consumers and intentionally implement their own input, retry and
budget choices; none of these policies is shipped by the runtime.
