# Public provider streaming consumer

This isolated Cargo workspace depends only on the public `causa` facade
and general consumer dependencies. Its real local HTTP server withholds the
terminal SSE event until the consumer observes text from each built-in gateway.
It checks `ModelGateway::stream` independently of runtime observation and uses
public provider `StreamLimits` configuration. A facade runtime test proves
that temporary text is observed before the final snapshot is committed.
Public decoder regressions cover byte boundaries, tool ordinal mapping,
identity conflicts, invalid completion sequences, cumulative usage and limits.

Run offline after dependencies have been cached:

```sh
cargo test --manifest-path tests/consumers/provider-streaming/Cargo.toml --locked
```

The path dependency validates the current checkout's public API. A separately
published registry release is outside this consumer's scope.
