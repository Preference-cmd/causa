# Site guide code

Every Rust block on the documentation site (`website/`) is read from this
workspace at build time, so CI compiles what the site shows. It depends only
on the public `causa` facade.

- `src/bin/*.rs` are whole programs. Those that run offline are executed by
  `tests/offline_programs.rs`; those that need a key or a network are only
  compiled.
- `src/*.rs` hold functions a page shows on their own.
- `// ANCHOR: name` and `// ANCHOR_END: name` mark the part a page shows;
  the marker lines never appear on the site. A page that names no anchor
  shows the whole file without the markers.

```sh
cargo test --manifest-path tests/consumers/site-guides/Cargo.toml --locked
```

The path dependency checks the current checkout's public API.
