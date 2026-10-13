# AGENTS.md — Causa

> The agent entry point for the Causa monorepo. `CLAUDE.md` is a symlink to
> this file — edit here only.

**Causa** is a minimal context kernel for Rust agents: it holds the facts,
your code holds the behavior. One Cargo workspace, six crates published at one
shared `0.0.x` version, with `causa` as the default entry. Dual-licensed
`MIT OR Apache-2.0`.

## Layout

```text
crates/causa               # facade: `kernel` always on; default = runtime + providers
crates/causa-kernel        # facts + contracts only (no I/O, no transport, no policy)
crates/causa-runtime       # optional turn runner, bound tool execution, processing chains
crates/causa-protocol      # wire-protocol translation, transport-free
crates/causa-provider      # reqwest ModelGateway adapters (implies protocol)
crates/causa-extension     # DynamicToolSource adapters; `mcp` feature (rmcp, on by default)
tests/consumers/           # separate workspaces that use the public facade
.github/workflows/         # ci.yml, website.yml, publish.yml (manual dispatch only)
.github/scripts/           # layering guard, layout guard, release.py and their tests
.github/RELEASING.md       # release checklist — never `cargo publish` by hand
website/                   # Astro Starlight docs site (its own pnpm project)
scripts/                   # proposal verifier and repo helpers (not published)
```

History, input admission, budgets, persistence and sessions are host-owned;
none of them belongs in the publish set.

## Fact sources

Read a fact here before asserting it — never from memory:

| fact | authoritative source |
|---|---|
| crate set, version, edition, MSRV | `Cargo.toml` `[workspace.package]` |
| facade feature selection | `crates/causa/Cargo.toml` `[features]` |
| driver policy surface and defaults | policy overview in `crates/causa-runtime/src/lib.rs` |
| example names and offline / key / server needs | the doc comment atop `crates/*/examples/*.rs` |
| release state and wire-shape breaks | `CHANGELOG.md` |
| CI jobs and gates | `.github/workflows/ci.yml`, `.github/scripts/` |

When a crate, feature or example changes, update the copies together:
`README.md` (crates, features and examples tables) and
`website/src/content/docs/crates.mdx`. When a CI check changes, update the
command list below.

## Commands

This list mirrors CI; a change is ready when all of it passes.

```bash
cargo test --workspace                    # full suite (also builds examples)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check                   # run `cargo fmt --all` first
cargo +1.96.0 check --workspace --all-targets          # MSRV
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps --locked

# The consumers are separate workspaces: the commands above do not reach
# them, and CI checks each one on its own.
cargo test --manifest-path tests/consumers/material-flow/Cargo.toml --locked
cargo test --manifest-path tests/consumers/material-flow/Cargo.toml --features runtime --locked
cargo test --manifest-path tests/consumers/provider-streaming/Cargo.toml --locked
cargo fmt --manifest-path tests/consumers/material-flow/Cargo.toml --all --check
cargo fmt --manifest-path tests/consumers/provider-streaming/Cargo.toml --all --check
cargo clippy --manifest-path tests/consumers/material-flow/Cargo.toml --features runtime --all-targets --locked -- -D warnings
cargo clippy --manifest-path tests/consumers/provider-streaming/Cargo.toml --all-targets --locked -- -D warnings
cargo +1.96.0 check --manifest-path tests/consumers/material-flow/Cargo.toml --features runtime --all-targets --locked
cargo +1.96.0 check --manifest-path tests/consumers/provider-streaming/Cargo.toml --all-targets --locked

bash .github/scripts/check-dependency-directions.sh   # layering guard
python3 .github/scripts/check-module-layout.py        # library source layout
python3 -m unittest discover -s .github/scripts -p 'test_guards.py'
python3 -m unittest discover -s .github/scripts -p 'test_release.py'
# CI also runs cargo-machete for unused dependencies; install it to run locally.

# Feature matrix:
cargo check -p causa
cargo check -p causa --no-default-features
cargo check -p causa --features full
cargo check -p causa-extension --no-default-features
```

Stable toolchain for development, MSRV **1.96**, edition 2024. Do not raise
`rust-version` or add a dependency without a reason stated in the commit.

Tests run offline: provider tests use `wiremock`, MCP tests use in-process
fixtures. The examples that need a live key or server (`quickstart`,
`mcp_tools`) are not meant to run in CI.

## Layering

```text
kernel <- protocol <- provider
kernel <- runtime
kernel <- extension
(*, extension) <- causa (facade; depended on by none)
```

The guard checks normal dependencies with all features enabled;
dev-dependencies are exempt, so tests may wire layers together. It cannot
check API ownership or where a policy lives — those need review.

1. `causa-kernel` is facts (`context`) and contracts (`ports`) only. It
   depends on no sibling and no transport or policy crate. Its direct normal
   dependencies are allowlisted: `async-trait`, `futures-core`, `serde`,
   `serde_json`, `thiserror`, `tokio-util`, `uuid`. A type belongs in `ports`
   iff third parties implement or call against it.
2. `causa-protocol` is pure translation: no transport, no driver.
3. `causa-runtime` drives the kernel and nothing else. A policy with one
   implementation is a config object or a documented opinion in the runtime
   `lib.rs` policy table, not a trait.
4. `provider` owns reqwest plumbing and `extension` owns `DynamicToolSource`
   adapters. Neither depends on the other or on the runtime.
5. Heavy dependencies go behind `dep:`-gated features (`rmcp` is the
   template). A new adapter is a new module and feature in `causa-extension`,
   not a new crate; examples and tests that use it need `required-features`.
6. Reference implementations of ports (stores, token counters, exporters)
   live in examples or hosts, never in the publish set.

In shell guards, match the facade as `"causa v"`; bare `causa` also matches
`causa-*`.

## Conventions

- `#![deny(unsafe_code)]` everywhere and `#![deny(missing_docs)]` on library
  targets. Write the docs with the API; the publish dry-run fails without
  them.
- Shipped comments and docstrings are self-contained. They state the invariant
  or rationale directly and never cite local-only documents, tickets or
  shorthands (for example anything under `.agents/`).
- Crate docs (`src/lib.rs`) state the layer contract and the policy surface.
- Naming: `causa-*` packages, `causa_*` imports, `McpToolSource`-style adapter
  names. `version`, `edition`, `authors`, `license` and `repository` are
  inherited from the workspace. `Cargo.lock` is committed. Each crate ships
  both license files as relative symlinks to the root copies.
- Before `0.1`, keep no backward compatibility: remove an obsolete
  implementation or format instead of adapting to it, and record the break
  and its migration at the top of `CHANGELOG.md`. From `0.1.0`, a wire-serde
  break needs a minor bump.
- Public type evolution is decided when the type is added, by who builds the
  value and who matches on it.
  - A struct with public fields that could gain one is `#[non_exhaustive]`
    and ships a `Default` or a `new` taking its required fields. Fields stay
    public, so a caller constructs, then assigns. Chainable `with_*` setters
    are fine where a value is usually built in one expression; each sets one
    public field and nothing else. A separate builder type is only for fields
    that must be checked together, and says so in its docs. Other crates in
    this workspace build values the same way outside callers do.
  - An enum callers only read (errors, events, interruption causes, error
    kinds) is `#[non_exhaustive]`. When only its own crate builds it, each
    struct-like variant is `#[non_exhaustive]` too, so a caller's pattern
    ends in `..` and a new field does not break it. An enum another crate
    builds (`StreamDelta`) keeps the attribute on the enum alone.
  - Fact vocabulary that a renderer or the runtime must be total over stays
    exhaustive, with a doc line saying so: `BlockContent`, `ContentPart`,
    `ModelItemContent`, `ToolResultStatus`, `ModelStopReason`,
    `ToolExecutionError`, `ControlError` and the option enums a renderer
    switches on. A new variant there is a breaking change.
  - A struct whose fields are its definition stays a plain literal:
    `InvocationId`, `ModelResponse`, `ContextFrame`, `Replacement`,
    `ToolCallId`.
  - Errors own the inputs of the failed operation and are returned unboxed;
    the workspace allows `clippy::result_large_err` for that reason.
- Facade and kernel `lib.rs` use explicit, namespaced re-exports, never
  globs, so a later addition cannot collide.
- Module layout: never `mod.rs` under `crates/*/src` (the layout guard
  rejects it; test fixtures may use `tests/common/mod.rs`). A module with one
  file is just `foo.rs`. Once it needs a second file it becomes `foo.rs` +
  `foo/`, where `foo.rs` is a thin index — docs, `mod` declarations,
  re-exports — and the code lives in `foo/*.rs`. A directory holding a single
  file is not created. Private submodules re-exported at `foo.rs` keep paths
  flat (`crate::driver::TurnOutcome`); `pub mod` is only for nested surfaces
  (`ports`, `context`).
- Commits follow Conventional Commits (`feat:`, `fix:`, `refactor!:`,
  `chore:`, `docs:`), one logical change each.
