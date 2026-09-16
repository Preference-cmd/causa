# AGENTS.md — Causa

> The agent entry point for the Causa monorepo. `CLAUDE.md` is a symlink to
> this file — edit here only.

**Causa** is a minimal context kernel for Rust agents: it holds the facts,
your code holds the behavior. One Cargo workspace, six crates, lockstep
versioned at `0.x`; `0.0.1` is the first experimental release, with `causa`
as the default entry. Dual-licensed `MIT OR Apache-2.0`.

## Layout

```text
crates/causa               # facade: `kernel` always on; default = runtime + providers
crates/causa-kernel        # facts + contracts only (no I/O, no transport, no policy)
crates/causa-runtime       # optional, reference components: execution stack (turn loop,
                           # dispatch, streaming, pause/resume) + session aggregate
                           # (ConversationState/Store, Session/SessionHandle/checkpoint)
                           # + budget/interaction seams and the unknown-outcome /
                           # output-retention configuration (Slice 13)
crates/causa-protocol      # wire-protocol translation, transport-free
crates/causa-provider      # reqwest ModelGateway adapters (implies protocol)
crates/causa-extension     # DynamicToolSource adapters; `mcp` feature (rmcp, on by default)
.github/workflows/ci.yml   # fmt / clippy / test / MSRV / features / docs / guards
.github/workflows/publish.yml  # manual dispatch ONLY — never `cargo publish` by hand
.github/scripts/check-dependency-directions.sh  # layering guard, runs in CI
.github/scripts/check-module-layout.py # rejects mod.rs in library sources
.github/scripts/release.py # version / archive checks and tagged release retries
CHANGELOG.md               # 0.0.x experimental; from 0.1 wire breaks bump minor
.github/RELEASING.md        # experimental release checklist and manual workflow
website/                   # Astro Starlight docs site (its own pnpm project + workflow)
scripts/                   # proposal verifier and repo helpers (not published)
```

User-facing feature map on the facade:

| selection | gets |
|---|---|
| `causa = "0.0.1"` (default) | kernel + runtime + providers |
| `features = ["full"]` | + extensions (MCP) |
| `default-features = false` | kernel only (offline audit / minimal embed) |
| `default-features = false, features = ["runtime"]` | kernel + offline driver |

## Fact sources

Before asserting a fact about the repo, read it here — never from memory:

| fact | authoritative source |
|---|---|
| crate set, version, edition, MSRV | `Cargo.toml` `[workspace.package]` |
| facade feature selection | `crates/causa/Cargo.toml` `[features]` |
| driver policy surface and defaults | policy overview in `crates/causa-runtime/src/lib.rs` and linked component/config docs |
| example names and offline / key / server needs | the doc comment atop `crates/*/examples/*.rs` |
| release state and wire-shape breaks | `CHANGELOG.md` |
| CI jobs and gates | `.github/workflows/ci.yml`, `.github/scripts/` |

When a crate, feature, or test command changes, update every copy:
`README.md` (crates + feature tables), `website/src/content/docs/crates.mdx`,
and this file together.

## Commands

```bash
cargo test --workspace                    # full suite (also builds examples)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check                   # CI fails otherwise; run `cargo fmt --all` first
bash .github/scripts/check-dependency-directions.sh   # layering guard, same as CI
python3 .github/scripts/check-module-layout.py        # library source layout
python3 -m unittest discover -s .github/scripts -p 'test_guards.py'
python3 -m unittest discover -s .github/scripts -p 'test_release.py'

# Feature-matrix checks (required when touching Cargo.toml / lib.rs / features):
cargo check -p causa
cargo check -p causa --no-default-features
cargo check -p causa --features full
cargo check -p causa-extension --no-default-features
```

Toolchain: stable for dev, MSRV **1.96** pinned by CI (`cargo check
--workspace --all-targets` on 1.96.0). Edition 2024. Do not raise
`rust-version` or add a dependency without a reason stated in the commit.

Tests run offline: provider tests use `wiremock`, MCP tests use
in-process fixtures. Examples needing live keys or servers (`quickstart`,
`mcp_tools`) are listed in `README.md` — do not "fix" them to run in CI.

## Layering

```text
kernel <- protocol <- provider
kernel <- runtime
kernel <- extension
(*, extension) <- causa (facade; depended on by none)
```

The guard checks normal family dependencies and named forbidden packages with
all features enabled. API ownership, policy placement and feature design also
require review; a dependency check cannot enforce their semantics.

1. `causa-kernel` never depends on transport, policy, or any sibling
   (`reqwest`, `rmcp`, `axum`, `tracing-subscriber` banned, plus all
   family crates). It is facts (`context`) + contracts (`ports`) only.
   A type belongs in `ports` iff third parties implement or call
   against it.
2. `causa-protocol` stays pure translation: no transport, no driver.
3. `causa-runtime` drives the kernel and nothing else. No trait seams
   for single-implementation policies — policy is a config object or a
   documented opinion in the runtime `lib.rs` policy table.
4. Edge crates are interop adapters (capabilities, not opinions):
   `provider` owns reqwest plumbing, `extension` owns
   `DynamicToolSource` adapters. They never depend on each other or on
   the runtime in the *normal* graph.
5. Heavy deps go behind `dep:`-gated features (`rmcp` in extension is
   the template). New adapters = new module + new feature in
   `causa-extension`, not new crates. Examples/tests touching an
   optional adapter need `required-features`.
6. Reference implementations of ports (stores, token counters,
   exporters) live in examples or hosts — never in the publish set.
7. Dev-dependencies are exempt from the guard by design (tests may wire
   layers together; the published graph may not).

When matching the facade in shell guards, use `"causa v"` (cargo-tree
rendering) — bare `causa` false-positives on `causa-*` names.

## Conventions

- `#![deny(unsafe_code)]` everywhere; `#![deny(missing_docs)]` on
  library targets. The `Publish` workflow's dry-run fails on public API
  without docs — write docs first, not after.
- Crate docs (`src/lib.rs`) state the layer contract and the policy
  surface; keep the README's crates table and facade feature table in sync
  when crates or features change (see Fact sources).
- Naming: `causa-*` packages, `causa_*` imports, `McpToolSource`-style
  adapter names. Inherited workspace metadata (`version`, `edition`,
  `authors`, `license`, `repository`) — never per-crate values.
- `Cargo.lock` is committed (workspace binary story + reproducible CI).
- All six crates ship `LICENSE-MIT` and `LICENSE-APACHE` via relative
  symlinks to the root copies; Cargo flattens them when packaging.
- During `0.0.x`, patch releases may break API / wire compatibility; record
  breaking changes and migrations at the top of CHANGELOG. Starting with
  `0.1.0`, wire-serde breaks require a minor bump. Checkpoint schema versions
  remain an independent validation boundary.
- Re-exports over globs: facade and kernel `lib.rs` use explicit,
  namespaced re-exports so future additions cannot collide.
- Library module layout: `foo.rs` + `foo/`, never `mod.rs` (the layout guard
  rejects it under `crates/*/src`; test fixtures may use `tests/common/mod.rs`).
  `foo.rs` is a thin index — docs, `mod` declarations,
  re-exports — and the code lives in `foo/*.rs`. Private submodules
  re-exported at `foo.rs` keep the path flat (`crate::session::WorkRef`);
  `pub mod` only for nested surfaces (`ports`, `context`).
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor!:`,
  `chore:`, `docs:`). One logical change per commit.
