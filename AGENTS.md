# AGENTS.md — Causa

> `derive, don't drift.` This file is the agent entry point for the Causa
> monorepo. `CLAUDE.md` is a symlink to this file — edit here only.

**Causa** is a small, principled agent kernel for Rust: facts, ports,
driver. One Cargo workspace, one GitHub repo
(`git@github.com:Preference-cmd/causa.git`), six crates, lockstep
versioned at `0.x`. Dual-licensed `MIT OR Apache-2.0`.

## Layout

```text
crates/causa               # facade: `kernel` always on; default = runtime + providers
crates/causa-kernel        # facts + contracts only (no I/O, no transport, no policy)
crates/causa-runtime       # reference driver (turn loop, dispatch, streaming, pause/resume)
crates/causa-protocol      # wire-protocol translation, transport-free
crates/causa-provider      # reqwest ModelGateway adapters (implies protocol)
crates/causa-extension     # DynamicToolSource adapters; `mcp` feature (rmcp, on by default)
.github/workflows/ci.yml   # fmt / clippy / test / MSRV / dependency-direction guard
.github/workflows/publish.yml  # manual dispatch ONLY — never `cargo publish` by hand
.github/scripts/check-dependency-directions.sh  # layering guard (tracked; force-added, see .gitignore note)
CHANGELOG.md               # Keep a Changelog; wire-serde breaks bump minor + flag at top
```

User-facing feature map on the facade:

| selection | gets |
|---|---|
| `causa = "0.1"` (default) | kernel + runtime + providers |
| `features = ["full"]` | + extensions (MCP) |
| `default-features = false` | kernel only (offline audit / minimal embed) |
| `default-features = false, features = ["runtime"]` | kernel + offline driver |

## Commands

```bash
cargo test --workspace                    # full suite (also builds examples)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check                   # CI fails otherwise; run `cargo fmt --all` first
bash .github/scripts/check-dependency-directions.sh   # layering guard, same as CI

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
in-process fixtures. Examples needing live keys/servers (`quickstart`,
`mcp_tools`) are documented in `README.md` — do not "fix" them to run in
CI.

## Layering (machine-enforced, first-principles 7.1)

```text
kernel <- protocol <- provider
kernel <- runtime
kernel <- extension
(*, extension) <- causa (facade; depended on by none)
```

Rules, asserted by the guard script on every CI push — not by review:

1. `causa-kernel` never depends on transport, policy, or any sibling
   (`reqwest`, `rmcp`, `axum`, `tracing-subscriber` banned, plus all
   family crates). It is facts (`context`) + contracts (`ports`) only.
   A type belongs in `ports` iff third parties implement or call
   against it.
2. `causa-protocol` stays pure translation: no transport, no driver.
3. `causa-runtime` drives the kernel and nothing else. No trait seams
   for single-implementation policies — policy is a config object or a
   documented opinion in `lib.rs`'s policy table (§7.6).
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
  library targets. Public API without docs fails CI-adjacent checks
  (`publish --dry-run`) — write docs first, not after.
- Crate docs (`src/lib.rs`) state the layer contract and the policy
  surface; keep the README's publish-set table in sync when crates or
  features change.
- Naming: `causa-*` packages, `causa_*` imports, `McpToolSource`-style
  adapter names. Inherited workspace metadata (`version`, `edition`,
  `authors`, `license`, `repository`) — never per-crate values.
- `Cargo.lock` is committed (workspace binary story + reproducible CI).
- Re-exports over globs: facade and kernel `lib.rs` use explicit,
  namespaced re-exports so future additions cannot collide.
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor!:`,
  `chore:`, `docs:`) with slice references where applicable
  (`feat(slice N …): …`). One logical change per commit.
- `.gitignore` anchors machine-local ignores to root (`/scripts/`,
  `.agents/`): `.github/scripts/*` MUST stay tracked. Never broaden an
  ignore pattern without `git status --short` proving nothing tracked
  is swallowed.

## Workflow

- Small, single-branch work: commit directly to `main` and push.
- Non-trivial behavior / architecture / contract change: record it in
  the local proposal pipeline (`.agents/pipelines/`, machine-local and
  gitignored — proposals live there, not in this repo). `dispatch.mjs`
  is retired; `verify-proposals.mjs` validates lifecycle files.
- `scripts/change-scope.mjs` crate mapping is stale (Reimagine-era
  names); update it before trusting its output on this repo.
- Release gate: functional completeness (multimodal I/O + subagents
  slices) per the slice 11 proposal. Publishing is the manual
  `Publish` workflow in dependency order, facade last — crates.io
  state is a human decision, not an agent default. Until release,
  push to GitHub only.
