# Experimental releases

Publish all six crates at the same `0.0.x` version. Most users depend only
on `causa`; Cargo resolves its layer dependencies. Session management and
subagent orchestration belong to applications; they are not Causa release gates.
The Publish workflow currently accepts only experimental `0.0.x` versions with a nonzero patch number.

## Prepare a version

1. Record breaking API/wire changes and migration instructions under
   `## [Unreleased]` in CHANGELOG. In `0.0.x`,
   patch releases may be incompatible; Cargo's `"0.0.3"` constraint does not
   automatically select `0.0.4`. There is no separate checkpoint schema version
   or automatic reader for the removed context/session formats.
2. Run `python3 .github/scripts/release.py prepare --version 0.0.4`. It dates
   the Unreleased notes as the release entry and opens an empty Unreleased
   section; sets `workspace.package.version` and every family dependency
   (including dev-dependencies) to the new version; refreshes the workspace
   and consumer lockfiles; and moves the version numbers in the README, the
   facade crate docs and the website installation pages. It commits nothing.
   Rewrite `website/src/content/docs/status.mdx` for the release by hand.
3. Keep each crate's two license symlinks pointing to the root license texts.
   Cargo flattens these into ordinary files in the published archives.
4. Run the checks below and review the complete diff. Preserve any unrelated
   working-tree edits when preparing the release commit.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo +1.96.0 check --workspace --all-targets --locked
bash .github/scripts/check-dependency-directions.sh
python3 .github/scripts/check-module-layout.py
python3 -m unittest discover -s .github/scripts -p 'test_guards.py'
python3 -m unittest discover -s .github/scripts -p 'test_release.py'
python3 .github/scripts/release.py check --version 0.0.4
cargo check -p causa --locked
cargo check -p causa --no-default-features --locked
cargo check -p causa --features full --locked
cargo check -p causa-extension --no-default-features --locked
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps --locked
cargo publish --workspace --dry-run --locked --registry crates-io
python3 .github/scripts/release.py packages --version 0.0.4
```

Use the actual version in these commands. Before committing, the local
dry-run can use `--allow-dirty`; the publishing workflow requires a clean
checkout. A workspace dry-run builds all six packaged crates against a
temporary registry, even before their dependencies exist on crates.io.
It does not upload anything. Do not replace it with separate per-crate
dry-runs for the first release.

## Publishing credentials

- Log in to crates.io and verify the account email address.
- For releases of the six existing crates, create a short-lived API token
  with only `publish-update`, scoped to `causa`, `causa-kernel`, `causa-runtime`,
  `causa-protocol`, `causa-provider` and `causa-extension`. Creating a new crate
  additionally needs `publish-new`. Add it to the repository secret
  `CARGO_REGISTRY_TOKEN`; the workflow injects it only into the upload step.
- The current crates.io Trusted Publishing setup requires an existing crate.
  After the first release, each crate can be configured for the same
  repository and publishing workflow, then the bootstrap token can be retired.

## Manual release

1. Finalize `## [0.0.4] - YYYY-MM-DD` in CHANGELOG and change the README/site
   status from “preparing” to “experimental release”. Commit the reviewed
   release changes and push them. CI must pass on that commit.
2. In GitHub Actions, run **Publish** on that commit's branch with its version
   and `publish` unchecked. This repeats CI and the complete package preflight.
3. Tag the same release commit `v0.0.4` and push the tag after approving the
   artifacts. Do not move a tag used for an upload.
4. Run **Publish** again, select the matching tag, enter `0.0.4`, and check
   `publish`. The workflow reruns CI for the selected commit before upload.
   After the upload it creates the GitHub release `v0.0.4 — Experimental`
   as a pre-release, with the dated CHANGELOG entry as its notes.
5. Check all six crates.io pages and docs.rs builds, then compile an external
   consumer using `causa = "0.0.4"` in default, kernel-only and full modes.

Actual uploads run only in this manually dispatched workflow. The default
is preflight-only. The upload helper refuses missing tokens, wrong tags,
dirty checkouts and unfinished release notes.

## Partial release recovery

Rerun the same workflow from the same tag. Before uploading anything, the
helper checks all six exact versions. It skips an existing version only if
its published VCS metadata identifies this clean commit; yanked versions,
different commits, malformed archives and registry errors stop the release.
Cargo publishes the remaining packages in dependency order and waits for
index visibility. If all six already exist from the tag, the upload is a no-op.
The GitHub release is created only when the tag has none; an existing release
is left as it is, so edit its notes on GitHub or delete it before a rerun.

This check uses the package's recorded VCS metadata; it is a retry guard,
not a cryptographic source attestation. A failed upload may already have
reached crates.io, so inspect the registry and rerun instead of moving the
tag or trying to overwrite a version.

References: [Cargo publishing](https://doc.rust-lang.org/cargo/reference/publishing.html),
[version constraints](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html),
[Trusted Publishing](https://crates.io/docs/trusted-publishing).
