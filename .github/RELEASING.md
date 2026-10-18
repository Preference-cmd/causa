# Releases

Publish all six crates at one shared version. Most users depend only on
`causa`; Cargo resolves its layer dependencies. Session management and
subagent orchestration belong to applications; they are not Causa release gates.

## Version rules

The Publish workflow accepts `0.0.x` (x ≥ 1) and `0.y.z` (y ≥ 1); 1.0 is a
separate decision. A new version must follow the latest dated CHANGELOG entry.

- A break of the public API or of the stable serialized set (`Context` and
  every type it contains) needs a minor bump, `0.1.z` to `0.2.0`. A patch
  release only fixes or adds without breaking, so a `"0.1"` constraint
  picks it up.
- The `0.0.x` series that ended with 0.0.7 bumped the patch number for every
  release, breaking ones included; Cargo's `"0.0.3"` constraint does not
  select `0.0.4`.

On GitHub, only a minor release (`v0.1.0`, `v0.2.0`) is a regular release,
titled with its tag. Every `0.0.x` release and every patch between minor
releases (`v0.1.1`) is a pre-release titled `v0.1.1 — Experimental`, so the
latest regular release is always the latest minor version.

## Prepare a version

1. Record breaking API/wire changes and migration instructions under
   `## [Unreleased]` in CHANGELOG, and choose the version by the rules above.
   There is no separate checkpoint schema version or automatic reader for the
   removed context/session formats.
2. Run `python3 .github/scripts/release.py prepare --version 0.0.4`. It dates
   the Unreleased notes as the release entry and opens an empty Unreleased
   section; sets `workspace.package.version` and every family dependency
   (including dev-dependencies) to the new version; refreshes the workspace
   and consumer lockfiles; and moves the version numbers in the README, the
   facade crate docs and the website installation pages. It commits nothing.
   Rewrite `website/src/content/docs/reference/versioning.mdx` for the release by hand.
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

Use the actual version in these commands. The release workflow repeats them
on the pushed commit; running them first catches a failure before the push.
Before committing, the local dry-run can use `--allow-dirty`; the publishing
workflow requires a clean checkout. A workspace dry-run builds all six packaged crates against a
temporary registry, even before their dependencies exist on crates.io.
It does not upload anything. Do not replace it with separate per-crate
dry-runs for the first release.

## Publishing credentials

Uploads use crates.io Trusted Publishing; the repository stores no token.

- In Settings → Environments, create `crates-io` with a required reviewer
  before the first release through this workflow: a job that names a
  missing environment creates it without protection.
- On crates.io, in each of the six crates' Settings → Trusted Publishing,
  add a GitHub configuration: owner `Preference-cmd`, repository `causa`,
  workflow `publish.yml`, environment `crates-io`. Renaming the workflow file
  or the environment breaks the upload until all six are updated.
- The upload job exchanges its OIDC identity through
  `rust-lang/crates-io-auth-action` for a token that expires after 30 minutes
  and is revoked when the job ends. The approval comes first, so no token
  exists before the upload is approved.
- Trusted Publishing works only for a crate that already exists. A new crate
  is first published with an API token scoped to it with `publish-new`, then
  configured as above, and the token is revoked.

## Release

1. Push the reviewed release commit from “Prepare a version” to `main`.
2. The **Publish** workflow runs on every push to `main`. Its `plan` job reads
   the workspace version and releases it only when CHANGELOG has the dated
   entry and the `v0.0.4` tag is missing or already names this commit; any
   other push stops there. A release reruns CI and the complete package
   preflight on the pushed commit, then tags it `v0.0.4` (annotated). A tag
   used for a release never moves; the workflow refuses to retag.
3. The upload waits for approval in the `crates-io` environment. Review the
   preflight output, then approve. The workflow publishes the missing crates
   and creates the GitHub release, labelled by the version rules above, with
   the dated CHANGELOG entry as its notes.
4. Check all six crates.io pages and docs.rs builds, then compile an external
   consumer using `causa = "0.0.4"` in default, kernel-only and full modes.

Uploads run only in this workflow, from `main` or a `v*` tag. The helpers
refuse to run outside GitHub Actions, and refuse a missing token, a tag on
another commit, dirty checkouts and unfinished release notes. Once the tag
names an earlier commit, later pushes at the same version release nothing.

## Partial release recovery

Rerun the failed jobs of the same run, or dispatch **Publish** on the `v0.0.4`
tag. The tag step is a no-op when the tag already names the commit. Before
uploading anything, the helper checks all six exact versions. It skips an
existing version only if its published VCS metadata identifies this clean
commit; yanked versions, different commits, malformed archives and registry
errors stop the release. Cargo publishes the remaining packages in dependency
order and waits for index visibility. If all six already exist from the tag,
the upload is a no-op. The GitHub release is created only when the tag has
none; an existing release is left as it is, so edit its notes on GitHub or
delete it before a rerun.

This check uses the package's recorded VCS metadata; it is a retry guard,
not a cryptographic source attestation. A failed upload may already have
reached crates.io, so inspect the registry and rerun instead of moving the
tag or trying to overwrite a version.

References: [Cargo publishing](https://doc.rust-lang.org/cargo/reference/publishing.html),
[version constraints](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html),
[Trusted Publishing](https://crates.io/docs/trusted-publishing).
