#!/usr/bin/env python3
"""Prepare, validate and resume a release of the crate family, and create its
GitHub release."""

import argparse
import datetime
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
FAMILY = {
    "causa", "causa-kernel", "causa-protocol", "causa-runtime",
    "causa-provider", "causa-extension",
}
USER_AGENT = "Causa-release (https://github.com/Preference-cmd/causa)"
# Every version number in these files names the current release, except the
# one before it in an "upgrade from" sentence; both move up together.
# `website/src/content/docs/status.mdx` describes each release in prose and is
# written by hand.
INSTALL_DOCS = (
    "README.md", "crates/causa/src/lib.rs",
    "website/src/content/docs/index.mdx", "website/src/content/docs/crates.mdx",
    "website/src/content/docs/examples.mdx", "website/src/content/docs/getting-started.mdx",
)
DATED_ENTRY = r"^## \[([0-9.]+)\] - \d{4}-\d{2}-\d{2}$"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def require_version(version):
    require(re.fullmatch(r"0\.(0\.[1-9][0-9]*|[1-9][0-9]*\.(0|[1-9][0-9]*))", version),
            "This workflow releases 0.0.x (x >= 1) and 0.y.z (y >= 1) versions; 1.0 is a separate decision")


def experimental(version):
    """Only a 0.y.0 version is a regular release; 0.0.x and the 0.y.z patches
    between minor versions are experimental pre-releases."""
    return not re.fullmatch(r"0\.[1-9][0-9]*\.0", version)


def numbers(version):
    return tuple(map(int, version.split(".")))


def workspace(version):
    require_version(version)
    metadata = json.loads(run("cargo", "metadata", "--no-deps", "--locked", "--format-version=1"))
    packages = [p for p in metadata["packages"] if p["id"] in metadata["workspace_members"]]
    require({p["name"] for p in packages} == FAMILY, "Unexpected workspace release set")
    for package in packages:
        name = package["name"]
        require(package["version"] == version, f"{name}: version differs from {version}")
        require(package["publish"] in (None, ["crates-io"]), f"{name}: not publishable to crates.io")
        for dependency in package["dependencies"]:
            if dependency["name"] in FAMILY:
                require(dependency["req"] in (f"^{version}", f"={version}"),
                        f"{name}: stale family dependency {dependency['name']}")
        for license_name in ("LICENSE-MIT", "LICENSE-APACHE"):
            path = Path(package["manifest_path"]).parent / license_name
            require(path.read_bytes() == (ROOT / license_name).read_bytes(),
                    f"{name}: missing or stale {license_name}")
    return metadata, packages


def check_packages(metadata, packages):
    """Inspect the actual archives after cargo's workspace dry-run."""
    directory = Path(metadata["target_directory"]) / "package"
    for package in packages:
        stem = f"{package['name']}-{package['version']}"
        archive = directory / "tmp-crate" / f"{stem}.crate"
        if not archive.exists():
            archive = directory / f"{stem}.crate"
        with tarfile.open(archive, "r:gz") as bundle:
            for name in ("LICENSE-MIT", "LICENSE-APACHE"):
                require(bundle.extractfile(f"{stem}/{name}").read() == (ROOT / name).read_bytes(),
                        f"{stem}: packaged {name} differs from the root license")
            require(bool(bundle.extractfile(f"{stem}/README.md").read()), f"{stem}: empty README")
            manifest = tomllib.loads(bundle.extractfile(f"{stem}/Cargo.toml").read().decode())
            require(manifest["package"]["version"] == package["version"], f"{stem}: stale archive")
        print(f"Verified archive: {stem}")


def dated_entry(changelog, version):
    """Find the heading of the dated CHANGELOG entry for `version`, if any."""
    return next((heading for heading in re.finditer(DATED_ENTRY, changelog, re.M)
                 if heading.group(1) == version), None)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def unpublished(version, commit):
    """Skip an existing version only when it came from this clean commit."""
    remaining = []
    for name in sorted(FAMILY):
        try:
            data = json.loads(fetch(f"https://crates.io/api/v1/crates/{name}/{version}"))
        except urllib.error.HTTPError as error:
            error.close()
            if error.code != 404:
                raise
            remaining.append(name)
            continue
        require(not data["version"]["yanked"], f"{name} {version} is yanked; inspect it before retrying")
        body = fetch(f"https://static.crates.io/crates/{name}/{name}-{version}.crate")
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as bundle:
            vcs = json.load(bundle.extractfile(f"{name}-{version}/.cargo_vcs_info.json"))["git"]
        require(vcs["sha1"] == commit and not vcs.get("dirty", False),
                f"{name} {version} already exists from different or dirty sources")
        print(f"Already published from this commit: {name} {version}")
    return remaining


def publish(version):
    require(os.environ.get("GITHUB_ACTIONS") == "true", "Upload only through the Publish workflow")
    require(bool(os.environ.get("CARGO_REGISTRY_TOKEN")),
            "Missing CARGO_REGISTRY_TOKEN from the Trusted Publishing exchange")
    require(not run("git", "status", "--porcelain"), "Release checkout must be clean")
    commit = run("git", "rev-parse", "HEAD")
    require(run("git", "rev-parse", f"refs/tags/v{version}^{{commit}}") == commit,
            "Release tag and checkout differ")
    changelog = (ROOT / "CHANGELOG.md").read_text()
    require(dated_entry(changelog, version),
            "Finalize the dated CHANGELOG release entry before tagging")
    remaining = unpublished(version, commit)
    if not remaining:
        print(f"All six crates at {version} are already published from this commit")
        return
    command = ["cargo", "publish", "--locked", "--registry", "crates-io"]
    for name in remaining:
        command.extend(["-p", name])
    # Cargo orders the selected packages by their dependencies and waits for
    # index visibility. A retry rechecks all six versions before any upload.
    subprocess.run(command, cwd=ROOT, check=True)


def notes(version):
    """Return the CHANGELOG section of one dated release, without its heading."""
    changelog = (ROOT / "CHANGELOG.md").read_text()
    heading = dated_entry(changelog, version)
    require(heading, f"CHANGELOG has no dated entry for {version}")
    rest = changelog[heading.end():]
    following = re.search(r"^## ", rest, re.M)
    body = (rest[:following.start()] if following else rest).strip()
    require(body, f"CHANGELOG entry for {version} is empty")
    return body + "\n"


def github_release(version):
    """Create the GitHub release for an uploaded version; a rerun changes nothing."""
    require(os.environ.get("GITHUB_ACTIONS") == "true", "Create releases only through the Publish workflow")
    tag = f"v{version}"
    require(run("git", "rev-parse", f"refs/tags/{tag}^{{commit}}") == run("git", "rev-parse", "HEAD"),
            "Release tag and checkout differ")
    body = notes(version)
    existing = subprocess.run(["gh", "release", "view", tag], cwd=ROOT, capture_output=True, text=True)
    if existing.returncode == 0:
        print(f"GitHub release {tag} already exists")
        return
    require("release not found" in existing.stderr.lower(),
            f"Could not look up release {tag}: {existing.stderr.strip()}")
    label = ["--prerelease", "--title", f"{tag} — Experimental"] if experimental(version) else ["--title", tag]
    subprocess.run(
        ["gh", "release", "create", tag, "--verify-tag", *label, "--notes-file", "-"],
        cwd=ROOT, input=body, text=True, check=True)
    print(f"Created GitHub release {tag}")


def remote_tag(version):
    """Return the commit the remote vVERSION tag points at, or None."""
    tag = f"refs/tags/v{version}"
    listing = run("git", "ls-remote", "origin", tag, f"{tag}^{{}}")
    refs = {name: commit for commit, name in (line.split("\t") for line in listing.splitlines())}
    # An annotated tag lists its object and, peeled, the commit it names.
    return refs.get(f"{tag}^{{}}", refs.get(tag))


def plan():
    """Decide whether this commit releases the workspace version.

    It does when CHANGELOG has the version's dated entry and the remote tag is
    missing or already names this commit (a rerun). Once the tag names an
    earlier commit, later commits at the same version release nothing.
    """
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text()
    if not dated_entry(changelog, version):
        return version, False, f"CHANGELOG has no dated entry for {version}"
    tagged = remote_tag(version)
    if tagged is None:
        return version, True, f"Releasing {version}: v{version} is not tagged yet"
    if tagged == run("git", "rev-parse", "HEAD"):
        return version, True, f"Resuming {version}: v{version} already names this commit"
    return version, False, f"{version} was released from {tagged[:12]}"


def tag(version):
    """Tag this commit vVERSION on the remote; a rerun on the same commit changes nothing."""
    require_version(version)
    require(os.environ.get("GITHUB_ACTIONS") == "true", "Tag releases only through the Publish workflow")
    require(not run("git", "status", "--porcelain"), "Release checkout must be clean")
    commit = run("git", "rev-parse", "HEAD")
    tagged = remote_tag(version)
    if tagged is not None:
        require(tagged == commit, f"v{version} already names {tagged}; a tag used for a release never moves")
        print(f"v{version} already names this commit")
        return
    kind = "experimental release" if experimental(version) else "release"
    run("git", "tag", "--annotate", f"v{version}", "--message", f"Causa v{version} {kind}", commit)
    run("git", "push", "origin", f"refs/tags/v{version}")
    print(f"Tagged {commit} as v{version}")


def finalize_changelog(changelog, version, date):
    """Date the unreleased notes as `version` and open an empty Unreleased section."""
    require(not re.search(rf"^## \[{re.escape(version)}\]", changelog, re.M),
            f"CHANGELOG already has an entry for {version}")
    heading = re.search(r"^## \[Unreleased\]$", changelog, re.M)
    require(heading, "CHANGELOG has no [Unreleased] section")
    rest = changelog[heading.end():]
    following = re.search(r"^## ", rest, re.M)
    require((rest[:following.start()] if following else rest).strip(),
            "The [Unreleased] section is empty; there is nothing to release")
    return changelog[:heading.end()] + f"\n\n## [{version}] - {date}" + rest


def bump_manifest(text, current, version):
    """Move the workspace version and the family dependency requirements."""
    family = "|".join(map(re.escape, sorted(FAMILY)))
    pattern = rf'^(version = |(?:{family}) = \{{.*\bversion = )"{re.escape(current)}"'
    return re.sub(pattern, rf'\1"{version}"', text, flags=re.M)


def shift_versions(text, previous, current, version, date):
    """Name `version` where the text named `current`, and `current` where it
    named `previous`, including the link anchor of the current release notes."""
    def anchor(number):
        return number.replace(".", "")
    text = re.sub(rf"#{anchor(current)}---\d{{4}}-\d{{2}}-\d{{2}}", f"#{anchor(version)}---{date}", text)
    moves = {current: version}
    if previous:
        moves[previous] = current
    pattern = rf"(?<![\d.])({'|'.join(map(re.escape, moves))})(?!\d|\.\d)"
    return re.sub(pattern, lambda match: moves[match.group(1)], text)


def prepare(version, date):
    """Move the manifests, lockfiles, CHANGELOG and install docs to `version`.

    Nothing is committed: the diff is for review. Every file is rewritten in
    memory first, so a refusal leaves the checkout untouched.
    """
    require_version(version)
    require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), "The release date must be YYYY-MM-DD")
    changelog = (ROOT / "CHANGELOG.md").read_text()
    released = re.findall(DATED_ENTRY, changelog, re.M)
    require(released, "CHANGELOG has no dated release to move from")
    current, previous = released[0], (released[1] if len(released) > 1 else None)
    require(numbers(version) > numbers(current), f"{version} does not follow the latest release {current}")
    edits = {ROOT / "CHANGELOG.md": finalize_changelog(changelog, version, date)}
    for path in [ROOT / "Cargo.toml", *sorted((ROOT / "crates").glob("*/Cargo.toml"))]:
        edits[path] = bump_manifest(path.read_text(), current, version)
    require(edits[ROOT / "Cargo.toml"] != (ROOT / "Cargo.toml").read_text(),
            f"The workspace version is not {current}, the latest dated CHANGELOG entry")
    for name in INSTALL_DOCS:
        path = ROOT / name
        edits[path] = shift_versions(path.read_text(), previous, current, version, date)
    for path, text in edits.items():
        path.write_text(text)
    # The consumer workspaces depend on the crates by path; their lockfiles
    # record the version too, and CI builds them with --locked.
    for manifest in [ROOT / "Cargo.toml", *sorted((ROOT / "tests/consumers").glob("*/Cargo.toml"))]:
        run("cargo", "update", "--workspace", "--manifest-path", str(manifest))
    workspace(version)
    print(f"Prepared {version} from {current}. Rewrite website/src/content/docs/status.mdx "
          "for this release by hand, then review the whole diff before committing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "check", "packages", "plan", "tag", "publish", "notes", "github-release"))
    parser.add_argument("--version")
    parser.add_argument("--date", default=datetime.date.today().isoformat(),
                        help="release date for prepare (default: today)")
    args = parser.parse_args()
    if args.command == "plan":
        version, release, reason = plan()
        print(reason)
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(f"version={version}\nrelease={str(release).lower()}\n")
        return
    if not args.version:
        parser.error(f"{args.command} requires --version")
    if args.command == "notes":
        sys.stdout.write(notes(args.version))
        return
    if args.command == "prepare":
        prepare(args.version, args.date)
        return
    if args.command == "tag":
        tag(args.version)
        return
    metadata, packages = workspace(args.version)
    if args.command == "packages":
        check_packages(metadata, packages)
    elif args.command == "publish":
        publish(args.version)
    elif args.command == "github-release":
        github_release(args.version)
    else:
        print(f"Validated all six crates at {args.version}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, tarfile.TarError, subprocess.CalledProcessError) as error:
        sys.exit(f"Release refused: {error}")
