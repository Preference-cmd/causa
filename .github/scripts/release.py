#!/usr/bin/env python3
"""Validate the experimental crate family and resume a tagged manual release."""

import argparse
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def workspace(version):
    require(re.fullmatch(r"0\.0\.[1-9][0-9]*", version),
            "This workflow only releases experimental 0.0.x versions (x >= 1)")
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
    require(os.environ.get("GITHUB_ACTIONS") == "true", "Upload only through the manual Publish workflow")
    require(os.environ.get("GITHUB_REF") == f"refs/tags/v{version}",
            f"Select the v{version} tag for an actual upload")
    require(bool(os.environ.get("CARGO_REGISTRY_TOKEN")), "Missing CARGO_REGISTRY_TOKEN repository secret")
    require(not run("git", "status", "--porcelain"), "Release checkout must be clean")
    commit = run("git", "rev-parse", "HEAD")
    require(run("git", "rev-parse", f"refs/tags/v{version}^{{commit}}") == commit,
            "Release tag and checkout differ")
    changelog = (ROOT / "CHANGELOG.md").read_text()
    require(re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.M),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "packages", "publish"))
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    metadata, packages = workspace(args.version)
    if args.command == "packages":
        check_packages(metadata, packages)
    elif args.command == "publish":
        publish(args.version)
    else:
        print(f"Validated all six crates at {args.version}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, tarfile.TarError, subprocess.CalledProcessError) as error:
        sys.exit(f"Release refused: {error}")
