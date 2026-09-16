"""Offline release tests: no test uploads or contacts crates.io."""

import copy
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import release

VERSION = "0.0.1"
COMMIT = "a" * 40


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.enterContext(patch.object(release, "ROOT", self.root))
        for license_name in ("LICENSE-MIT", "LICENSE-APACHE"):
            (self.root / license_name).write_text(license_name)
        (self.root / "CHANGELOG.md").write_text("## [0.0.1] - 2026-09-16\n")
        self.packages = []
        for name in sorted(release.FAMILY):
            directory = self.root / name
            directory.mkdir()
            for license_name in ("LICENSE-MIT", "LICENSE-APACHE"):
                (directory / license_name).symlink_to(self.root / license_name)
            self.packages.append({
                "id": name, "name": name, "version": VERSION, "publish": None,
                "manifest_path": str(directory / "Cargo.toml"),
                "dependencies": [{"name": "causa-kernel", "req": "^0.0.1", "kind": "dev"}],
            })
        self.metadata = {
            "packages": self.packages, "workspace_members": sorted(release.FAMILY),
            "target_directory": str(self.root / "target"),
        }

    def archive(self, name, commit=COMMIT, dirty=False, omit=None):
        files = {
            "LICENSE-MIT": b"LICENSE-MIT", "LICENSE-APACHE": b"LICENSE-APACHE",
            "README.md": b"Causa", "Cargo.toml": b'[package]\nversion = "0.0.1"\n',
            ".cargo_vcs_info.json": json.dumps({"git": {"sha1": commit, "dirty": dirty}}).encode(),
        }
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as bundle:
            for filename, contents in files.items():
                if filename != omit:
                    entry = tarfile.TarInfo(f"{name}-{VERSION}/{filename}")
                    entry.size = len(contents)
                    bundle.addfile(entry, io.BytesIO(contents))
        return output.getvalue()

    def registry(self, published, commit=COMMIT, dirty=False, yanked=False):
        def fetch(url):
            if "/api/v1/" in url:
                name = url.split("/")[-2]
                if name not in published:
                    raise HTTPError(url, 404, "not found", {}, None)
                return json.dumps({"version": {"yanked": yanked}}).encode()
            name = url.split("/")[-2]
            return self.archive(name, commit=commit, dirty=dirty)
        return fetch

    def allow_publish(self):
        self.enterContext(patch.dict(os.environ, {
            "GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/tags/v0.0.1",
            "CARGO_REGISTRY_TOKEN": "offline-test-token",
        }, clear=True))
        self.enterContext(patch.object(release, "run", side_effect=lambda *args:
                                      "" if args[1] == "status" else COMMIT))

    def test_workspace_accepts_family_and_rejects_version_or_dependency_drift(self):
        with patch.object(release, "run", return_value=json.dumps(self.metadata)):
            self.assertEqual(len(release.workspace(VERSION)[1]), 6)
        for field in ("package", "dev-dependency", "publish"):
            metadata = copy.deepcopy(self.metadata)
            package = metadata["packages"][0]
            if field == "package":
                package["version"] = "0.1.0"
            elif field == "dev-dependency":
                package["dependencies"][0]["req"] = "^0.1"
            else:
                package["publish"] = []
            with self.subTest(field=field), patch.object(release, "run", return_value=json.dumps(metadata)):
                with self.assertRaises(ValueError):
                    release.workspace(VERSION)

    def test_nonexperimental_version_is_refused_before_cargo(self):
        with patch.object(release, "run") as run:
            for version in ("0.1.0", "0.0.0", "0.0.01", "0.0.1;echo bad"):
                with self.subTest(version=version), self.assertRaises(ValueError):
                    release.workspace(version)
            run.assert_not_called()

    def test_packaged_licenses_are_checked_in_the_archive(self):
        directory = self.root / "target/package/tmp-crate"
        directory.mkdir(parents=True)
        for package in self.packages:
            (directory / f"{package['name']}-{VERSION}.crate").write_bytes(self.archive(package["name"]))
        release.check_packages(self.metadata, self.packages)
        (directory / f"causa-{VERSION}.crate").write_bytes(self.archive("causa", omit="LICENSE-MIT"))
        with self.assertRaises(KeyError):
            release.check_packages(self.metadata, self.packages)

    def test_partial_release_only_selects_missing_packages(self):
        published = {"causa-kernel", "causa-runtime"}
        with patch.object(release, "fetch", side_effect=self.registry(published)):
            self.assertEqual(set(release.unpublished(VERSION, COMMIT)), release.FAMILY - published)

    def test_existing_foreign_dirty_or_yanked_version_stops_retry(self):
        for options in ({"commit": "b" * 40}, {"dirty": True}, {"yanked": True}):
            with self.subTest(options=options), patch.object(
                release, "fetch", side_effect=self.registry(release.FAMILY, **options)
            ):
                with self.assertRaises(ValueError):
                    release.unpublished(VERSION, COMMIT)

    def test_registry_errors_are_not_treated_as_missing_versions(self):
        error = HTTPError("https://crates.io", 503, "unavailable", {}, None)
        with patch.object(release, "fetch", side_effect=error), self.assertRaises(HTTPError):
            release.unpublished(VERSION, COMMIT)

    def test_upload_refuses_local_execution_wrong_tag_and_missing_token(self):
        valid = {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/tags/v0.0.1",
                 "CARGO_REGISTRY_TOKEN": "offline-test-token"}
        for key in valid:
            environment = {k: v for k, v in valid.items() if k != key}
            with self.subTest(key=key), patch.dict(os.environ, environment, clear=True), \
                    patch.object(release, "unpublished") as lookup, patch.object(release.subprocess, "run") as upload:
                with self.assertRaises(ValueError):
                    release.publish(VERSION)
                lookup.assert_not_called()
                upload.assert_not_called()

    def test_unfinalized_notes_prevent_upload(self):
        self.allow_publish()
        (self.root / "CHANGELOG.md").write_text("## [Unreleased]\n")
        with patch.object(release, "unpublished") as lookup, patch.object(release.subprocess, "run") as upload:
            with self.assertRaisesRegex(ValueError, "Finalize"):
                release.publish(VERSION)
            lookup.assert_not_called()
            upload.assert_not_called()

    def test_dirty_checkout_or_moved_tag_prevents_upload(self):
        self.allow_publish()
        for replies in ([" M Cargo.toml"], ["", COMMIT, "b" * 40]):
            with self.subTest(replies=replies), patch.object(release, "run", side_effect=replies), \
                    patch.object(release, "unpublished") as lookup, patch.object(release.subprocess, "run") as upload:
                with self.assertRaises(ValueError):
                    release.publish(VERSION)
                lookup.assert_not_called()
                upload.assert_not_called()

    def test_first_release_selects_all_six_missing_packages(self):
        with patch.object(release, "fetch", side_effect=self.registry(set())):
            self.assertEqual(set(release.unpublished(VERSION, COMMIT)), release.FAMILY)

    def test_publish_passes_only_remaining_packages_to_cargo(self):
        self.allow_publish()
        with patch.object(release, "unpublished", return_value=["causa-provider", "causa"]), \
                patch.object(release.subprocess, "run") as upload:
            release.publish(VERSION)
            upload.assert_called_once_with(
                ["cargo", "publish", "--locked", "--registry", "crates-io",
                 "-p", "causa-provider", "-p", "causa"], cwd=self.root, check=True,
            )

    def test_completed_release_is_a_no_op(self):
        self.allow_publish()
        with patch.object(release, "unpublished", return_value=[]), patch.object(release.subprocess, "run") as upload:
            release.publish(VERSION)
            upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
