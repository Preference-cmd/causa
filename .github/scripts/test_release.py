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

    def test_notes_are_the_section_of_one_dated_release(self):
        (self.root / "CHANGELOG.md").write_text(
            "# Changelog\n\n## [Unreleased]\n\n## [0.0.2] - 2026-10-02\n\nSecond.\n\n"
            "### Breaking\n\n- A change.\n\n## [0.0.1] - 2026-09-16\n\nFirst.\n")
        self.assertEqual(release.notes("0.0.2"), "Second.\n\n### Breaking\n\n- A change.\n")
        self.assertEqual(release.notes("0.0.1"), "First.\n")
        for version, text in (("0.0.3", "## [0.0.3]\n\nUndated.\n"), ("0.0.3", "## [0.0.3] - 2026-10-12\n\n")):
            (self.root / "CHANGELOG.md").write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                release.notes(version)

    def release_tree(self, unreleased="### Breaking\n\n- A change.\n"):
        files = {
            "CHANGELOG.md": f"# Changelog\n\n## [Unreleased]\n\n{unreleased}\n"
                            "## [0.0.2] - 2026-10-02\n\nSecond.\n\n## [0.0.1] - 2026-09-16\n\nFirst.\n",
            "Cargo.toml": '[workspace.package]\nversion = "0.0.2"\nrust-version = "1.96"\n\n'
                          '[workspace.dependencies]\nother = { version = "0.0.2" }\n',
            "crates/causa/Cargo.toml": '[dependencies]\n'
                                       'causa-kernel = { path = "../causa-kernel", version = "0.0.2" }\n'
                                       'causa-runtime = { path = "../causa-runtime", version = "0.0.2", optional = true }\n'
                                       'serde = { version = "0.0.2" }\n',
            "README.md": "> **Experimental 0.0.2.** See the [notes](https://x/blob/v0.0.2/CHANGELOG.md"
                         "#002---2026-10-02).\n\ncargo add causa@0.0.2\n\n"
                         'Cargo does not upgrade `"0.0.1"` to `0.0.2`. Not 0.0.20, 10.0.2 or 1.96.0.\n',
        }
        for name in release.INSTALL_DOCS[1:]:
            files[name] = "`causa = \"0.0.2\"`\n"
        for name, text in files.items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(text)
        return files

    def test_prepare_moves_versions_changelog_and_install_docs(self):
        self.release_tree()
        with patch.object(release, "run") as cargo, patch.object(release, "workspace") as check:
            release.prepare("0.0.3", "2026-10-14")
        check.assert_called_once_with("0.0.3")
        self.assertEqual([call.args[:3] for call in cargo.call_args_list],
                         [("cargo", "update", "--workspace")])
        read = lambda name: (self.root / name).read_text()
        self.assertIn("## [Unreleased]\n\n## [0.0.3] - 2026-10-14\n\n### Breaking\n\n- A change.\n\n"
                      "## [0.0.2] - 2026-10-02", read("CHANGELOG.md"))
        self.assertEqual(read("Cargo.toml"), '[workspace.package]\nversion = "0.0.3"\nrust-version = "1.96"\n\n'
                                             '[workspace.dependencies]\nother = { version = "0.0.2" }\n')
        self.assertEqual(read("crates/causa/Cargo.toml"),
                         '[dependencies]\ncausa-kernel = { path = "../causa-kernel", version = "0.0.3" }\n'
                         'causa-runtime = { path = "../causa-runtime", version = "0.0.3", optional = true }\n'
                         'serde = { version = "0.0.2" }\n')
        self.assertEqual(read("README.md"),
                         "> **Experimental 0.0.3.** See the [notes](https://x/blob/v0.0.3/CHANGELOG.md"
                         "#003---2026-10-14).\n\ncargo add causa@0.0.3\n\n"
                         'Cargo does not upgrade `"0.0.2"` to `0.0.3`. Not 0.0.20, 10.0.2 or 1.96.0.\n')
        for name in release.INSTALL_DOCS[1:]:
            self.assertEqual(read(name), "`causa = \"0.0.3\"`\n")

    def test_prepare_refuses_before_writing_anything(self):
        cases = {
            "empty unreleased notes": ({"unreleased": ""}, "0.0.3", "2026-10-14"),
            "already dated version": ({}, "0.0.2", "2026-10-14"),
            "nonexperimental version": ({}, "0.1.0", "2026-10-14"),
            "malformed date": ({}, "0.0.3", "14.10.2026"),
        }
        for label, (tree, version, date) in cases.items():
            with self.subTest(label):
                files = self.release_tree(**tree)
                with patch.object(release, "run") as cargo, self.assertRaises(ValueError):
                    release.prepare(version, date)
                cargo.assert_not_called()
                for name, text in files.items():
                    self.assertEqual((self.root / name).read_text(), text, name)

    def test_prepare_refuses_a_workspace_behind_the_latest_entry(self):
        files = self.release_tree()
        (self.root / "Cargo.toml").write_text(files["Cargo.toml"].replace("0.0.2", "0.0.1"))
        with patch.object(release, "run") as cargo, self.assertRaisesRegex(ValueError, "workspace version"):
            release.prepare("0.0.3", "2026-10-14")
        cargo.assert_not_called()
        self.assertEqual((self.root / "CHANGELOG.md").read_text(), files["CHANGELOG.md"])

    def github(self, lookup):
        environment = {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/tags/v0.0.1"}
        self.enterContext(patch.dict(os.environ, environment, clear=True))
        (self.root / "CHANGELOG.md").write_text("## [0.0.1] - 2026-09-16\n\nFirst.\n")
        return self.enterContext(patch.object(release.subprocess, "run", side_effect=[lookup, None]))

    def test_github_release_is_created_from_the_notes_once(self):
        missing = release.subprocess.CompletedProcess([], 1, "", "release not found\n")
        gh = self.github(missing)
        release.github_release(VERSION)
        self.assertEqual(gh.call_count, 2)
        gh.assert_called_with(
            ["gh", "release", "create", "v0.0.1", "--verify-tag", "--prerelease",
             "--title", "v0.0.1 — Experimental", "--notes-file", "-"],
            cwd=self.root, input="First.\n", text=True, check=True)

    def test_existing_github_release_is_left_alone(self):
        gh = self.github(release.subprocess.CompletedProcess([], 0, "v0.0.1", ""))
        release.github_release(VERSION)
        self.assertEqual(gh.call_count, 1)

    def test_github_release_stops_on_a_failed_lookup_or_outside_the_tagged_workflow(self):
        gh = self.github(release.subprocess.CompletedProcess([], 1, "", "HTTP 503\n"))
        with self.assertRaisesRegex(ValueError, "look up"):
            release.github_release(VERSION)
        self.assertEqual(gh.call_count, 1)
        for environment in ({"GITHUB_REF": "refs/tags/v0.0.1"},
                            {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/main"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ValueError):
                    release.github_release(VERSION)
        self.assertEqual(gh.call_count, 1)


if __name__ == "__main__":
    unittest.main()
