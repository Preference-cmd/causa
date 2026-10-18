"""Exercise CI guards with isolated source layouts and cargo-tree fixtures."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def layout(self):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "check-module-layout.py")],
            cwd=self.root, capture_output=True, text=True,
        )

    def test_library_index_and_test_fixture_are_allowed(self):
        (self.root / "crates/sample/src/session").mkdir(parents=True)
        (self.root / "crates/sample/src/session.rs").touch()
        fixture = self.root / "crates/sample/tests/common/mod.rs"
        fixture.parent.mkdir(parents=True)
        fixture.touch()
        self.assertEqual(self.layout().returncode, 0)

    def test_nested_library_mod_is_rejected(self):
        forbidden = self.root / "crates/sample/src/session/nested/mod.rs"
        forbidden.parent.mkdir(parents=True)
        forbidden.touch()
        result = self.layout()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("crates/sample/src/session/nested/mod.rs", result.stderr)

    def test_wrong_working_directory_is_rejected(self):
        self.assertNotEqual(self.layout().returncode, 0)

    def dependencies(self, forbidden="", cargo_error=False, kernel_extra=""):
        # These fixtures model cargo's normal tree. A forbidden optional edge
        # appears only with --all-features, so omitting the flag loses detection.
        cargo = self.root / "cargo"
        cargo.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "if os.environ['CARGO_PROBE_ERROR'] == '1': sys.exit(42)\n"
            "package = sys.argv[sys.argv.index('-p') + 1]\n"
            "print(package + ' v0.0.1')\n"
            "if package != 'causa-kernel': print('`-- causa-kernel v0.0.1')\n"
            "if package == 'causa-kernel':\n"
            "    for dep in ['async-trait', 'futures-core', 'serde', 'serde_json', 'thiserror', 'tokio-util', 'uuid']:\n"
            "        print('|-- ' + dep + ' v0.0.1')\n"
            "    if os.environ.get('CARGO_PROBE_KERNEL_EXTRA'):\n"
            "        print('`-- ' + os.environ['CARGO_PROBE_KERNEL_EXTRA'] + ' v0.0.1')\n"
            "if package == 'causa-runtime' and '--all-features' in sys.argv:\n"
            "    if os.environ['CARGO_PROBE_EDGE']:\n"
            "        print('`-- ' + os.environ['CARGO_PROBE_EDGE'] + ' v0.0.1')\n"
        )
        cargo.chmod(0o755)
        env = dict(os.environ, PATH=f"{self.root}{os.pathsep}{os.environ['PATH']}",
                   CARGO_PROBE_EDGE=forbidden, CARGO_PROBE_ERROR=str(int(cargo_error)),
                   CARGO_PROBE_KERNEL_EXTRA=kernel_extra)
        return subprocess.run(
            ["bash", str(SCRIPTS / "check-dependency-directions.sh")],
            cwd=self.root, env=env, capture_output=True, text=True,
        )

    def test_kernel_only_runtime_passes(self):
        self.assertEqual(self.dependencies().returncode, 0)

    def test_kernel_direct_tokio_dependency_is_rejected(self):
        result = self.dependencies(kernel_extra="tokio")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("causa-kernel has disallowed direct dependency: tokio", result.stderr)

    def test_optional_runtime_protocol_dependency_is_rejected(self):
        result = self.dependencies("causa-protocol")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("causa-runtime must not depend on causa-protocol", result.stderr)

    def test_runtime_transport_dependency_is_rejected(self):
        result = self.dependencies("reqwest")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("causa-runtime must not depend on reqwest", result.stderr)

    def test_cargo_failure_is_not_a_successful_check(self):
        self.assertNotEqual(self.dependencies(cargo_error=True).returncode, 0)


class DocsLinkTests(unittest.TestCase):
    """Site links to docs.rs must name pages the local rustdoc build has."""

    STUB = (
        '<meta http-equiv="refresh" '
        'content="0;URL=../../causa_kernel/struct.Context.html">'
    )

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.write("target/doc/causa_kernel/index.html", "<main></main>")
        self.write(
            "target/doc/causa_kernel/struct.Context.html",
            '<h2 id="method.edit">edit</h2>',
        )
        self.write("target/doc/causa_kernel/context/struct.Context.html", self.STUB)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def check(self, page):
        self.write("website/src/content/docs/guides/page.mdx", page)
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "check-docs-links.py")],
            cwd=self.root, capture_output=True, text=True,
        )

    def test_existing_pages_urls_and_run_it_paths_pass(self):
        result = self.check(
            "[kernel](https://docs.rs/causa-kernel) and\n"
            "[edit](https://docs.rs/causa-kernel/latest/causa_kernel/struct.Context.html#method.edit).\n"
            "docs={['causa-kernel/latest/causa_kernel/struct.Context.html']}\n"
            "[rmcp](https://docs.rs/rmcp/latest/rmcp/struct.Missing.html)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3 checked", result.stdout)

    def test_missing_item_is_rejected(self):
        result = self.check("docs={['causa-kernel/latest/causa_kernel/struct.Gone.html']}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("page.mdx:1", result.stderr)

    def test_unknown_crate_is_rejected(self):
        self.assertNotEqual(self.check("https://docs.rs/causa-kernal").returncode, 0)

    def test_redirect_stub_names_the_canonical_page(self):
        result = self.check(
            "https://docs.rs/causa-kernel/latest/causa_kernel/context/struct.Context.html"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("causa-kernel/latest/causa_kernel/struct.Context.html instead", result.stderr)

    def test_missing_fragment_is_rejected(self):
        result = self.check(
            "https://docs.rs/causa-kernel/latest/causa_kernel/struct.Context.html#method.gone"
        )
        self.assertNotEqual(result.returncode, 0)

    def test_missing_rustdoc_output_is_rejected(self):
        shutil.rmtree(self.root / "target")
        self.assertNotEqual(self.check("").returncode, 0)


class CommitMessageTests(unittest.TestCase):
    """The commit-msg hook and the CI range check share one rule set."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = dict(
            os.environ,
            GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
            GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com",
            GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
        )
        self.git("init", "-q", "-b", "main")
        self.commit("chore: start", {"README.md": "start"})

    def git(self, *args):
        return subprocess.run(
            ["git", *args], cwd=self.root, env=self.env,
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    def stage(self, files):
        for name, text in files.items():
            (self.root / name).write_text(text)
            self.git("add", name)

    def commit(self, message, files):
        self.stage(files)
        self.git("commit", "-q", "--no-verify", "-m", message)
        return self.git("rev-parse", "HEAD")

    def check(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "check-commit-messages.py"), *args],
            cwd=self.root, env=self.env, capture_output=True, text=True,
        )

    def message(self, text, staged=None):
        self.stage(staged or {})
        path = self.root / ".git" / "COMMIT_EDITMSG"
        path.write_text(text)
        return self.check("message", str(path))

    def test_conventional_titles_pass(self):
        for title in ["feat: add x", "fix(provider): y", "ci: z", "refactor!: w"]:
            with self.subTest(title=title):
                staged = {"CHANGELOG.md": title} if "!" in title else None
                self.assertEqual(self.message(title + "\n", staged).returncode, 0)

    def test_free_form_title_is_rejected(self):
        result = self.message("update stuff\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Conventional Commits", result.stderr)

    def test_merge_and_revert_titles_are_exempt(self):
        for title in ["Merge branch 'x'", 'Revert "feat: add x"']:
            with self.subTest(title=title):
                self.assertEqual(self.message(title + "\n").returncode, 0)

    def test_comment_lines_are_ignored(self):
        text = "# Please enter the commit message\nfeat: add x\n# trailing note\n"
        self.assertEqual(self.message(text).returncode, 0)

    def test_breaking_title_needs_a_staged_changelog(self):
        self.stage({"lib.rs": "x"})
        result = self.message("feat!: drop x\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CHANGELOG.md", result.stderr)
        self.assertEqual(self.message("feat!: drop x\n", {"CHANGELOG.md": "x"}).returncode, 0)

    def test_breaking_footer_needs_a_staged_changelog(self):
        text = "feat: drop x\n\nBREAKING CHANGE: x is gone\n"
        self.assertNotEqual(self.message(text, {"lib.rs": "x"}).returncode, 0)

    def test_amending_a_breaking_commit_keeps_its_changelog(self):
        self.commit("feat!: drop x", {"CHANGELOG.md": "x", "lib.rs": "x"})
        self.assertEqual(self.message("feat!: drop x\n", {"lib.rs": "y"}).returncode, 0)

    def test_range_with_breaking_commit_and_no_changelog_fails(self):
        base = self.git("rev-parse", "HEAD")
        self.commit("feat!: drop x", {"lib.rs": "x"})
        head = self.commit("docs: note", {"notes.md": "x"})
        result = self.check("range", base, head)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("feat!: drop x", result.stderr)

    def test_range_with_changelog_in_a_later_commit_passes(self):
        base = self.git("rev-parse", "HEAD")
        self.commit("feat!: drop x", {"lib.rs": "x"})
        head = self.commit("docs: record the break", {"CHANGELOG.md": "x"})
        self.assertEqual(self.check("range", base, head).returncode, 0)

    def test_range_reads_the_breaking_footer(self):
        base = self.git("rev-parse", "HEAD")
        head = self.commit("feat: drop x\n\nBREAKING CHANGE: x is gone", {"lib.rs": "x"})
        self.assertNotEqual(self.check("range", base, head).returncode, 0)

    def test_range_ignores_titles_and_merges(self):
        base = self.git("rev-parse", "HEAD")
        self.git("checkout", "-q", "-b", "side")
        self.commit("update stuff", {"side.md": "x"})
        self.git("checkout", "-q", "main")
        self.commit("fix: y", {"main.md": "y"})
        self.git("merge", "-q", "--no-ff", "--no-verify", "-m", "Merge branch 'side'", "side")
        head = self.git("rev-parse", "HEAD")
        self.assertEqual(self.check("range", base, head).returncode, 0)

    def test_unknown_base_checks_head_alone(self):
        head = self.commit("feat!: drop x", {"lib.rs": "x"})
        self.assertNotEqual(self.check("range", "0" * 40, head).returncode, 0)
        head = self.commit("feat!: drop y", {"CHANGELOG.md": "y"})
        self.assertEqual(self.check("range", "f" * 40, head).returncode, 0)


if __name__ == "__main__":
    unittest.main()
