"""Exercise CI guards with isolated source layouts and cargo-tree fixtures."""

import os
from pathlib import Path
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

    def dependencies(self, forbidden="", cargo_error=False):
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
            "if package == 'causa-runtime' and '--all-features' in sys.argv:\n"
            "    if os.environ['CARGO_PROBE_EDGE']:\n"
            "        print('`-- ' + os.environ['CARGO_PROBE_EDGE'] + ' v0.0.1')\n"
        )
        cargo.chmod(0o755)
        env = dict(os.environ, PATH=f"{self.root}{os.pathsep}{os.environ['PATH']}",
                   CARGO_PROBE_EDGE=forbidden, CARGO_PROBE_ERROR=str(int(cargo_error)))
        return subprocess.run(
            ["bash", str(SCRIPTS / "check-dependency-directions.sh")],
            cwd=self.root, env=env, capture_output=True, text=True,
        )

    def test_kernel_only_runtime_passes(self):
        self.assertEqual(self.dependencies().returncode, 0)

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


if __name__ == "__main__":
    unittest.main()
