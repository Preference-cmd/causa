#!/usr/bin/env python3
"""Reject mod.rs in library sources; test fixtures use their own layout."""

from pathlib import Path
import sys


def main() -> int:
    sources = sorted(Path("crates").glob("*/src"))
    if not sources:
        print("FAIL: run from the repository root (no crates/*/src)", file=sys.stderr)
        return 1
    forbidden = sorted(path for source in sources for path in source.rglob("mod.rs"))
    if forbidden:
        for path in forbidden:
            print(f"FAIL: use foo.rs + foo/ instead of {path}", file=sys.stderr)
        return 1
    print("library module layout OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
