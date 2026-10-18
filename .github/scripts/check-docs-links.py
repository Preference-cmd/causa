#!/usr/bin/env python3
"""Check the site's docs.rs links to Causa crates against local rustdoc output.

Run from the repository root after
`cargo doc --workspace --all-features --no-deps --locked`. Every link to a
`causa*` crate, written as a full `https://docs.rs/...` URL or as a quoted
path after `https://docs.rs/` (the guides' run-it boxes), must name a page
rustdoc generated. A redirect stub fails too: it sits at a private module
path, and the message names the canonical page to link instead. A
`#fragment` must be an id on the page. Links to other crates are skipped.
"""

from pathlib import Path
import re
import sys

DOC = Path("target/doc")
SOURCES = ["website/src/content/docs", "README.md"]
URL = re.compile(r"https://docs\.rs/(causa[a-z-]*)((?:/[^\s'\"`)<>\]]*)?)")
QUOTED_PATH = re.compile(r"['\"](causa[a-z-]*)(/[^/'\"\s]+/causa[a-z_]*/[^'\"\s]*)['\"]")
REDIRECT = re.compile(r'http-equiv="refresh" content="0;URL=([^"]+)"')


def files():
    for source in map(Path, SOURCES):
        if source.is_file():
            yield source
        elif source.is_dir():
            yield from sorted(p for p in source.rglob("*") if p.suffix in {".md", ".mdx"})


def links(text):
    for pattern in (URL, QUOTED_PATH):
        for match in pattern.finditer(text):
            yield text.count("\n", 0, match.start()) + 1, match.group(1), match.group(2)


def problem(crate, rest):
    """Return why `docs.rs/<crate><rest>` has no page, or None if it has one."""
    ident = crate.replace("-", "_")
    rest, _, fragment = rest.partition("#")
    rest = rest.rstrip(".,;:")
    parts = [p for p in rest.split("/") if p]
    if not parts:
        page = DOC / ident / "index.html"
    else:
        # parts[0] is the version: `latest` or a release number.
        if len(parts) < 2 or parts[1] != ident:
            return f"expected /<version>/{ident}/ after the crate name"
        page = DOC.joinpath(*parts[1:])
        if rest.endswith("/") or len(parts) == 2:
            page = page / "index.html"
    if not page.is_file():
        return f"rustdoc generated no {page}"
    html = page.read_text(encoding="utf-8")
    redirect = REDIRECT.search(html)
    if redirect:
        target = (page.parent / redirect.group(1)).resolve().relative_to(DOC.resolve())
        return f"redirect stub; link {crate}/latest/{target.as_posix()} instead"
    if fragment and f'id="{fragment}"' not in html:
        return f"{page} has no id {fragment!r}"
    return None


def main() -> int:
    if not DOC.is_dir():
        print("FAIL: no target/doc; run cargo doc first, from the repository root", file=sys.stderr)
        return 1
    checked = failed = 0
    for path in files():
        for line, crate, rest in links(path.read_text(encoding="utf-8")):
            checked += 1
            reason = problem(crate, rest)
            if reason:
                failed += 1
                print(f"FAIL: {path}:{line}: docs.rs/{crate}{rest}: {reason}", file=sys.stderr)
    if failed:
        return 1
    print(f"docs.rs links OK ({checked} checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
