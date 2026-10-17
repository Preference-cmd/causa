"""Check Conventional Commit titles and that breaking changes update CHANGELOG.md.

Two entry points share the rules below:

    check-commit-messages.py message FILE     # commit-msg hook (lefthook)
    check-commit-messages.py range BASE HEAD  # CI, over a pushed range

`message` checks the title of the proposed message. When the message marks
a breaking change, CHANGELOG.md must be staged, or the commit must amend a
HEAD with the same title that already changed it.

`range` checks only the CHANGELOG rule, so commits made before the titles
were checked do not fail CI: when any commit in BASE..HEAD marks a breaking
change, CHANGELOG.md must change within the range. A BASE that is all zeros
or not in the history (a new branch, a force push) leaves HEAD alone to
check.
"""

import re
import subprocess
import sys

TYPES = ("feat", "fix", "refactor", "perf", "docs", "test", "chore", "ci", "build")
TITLE = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([^()]+\))?(?P<bang>!)?: \S")
EXEMPT = re.compile(r'^(?:Merge |Revert ")')
FOOTER = re.compile(r"^BREAKING[ -]CHANGE: ", re.MULTILINE)
CHANGELOG = "CHANGELOG.md"


def strip_comments(message):
    """The message git records: comment lines and the cut-off scissors removed."""
    kept = []
    for line in message.splitlines():
        if line.startswith("# ------------------------ >8 ------------------------"):
            break
        if not line.startswith("#"):
            kept.append(line)
    return "\n".join(kept).strip()


def title_of(message):
    return message.split("\n", 1)[0].strip()


def title_error(message):
    """None when the title is acceptable, otherwise the reason."""
    title = title_of(message)
    if EXEMPT.match(title) or TITLE.match(title):
        return None
    return (
        f"commit title does not follow Conventional Commits: {title!r}\n"
        f"expected '<type>[(scope)][!]: <summary>' with type one of {', '.join(TYPES)}"
    )


def is_breaking(message):
    match = TITLE.match(title_of(message))
    return bool(match and match.group("bang")) or bool(FOOTER.search(message))


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def is_commit(rev):
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True
    )
    return probe.returncode == 0


def check_message(path):
    with open(path, encoding="utf-8") as handle:
        message = strip_comments(handle.read())
    if not message:
        return 0  # git aborts an empty message itself
    error = title_error(message)
    if error:
        print(error, file=sys.stderr)
        return 1
    if not is_breaking(message):
        return 0
    if CHANGELOG in git("diff", "--cached", "--name-only").split():
        return 0
    if is_commit("HEAD") and title_of(git("log", "-1", "--format=%B", "HEAD")) == title_of(message):
        if CHANGELOG in git("show", "--format=", "--name-only", "HEAD").split():
            return 0  # amending a breaking commit that already updated it
    print(
        f"breaking change without {CHANGELOG}: record the break and its "
        f"migration under [Unreleased] and stage it with this commit",
        file=sys.stderr,
    )
    return 1


def check_range(base, head):
    if set(base) == {"0"} or not is_commit(base):
        revs, base = [head], f"{head}^" if is_commit(f"{head}^") else None
    else:
        revs = git("rev-list", f"{base}..{head}").split()
    breaking = []
    for rev in revs:
        message = git("log", "-1", "--format=%B", rev)
        if is_breaking(message):
            breaking.append(f"{rev[:12]} {title_of(message)}")
    if not breaking:
        return 0
    if base is None:
        changed = git("show", "--format=", "--name-only", head).split()
    else:
        changed = git("diff", "--name-only", base, head).split()
    if CHANGELOG in changed:
        return 0
    print(f"breaking commits without a {CHANGELOG} change in the same push:", file=sys.stderr)
    for line in breaking:
        print(f"  {line}", file=sys.stderr)
    return 1


def main(argv):
    if len(argv) == 2 and argv[0] == "message":
        return check_message(argv[1])
    if len(argv) == 3 and argv[0] == "range":
        return check_range(argv[1], argv[2])
    print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
