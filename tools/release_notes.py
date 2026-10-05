# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The release notes of a version: its section of CHANGELOG.md.

    python tools/release_notes.py 1.0.0rc1 > notes.md
    python tools/release_notes.py --check v1.0.0rc1   # the tag matches the packages

Used by the release workflow (.github/workflows/release.yml) for the GitHub
release, and by ``--check`` to stop a release whose tag, package version and
changelog do not agree.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def notes(changelog: str, version: str) -> str:
    """The text under ``## <version>`` (up to the next ``## ``), or "" if there is no
    such section."""
    lines = changelog.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            title = line[3:].strip()
            inside = title == version or title.startswith(version + " ")
            continue
        if inside:
            out.append(line)
    return "\n".join(out).strip()


def package_version(init: Path) -> str:
    match = re.search(r'^__version__ = "([^"]+)"', init.read_text(encoding="utf-8"), re.M)
    if not match:
        raise SystemExit(f"no __version__ in {init}")
    return match.group(1)


def check(tag: str) -> list[str]:
    """What is wrong with releasing ``tag`` (v<version of lagoon-viz>); empty if fine."""
    problems = []
    version = package_version(ROOT / "src" / "lagoonviz" / "__init__.py")
    if tag.removeprefix("v") != version:
        problems.append(f"tag {tag} is not v{version} (src/lagoonviz/__init__.py)")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not notes(changelog, version):
        problems.append(f"CHANGELOG.md has no section '## {version}'")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--check":
        problems = check(argv[1])
        for problem in problems:
            print(f"release check: {problem}", file=sys.stderr)
        return 1 if problems else 0
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    text = notes((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), argv[0].removeprefix("v"))
    if not text:
        print(f"no section for {argv[0]} in CHANGELOG.md", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
