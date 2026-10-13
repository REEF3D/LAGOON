# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Numbered files as series, as ParaView lists them: ``REEF3D-NHFLOW-FSF-..pvtp``.

:func:`file_groups` lists a folder's sub-folders, its numbered files grouped by
everything but the last number of their name (a series: one file per output),
and its other files. The pieces of parallel output (``.vtp``/``.vtu`` next to the
``.pvtp``/``.pvtu`` that name them) are left out where the parallel files are.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: the files LAGOON opens as output
OUTPUT_SUFFIXES = (".pvtu", ".pvtp", ".vtu", ".vtp", ".vtk", ".vtm")
_NUMBERED = re.compile(r"^(?P<head>.*?)(?P<number>\d+)(?P<tail>\D*)$")


@dataclass
class FileGroup:
    """A series of numbered files, a single file or a folder."""

    kind: str  # "folder", "series", "file"
    label: str  # "REEF3D-NHFLOW-FSF-..pvtp", a file or folder name
    paths: list[Path] = field(default_factory=list)
    #: first and last number of a series
    first: int = 0
    last: int = 0
    size: int = 0  # bytes of all its files (folders: 0)

    @property
    def path(self) -> Path:
        """What opening it opens: its first file (the series follows), the folder."""
        return self.paths[0]

    def describe(self) -> str:
        if self.kind == "folder":
            return "folder"
        if self.kind == "series":
            return f"{len(self.paths)} files, {self.first} – {self.last}, {_size(self.size)}"
        return _size(self.size)


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def file_groups(folder: str | Path, suffixes: tuple[str, ...] = OUTPUT_SUFFIXES) -> list[FileGroup]:
    """The folder's sub-folders (A-Z), then its series and single files with one
    of ``suffixes`` (A-Z); hidden files are left out."""
    folder = Path(folder)
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name.lower())
    except OSError:
        return []
    folders = [
        FileGroup("folder", e.name, [Path(e.path)])
        for e in entries
        if not e.name.startswith(".") and e.is_dir(follow_symlinks=True)
    ]
    files = [
        e
        for e in entries
        if not e.name.startswith(".")
        and e.is_file(follow_symlinks=True)
        and Path(e.name).suffix.lower() in suffixes
    ]
    parallel = {Path(e.name).suffix.lower() for e in files} & {".pvtu", ".pvtp"}
    if parallel:  # the pieces belong to the parallel files that name them
        pieces = (
            {".vtu"}
            if parallel == {".pvtu"}
            else {".vtp"}
            if parallel == {".pvtp"}
            else {".vtu", ".vtp"}
        )
        files = [e for e in files if Path(e.name).suffix.lower() not in pieces]
    series: dict[tuple[str, str], list] = {}
    singles = []
    for entry in files:
        match = _NUMBERED.match(entry.name)
        if match is None:
            singles.append(entry)
            continue
        series.setdefault((match["head"], match["tail"]), []).append((int(match["number"]), entry))
    groups = []
    for (head, tail), members in series.items():
        if len(members) == 1:
            singles.append(members[0][1])
            continue
        members.sort(key=lambda m: m[0])
        groups.append(
            FileGroup(
                "series",
                f"{head}..{tail[1:] if tail.startswith('.') else tail}",  # ParaView's way
                [Path(e.path) for _, e in members],
                members[0][0],
                members[-1][0],
                sum(_stat_size(e) for _, e in members),
            )
        )
    groups += [FileGroup("file", e.name, [Path(e.path)], size=_stat_size(e)) for e in singles]
    groups.sort(key=lambda g: g.label.lower())
    return folders + groups


def _stat_size(entry) -> int:
    try:
        return entry.stat().st_size
    except OSError:
        return 0
