# SPDX-License-Identifier: GPL-3.0-or-later
"""Free surface lines: REEF3D's ``REEF3D_*_WSFLINE`` output (P 52, lines along x),
``REEF3D_*_WSFLINE_Y`` (P 56, along y) and NHFLOW's time-averaged lines
(``REEF3D_NHFLOW_TIME_AVG_WSFLINE``, ``..._Y``: P 146, P 147).

Each file is one moment: ``simtime: t``, the number of lines and their y
positions, then a table with, per line, ``X n`` (position along x), ``P n``
(the computed free surface elevation) and, from FNPF, ``W n`` (wave theory)::

    simtime:  1.44584344884
    number of wsf-lines:  1

    line_No     y_coord
    1    0.005

    X 1   P 1   W 1
    0.0145833 0.4097076 0.4097319
    ...

Lines along y have ``Y n`` (or one ``y_coord`` column, CFD) in place of ``X n``.
Files are numbered by iteration, not by output, so the time comes from
``simtime``. Only the headers are read when a case opens; a line is read when it
is shown. Time-averaged lines are written once per line, at the end of their
averaging window: one set per line.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_FOLDER = re.compile(r"^REEF3D_(?P<solver>[A-Z0-9]+)_(?P<avg>TIME_AVG_)?WSFLINE(?P<y>_Y)?$")
_COLUMN = re.compile(r"([A-Za-z]+)\s+(\d+)")


@dataclass
class ProfileLine:
    """One line of a free surface line file at one moment."""

    number: int
    y: float  # where the line lies across: its y (lines along x) or x (along y)
    x: np.ndarray  # (n,) along the line: x (lines along x) or y (along y)
    eta: np.ndarray  # (n,) computed elevation
    theory: np.ndarray | None = None  # (n,) wave theory (FNPF), if written
    axis: str = "x"  # the direction the line runs in

    @property
    def across(self) -> str:
        """The coordinate that places the line ("y" for a line along x)."""
        return "y" if self.axis == "x" else "x"


@dataclass
class Profile:
    """All lines of one file: one moment."""

    path: Path
    time: float
    lines: dict[int, ProfileLine]


def read_header(path: str | Path) -> tuple[float, dict[int, float]]:
    """(simtime, {line number: y}) from the first lines of a file."""
    time = float("nan")
    ys: dict[int, float] = {}
    with open(path, errors="replace") as handle:
        inside = False
        for _ in range(200):
            line = handle.readline()
            if not line:
                break
            tokens = line.split()
            if not tokens:
                if ys:
                    break
                continue
            if tokens[0].lower().startswith("simtime"):
                time = float(tokens[-1])
            elif tokens[0].lower().startswith("line_no"):
                inside = True
            elif inside and len(tokens) >= 2:
                try:
                    ys[int(tokens[0])] = float(tokens[1])
                except ValueError:
                    break  # e.g. "2  Wave Theory": the end of the table
            elif tokens[0] in ("X", "Y") or tokens[0].lower() in ("x_coord", "y_coord"):
                break
    return time, ys


def read_profile(path: str | Path, axis: str = "x") -> Profile:
    """Read one free surface line file (lines along ``axis``)."""
    path = Path(path)
    time, ys = read_header(path)
    lines = path.read_text(errors="replace").splitlines()
    header_at = None
    for i, line in enumerate(lines):
        first = line.split()[:1]
        if first and (first[0] in ("X", "Y") or first[0].lower() in ("y_coord", "x_coord")):
            header_at = i
            break
    if header_at is None:
        raise ValueError(f"{path.name}: no 'X 1  P 1' header")
    found_columns = [(k.upper(), int(n)) for k, n in _COLUMN.findall(lines[header_at])]
    if lines[header_at].split()[0].lower().endswith("_coord"):
        # one coordinate column named for all lines (CFD's lines along y); the rows
        # hold the coordinate and the elevation of every line
        found_columns = [
            c
            for n in (num for kind, num in found_columns if kind == "P")
            for c in (("X", n), ("P", n))
        ]
    columns = [("X" if kind == "Y" else kind, number) for kind, number in found_columns]
    rows = [line.split() for line in lines[header_at + 1 :] if line.strip()]
    numbers = sorted({c[1] for c in columns})
    if rows and "W" not in {c[0] for c in columns}:
        if sum(len(r) == len(columns) + len(numbers) for r in rows) > len(rows) // 2:
            # the wave theory written (P 53) but not named
            columns = [c for n in numbers for c in (("X", n), ("P", n), ("W", n))]
    n = len(columns)
    kept = [r for r in rows if len(r) == n]  # (a row a line has no point in: left out)
    table = np.array([[_number(v) for v in r] for r in kept], np.float64).reshape(-1, n)
    table = np.where(np.abs(table) >= 1e19, np.nan, table)  # REEF3D's "no value"
    found: dict[int, ProfileLine] = {}
    for number in numbers:
        index = {kind: i for i, (kind, num) in enumerate(columns) if num == number}
        if "X" not in index or "P" not in index:
            continue
        x = table[:, index["X"]]
        keep = np.isfinite(x)
        theory = table[keep, index["W"]] if "W" in index else None
        found[number] = ProfileLine(
            number, ys.get(number, 0.0), x[keep], table[keep, index["P"]], theory, axis
        )
    return Profile(path, time, found)


def _number(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return float("nan")


@dataclass
class ProfileSet:
    """The free surface line files of a case folder, by time."""

    key: str
    folder: Path
    solver: str
    paths: list[Path] = field(default_factory=list)
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ys: dict[int, float] = field(default_factory=dict)
    #: the direction the lines run in
    axis: str = "x"
    #: time-averaged (P 146, P 147): one line, its number
    averaged: int = 0

    def __post_init__(self) -> None:
        self._cache: dict[Path, Profile] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """What REEF3D calls them (P 52, P 56, P 146, P 147)."""
        if self.averaged:
            name = f"Time-averaged free surface line {self.averaged}"
        else:
            name = "Free surface lines"
        return name + (" in y" if self.axis == "y" else "")

    @property
    def label(self) -> str:
        return f"{self.solver} {self.name.lower()} ({len(self.paths)} times, {self.folder.name})"

    def written_span(self) -> tuple[float, float]:
        """(first, last) modification time of the files."""
        stamps = [p.stat().st_mtime for p in (self.paths[0], self.paths[-1])] if self.paths else []
        return (min(stamps), max(stamps)) if stamps else (0.0, 0.0)

    def index_at(self, time: float) -> int:
        """The file shown at a time: the last one written at or before it."""
        index = int(np.searchsorted(self.times, time + 1e-9, side="right")) - 1
        return min(max(index, 0), len(self.paths) - 1)

    def at(self, time: float) -> Profile:
        """The profile at a time (read on first use, then kept)."""
        path = self.paths[self.index_at(time)]
        with self._lock:
            profile = self._cache.get(path)
        if profile is None:
            profile = read_profile(path, self.axis)
            with self._lock:
                if len(self._cache) > 400:
                    self._cache.clear()
                self._cache[path] = profile
        return profile

    def value_range(self, line: int, samples: int = 12) -> tuple[float, float]:
        """Elevation range of a line over a spread of times (for a steady axis)."""
        picks = np.unique(np.linspace(0, len(self.paths) - 1, samples).round().astype(int))
        lows, highs = [], []
        for index in picks:
            profile = self.at(float(self.times[index]))
            item = profile.lines.get(line)
            if item is None:
                continue
            for values in (item.eta, item.theory):
                if values is not None and np.isfinite(values).any():
                    lows.append(np.nanmin(values))
                    highs.append(np.nanmax(values))
        if not lows:
            return 0.0, 1.0
        low, high = float(min(lows)), float(max(highs))
        return (low - 0.5, high + 0.5) if high - low < 1e-12 else (low, high)


def _iteration(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]) if numbers else -1, path.name)


def _latest_run(paths: list[Path], short: float = 0.1) -> list[Path]:
    """The files of the run to show, when REEF3D ran more than once in the folder.

    A run writes in iteration order, so modification times rise with the number; a
    later run overwrites from the start, and the files it did not reach are older
    than the one before them. As for surfaces, the latest run is shown unless it is
    a short start (fewer than ``short`` times the files of the longest).
    """
    if len(paths) < 2:
        return paths
    stamps = []
    for path in paths:
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            stamps.append(0.0)
    runs = [[0]]
    for i in range(1, len(paths)):
        if stamps[i] < stamps[i - 1] - 1.0:
            runs.append([])
        runs[-1].append(i)
    if len(runs) == 1:
        return paths
    latest = max(runs, key=lambda run: max(stamps[i] for i in run))
    longest = max(runs, key=len)
    chosen = latest if len(latest) >= short * len(longest) else longest
    return [paths[i] for i in chosen]


def find_profiles(case_dir: Path) -> dict[str, ProfileSet]:
    """The free surface line folders of a case (``REEF3D_*_WSFLINE``, ``..._Y`` and
    NHFLOW's time-averaged ones), with the time of every file."""
    found: dict[str, ProfileSet] = {}
    if not case_dir.is_dir():
        return found
    for folder in sorted(case_dir.iterdir()):
        match = _FOLDER.match(folder.name)
        if not match or not folder.is_dir():
            continue
        solver = match.group("solver")
        axis = "y" if match.group("y") else "x"
        averaged = bool(match.group("avg"))
        groups: dict[int, list[Path]] = {}
        for path in sorted(folder.glob("*.dat"), key=_iteration):
            # time-averaged: one file per line (REEF3D-NHFLOW-TimeAvgWsfLine-<line>-<it>)
            numbers = re.findall(r"\d+", path.stem) if averaged else []
            groups.setdefault(int(numbers[-2]) if len(numbers) >= 2 else 0, []).append(path)
        for line, paths in sorted(groups.items()):
            entries = []
            ys: dict[int, float] = {}
            for path in _latest_run(paths) if not averaged else paths:
                try:
                    time, line_ys = read_header(path)
                except (OSError, ValueError):
                    continue
                if np.isfinite(time):
                    entries.append((time, path))
                    ys = line_ys or ys
            if not entries:
                continue
            entries.sort(key=lambda e: e[0])
            key = f"{solver.lower()}_{'time_avg_' if averaged else ''}wsfline"
            key += ("_y" if axis == "y" else "") + (f"_{line}" if averaged else "")
            found[key] = ProfileSet(
                key,
                folder,
                solver,
                [p for _, p in entries],
                np.array([t for t, _ in entries]),
                ys,
                axis,
                line if averaged else 0,
            )
    return found
