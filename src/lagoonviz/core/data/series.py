# SPDX-License-Identifier: GPL-3.0-or-later
"""REEF3D time series: wave gauges, 6DOF motions and forces, other force files.

REEF3D writes these as text tables (``.dat``): an optional preamble (gauge
positions, coefficients), then a header line of column names that starts with
``time`` (or ``it``, ``time``), then one row per time step. Columns are separated
by tabs and spaces; names may carry units (``Ue [m/s]``).

    REEF3D_NHFLOW_WSF/REEF3D-NHFLOW-WSF-HG.dat   wave gauges P1..Pn, x/y in the preamble
    REEF3D_NHFLOW_6DOF/REEF3D_6DOF_forces_0.dat  Fx Fy Fz Mx My Mz ... of body 0
    REEF3D_NHFLOW_6DOF/REEF3D_6DOF_position_0.dat, _velocity_0.dat, _mooring_force_k.dat
    REEF3D_FNPF_Force_ALE/REEF3D_FNPF_ALE_Force-1.dat

:func:`find_series` lists such files in a case folder; :func:`read_series` reads
one into a :class:`TimeSeries`.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_FOLDER = re.compile(r"^REEF3D_(?P<solver>[A-Z0-9]+)_(?P<what>.+)$")
#: folders that hold time series, and how their files are labelled
SERIES_FOLDERS = {
    "WSF": "wave gauges",
    "6DOF": "6DOF",
    "Force_ALE": "forces",
    "Force": "forces",
    "VRANS_Force": "VRANS forces",
    "Probes": "probes",
    "ProbePoint": "probes",
    "PressureProbe": "probes",
    "Runup": "run-up",
    "RUNUP": "run-up",
    "Sediment": "sediment",
}
#: REEF3D's folder of the wave theory's time series (P 58), of no solver
WAVE_THEORY_FOLDER = "REEF3D_Log-Wave"
#: what REEF3D calls its print-outs (the User Guide's names, P 50 - P 140), by file
#: name (lower case); {n}: the number in the name
REEF3D_NAMES = (
    (r"wsf-hg-theory", "Wave gauges (theory)"),
    (r"wsf-hg", "Wave gauges"),
    (r"vel-probe-theory-(\d+)", "Velocity probe {n} (theory)"),
    (r"vel-probe-(\d+)", "Velocity probe {n}"),
    (r"probe[_-]press(?:ure)?-(\d+)", "Pressure probe {n}"),
    (r"sflow-probe-point-(\d+)", "Depth-averaged point probe {n}"),
    (r"probe-point-(\d+)", "Point probe {n}"),
    (r"ale_force-(\d+)", "Morison force {n}"),
    (r"vrans", "VRANS forces"),
    (r"_force-(\d+)$", "Force box {n}"),
    (r"runup-max-x", "Maximum run-up gauges"),
    (r"runup-x", "Run-up gauges"),
    (r"runup-(\d+)", "Cylinder run-up {n}"),
    (r"sediment-point", "Bed level gauges"),
    (r"sediment-max", "Maximum bed change"),
    (r"sediment-bedshear-max", "Maximum bed shear stress"),
    (r"sediment-bedshear", "Bed shear stress gauges"),
    (r"wave-timeseries-(\d+)", "Theoretical wave time series {n}"),
)
_UNIT = re.compile(r"^(?P<name>.*?)\s*\[(?P<unit>[^\]]*)\]\s*$")


@dataclass
class TimeSeries:
    """A table of values over time."""

    path: Path
    time: np.ndarray  # (n,) seconds
    columns: dict[str, np.ndarray]  # name -> (n,) values
    units: dict[str, str] = field(default_factory=dict)
    #: gauge positions from the preamble: name -> (x, y) or (x, y, z)
    locations: dict[str, tuple[float, ...]] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.columns)

    def __len__(self) -> int:
        return int(self.time.size)

    def value_at(self, name: str, time: float) -> float:
        """Linearly interpolated value at a time (NaN outside the record)."""
        t = self.time
        if t.size == 0 or time < t[0] or time > t[-1]:
            return float("nan")
        return float(np.interp(time, t, self.columns[name]))


@dataclass
class SeriesFile:
    """A time-series file found in a case, read on first use."""

    key: str
    path: Path
    solver: str
    group: str  # "wave gauges", "6DOF", "forces", ...

    def __post_init__(self) -> None:
        self._data: TimeSeries | None = None
        self._mtime = 0.0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """A short name for the scene tree, as REEF3D calls the print-out ("Wave
        gauges", "Velocity probe 1", "Run-up gauges"), or e.g. "Body 0 forces"."""
        stem = self.path.stem
        lower = stem.lower()
        for pattern, name in REEF3D_NAMES:
            found = re.search(pattern, lower)
            if found:
                return name.format(n=found.group(1) if found.groups() else "")
        number = re.search(r"(\d+)$", stem)
        index = number.group(1) if number else ""
        lower = stem.lower()
        if "mooring" in lower:
            return f"Mooring line {index}".strip()
        for word, what in (("forces", "forces"), ("position", "motion"), ("velocity", "velocity")):
            if self.group == "6DOF" and word in lower:
                return f"Body {index} {what}".replace("  ", " ")
        if self.group in ("forces", "VRANS forces"):
            return f"{self.group.capitalize()} {index}".strip()
        cleaned = re.sub(r"^REEF3D[_-]", "", stem).replace("_", " ").replace("-", " ")
        return cleaned

    @property
    def label(self) -> str:
        return f"{self.solver} {self.name} ({self.path.name})"

    def read(self) -> TimeSeries:
        """The table, read again if the file changed (a run still writing)."""
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
            except OSError:
                mtime = 0.0
            if self._data is None or mtime != self._mtime:
                self._data = read_series(self.path)
                self._mtime = mtime
            return self._data


# ----------------------------------------------------------------- parsing
def _split(line: str) -> list[str]:
    """Column names: tab separated, or runs of 2+ spaces, or single spaces."""
    if "\t" in line:
        parts = [p.strip() for p in line.split("\t")]
    else:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) == 1:
            parts = line.split()
    return [p for p in parts if p]


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _locations(preamble: list[str]) -> dict[str, tuple[float, ...]]:
    """Gauge positions: rows ``i x y [z]`` after an ``x_coord`` header (``x y [T]``
    without the number in the wave theory's time series: numbered in order)."""
    found: dict[str, tuple[float, ...]] = {}
    inside = 0  # the header's number of names
    for line in preamble:
        tokens = line.split()
        if not tokens:
            if found:
                break
            continue
        if tokens[0].lower().startswith("x_coord"):
            inside = len(tokens)
            continue
        if inside and all(_is_number(t) for t in tokens) and len(tokens) >= 2:
            if len(tokens) > inside:  # numbered
                found[f"P{int(float(tokens[0]))}"] = tuple(float(t) for t in tokens[1:])
            else:
                found[f"P{len(found) + 1}"] = tuple(float(t) for t in tokens)
    return found


def _time_header(tokens: list[str]) -> bool:
    """A header line: ``time ...``, ``t ...`` (the probes) or ``it time ...``."""
    if not tokens:
        return False
    first = tokens[0].lower()
    if first in ("time", "time[s]", "t", "t[s]"):
        return len(tokens) == 1 or tokens[1] not in ("=", ":")
    return first == "it" and len(tokens) > 1 and tokens[1].lower() == "time"


def _numbers(line: str) -> int:
    """How many numbers a line holds (0: empty or not all numbers)."""
    tokens = line.split()
    return len(tokens) if tokens and all(_is_number(t) for t in tokens) else 0


def read_series(path: str | Path) -> TimeSeries:
    """Read a REEF3D ``.dat`` time-series table.

    The table follows a header of column names that starts with ``time`` (``t`` in
    the probes', ``it time`` in some force files). Without one (run-up gauges name
    only their columns, the CFD pressure probes nothing), the table is the block of
    numbers at the end of the file, its first column the time and the others named
    by the line above it (else ``value 1``, ...)."""
    path = Path(path)
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    header_at = None
    for number, line in enumerate(lines):
        if _time_header(line.split()):
            header_at = number
            break
    if header_at is not None:
        names = _split(lines[header_at])
        names[0] = "time" if names[0].lower() in ("t", "t[s]", "time[s]") else names[0]
        body_from = header_at + 1
        count = next((_numbers(line) for line in lines[body_from:] if line.strip()), 0)
        if count and count != len(names):
            # a header that does not fit the rows (NHFLOW's pressure probes are headed
            # "t U V W" and hold the time and the pressure): named by their number
            names = _generic(path, count)
    else:
        names, body_from = _untitled(path, lines)
    body = "\n".join(lines[body_from:])
    try:
        values = np.array(body.split(), dtype=np.float64) if body.strip() else np.zeros(0)
    except ValueError:  # words among the numbers: the rows of numbers only
        rows = [line.split() for line in lines[body_from:] if _numbers(line) == len(names)]
        values = np.array(rows, dtype=np.float64).ravel() if rows else np.zeros(0)
    n_columns = len(names)
    rows = values.size // n_columns
    if rows * n_columns != values.size:
        # a row being written by a running simulation: drop the incomplete tail
        values = values[: rows * n_columns]
    table = values.reshape(rows, n_columns)
    # bare names and units
    clean, units = [], {}
    for name in names:
        match = _UNIT.match(name)
        bare = match.group("name").strip() if match else name
        clean.append(bare)
        if match:
            units[bare] = match.group("unit")
    time_column = [c.lower() for c in clean].index("time")
    time = table[:, time_column]
    columns = {
        name: np.where(np.abs(table[:, i]) >= 1e19, np.nan, table[:, i])  # REEF3D's -1e20: no value
        for i, name in enumerate(clean)
        if i != time_column and name.lower() != "it"
    }
    return TimeSeries(path, time, columns, units, _locations(lines[: max(body_from - 1, 0)]))


def _untitled(path: Path, lines: list[str]) -> tuple[list[str], int]:
    """Column names and the first row of a table without a time header: the block of
    rows of numbers at the end of the file (two at least, the time rising)."""
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    if not end:
        raise ValueError(f"{path.name}: no header line starting with 'time'")
    count = _numbers(lines[end - 1])
    if end >= 2 and _numbers(lines[end - 2]) and _numbers(lines[end - 2]) != count:
        count = _numbers(lines[end - 2])  # the last row half written
    start = end - 1  # (the last row, perhaps half written)
    while start > 0 and _numbers(lines[start - 1]) == count:
        start -= 1
    rows = [lines[i].split() for i in range(start, end) if count and _numbers(lines[i]) == count]
    times = [float(row[0]) for row in rows]
    if count < 2 or len(rows) < 2 or any(b < a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError(f"{path.name}: no header line starting with 'time'")
    above = start - 1
    while above >= 0 and not lines[above].strip():
        above -= 1
    named = _split(lines[above]) if above >= 0 and not _numbers(lines[above]) else []
    if len(named) == count:
        names = ["time", *named[1:]]
    elif len(named) == count - 1:  # the columns named, the time not
        names = ["time", *named]
    else:
        names = _generic(path, count)
    return names, start


def _generic(path: Path, count: int) -> list[str]:
    """Names of ``count`` columns without a header that fits: the time and the
    pressure of a pressure probe, else the time and ``value 1``, ..."""
    if count == 2 and "press" in path.stem.lower():
        return ["time", "p"]
    return ["time", *(f"value {i}" for i in range(1, count))]


# ----------------------------------------------------------------- discovery
def find_series(case_dir: Path) -> dict[str, SeriesFile]:
    """Time-series files in the ``REEF3D_*`` folders of a case."""
    found: dict[str, SeriesFile] = {}
    if not case_dir.is_dir():
        return found
    for folder in sorted(case_dir.iterdir()):
        if folder.name == WAVE_THEORY_FOLDER and folder.is_dir():
            solver, group = "REEF3D", "wave theory"
        else:
            match = _FOLDER.match(folder.name)
            if not match or not folder.is_dir():
                continue
            what = match.group("what")
            group = SERIES_FOLDERS.get(what)
            if group is None and what.endswith("Force"):
                group = "forces"
            if group is None:
                continue
            solver = match.group("solver")
        for path in sorted(folder.glob("*.dat")):
            try:
                if path.stat().st_size < 8:
                    continue
            except OSError:
                continue
            stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
            stem = stem.removeprefix("reef3d_")
            key = f"{solver.lower()}_{stem}" if not stem.startswith(solver.lower()) else stem
            found[key] = SeriesFile(key, path, solver, group)
    return found


def decimate(time: np.ndarray, values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Min and max per bin, so a line of millions of points keeps its envelope.

    Each bin becomes two points at its mean time (its minimum, then its maximum),
    which draws as one vertical stroke per pixel column.
    """
    n = time.size
    if n <= 2 * bins:
        return time, values
    edges = np.linspace(0, n, bins + 1).astype(np.int64)[:-1]
    with np.errstate(invalid="ignore"):
        low = np.fmin.reduceat(values, edges)
        high = np.fmax.reduceat(values, edges)
    counts = np.diff(np.append(edges, n))
    centre = np.add.reduceat(time, edges) / counts
    out_t = np.repeat(centre, 2)
    out_v = np.empty(2 * edges.size)
    out_v[0::2], out_v[1::2] = low, high
    return out_t, out_v


#: columns plotted when an item names none
_DEFAULT_COLUMNS = {
    "forces": ("Fx", "Fz"),
    "position": ("ZG",),
    "velocity": ("We",),
    "mooring": ("T",),
}


def default_columns(series: TimeSeries) -> list[str]:
    """Sensible first columns to plot: heave for positions, Fx/Fz for forces, P1-P4."""
    stem = series.path.stem.lower()
    for word, names in _DEFAULT_COLUMNS.items():
        if word in stem:
            chosen = [n for n in names if n in series.columns]
            if chosen:
                return chosen
    if series.locations:  # wave gauges: the first few, so the 3D view shows where they are
        return [n for n in series.names if n in series.locations][:4] or series.names[:1]
    return series.names[:1]


def parse_columns(text: str, series: TimeSeries) -> list[str]:
    """Column names from ``"P1, P4"`` (empty: the defaults); unknown names are skipped."""
    names = [n.strip() for n in text.replace(";", ",").split(",") if n.strip()]
    if not names:
        return default_columns(series)
    return [n for n in names if n in series.columns]
