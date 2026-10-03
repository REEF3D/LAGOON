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
}
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
        """A short name for the scene tree, e.g. "Body 0 forces" or "Wave gauges"."""
        stem = self.path.stem
        if self.group == "wave gauges":
            return "Wave gauges"
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
    """Gauge positions: rows ``i x y [z]`` after an ``x_coord`` header."""
    found: dict[str, tuple[float, ...]] = {}
    inside = False
    for line in preamble:
        tokens = line.split()
        if not tokens:
            if found:
                break
            continue
        if tokens[0].lower().startswith("x_coord"):
            inside = True
            continue
        if inside and all(_is_number(t) for t in tokens) and len(tokens) >= 3:
            found[f"P{int(float(tokens[0]))}"] = tuple(float(t) for t in tokens[1:])
    return found


def read_series(path: str | Path) -> TimeSeries:
    """Read a REEF3D ``.dat`` time-series table."""
    path = Path(path)
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    header_at = None
    for number, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        first = tokens[0].lower()
        if first == "time" or (first == "it" and len(tokens) > 1 and tokens[1].lower() == "time"):
            header_at = number
            break
    if header_at is None:
        raise ValueError(f"{path.name}: no header line starting with 'time'")
    names = _split(lines[header_at])
    body = "\n".join(lines[header_at + 1 :])
    values = np.array(body.split(), dtype=np.float64) if body.strip() else np.zeros(0)
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
    return TimeSeries(path, time, columns, units, _locations(lines[:header_at]))


# ----------------------------------------------------------------- discovery
def find_series(case_dir: Path) -> dict[str, SeriesFile]:
    """Time-series files in the ``REEF3D_*`` folders of a case."""
    found: dict[str, SeriesFile] = {}
    if not case_dir.is_dir():
        return found
    for folder in sorted(case_dir.iterdir()):
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
    """Sensible first columns to plot: heave for positions, Fx/Fz for forces, P1, ..."""
    stem = series.path.stem.lower()
    for word, names in _DEFAULT_COLUMNS.items():
        if word in stem:
            chosen = [n for n in names if n in series.columns]
            if chosen:
                return chosen
    return series.names[:1]


def parse_columns(text: str, series: TimeSeries) -> list[str]:
    """Column names from ``"P1, P4"`` (empty: the defaults); unknown names are skipped."""
    names = [n.strip() for n in text.replace(";", ",").split(",") if n.strip()]
    if not names:
        return default_columns(series)
    return [n for n in names if n in series.columns]
