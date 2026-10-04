# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""Time series and wave profiles in a LAGOON store (SPEC.md, sections 7 and 8).

A *table* is a time series file of a run (wave gauges, 6DOF forces and motions,
force files): one ``time`` array and one column per quantity. A *profile set* is
the free surface along lines at many times (REEF3D's WSFLINE output).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lagoon_format.zarr3 import Array, Group, codecs_for

#: rows per chunk of a table column
TABLE_ROWS = 65536
#: profile times per chunk
PROFILE_TIMES = 16


def _codecs(itemsize: int):
    return codecs_for("gzip", 1, itemsize)


# ======================================================================== tables
@dataclass
class Table:
    """One time series table of a store, read lazily by column."""

    group: Group
    name: str = ""
    kind: str = ""  # "wave gauges", "6DOF", "forces", ...
    columns: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    locations: dict[str, tuple[float, ...]] = field(default_factory=dict)
    source: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, group: Group) -> Table:
        info = group.attrs.get("lagoon", {})
        return cls(
            group,
            name=str(info.get("name", group.path.name)),
            kind=str(info.get("group", "")),
            columns=list(info.get("columns", [])),
            units=dict(info.get("units", {})),
            locations={k: tuple(v) for k, v in info.get("locations", {}).items()},
            source=str(info.get("source", "")),
            info=info,
        )

    @property
    def key(self) -> str:
        return self.group.path.name

    @property
    def time(self) -> np.ndarray:
        return self.group.array("time")[:]

    def __len__(self) -> int:
        return self.group.array("time").shape[0]

    def column(self, name: str) -> np.ndarray:
        """One column over all rows (reads only its chunks)."""
        return self.group.array("values")[:, self.columns.index(name)]

    def read(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """time and all columns."""
        values = self.group.array("values")[:]
        return self.time, {name: values[:, i] for i, name in enumerate(self.columns)}


def write_table(
    parent: Group,
    key: str,
    time: np.ndarray,
    columns: dict[str, np.ndarray],
    *,
    name: str = "",
    kind: str = "",
    units: dict[str, str] | None = None,
    locations: dict[str, tuple[float, ...]] | None = None,
    source: str = "",
    extra: dict[str, Any] | None = None,
) -> Table:
    """Write (or replace) a table: ``time`` (n,) and ``values`` (n, columns), float64."""
    time = np.asarray(time, np.float64)
    names = list(columns)
    values = (
        np.stack([np.asarray(columns[c], np.float64) for c in names], axis=1)
        if names
        else (np.zeros((time.size, 0)))
    )
    info: dict[str, Any] = {
        "kind": "table",
        "name": name or key,
        "group": kind,
        "columns": names,
        "units": dict(units or {}),
        "locations": {k: [float(v) for v in xyz] for k, xyz in (locations or {}).items()},
        "rows": int(time.size),
    }
    if source:
        info["source"] = source
    info.update(extra or {})
    group = Group.create(parent.path / key, {"lagoon": info})
    rows = max(min(TABLE_ROWS, time.size), 1)
    Array.create(
        group.path / "time",
        [time.size],
        "float64",
        [rows],
        codecs=_codecs(8),
        dimension_names=["time"],
        attributes={"units": "s"},
    )[:] = time
    Array.create(
        group.path / "values",
        list(values.shape),
        "float64",
        [rows, 1],
        codecs=_codecs(8),
        dimension_names=["time", "column"],
        attributes={"columns": names},
    )[:] = values
    return Table.open(group)


# ======================================================================== profiles
@dataclass
class ProfileLineData:
    number: int
    y: float
    x: np.ndarray
    group: Group

    def eta(self, index: int) -> np.ndarray:
        return self.group.array("eta")[index]

    def theory(self, index: int) -> np.ndarray | None:
        if "theory" not in self.group:
            return None
        return self.group.array("theory")[index]


@dataclass
class ProfileSet:
    """Wave profiles along lines over time (one group of a store)."""

    group: Group
    times: np.ndarray
    lines: dict[int, ProfileLineData]
    source: str = ""

    @classmethod
    def open(cls, group: Group) -> ProfileSet:
        info = group.attrs.get("lagoon", {})
        lines = {}
        for number, spec in info.get("lines", {}).items():
            line_group = group.group(f"line_{number}")
            x = line_group.array("x")[:]
            lines[int(number)] = ProfileLineData(
                int(number), float(spec.get("y", 0.0)), x, line_group
            )
        n = group.array("time").shape[0]
        return cls(group, group.array("time")[:n], lines, str(info.get("source", "")))

    @property
    def key(self) -> str:
        return self.group.path.name


class ProfileWriter:
    """Appends wave profiles (all lines at one time) to a profile set."""

    def __init__(self, group: Group):
        self.group = group

    @classmethod
    def create(
        cls,
        parent: Group,
        key: str,
        lines: dict[int, tuple[float, np.ndarray, bool]],
        source: str = "",
    ) -> ProfileWriter:
        """``lines``: number -> (y, x along the line, has a wave theory column)."""
        info = {
            "kind": "profiles",
            "lines": {
                str(n): {"y": float(y), "points": int(len(x))} for n, (y, x, _) in lines.items()
            },
        }
        if source:
            info["source"] = source
        group = Group.create(parent.path / key, {"lagoon": info})
        Array.create(
            group.path / "time",
            [0],
            "float64",
            [4096],
            dimension_names=["time"],
            attributes={"units": "s"},
        )
        for number, (_y, x, theory) in lines.items():
            line = Group.create(group.path / f"line_{number}")
            x = np.asarray(x, np.float64)
            Array.create(line.path / "x", [x.size], "float64", dimension_names=["point"])[:] = x
            names = ["eta", "theory"] if theory else ["eta"]
            for name in names:
                Array.create(
                    line.path / name,
                    [0, x.size],
                    "float32",
                    [PROFILE_TIMES, max(x.size, 1)],
                    codecs=_codecs(4),
                    dimension_names=["time", "point"],
                    attributes={"units": "m"},
                )
        return cls(group)

    def append(
        self, times: list[float], profiles: list[dict[int, tuple[np.ndarray, np.ndarray | None]]]
    ) -> None:
        """Profiles at ``times``: per time, line number -> (eta, theory or None)."""
        if not times:
            return
        time_array = self.group.array("time")
        start = time_array.shape[0]
        stop = start + len(times)
        info = self.group.attrs["lagoon"]
        for number in info["lines"]:
            line = self.group.group(f"line_{number}")
            for name in ("eta", "theory"):
                if name not in line:
                    continue
                array = line.array(name)
                width = array.shape[1]
                block = np.full((len(times), width), np.nan, np.float32)
                for row, profile in enumerate(profiles):
                    entry = profile.get(int(number))
                    if entry is None:
                        continue
                    values = entry[0] if name == "eta" else entry[1]
                    if values is not None:
                        n = min(width, len(values))
                        block[row, :n] = values[:n]
                array.resize([stop, width])
                array[start:stop] = block
        time_array.resize([stop])
        time_array[start:stop] = np.asarray(times, np.float64)  # last: readers count by it
