# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""Particles and other point sets that change from output to output (SPEC.md, section 10).

Particle clouds (CPM, NHFLOW particles), DEM elements, ice floes, rod trees and
mooring lines have a different number of points at every output. A *particle set*
keeps the points of all outputs one after another in growing arrays, with the end
of every output's points in ``point_end``. Objects made of points (polygons of DEM
elements and ice floes, lines of rods and mooring lines) also have cells; as these
mostly stay the same from one output to the next, a cell set is stored only when
it changes and each output names the cell set it uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lagoon_format.zarr3 import Array, Group, codecs_for

#: rows per inner chunk and per shard of the growing arrays
ROWS = 65536
SHARD_ROWS = 16 * ROWS
#: outputs per chunk of the per-output arrays
OUTPUT_ROWS = 4096
#: kept in memory before a write: points (or cells, or connectivity entries)
BUFFER_ROWS = SHARD_ROWS
#: kinds of cells
CELLS = ("polys", "lines")
#: growing arrays up to this many rows are read whole once and kept (small sets:
#: mooring lines, DEM elements); larger ones are read output by output
SMALL_ROWS = 4 * ROWS


def _codecs(dtype: np.dtype):
    return codecs_for("gzip", 1, np.dtype(dtype).itemsize)


def _growing(path, dtype, components: int, name: str, units: str = "") -> Array:
    """An empty array that grows along its first dimension (sharded)."""
    shape = [0, components] if components > 1 else [0]
    shard = [SHARD_ROWS, components] if components > 1 else [SHARD_ROWS]
    inner = [ROWS, components] if components > 1 else [ROWS]
    dims = [name, "component"] if components > 1 else [name]
    return Array.create(
        path,
        shape,
        dtype,
        shard,
        inner=inner,
        codecs=_codecs(dtype),
        dimension_names=dims,
        attributes={"units": units} if units else {},
    )


def _per_output(path, dtype, name: str = "time", units: str = "") -> Array:
    return Array.create(
        path,
        [0],
        dtype,
        [OUTPUT_ROWS],
        codecs=_codecs(dtype),
        dimension_names=[name],
        attributes={"units": units} if units else {},
    )


def array_name(name: str) -> str:
    """The array of a field: its name with characters other than letters, digits,
    ``_``, ``.`` and ``-`` replaced by ``_`` (REEF3D writes "fluid velocity")."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "_"


def _fields_json(fields: dict[str, tuple[int, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for name, (c, d) in fields.items():
        array = base = array_name(name)
        n = 2
        while array in used:
            array, n = f"{base}_{n}", n + 1
        used.add(array)
        out[name] = {"components": int(c), "data_type": np.dtype(d).name, "array": array}
    return out


def _fields(info: dict[str, Any]) -> dict[str, tuple[int, np.dtype]]:
    return {
        name: (int(v.get("components", 1)), np.dtype(v.get("data_type", "float32")))
        for name, v in info.items()
    }


def _arrays(info: dict[str, Any]) -> dict[str, str]:
    return {name: str(v.get("array", array_name(name))) for name, v in info.items()}


def _append(array: Array, values: np.ndarray) -> None:
    """Values after the array's last row."""
    if not len(values):
        return
    start = array.shape[0]
    array.resize([start + len(values), *array.shape[1:]])
    array[start : start + len(values)] = values


# ======================================================================== reading
@dataclass
class ParticleSet:
    """One particle set of a store (a group ``particles/<key>``)."""

    group: Group
    info: dict[str, Any]
    _ends: dict[str, np.ndarray] = field(default_factory=dict)
    _count: int = -1
    _arrays: dict[str, Array] = field(default_factory=dict)
    _whole: dict[str, np.ndarray] = field(default_factory=dict)
    _cell_sets: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def open(cls, group: Group) -> ParticleSet:
        return cls(group, group.attrs.get("lagoon", {}))

    @property
    def key(self) -> str:
        return self.group.path.name

    @property
    def cells(self) -> str | None:
        """``"polys"``, ``"lines"`` or None (points only)."""
        return self.info.get("cells")

    @property
    def point_fields(self) -> dict[str, tuple[int, np.dtype]]:
        return _fields(self.info.get("point_fields", {}))

    @property
    def cell_fields(self) -> dict[str, tuple[int, np.dtype]]:
        return _fields(self.info.get("cell_fields", {}))

    def __len__(self) -> int:
        return self.group.array("time").shape[0]  # time is extended last

    def _per_output(self, name: str) -> np.ndarray:
        n = len(self)
        if n != self._count:  # grown since: read again what was kept
            self._ends = {}
            self._arrays = {}
            self._whole = {}
            self._count = n
        if name not in self._ends:
            self._ends[name] = self._array(name)[:n]
        return self._ends[name]

    def _array(self, name: str) -> Array:
        array = self._arrays.get(name)
        if array is None:
            array = self._arrays[name] = self.group.array(name)
        return array

    def _rows(self, name: str, low: int, high: int) -> np.ndarray:
        """Rows low:high of a growing array; a small array is read whole once."""
        whole = self._whole.get(name)
        if whole is not None and high <= len(whole):
            return whole[low:high].copy()  # the kept array stays as read
        array = self._array(name)
        if array.shape[0] <= SMALL_ROWS:
            whole = self._whole[name] = array[:]
            return whole[low:high].copy()  # the kept array stays as read
        return array[low:high]

    @property
    def times(self) -> np.ndarray:
        return self._per_output("time")

    @property
    def steps(self) -> np.ndarray:
        return self._per_output("step")

    def _span(self, t: int, ends: str) -> tuple[int, int]:
        end = self._per_output(ends)
        return (int(end[t - 1]) if t > 0 else 0), int(end[t])

    def count(self, t: int) -> int:
        """The number of points at output t."""
        low, high = self._span(t, "point_end")
        return high - low

    def positions(self, t: int) -> np.ndarray:
        low, high = self._span(t, "point_end")
        return self._rows("position", low, high)

    def point_field(self, name: str, t: int) -> np.ndarray:
        low, high = self._span(t, "point_end")
        array = _arrays(self.info.get("point_fields", {}))[name]
        return self._rows(f"point_data/{array}", low, high)

    def cell_set(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Connectivity and offsets (VTK: end of each cell) of output t's cells, with
        the output's point numbers."""
        if not self.cells:
            empty = np.zeros(0, np.int64)
            return empty, empty
        number = int(self._per_output("cell_set")[t])
        kept = self._cell_sets.get(number)  # cell sets never change once written
        if kept is not None:
            return kept
        cell_end = self._array("cell_sets/cell_end")[: number + 1]
        conn_end = self._array("cell_sets/connectivity_end")[: number + 1]
        c0 = int(cell_end[number - 1]) if number else 0
        k0 = int(conn_end[number - 1]) if number else 0
        offsets = self._rows("cell_sets/offsets", c0, int(cell_end[number])).astype(np.int64)
        connectivity = self._rows("cell_sets/connectivity", k0, int(conn_end[number]))
        kept = (connectivity.astype(np.int64), offsets)
        if len(self._cell_sets) > 64:
            self._cell_sets.clear()
        self._cell_sets[number] = kept
        return kept

    def cell_field(self, name: str, t: int) -> np.ndarray:
        low, high = self._span(t, "cell_data_end")
        array = _arrays(self.info.get("cell_fields", {}))[name]
        return self._rows(f"cell_data/{array}", low, high)


# ======================================================================== writing
class ParticleWriter:
    """Appends outputs to a particle set; rows are kept and written in large pieces."""

    def __init__(self, group: Group):
        self.group = group
        self.info = group.attrs.get("lagoon", {})
        self.stored = ParticleSet(group, self.info)
        self.point_fields = self.stored.point_fields
        self.cell_fields = self.stored.cell_fields
        self.cells = self.stored.cells
        self._point_arrays = _arrays(self.info.get("point_fields", {}))
        self._cell_arrays = _arrays(self.info.get("cell_fields", {}))
        n = len(self.stored)
        self._point_end = int(self.stored._per_output("point_end")[-1]) if n else 0
        self._cell_data_end = (
            int(self.stored._per_output("cell_data_end")[-1]) if n and self.cell_fields else 0
        )
        self._sets = self.group.array("cell_sets/cell_end").shape[0] if self.cells else 0
        self._last_cells: tuple[np.ndarray, np.ndarray] | None = None
        if self.cells and n:
            self._last_cells = self.stored.cell_set(n - 1)
            self._last_set = int(self.stored._per_output("cell_set")[-1])
        else:
            self._last_set = -1
        sets = self.group.group("cell_sets") if self.cells else None
        self._cell_end = int(sets.array("cell_end")[self._sets - 1]) if self._sets else 0
        self._conn_end = int(sets.array("connectivity_end")[self._sets - 1]) if self._sets else 0
        self._clear()

    def _clear(self) -> None:
        self._rows: dict[str, list] = {
            k: [] for k in ("time", "step", "point_end", "cell_set", "cell_data_end")
        }
        self._points: dict[str, list[np.ndarray]] = {"position": []}
        self._points.update({f"point_data/{a}": [] for a in self._point_arrays.values()})
        self._points.update({f"cell_data/{a}": [] for a in self._cell_arrays.values()})
        self._new_sets: dict[str, list] = {
            k: [] for k in ("cell_end", "connectivity_end", "offsets", "connectivity")
        }
        self._buffered = 0

    @classmethod
    def create(
        cls,
        parent: Group,
        key: str,
        point_fields: dict[str, tuple[int, Any]],
        cells: str | None = None,
        cell_fields: dict[str, tuple[int, Any]] | None = None,
        source: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> ParticleWriter:
        """A new particle set. ``point_fields`` / ``cell_fields``: name -> (components,
        data type); ``cells``: None (points only), ``"polys"`` or ``"lines"``."""
        if cells is not None and cells not in CELLS:
            raise ValueError(f"cells must be one of {CELLS}, not {cells!r}")
        cell_fields = dict(cell_fields or {}) if cells else {}
        info: dict[str, Any] = {
            "kind": "particles",
            "cells": cells,
            "point_fields": _fields_json(point_fields),
            "cell_fields": _fields_json(cell_fields),
        }
        point_arrays = _arrays(info["point_fields"])
        cell_arrays = _arrays(info["cell_fields"])
        if source:
            info["source"] = source
        info.update(attributes or {})
        group = Group.create(parent.path / key, {"lagoon": info})
        _per_output(group.path / "time", "float64", units="s")
        _per_output(group.path / "step", "int64")
        _per_output(group.path / "point_end", "int64")
        _growing(group.path / "position", "float32", 3, "point", "m")
        Group.create(group.path / "point_data")
        for name, (components, dtype) in point_fields.items():
            _growing(group.path / "point_data" / point_arrays[name], dtype, components, "point")
        if cells:
            _per_output(group.path / "cell_set", "int64")
            sets = Group.create(group.path / "cell_sets")
            _per_output(sets.path / "cell_end", "int64", "set")
            _per_output(sets.path / "connectivity_end", "int64", "set")
            _growing(sets.path / "offsets", "int32", 1, "cell")
            _growing(sets.path / "connectivity", "int32", 1, "corner")
            Group.create(group.path / "cell_data")
            if cell_fields:
                _per_output(group.path / "cell_data_end", "int64")
            for name, (components, dtype) in cell_fields.items():
                _growing(group.path / "cell_data" / cell_arrays[name], dtype, components, "cell")
        return cls(group)

    def append(
        self,
        time: float,
        step: int,
        positions: np.ndarray,
        point_data: dict[str, np.ndarray] | None = None,
        connectivity: np.ndarray | None = None,
        offsets: np.ndarray | None = None,
        cell_data: dict[str, np.ndarray] | None = None,
    ) -> None:
        """One output: its points, their arrays and (for objects) its cells, with the
        output's point numbers, and the cells' arrays."""
        positions = np.asarray(positions, np.float32).reshape(-1, 3)
        n = len(positions)
        point_data = point_data or {}
        for name, (components, dtype) in self.point_fields.items():
            if name not in point_data:
                raise ValueError(f"output {step}: no point array {name!r}")
            values = np.asarray(point_data[name], dtype)
            if len(values) != n:
                raise ValueError(f"output {step}: {name!r} has {len(values)} values for {n} points")
            self._points[f"point_data/{self._point_arrays[name]}"].append(
                values.reshape(n, components) if components > 1 else values.reshape(n)
            )
        self._points["position"].append(positions)
        self._point_end += n
        self._rows["point_end"].append(self._point_end)
        if self.cells:
            connectivity = np.asarray(
                connectivity if connectivity is not None else [], np.int32
            ).ravel()
            offsets = np.asarray(offsets if offsets is not None else [], np.int32).ravel()
            last = self._last_cells
            if not (
                last is not None
                and np.array_equal(last[0], connectivity)
                and np.array_equal(last[1], offsets)
            ):
                self._cell_end += len(offsets)
                self._conn_end += len(connectivity)
                self._new_sets["cell_end"].append(self._cell_end)
                self._new_sets["connectivity_end"].append(self._conn_end)
                self._new_sets["offsets"].append(offsets)
                self._new_sets["connectivity"].append(connectivity)
                self._last_cells = (connectivity, offsets)
                self._last_set = self._sets + len(self._new_sets["cell_end"]) - 1
                self._buffered += len(connectivity)
            self._rows["cell_set"].append(self._last_set)
            cell_data = cell_data or {}
            cells = len(offsets)
            for name, (components, dtype) in self.cell_fields.items():
                if name not in cell_data:
                    raise ValueError(f"output {step}: no cell array {name!r}")
                values = np.asarray(cell_data[name], dtype)
                if len(values) != cells:
                    raise ValueError(f"output {step}: {name!r} has {len(values)} values")
                self._points[f"cell_data/{self._cell_arrays[name]}"].append(
                    values.reshape(cells, components) if components > 1 else values.reshape(cells)
                )
            if self.cell_fields:
                self._cell_data_end += cells
                self._rows["cell_data_end"].append(self._cell_data_end)
        self._rows["time"].append(float(time))
        self._rows["step"].append(int(step))
        self._buffered += n
        if self._buffered >= BUFFER_ROWS or len(self._rows["time"]) >= OUTPUT_ROWS:
            self.flush()

    def flush(self) -> None:
        """Write what is kept: the rows first, the per-output arrays after, ``time``
        last (readers count the outputs by it)."""
        if not self._rows["time"]:
            return
        for name, parts in self._points.items():
            if parts:
                _append(self.group.array(name), np.concatenate(parts))
        if self.cells and self._new_sets["cell_end"]:
            sets = self.group.group("cell_sets")
            for name in ("offsets", "connectivity"):
                _append(sets.array(name), np.concatenate(self._new_sets[name]))
            for name in ("cell_end", "connectivity_end"):
                _append(sets.array(name), np.asarray(self._new_sets[name], np.int64))
            self._sets += len(self._new_sets["cell_end"])
        for name in ("step", "point_end", "cell_set", "cell_data_end", "time"):
            rows = self._rows[name]
            if rows:
                dtype = np.float64 if name == "time" else np.int64
                _append(self.group.array(name), np.asarray(rows, dtype))
        self._clear()

    def close(self) -> None:
        self.flush()
