# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""Adaptive mesh refinement output (SPEC.md, section 11).

REEF3D's AMR solvers (FNPF, NHFLOW, SFLOW with ``A 270``) write their free surface
as a set of 2D rectilinear grids per output: level 0 (one grid per MPI rank) and
the refined patches, which come and go as the solution is regridded. CFD writes
the refined blocks of its volume as 3D grids (``dims`` 3; its level 0 is the
volume). An *AMR set* ``amr/<key>`` keeps the grids of all outputs one after
another: a table of the grids (refinement level, MPI rank, size, where its
coordinates and cells end), the node coordinates in ``x``, ``y`` (and ``z``), and
the cell fields.

    writer = AmrWriter.create(parent, "nhflow_amr", {"eta": (1, "float32")})
    writer.append(0.5, 1, [AmrGrid(0, x, y, {"eta": eta}), AmrGrid(1, xp, yp, {...})])
    writer.close()
    grids = AmrSet.open(parent.group("nhflow_amr")).grids(0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lagoon_format.particles import (
    BUFFER_ROWS,
    OUTPUT_ROWS,
    SMALL_ROWS,
    _append,
    _arrays,
    _fields,
    _fields_json,
    _growing,
    _per_output,
)
from lagoon_format.zarr3 import Array, Group

#: the per-grid table: name -> (data type, components)
GRID_TABLE = {
    "level": ("int32", 1),
    "rank": ("int32", 1),
    "size": ("int32", 2),
    "x_end": ("int64", 1),
    "y_end": ("int64", 1),
    "cell_end": ("int64", 1),
}
#: the per-grid table of a set of 3D grids (``dims`` 3)
GRID_TABLE_3D = {**GRID_TABLE, "size": ("int32", 3), "z_end": ("int64", 1)}


def grid_table(dims: int) -> dict[str, tuple[str, int]]:
    """The per-grid table of a set of 2D or 3D grids."""
    return GRID_TABLE_3D if int(dims) == 3 else GRID_TABLE


@dataclass
class AmrGrid:
    """One grid of an output: ``x`` (nx + 1) and ``y`` (ny + 1) node coordinates, cell
    fields of shape (ny, nx[, components]) with x fastest, its refinement level and the
    MPI rank that wrote it (-1: not known). A 3D grid (a block of CFD's volume) has
    ``z`` (nz + 1) node coordinates too, its fields of shape (nz, ny, nx[, c])."""

    level: int
    x: np.ndarray
    y: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    rank: int = -1
    z: np.ndarray | None = None

    @property
    def nz(self) -> int:
        """Cells in z; 0 for a 2D grid."""
        return 0 if self.z is None else int(np.asarray(self.z).size) - 1

    @property
    def cells(self) -> int:
        return self.nx * self.ny * max(self.nz, 1)

    @property
    def nx(self) -> int:
        return int(np.asarray(self.x).size) - 1

    @property
    def ny(self) -> int:
        return int(np.asarray(self.y).size) - 1

    @property
    def box(self) -> tuple[float, float, float, float]:
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])


def covered(grid, grids, boxes: Boxes | None = None) -> np.ndarray:
    """(ny, nx) True for the cells of ``grid`` whose centre lies in a grid of a higher
    level among ``grids`` (anything with ``level``, ``x`` and ``y``): the cells that
    are not leaf cells. ``boxes``: :class:`Boxes` of ``grids``, made once when
    asking for all of them (thousands of patches: only the finer grids that
    overlap are looked at, found for all at once)."""
    x = np.asarray(grid.x, np.float64)
    y = np.asarray(grid.y, np.float64)
    mask = np.zeros((y.size - 1, x.size - 1), bool)
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    if boxes is None:
        boxes = Boxes(grids)
    for other in boxes.finer_over(grid.level, x[0], x[-1], y[0], y[-1]):
        x0, x1, y0, y1 = boxes.box[other]
        i0, i1 = np.searchsorted(xc, x0, "left"), np.searchsorted(xc, x1, "right")
        j0, j1 = np.searchsorted(yc, y0, "left"), np.searchsorted(yc, y1, "right")
        if i0 < i1 and j0 < j1:
            mask[j0:j1, i0:i1] = True
    return mask


class Boxes:
    """The extents and levels of a set of AMR grids, to find the finer grids over
    a grid at once (:func:`covered`)."""

    def __init__(self, grids) -> None:
        self.level = np.array([int(g.level) for g in grids], np.int64)
        self.box = np.array(
            [(float(g.x[0]), float(g.x[-1]), float(g.y[0]), float(g.y[-1])) for g in grids],
            np.float64,
        ).reshape(-1, 4)

    def finer_over(self, level: int, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        """Indices of the grids of a higher level whose box overlaps (x0..x1, y0..y1)."""
        b = self.box
        hit = (self.level > level) & (b[:, 1] >= x0) & (b[:, 0] <= x1)
        hit &= (b[:, 3] >= y0) & (b[:, 2] <= y1)
        return np.flatnonzero(hit)


# ======================================================================== reading
@dataclass
class AmrSet:
    """One AMR set of a store (a group ``amr/<key>``)."""

    group: Group
    info: dict[str, Any]
    _count: int = -1
    _per: dict[str, np.ndarray] = field(default_factory=dict)
    _arrays: dict[str, Array] = field(default_factory=dict)
    _whole: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def open(cls, group: Group) -> AmrSet:
        return cls(group, group.attrs.get("lagoon", {}))

    @property
    def key(self) -> str:
        return self.group.path.name

    @property
    def fields(self) -> dict[str, tuple[int, np.dtype]]:
        """The cell fields: name -> (components, data type)."""
        return _fields(self.info.get("fields", {}))

    @property
    def dims(self) -> int:
        """2: grids of a surface; 3: blocks of a volume."""
        return int(self.info.get("dims", 2))

    def __len__(self) -> int:
        return self.group.array("time").shape[0]  # time is extended last

    def _fresh(self) -> None:
        n = len(self)
        if n != self._count:  # grown since: read again what was kept
            self._per, self._arrays, self._whole = {}, {}, {}
            self._count = n

    def _array(self, name: str) -> Array:
        array = self._arrays.get(name)
        if array is None:
            array = self._arrays[name] = self.group.array(name)
        return array

    def _per_output(self, name: str) -> np.ndarray:
        self._fresh()
        if name not in self._per:
            self._per[name] = self._array(name)[: self._count]
        return self._per[name]

    def _rows(self, name: str, low: int, high: int) -> np.ndarray:
        """Rows low:high of a growing array; a small array is read whole once."""
        self._fresh()
        whole = self._whole.get(name)
        if whole is not None and high <= len(whole):
            return whole[low:high].copy()
        array = self._array(name)
        if array.shape[0] <= SMALL_ROWS:
            whole = self._whole[name] = array[:]
            return whole[low:high].copy()
        return array[low:high]

    @property
    def times(self) -> np.ndarray:
        return self._per_output("time")

    @property
    def steps(self) -> np.ndarray:
        return self._per_output("step")

    def _grid_span(self, t: int) -> tuple[int, int]:
        end = self._per_output("grid_end")
        return (int(end[t - 1]) if t > 0 else 0), int(end[t])

    def table(self, t: int) -> dict[str, np.ndarray]:
        """The grids of output t: ``level``, ``rank``, ``size`` (nx, ny cells), and
        the start and end of each grid's ``x``, ``y`` and cells."""
        g0, g1 = self._grid_span(t)
        rows = {name: self._rows(f"grids/{name}", g0, g1) for name in grid_table(self.dims)}
        axes = ("x", "y", "z") if self.dims == 3 else ("x", "y")
        for name in (*axes, "cell"):
            before = self._rows(f"grids/{name}_end", g0 - 1, g0) if g0 > 0 else np.zeros(1)
            ends = rows[f"{name}_end"].astype(np.int64)
            rows[f"{name}_start"] = np.concatenate([before.astype(np.int64), ends[:-1]])
        return rows

    def count(self, t: int) -> int:
        """The number of grids at output t."""
        g0, g1 = self._grid_span(t)
        return g1 - g0

    def grids(self, t: int, fields: list[str] | None = None) -> list[AmrGrid]:
        """The grids of output t with the cell fields asked for (None: all)."""
        rows = self.table(t)
        if not len(rows["level"]):
            return []
        names = list(self.fields) if fields is None else [f for f in fields if f in self.fields]
        arrays = _arrays(self.info.get("fields", {}))
        x = self._rows("x", int(rows["x_start"][0]), int(rows["x_end"][-1]))
        y = self._rows("y", int(rows["y_start"][0]), int(rows["y_end"][-1]))
        c0, c1 = int(rows["cell_start"][0]), int(rows["cell_end"][-1])
        values = {name: self._rows(f"cell_data/{arrays[name]}", c0, c1) for name in names}
        x0, y0 = int(rows["x_start"][0]), int(rows["y_start"][0])
        volume = self.dims == 3
        if volume:
            z = self._rows("z", int(rows["z_start"][0]), int(rows["z_end"][-1]))
            z0 = int(rows["z_start"][0])
        grids = []
        for g in range(len(rows["level"])):
            size = [int(v) for v in rows["size"][g]]
            nx, ny = size[0], size[1]
            xs = slice(int(rows["x_start"][g]) - x0, int(rows["x_end"][g]) - x0)
            ys = slice(int(rows["y_start"][g]) - y0, int(rows["y_end"][g]) - y0)
            cs = slice(int(rows["cell_start"][g]) - c0, int(rows["cell_end"][g]) - c0)
            shape = (size[2], ny, nx) if volume else (ny, nx)
            grid_fields = {}
            for name in names:
                part = values[name][cs]
                grid_fields[name] = part.reshape((*shape, *part.shape[1:]))
            zs = slice(int(rows["z_start"][g]) - z0, int(rows["z_end"][g]) - z0) if volume else None
            grids.append(
                AmrGrid(
                    level=int(rows["level"][g]),
                    x=np.asarray(x[xs], np.float64),
                    y=np.asarray(y[ys], np.float64),
                    fields=grid_fields,
                    rank=int(rows["rank"][g]),
                    z=np.asarray(z[zs], np.float64) if volume else None,
                )
            )
        return grids


# ======================================================================== writing
class AmrWriter:
    """Appends outputs to an AMR set; rows are kept and written in large pieces."""

    def __init__(self, group: Group):
        self.group = group
        self.info = group.attrs.get("lagoon", {})
        self.stored = AmrSet(group, self.info)
        self.fields = self.stored.fields
        self.dims = self.stored.dims
        self._table = grid_table(self.dims)
        self._field_arrays = _arrays(self.info.get("fields", {}))
        n = len(self.stored)
        self._grid_end = int(self.stored._per_output("grid_end")[-1]) if n else 0
        ends = {}
        for name in ("x_end", "y_end", "z_end", "cell_end"):
            if name not in self._table:
                ends[name] = 0
                continue
            array = self.group.array(f"grids/{name}")
            ends[name] = int(array[self._grid_end - 1 : self._grid_end][0]) if self._grid_end else 0
        self._x_end, self._y_end, self._cell_end = ends["x_end"], ends["y_end"], ends["cell_end"]
        self._z_end = ends["z_end"]
        self._clear()

    def _clear(self) -> None:
        self._per: dict[str, list] = {k: [] for k in ("time", "step", "grid_end")}
        self._grid_rows: dict[str, list] = {k: [] for k in self._table}
        self._rows: dict[str, list[np.ndarray]] = {"x": [], "y": []}
        if self.dims == 3:
            self._rows["z"] = []
        self._rows.update({f"cell_data/{a}": [] for a in self._field_arrays.values()})
        self._buffered = 0

    @classmethod
    def create(
        cls,
        parent: Group,
        key: str,
        fields: dict[str, tuple[int, Any]],
        source: str = "",
        attributes: dict[str, Any] | None = None,
        dims: int = 2,
    ) -> AmrWriter:
        """A new AMR set. ``fields``: name -> (components, data type) of the cell fields;
        ``dims`` 3: the blocks of a volume (3D grids)."""
        info: dict[str, Any] = {"kind": "amr", "fields": _fields_json(fields)}
        if source:
            info["source"] = source
        if int(dims) == 3:
            info["dims"] = 3
        info.update(attributes or {})
        arrays = _arrays(info["fields"])
        group = Group.create(parent.path / key, {"lagoon": info})
        _per_output(group.path / "time", "float64", units="s")
        _per_output(group.path / "step", "int64")
        _per_output(group.path / "grid_end", "int64")
        table = Group.create(group.path / "grids")
        for name, (dtype, components) in grid_table(dims).items():
            _growing(table.path / name, dtype, components, "grid")
        for axis in ("x", "y", "z") if int(dims) == 3 else ("x", "y"):
            _growing(group.path / axis, "float64", 1, "node", "m")
        Group.create(group.path / "cell_data")
        for name, (components, dtype) in fields.items():
            _growing(group.path / "cell_data" / arrays[name], dtype, components, "cell")
        return cls(group)

    def append(self, time: float, step: int, grids: list[AmrGrid]) -> None:
        """One output: all its grids, level 0 first as REEF3D lists them."""
        volume = self.dims == 3
        for grid in grids:
            nx, ny, nz = grid.nx, grid.ny, grid.nz
            if nx < 1 or ny < 1 or (volume and nz < 1):
                raise ValueError(f"output {step}: a grid of {nx} x {ny} x {nz} cells")
            if not volume and nz:
                raise ValueError(f"output {step}: a 3D grid in a set of 2D grids")
            n = grid.cells
            for name, (components, dtype) in self.fields.items():
                if name not in grid.fields:
                    raise ValueError(f"output {step}: a grid without {name!r}")
                values = np.asarray(grid.fields[name], dtype)
                if values.size != n * components:
                    size = f"{nx} x {ny}" + (f" x {nz}" if volume else "")
                    raise ValueError(
                        f"output {step}: {name!r} has {values.size} values for {size} cells"
                    )
                shape = (n, components) if components > 1 else (n,)
                self._rows[f"cell_data/{self._field_arrays[name]}"].append(values.reshape(shape))
            self._rows["x"].append(np.asarray(grid.x, np.float64).ravel())
            self._rows["y"].append(np.asarray(grid.y, np.float64).ravel())
            self._x_end += nx + 1
            self._y_end += ny + 1
            self._cell_end += n
            self._grid_rows["level"].append(int(grid.level))
            self._grid_rows["rank"].append(int(grid.rank))
            self._grid_rows["size"].append((nx, ny, nz) if volume else (nx, ny))
            self._grid_rows["x_end"].append(self._x_end)
            self._grid_rows["y_end"].append(self._y_end)
            if volume:
                self._rows["z"].append(np.asarray(grid.z, np.float64).ravel())
                self._z_end += nz + 1
                self._grid_rows["z_end"].append(self._z_end)
            self._grid_rows["cell_end"].append(self._cell_end)
            self._buffered += n
        self._grid_end += len(grids)
        self._per["grid_end"].append(self._grid_end)
        self._per["time"].append(float(time))
        self._per["step"].append(int(step))
        if self._buffered >= BUFFER_ROWS or len(self._per["time"]) >= OUTPUT_ROWS:
            self.flush()

    def flush(self) -> None:
        """Write what is kept: the rows first, the grid table, the per-output arrays,
        ``time`` last (readers count the outputs by it)."""
        if not self._per["time"]:
            return
        for name, parts in self._rows.items():
            if parts:
                _append(self.group.array(name), np.concatenate(parts))
        for name, (dtype, components) in self._table.items():
            rows = self._grid_rows[name]
            if rows:
                values = np.asarray(rows, dtype)
                _append(
                    self.group.array(f"grids/{name}"),
                    values.reshape(len(rows), components) if components > 1 else values,
                )
        for name in ("step", "grid_end", "time"):
            dtype = np.float64 if name == "time" else np.int64
            _append(self.group.array(name), np.asarray(self._per[name], dtype))
        self._clear()

    def close(self) -> None:
        self.flush()
