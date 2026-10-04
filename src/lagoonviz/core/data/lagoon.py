# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""LAGOON stores as datasets: the same interface as REEF3D's VTK output.

Each block of a store output plays the part of one MPI piece: it is handed to the
VTK code paths (surfaces, volume boundaries, slices, iso-surfaces, streamlines)
as a piece with the same point order and hexahedral cells REEF3D writes, built
from the structured arrays. A probe reads one small chunk per output.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from lagoon_format.series import ProfileSet as StoredProfiles
from lagoon_format.series import Table
from lagoon_format.store import Output, Store, is_store, sigma_heights

from lagoonviz.core.data.mesh import SurfaceMesh
from lagoonviz.core.data.profiles import Profile, ProfileLine, ProfileSet
from lagoonviz.core.data.reef3d import Dataset, Frame, _joined
from lagoonviz.core.data.series import SeriesFile, TimeSeries
from lagoonviz.core.data.vtkxml import Piece

VTK_HEXAHEDRON = 12


def _hexahedra(nz: int, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells of a block whose points are ordered level, y, x (x fastest)."""
    k, j, i = np.meshgrid(np.arange(nz - 1), np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    base = ((k * ny + j) * nx + i).ravel()
    up = ny * nx
    corners = np.stack(
        [
            base,
            base + 1,
            base + nx + 1,
            base + nx,
            base + up,
            base + up + 1,
            base + up + nx + 1,
            base + up + nx,
        ],
        axis=1,
    )
    count = len(base)
    return (
        corners.ravel().astype(np.int64),
        np.arange(1, count + 1, dtype=np.int64) * 8,
        np.full(count, VTK_HEXAHEDRON, np.uint8),
    )


def _quads(ny: int, nx: int, missing: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Quadrilaterals of a surface block (points ordered y, x), leaving out those
    with a corner that has no point."""
    j, i = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    base = (j * nx + i).ravel()
    corners = np.stack([base, base + 1, base + nx + 1, base + nx], axis=1)
    if missing is not None and missing.any():
        corners = corners[~missing.ravel()[corners].any(axis=1)]
    return corners.ravel().astype(np.int64), np.arange(1, len(corners) + 1, dtype=np.int64) * 4


def _face(points: np.ndarray, data: dict[str, np.ndarray], flip: bool) -> SurfaceMesh:
    """A structured face (rows, cols, 3) as triangles; ``flip`` turns its normals."""
    rows, cols = points.shape[:2]
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    base = (r * cols + c).ravel()
    quads = np.stack([base, base + 1, base + cols + 1, base + cols], axis=1)
    flat = points.reshape(-1, 3).astype(np.float32)
    missing = ~np.isfinite(flat).all(axis=1)
    if missing.any():
        quads = quads[~missing[quads].any(axis=1)]
    if flip:
        quads = quads[:, ::-1]
    triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).astype(np.uint32)
    values = {
        name: v.reshape(rows * cols, -1) if v.ndim == 3 else v.reshape(-1)
        for name, v in data.items()
    }
    return SurfaceMesh(flat, triangles, values)


class StoreDataset(Dataset):
    """One output of a LAGOON store (volume, free surface, bed)."""

    source = "lagoon"

    def __init__(self, store: Store, output: Output, key: str):
        self.store = store
        self.output = output
        self._cells_cache: dict[tuple, tuple] = {}
        self._cells_lock = threading.Lock()
        frames = self._frames()
        super().__init__(key, store.solver, output.role, frames, output.group.path)
        self.times = output.times.copy()
        self.run_id = store.run_id
        self.run_started = str(store.run.get("started", "")) if store.run else ""

    def _frames(self) -> list[Frame]:
        blocks = self.output.group.path / "blocks"
        return [
            Frame(
                int(step),
                self.store.path,
                [blocks / f"b{b.number:04d}" / f"t{t}" for b in self.output.blocks],
            )
            for t, step in enumerate(self.output.steps)
        ]

    def refresh(self) -> bool:
        """Pick up outputs written since (a store still being written); True if any."""
        before = len(self.output.times)
        if self.output.refresh() == before:
            return False
        self.frames = self._frames()
        self.times = self.output.times.copy()
        self._step_array = None
        return True

    # ------------------------------------------------------------- description
    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        kind = "PolyData" if self.output.grid == "surface" else "UnstructuredGrid"
        return kind, dict(self.output.variables)

    @property
    def label(self) -> str:
        return super().label + " (LAGOON)"

    def written_span(self) -> tuple[float, float]:
        if self._span is None:
            try:
                first = (self.output.group.path / "zarr.json").stat().st_mtime
                last = (self.output.group.path / "time" / "zarr.json").stat().st_mtime
            except OSError:
                first = last = 0.0
            self._span = (min(first, last), max(first, last))
        return self._span

    # ------------------------------------------------------------- pieces
    def _piece(self, path: Path, point_arrays: list[str] | None = None, with_cells: bool = True):
        """A block at an output, as the VTK piece REEF3D would have written."""
        number, t = int(path.parent.name[1:]), int(path.name[1:])
        output = self.output
        block = output.blocks[number]
        (i0, j0), (i1, j1) = block.start, block.stop
        xs = output.x[i0:i1].astype(np.float32)
        ys = output.y[j0:j1].astype(np.float32)
        z = output.heights(t, block)
        volume = output.grid != "surface"
        if volume:
            nz = z.shape[0]
            x = np.broadcast_to(xs[None, None, :], z.shape)
            y = np.broadcast_to(ys[None, :, None], z.shape)
        else:
            nz = 1
            x = np.broadcast_to(xs[None, :], z.shape)
            y = np.broadcast_to(ys[:, None], z.shape)
        points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
        names = list(output.variables) if point_arrays is None else point_arrays
        data = {}
        for name in names:
            if name not in output.variables:
                raise KeyError(f"{output.name}: no variable {name!r}")
            values = output.read_block(name, t, block)
            components = output.variables[name]
            data[name] = values.reshape(-1, components) if components > 1 else values.ravel()
        if not with_cells:
            empty = np.zeros(0, np.int64)
            types = np.zeros(0, np.uint8) if volume else None
            return Piece(self.kind, points, empty, empty, types, data)
        ny, nx = len(ys), len(xs)
        key = (number, nz, ny, nx)
        with self._cells_lock:
            cells = self._cells_cache.get(key)
            if cells is None:
                if volume:
                    cells = _hexahedra(nz, ny, nx)
                else:
                    connectivity, offsets = _quads(ny, nx, np.isnan(z))
                    cells = (connectivity, offsets, None)
                self._cells_cache[key] = cells
        connectivity, offsets, types = cells
        return Piece(self.kind, points, connectivity, offsets, types, data)

    # ------------------------------------------------------------- volume faces
    def read(self, index: int, variables: list[str] | None = None) -> SurfaceMesh:
        """The outer faces of a σ-grid volume straight from the store: the bottom and
        top levels and the four sides, so only those levels and the tiles along the
        sides are read. Surfaces are read as their pieces."""
        output = self.output
        names = list(output.variables) if variables is None else self._arrays(variables) or []
        if output.grid == "surface":
            return self._surface(index, names)
        if output.grid not in ("sigma", "cartesian"):
            return super().read(index, variables)
        nz, ny, nx = output.shape
        faces = []
        x, y = output.x.astype(np.float32), output.y.astype(np.float32)
        for level, flip in ((0, True), (nz - 1, False)):
            z = output.heights(index, level=level)
            data = {n: output.read(n, index, level=level) for n in names}
            gx, gy = np.meshgrid(x, y)
            faces.append(_face(np.stack([gx, gy, z], axis=-1), data, flip))
        for axis, line, flip in (
            (0, 0, True),
            (0, nx - 1, False),
            (1, 0, False),
            (1, ny - 1, True),
        ):
            z, data = self._side(index, axis, line, names)
            along = y if axis == 0 else x
            a = np.broadcast_to(along[None, :], z.shape)
            p = np.full(z.shape, (x if axis == 0 else y)[line], np.float32)
            points = np.stack([p, a, z] if axis == 0 else [a, p, z], axis=-1)
            faces.append(_face(points, data, flip))
        return _joined(faces)

    def _surface(self, index: int, names: list[str]) -> SurfaceMesh:
        """A free surface or bed as one grid: the blocks joined, two triangles per
        grid cell whose corners all have a point (made once)."""
        output = self.output
        if self.role == "free_surface" and "eta" in output.variables and "eta" not in names:
            names = [*names, "eta"]  # needed for the vertical scale
        z = output.heights(index)
        gx, gy = np.meshgrid(output.x.astype(np.float32), output.y.astype(np.float32))
        data = {name: output.read(name, index) for name in names}
        points = np.stack([gx, gy, z], axis=-1)
        with self._cells_lock:
            mesh = self._cells_cache.get("surface")
            if mesh is None:
                mesh = self._cells_cache["surface"] = _face(points, {}, False).triangles
        values = {
            n: v.reshape(z.size, -1) if v.ndim == 3 else v.reshape(-1) for n, v in data.items()
        }
        return SurfaceMesh(points.reshape(-1, 3).astype(np.float32), mesh, values).with_normals()

    def _side(self, t: int, axis: int, line: int, names: list[str]):
        """Heights (nz, n) and fields (nz, n[, 3]) on grid line ``line`` across ``axis``
        (x = x[line] for axis 0), from the blocks that hold it."""
        output = self.output
        nz, ny, nx = output.shape
        n = ny if axis == 0 else nx
        z = np.full((nz, n), np.nan, np.float32)
        data = {
            name: np.full(
                (nz, n, output.variables[name]) if output.variables[name] > 1 else (nz, n),
                np.nan,
                np.float32,
            )
            for name in names
        }
        # the highest numbered first: where blocks overlap, the lowest numbered wins
        for block in sorted(output.blocks, key=lambda b: -b.number):
            local = line - block.start[axis]
            if not 0 <= local < block.size[axis]:
                continue
            other = 1 - axis
            lo, hi = block.start[other], block.stop[other]
            rows = slice(block.k0, block.k0 + block.nz)

            def at(name, levels, block=block, local=local, width=hi - lo):
                return self._on_line(t, block, axis, local, name, levels, width)

            if output.grid == "cartesian":
                heights = np.broadcast_to(output.z[rows, None], (block.nz, hi - lo))
            else:
                bed = at("z_bed", False).astype(np.float64)
                top = at("z_surface", False)
                heights = bed[None] + output.sigma[:, None] * (top - bed)[None]
                heights += at("z_offset", True)
            z[rows, lo:hi] = heights
            for name in names:
                data[name][rows, lo:hi] = at(name, True)
        return z, data

    def _on_line(self, t, block, axis, local, name, levels, width) -> np.ndarray:
        """A block's array on its grid line ``local`` across ``axis``."""
        output = self.output
        if name == "z_offset" and not output.has(name, block.number):
            return np.zeros((block.nz, width), np.float32)
        head = (t, slice(None)) if levels else (t,)
        key = head + ((slice(None), local) if axis == 0 else (local, slice(None)))
        return output.array(block, name)[key]

    # ------------------------------------------------------------- slices
    def read_slice(
        self, index: int, axis: int, position: float, variables: list[str] | None = None
    ) -> SurfaceMesh:
        """A vertical slice of a σ-grid read straight from the store: only the two grid
        lines around the plane (the tiles holding them) are read, and the section is
        their interpolation (exactly what cutting REEF3D's hexahedra gives)."""
        output = self.output
        if output.grid not in ("sigma", "cartesian") or axis not in (0, 1):
            return super().read_slice(index, axis, position, variables)
        lines = output.x if axis == 0 else output.y
        if len(lines) < 2 or not lines[0] <= position <= lines[-1]:
            return super().read_slice(index, axis, position, variables)
        i = int(np.clip(np.searchsorted(lines, position) - 1, 0, len(lines) - 2))
        w = float((position - lines[i]) / (lines[i + 1] - lines[i]))
        names = self._arrays(variables) or []
        parts = []
        for block in output.blocks:
            low = i - block.start[axis]
            if low < 0 or low + 1 >= block.size[axis]:
                continue  # the plane does not cross this block's cells
            parts.append(self._slice_block(index, block, axis, low, w, position, names))
        if not parts:
            return super().read_slice(index, axis, position, variables)
        return _joined(parts)

    def _slice_block(self, t, block, axis, low, w, position, names) -> SurfaceMesh:
        output = self.output
        pair = slice(low, low + 2)

        def two(name: str, levels: bool) -> np.ndarray:
            """The block's array at the two grid lines, then mixed: (…, n_along)."""
            array = output.array(block, name)
            head = (t, slice(None)) if levels else (t,)
            key = head + ((slice(None), pair) if axis == 0 else (pair, slice(None)))
            values = array[key]
            where = -1 if axis == 0 else -2
            if values.ndim > (3 if levels else 2):  # vector components last
                where -= 1
            a = np.take(values, 0, axis=where).astype(np.float64)
            b = np.take(values, 1, axis=where).astype(np.float64)
            return (1 - w) * a + w * b

        (i0, j0), (i1, j1) = block.start, block.stop
        if output.grid == "cartesian":
            n_along = (j1 - j0) if axis == 0 else (i1 - i0)
            levels = output.z[block.k0 : block.k0 + block.nz].astype(np.float32)
            z = np.repeat(levels[:, None], n_along, axis=1)
        else:
            bed, top = two("z_bed", False), two("z_surface", False)
            offset = two("z_offset", True) if output.has("z_offset", block.number) else None
            z = sigma_heights(bed, top, output.sigma, offset).astype(np.float32)  # (nz, n)
        along = (output.y[j0:j1] if axis == 0 else output.x[i0:i1]).astype(np.float32)
        nz, n = z.shape
        a = np.broadcast_to(along[None, :], z.shape)
        p = np.full(z.shape, position, np.float32)
        x, y = (p, a) if axis == 0 else (a, p)
        points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        data = {}
        for name in names:
            values = two(name, True).astype(np.float32)
            components = output.variables[name]
            data[name] = values.reshape(-1, components) if components > 1 else values.ravel()
        k, j = np.meshgrid(np.arange(nz - 1), np.arange(n - 1), indexing="ij")
        base = (k * n + j).ravel()
        quads = np.stack([base, base + 1, base + n + 1, base + n], axis=1)
        missing = ~np.isfinite(points).all(axis=1)
        if missing.any():
            quads = quads[~missing[quads].any(axis=1)]
        triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        return SurfaceMesh(points, triangles.astype(np.uint32), data)

    # ------------------------------------------------------------- probes
    def point_series(self, variable: str, x: float, y: float) -> tuple[tuple, np.ndarray]:
        """A variable over all outputs at the grid point nearest to (x, y) (the top
        level of a volume): one small chunk per output."""
        name, _, component = variable.partition(".")
        if name not in self.components:
            raise KeyError(f"no variable {name!r} in {self.label}")
        output = self.output
        i, j = output.nearest(x, y)
        values = np.asarray(output.series(name, i, j), np.float64)
        if output.grid in ("sigma", "cartesian"):
            z = float(output.column_heights(i, j)[0, -1])
        else:
            z = float(output.series("z", i, j)[0])
        where = (float(output.x[i]), float(output.y[j]), z)
        if values.ndim == 1:
            return where, values
        if component:
            return where, values[:, "xyz".index(component)]
        return where, np.linalg.norm(values, axis=1)


def store_datasets(path: str | Path) -> list[StoreDataset]:
    """The datasets of a LAGOON store, keyed like REEF3D's (``fnpf_volume``)."""
    store = Store(path)
    return [
        StoreDataset(
            store, output, output.info.get("dataset") or f"{store.solver.lower()}_{output.role}"
        )
        for output in store.outputs.values()
        if len(output.times)
    ]


def find_stores(case_dir: Path) -> list[Path]:
    """LAGOON stores in a case folder (``REEF3D_<SOLVER>.lagoon``)."""
    if not case_dir.is_dir():
        return []
    return sorted(p for p in case_dir.glob("*.lagoon") if p.is_dir() and is_store(p))


# ----------------------------------------------------------------- time series
class StoreSeriesFile(SeriesFile):
    """A time series table of a store, in place of the ``.dat`` file it came from."""

    def __init__(self, key: str, path: Path, solver: str, group: str, table: Table):
        super().__init__(key, path, solver, group)
        self.table = table
        self._series: TimeSeries | None = None

    @property
    def name(self) -> str:
        return self.table.name or super().name

    @property
    def label(self) -> str:
        return f"{self.solver} {self.name} (LAGOON)"

    def read(self) -> TimeSeries:
        with self._lock:
            if self._series is None:
                time, columns = self.table.read()
                self._series = TimeSeries(
                    self.path, time, columns, dict(self.table.units), dict(self.table.locations)
                )
            return self._series


class StoreProfileSet(ProfileSet):
    """Wave profiles of a store, in place of the WSFLINE files."""

    def __init__(self, key: str, folder: Path, solver: str, stored: StoredProfiles):
        times = np.asarray(stored.times, np.float64)
        super().__init__(
            key,
            folder,
            solver,
            [folder / f"t{i}" for i in range(len(times))],  # one name per time
            times,
            {n: line.y for n, line in stored.lines.items()},
        )
        self.stored = stored

    @property
    def label(self) -> str:
        return f"{self.solver} wave profiles ({len(self.paths)} times, LAGOON)"

    def written_span(self) -> tuple[float, float]:
        try:
            stamp = (self.stored.group.path / "time" / "zarr.json").stat().st_mtime
        except OSError:
            stamp = 0.0
        return stamp, stamp

    def at(self, time: float) -> Profile:
        index = self.index_at(time)
        lines = {
            n: ProfileLine(
                n, line.y, line.x, line.eta(index).astype(np.float64), _maybe(line.theory(index))
            )
            for n, line in self.stored.lines.items()
        }
        return Profile(self.paths[index], float(self.times[index]), lines)


def _maybe(values):
    return None if values is None else np.asarray(values, np.float64)


def _unchanged(path: Path, info: dict) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    return info.get("source_bytes") == stat.st_size and info.get("source_mtime") == stat.st_mtime


def stored_series(stores: list[Path], series: dict, profiles: dict) -> tuple[dict, dict]:
    """Time series and wave profiles from stores, in place of the text files they
    came from where those did not change since (or are gone)."""
    series, profiles = dict(series), dict(profiles)
    for path in stores:
        try:
            store = Store(path)
        except (OSError, ValueError):
            continue
        for key, table in store.tables.items():
            source = store.path.parent / table.source if table.source else table.group.path
            current = series.get(key)
            if current is not None and not _unchanged(current.path, table.info):
                continue  # the file is newer: a run still writing it
            where = source if source.exists() else table.group.path
            series[key] = StoreSeriesFile(key, where, store.solver, table.kind, table)
        for key, stored in store.profiles.items():
            current = profiles.get(key)
            if (
                current is not None
                and len(current.times)
                and (not len(stored.times) or current.times[-1] > stored.times[-1] + 1e-9)
            ):
                continue  # more profiles in the files
            folder = store.path.parent / (stored.source or key)
            profiles[key] = StoreProfileSet(key, folder, store.solver, stored)
    return series, profiles
