# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The bed of REEF3D's CPM sediment model (CFD, S 10 1 with the bedload layer Q 58).

With the parcel output (Q 180, Q 182) each MPI rank writes its columns to
``REEF3D_CFD_CPM_Particle/REEF3D-CFD-CPM-Bed-<output>-<rank>.dat``::

    # time 12.5
    # x y z_bed tau_x tau_y
    0.005 0.005 0.1021 0.31 0.02
    ...

one line per column (cell centre), the bed level and the bed shear stress [Pa] of
the bedload layer. LAGOON joins the ranks' columns into one surface: the bed level
``z_bed``, the bed shear stress ``tau`` (a vector, tau_x tau_y 0) and the bed's
change since the first output, ``bed_change``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from lagoonviz.core.data.mesh import SurfaceMesh, grid_normals, grid_points, grid_triangles
from lagoonviz.core.data.reef3d import Dataset, Frame

#: ``REEF3D-CFD-CPM-Bed-00000012-000003.dat``
BED_FILE = re.compile(r"^REEF3D-(?P<solver>[A-Z0-9]+)-CPM-Bed-(?P<step>\d+)-(?P<rank>\d+)\.dat$")
#: the role of the CPM bed (shown as a bed, beside the CFD's own)
ROLE = "cpm_bed"
VARIABLES = {"z_bed": 1, "bed_change": 1, "tau": 3}


def read_time(path: str | Path) -> float | None:
    """The time in a bed file's first lines (``# time 12.5``)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _ in range(3):
                found = re.match(r"#\s*time\s+(\S+)", handle.readline())
                if found:
                    return float(found.group(1))
    except (OSError, ValueError):
        pass
    return None


def read_bed_file(path: str | Path) -> tuple[float | None, np.ndarray]:
    """One rank's bed file: (its time, the columns (n, 5): x y z_bed tau_x tau_y)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    time = None
    body = []
    for line in text.splitlines():
        if line.startswith("#"):
            found = re.match(r"#\s*time\s+(\S+)", line)
            if found:
                try:
                    time = float(found.group(1))
                except ValueError:
                    pass
            continue
        body.append(line)
    values = np.array(" ".join(body).split(), dtype=np.float64)
    if values.size % 5:
        values = values[: values.size - values.size % 5]  # a file still being written
    return time, values.reshape(-1, 5)


class CpmBedDataset(Dataset):
    """The CPM bed of a run over its outputs: the ranks' columns as one surface."""

    def __init__(self, key: str, solver: str, frames: list[Frame], folder: Path):
        super().__init__(key, solver, ROLE, frames, folder)
        self._grid: tuple[np.ndarray, np.ndarray] | None = None
        self._first: np.ndarray | None = None

    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        return "PolyData", dict(VARIABLES)

    @property
    def surface_like(self) -> bool:
        return True

    def default_variable(self) -> str:
        return "bed_change"

    def _columns(self, index: int) -> np.ndarray:
        return np.concatenate([read_bed_file(p)[1] for p in self.frames[index].pieces])

    def _levels(self, columns: np.ndarray) -> list[np.ndarray]:
        """The (ny, nx) grids of the columns: z_bed, tau_x, tau_y (NaN where none)."""
        if self._grid is None:
            self._grid = (
                np.unique(np.round(columns[:, 0], 9)),
                np.unique(np.round(columns[:, 1], 9)),
            )
        xs, ys = self._grid
        i = np.searchsorted(xs, np.round(columns[:, 0], 9))
        j = np.searchsorted(ys, np.round(columns[:, 1], 9))
        ok = (i < xs.size) & (j < ys.size)
        ok[ok] &= np.isclose(xs[i[ok]], columns[ok, 0]) & np.isclose(ys[j[ok]], columns[ok, 1])
        grids = []
        for c in (2, 3, 4):
            g = np.full((ys.size, xs.size), np.nan)
            g[j[ok], i[ok]] = columns[ok, c]
            grids.append(g)
        return grids

    def read(self, index: int, variables: list[str] | None = None) -> SurfaceMesh:
        z, tau_x, tau_y = self._levels(self._columns(index))
        if self._first is None:
            self._first = z if index == 0 else self._levels(self._columns(0))[0]
        xs, ys = self._grid
        if ys.size == 1 or xs.size == 1:  # a 2D run: its columns as a strip a cell wide
            z, tau_x, tau_y = (
                np.repeat(g, 2, axis=0 if ys.size == 1 else 1) for g in (z, tau_x, tau_y)
            )
            first = np.repeat(self._first, 2, axis=0 if ys.size == 1 else 1)
            width = _spacing(xs if ys.size == 1 else ys)
            xs, ys = _strip(xs, width), _strip(ys, width)
        else:
            first = self._first
        missing = ~np.isfinite(z)
        filled = np.where(missing, np.nanmean(z) if (~missing).any() else 0.0, z)
        points = grid_points(xs, ys, filled.astype(np.float32))
        tau = np.stack([tau_x, tau_y, np.zeros_like(tau_x)], axis=-1).reshape(-1, 3)
        data = {
            "z_bed": filled.astype(np.float32).ravel(),
            "bed_change": np.nan_to_num(z - first).astype(np.float32).ravel(),
            "tau": np.nan_to_num(tau).astype(np.float32),
        }
        triangles = grid_triangles(*z.shape, missing if missing.any() else None)
        normals = grid_normals(filled, xs, ys).reshape(-1, 3)
        return SurfaceMesh(points, triangles, data, normals=normals)

    def read_values(self, index: int, variable: str) -> np.ndarray | None:
        return self.read(index, [variable]).scalar(variable)


def _spacing(centres: np.ndarray) -> float:
    """The cells' size along an axis (1 for a single cell)."""
    return float(np.median(np.diff(centres))) if centres.size > 1 else 1.0


def _strip(centres: np.ndarray, width: float) -> np.ndarray:
    """A single row of cells (a 2D run) as a strip ``width`` wide (the cells' size
    along the other axis: the file has no width across); more cells as they are."""
    if centres.size > 1:
        return centres
    return np.array([centres[0] - 0.5 * width, centres[0] + 0.5 * width])


def bed_frames(folder: Path) -> dict[str, list[Frame]]:
    """The CPM bed files of a folder, grouped by solver: per output a frame of its
    ranks' files (rank 1 first)."""
    found: dict[str, dict[int, dict[int, Path]]] = {}
    for path in folder.iterdir():
        match = BED_FILE.match(path.name)
        if match:
            steps = found.setdefault(match.group("solver"), {})
            steps.setdefault(int(match.group("step")), {})[int(match.group("rank"))] = path
    out = {}
    for solver, steps in found.items():
        out[solver] = [
            Frame(step, ranks[min(ranks)], [ranks[r] for r in sorted(ranks)])
            for step, ranks in sorted(steps.items())
        ]
    return out


def bed_datasets(folder: Path) -> list[Dataset]:
    """The CPM beds in a ``REEF3D_CFD_CPM_Particle`` folder, with their times."""
    datasets: list[Dataset] = []
    if not folder.is_dir():
        return datasets
    for solver, frames in bed_frames(folder).items():
        dataset = CpmBedDataset(f"{solver.lower()}_{ROLE}", solver, frames, folder)
        times = [read_time(frame.pieces[0]) for frame in frames]
        if times and all(t is not None for t in times):
            order = np.argsort(times, kind="stable")
            dataset.frames = [frames[k] for k in order]
            dataset.times = np.asarray(times, np.float64)[order]
        datasets.append(dataset)
    return datasets
