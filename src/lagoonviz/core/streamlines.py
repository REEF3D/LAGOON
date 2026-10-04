# SPDX-License-Identifier: GPL-3.0-or-later
"""Streamlines through REEF3D volume output (NumPy only).

REEF3D's grids are made of vertical columns: a regular (or stretched) grid in x and
y, and in every column the same number of points over the height (fixed levels for
CFD, σ-levels between bed and free surface for NHFLOW and FNPF). Knowing that, a
point is found by binary search in x and y and by counting levels in its four
neighbouring columns, which is fast for many points at once without any search
tree. The volume of every MPI rank is joined first (shared points once).

Lines are traced with the midpoint rule (second order), forwards and backwards from
seeds spread over a plane, and stop where they leave the water or the domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class NotColumnGrid(ValueError):
    """The points are not columns with the same number of levels."""


@dataclass
class ColumnPlan:
    """Where every point of a volume goes in its columns, worked out once.

    REEF3D's grids do not change their columns between outputs (σ-grids only move
    their levels up and down), so the grouping, the order of the levels and the
    choice among the copies of a column on an MPI boundary are kept; a frame then
    costs only picking its values in this order. Dry columns (FNPF puts all levels
    of a dry column onto the bed) are found per frame.
    """

    xs: np.ndarray  # (nx,)
    ys: np.ndarray  # (ny,)
    index: np.ndarray  # (nx, ny, nz) into the joined points of the pieces; -1: no column

    @classmethod
    def build(cls, pieces: list[np.ndarray]) -> ColumnPlan:
        points = np.concatenate([np.asarray(p, dtype=np.float64) for p in pieces])
        piece = np.repeat(np.arange(len(pieces)), [len(p) for p in pieces])
        # millimetres: float32 coordinates of different MPI ranks differ in the last digits
        xs, ix = np.unique(np.round(points[:, 0], 3), return_inverse=True)
        ys, iy = np.unique(np.round(points[:, 1], 3), return_inverse=True)
        nx, ny = len(xs), len(ys)
        if nx < 2 or ny < 2:
            raise NotColumnGrid("the volume is not made of columns")
        column = ix.ravel() * ny + iy.ravel()
        # a column on an MPI boundary is written by two ranks: keep the first one's
        first = np.full(nx * ny, len(pieces))
        np.minimum.at(first, column, piece)
        keep = piece == first[column]
        counts = np.bincount(column[keep], minlength=nx * ny)
        levels = np.bincount(counts[counts > 1])
        nz = int(np.argmax(levels)) if levels.size else 0  # the usual number of levels
        if nz < 2 or levels[nz] < 0.5 * np.count_nonzero(counts > 1):
            raise NotColumnGrid("the columns of the volume differ in their number of levels")
        good = counts == nz
        chosen = np.flatnonzero(keep & good[column])
        order = chosen[np.lexsort((points[chosen, 2], column[chosen]))]  # rising in each column
        index = np.full((nx * ny, nz), -1, np.int64)
        index[good] = order.reshape(-1, nz)
        return cls(xs.astype(np.float64), ys.astype(np.float64), index.reshape(nx, ny, nz))

    def accepts(self, sizes: tuple[int, ...]) -> bool:
        return self.index.max() < sum(sizes)

    def grid(self, points: np.ndarray, fields: dict[str, np.ndarray]) -> ColumnGrid:
        """The grid of one frame: the joined points and fields of its pieces."""
        points = np.asarray(points)
        present = self.index >= 0
        where = np.where(present, self.index, 0)
        z = np.where(present, points[where, 2], np.nan).astype(np.float64)
        dry = (z[..., -1] - z[..., 0]) < 1e-6  # all levels on the bed: no water
        z[dry] = np.nan
        shaped = {}
        for name, values in fields.items():
            values = np.asarray(values)[where].astype(np.float64)
            values[~present] = 0.0
            shaped[name] = values
        return ColumnGrid(self.xs, self.ys, z, shaped)


@dataclass
class ColumnGrid:
    xs: np.ndarray  # (nx,)
    ys: np.ndarray  # (ny,)
    z: np.ndarray  # (nx, ny, nz) heights of the levels, rising
    fields: dict[str, np.ndarray]  # name -> (nx, ny, nz) or (nx, ny, nz, c)

    @classmethod
    def from_points(cls, points: np.ndarray, fields: dict[str, np.ndarray]) -> ColumnGrid:
        """A grid from the points of one piece (see :class:`ColumnPlan` for several)."""
        return ColumnPlan.build([points]).grid(points, fields)

    @property
    def cell(self) -> float:
        """A typical horizontal cell size."""
        return float(min(np.median(np.diff(self.xs)), np.median(np.diff(self.ys))))

    def sample(self, name: str, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Values of a field at points (trilinear in the columns' levels) and whether
        each point lies inside the water."""
        values = self.fields[name]
        p = np.asarray(points, dtype=np.float64)
        n = len(p)
        i = np.clip(np.searchsorted(self.xs, p[:, 0]) - 1, 0, len(self.xs) - 2)
        j = np.clip(np.searchsorted(self.ys, p[:, 1]) - 1, 0, len(self.ys) - 2)
        fx = (p[:, 0] - self.xs[i]) / (self.xs[i + 1] - self.xs[i])
        fy = (p[:, 1] - self.ys[j]) / (self.ys[j + 1] - self.ys[j])
        inside = (fx >= 0) & (fx <= 1) & (fy >= 0) & (fy <= 1)
        shape = (n, *values.shape[3:])
        result = np.zeros(shape)
        for di, dj, weight in (
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            levels = self.z[i + di, j + dj]  # (n, nz)
            k = np.clip((levels <= p[:, 2:3]).sum(axis=1) - 1, 0, levels.shape[1] - 2)
            rows = np.arange(n)
            low, high = levels[rows, k], levels[rows, k + 1]
            fz = (p[:, 2] - low) / np.where(high > low, high - low, 1.0)
            inside &= (fz >= -1e-9) & (fz <= 1 + 1e-9) & np.isfinite(low) & np.isfinite(high)
            column = values[i + di, j + dj]
            a, b = column[rows, k], column[rows, k + 1]
            fz_ = np.clip(fz, 0.0, 1.0).reshape(-1, *([1] * (a.ndim - 1)))
            w = weight.reshape(-1, *([1] * (a.ndim - 1)))
            result += w * (a + fz_ * (b - a))
        return result, inside

    def seeds(self, axis: int, position: float, count: int) -> np.ndarray:
        """About ``count`` seeds spread evenly over the plane ``x[axis] = position``
        (in the water: over the height of each column, between its first and last level)."""
        count = max(int(count), 1)
        if axis == 2:  # a horizontal plane
            nx = max(int(round(np.sqrt(count * np.ptp(self.xs) / max(np.ptp(self.ys), 1e-12)))), 1)
            ny = max(count // nx, 1)
            gx, gy = np.meshgrid(
                np.linspace(self.xs[0], self.xs[-1], nx + 2)[1:-1],
                np.linspace(self.ys[0], self.ys[-1], ny + 2)[1:-1],
                indexing="ij",
            )
            return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, position)], axis=1)
        along = self.ys if axis == 0 else self.xs
        n_along = max(int(round(np.sqrt(count * 4))), 1)
        n_up = max(count // n_along, 1)
        u = np.linspace(along[0], along[-1], n_along + 2)[1:-1]
        seeds = []
        for value in u:
            point = (
                np.array([[position, value, 0.0]])
                if axis == 0
                else np.array([[value, position, 0.0]])
            )
            bottom, top = self._column_range(point[0])
            for s in np.linspace(0.0, 1.0, n_up + 2)[1:-1]:
                seed = point[0].copy()
                seed[2] = bottom + s * (top - bottom)
                seeds.append(seed)
        seeds = np.array(seeds, dtype=np.float64).reshape(-1, 3)
        return seeds[np.isfinite(seeds).all(axis=1)]  # not in dry columns

    def _column_range(self, point: np.ndarray) -> tuple[float, float]:
        i = int(np.clip(np.searchsorted(self.xs, point[0]) - 1, 0, len(self.xs) - 1))
        j = int(np.clip(np.searchsorted(self.ys, point[1]) - 1, 0, len(self.ys) - 1))
        return float(self.z[i, j, 0]), float(self.z[i, j, -1])


def trace(
    grid: ColumnGrid,
    vector: str,
    seeds: np.ndarray,
    step: float | None = None,
    steps: int = 400,
) -> list[np.ndarray]:
    """Streamlines through ``seeds``: forwards and backwards with the midpoint rule,
    ``step`` (default: a third of a cell) per step in space, until they leave the
    water. Returns one ``(m, 3)`` polyline per seed (2 points at least)."""
    step = grid.cell / 3 if step is None else float(step)
    halves = []
    for direction in (1.0, -1.0):
        position = np.array(seeds, dtype=np.float64)
        alive = np.ones(len(position), bool)
        path = [position.copy()]
        for _ in range(steps):
            if not alive.any():
                break
            velocity, inside = grid.sample(vector, position)
            speed = np.linalg.norm(velocity, axis=1)
            alive &= inside & (speed > 1e-12)
            unit = velocity / np.where(speed > 0, speed, 1.0)[:, None]
            middle = position + 0.5 * step * direction * unit
            v_mid, inside_mid = grid.sample(vector, middle)
            s_mid = np.linalg.norm(v_mid, axis=1)
            alive &= inside_mid & (s_mid > 1e-12)
            unit_mid = v_mid / np.where(s_mid > 0, s_mid, 1.0)[:, None]
            moved = position + step * direction * unit_mid
            _, inside_end = grid.sample(vector, moved)
            alive &= inside_end  # a line stops at the last point still in the water
            position = np.where(alive[:, None], moved, position)
            path.append(position.copy())
        halves.append(np.stack(path, axis=1))  # (seeds, steps + 1, 3)
    lines = []
    for forward, backward in zip(halves[0], halves[1], strict=True):
        line = np.concatenate([backward[::-1], forward[1:]])
        keep = np.concatenate([[True], np.any(np.diff(line, axis=0) != 0, axis=1)])
        line = line[keep]  # the steps after a line stopped repeat its end
        if len(line) >= 2:
            lines.append(line)
    return lines


@dataclass
class SheetGrid:
    """A surface over a regular x-y grid (REEF3D's free surface): its height and
    fields per grid point, NaN where there is no water (dry or missing)."""

    xs: np.ndarray  # (nx,)
    ys: np.ndarray  # (ny,)
    z: np.ndarray  # (nx, ny), NaN where dry
    fields: dict[str, np.ndarray]  # name -> (nx, ny) or (nx, ny, c)

    @classmethod
    def from_points(
        cls, points: np.ndarray, fields: dict[str, np.ndarray], wet: np.ndarray | None = None
    ) -> SheetGrid:
        points = np.asarray(points, dtype=np.float64)
        xs, ix = np.unique(np.round(points[:, 0], 3), return_inverse=True)
        ys, iy = np.unique(np.round(points[:, 1], 3), return_inverse=True)
        nx, ny = len(xs), len(ys)
        if nx < 2 or ny < 2:
            raise NotColumnGrid("the surface is not on an x-y grid")
        cell = ix.ravel() * ny + iy.ravel()
        if len(np.unique(cell)) < 0.5 * len(points):
            raise NotColumnGrid("the surface is not on an x-y grid")
        ok = np.ones(len(points), bool) if wet is None else np.asarray(wet, bool)
        z = np.full(nx * ny, np.nan)
        z[cell[ok]] = points[ok, 2]
        shaped = {}
        for name, values in fields.items():
            values = np.asarray(values, dtype=np.float64)
            full = np.zeros((nx * ny, *values.shape[1:]))
            full[cell[ok]] = values[ok]
            shaped[name] = full.reshape(nx, ny, *values.shape[1:])
        return cls(xs.astype(np.float64), ys.astype(np.float64), z.reshape(nx, ny), shaped)

    @property
    def cell(self) -> float:
        return float(min(np.median(np.diff(self.xs)), np.median(np.diff(self.ys))))

    def _corners(self, p: np.ndarray):
        i = np.clip(np.searchsorted(self.xs, p[:, 0]) - 1, 0, len(self.xs) - 2)
        j = np.clip(np.searchsorted(self.ys, p[:, 1]) - 1, 0, len(self.ys) - 2)
        fx = (p[:, 0] - self.xs[i]) / (self.xs[i + 1] - self.xs[i])
        fy = (p[:, 1] - self.ys[j]) / (self.ys[j + 1] - self.ys[j])
        inside = (fx >= 0) & (fx <= 1) & (fy >= 0) & (fy <= 1)
        weights = (
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        )
        for di, dj, _ in weights:
            inside &= np.isfinite(self.z[i + di, j + dj])  # all four corners wet
        return i, j, weights, inside

    def sample(self, name: str, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear values at points (x and y used) and whether they are on wet water."""
        values = self.z if name == "z" else self.fields[name]
        p = np.asarray(points, dtype=np.float64)
        i, j, weights, inside = self._corners(p)
        result = np.zeros((len(p), *values.shape[2:]))
        for di, dj, weight in weights:
            corner = np.nan_to_num(values[i + di, j + dj])
            result += weight.reshape(-1, *([1] * (corner.ndim - 1))) * corner
        return result, inside

    def seeds(self, count: int) -> np.ndarray:
        """About ``count`` seeds on a regular grid over the wet surface."""
        count = max(int(count), 1)
        width, depth = np.ptp(self.xs), np.ptp(self.ys)
        nx = max(int(round(np.sqrt(count * width / max(depth, 1e-12)))), 1)
        ny = max(int(round(count / nx)), 1)
        gx, gy = np.meshgrid(
            np.linspace(self.xs[0], self.xs[-1], nx + 2)[1:-1],
            np.linspace(self.ys[0], self.ys[-1], ny + 2)[1:-1],
            indexing="ij",
        )
        seeds = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        z, inside = self.sample("z", seeds)
        seeds[:, 2] = z
        return seeds[inside]


def trace_on_sheet(
    sheet: SheetGrid, vector: str, seeds: np.ndarray, step: float | None = None, steps: int = 400
) -> list[np.ndarray]:
    """Streamlines of the horizontal part of a vector field on a surface: traced in x
    and y with the midpoint rule, lying on the surface. Like :func:`trace`."""
    step = sheet.cell / 3 if step is None else float(step)

    def horizontal(points):
        velocity, inside = sheet.sample(vector, points)
        velocity = velocity[:, :2]
        speed = np.linalg.norm(velocity, axis=1)
        return velocity / np.where(speed > 0, speed, 1.0)[:, None], inside & (speed > 1e-12)

    halves = []
    for direction in (1.0, -1.0):
        position = np.array(seeds, dtype=np.float64)
        alive = np.ones(len(position), bool)
        path = [position.copy()]
        for _ in range(steps):
            if not alive.any():
                break
            unit, ok = horizontal(position)
            alive &= ok
            middle = position.copy()
            middle[:, :2] += 0.5 * step * direction * unit
            unit_mid, ok_mid = horizontal(middle)
            alive &= ok_mid
            moved = position.copy()
            moved[:, :2] += step * direction * unit_mid
            z, ok_end = sheet.sample("z", moved)
            alive &= ok_end
            moved[:, 2] = z
            position = np.where(alive[:, None], moved, position)
            path.append(position.copy())
        halves.append(np.stack(path, axis=1))
    lines = []
    for forward, backward in zip(halves[0], halves[1], strict=True):
        line = np.concatenate([backward[::-1], forward[1:]])
        keep = np.concatenate([[True], np.any(np.diff(line, axis=0) != 0, axis=1)])
        line = line[keep]
        if len(line) >= 2:
            lines.append(line)
    return lines
