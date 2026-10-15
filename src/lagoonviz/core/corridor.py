# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""A river corridor: two banks shaped by points that are dragged, and the
curvilinear grid between them that REEF3D reads with ``G 50 2`` (``curvgrid.dat``).

The banks are seen looking downstream: the **left** and the **right** bank, each a
list of points from upstream (index 0) to downstream. A bank is the curve the
points shape (``curve``):

- ``"through"``: a centripetal Catmull-Rom spline through every point (no loops,
  no overshoot between close points);
- ``"bspline"``: a clamped cubic B-spline the points pull on (smoother; it passes
  through the first and last point only).

The grid has ``cells_along`` (i, downstream) x ``cells_across`` (j, from the right
bank to the left) cells: each bank is cut into equal lengths, the nodes between
them on straight lines across (``smoothing`` > 0: then made smoother and more even
by Winslow's elliptic equations, the banks and the two ends kept). j = 0 is the
right bank, so the cells turn counter-clockwise (positive area), as REEF3D wants.

The bed (zb at the nodes) is geo.dat's points interpolated (``bed = "geo"``), or a
flat bed falling from ``z_start`` at the upstream end by ``slope`` (m/m) along the
river (``bed = "slope"``).

``curvgrid.dat``: "ni nj", then x y zb of every node, j = 0 .. nj, i = 0 .. ni with
i fastest (REEF3D's lexer_geometry.cpp). DIVEMesh's grid must be the logical box of
the curvilinear one: ni x nj cells (B 10 and B 1 of control.txt).

The corridor is kept in ``corridor.json`` in the case folder.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

FILE = "corridor.json"
GRID_FILE = "curvgrid.dat"
CURVES = ("through", "bspline")
BEDS = ("slope", "geo")
#: points per bank: the fewest and the most
MIN_POINTS, MAX_POINTS = 4, 60
#: samples of the dense curve per segment between two points
DENSE = 24


# ====================================================================== curves
def catmull_rom(points, samples: int = DENSE) -> np.ndarray:
    """A centripetal Catmull-Rom spline through the points (n, 2): (m, 2)."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return _straight(p, samples)
    # phantom ends: the first and last segment continue straight on
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = [p[:1]]
    for k in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[k - 1], ext[k], ext[k + 1], ext[k + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0) ** 0.5, 1e-9)
        t2 = t1 + max(np.linalg.norm(p2 - p1) ** 0.5, 1e-9)
        t3 = t2 + max(np.linalg.norm(p3 - p2) ** 0.5, 1e-9)
        t = np.linspace(t1, t2, samples + 1)[1:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def bspline(points, samples: int = DENSE, degree: int = 3) -> np.ndarray:
    """A clamped (open uniform) B-spline of the control points (n, 2): (m, 2); it
    starts at the first point and ends at the last."""
    p = np.asarray(points, float)
    n = len(p)
    if n < 3:
        return _straight(p, samples)
    k = min(degree, n - 1)
    inner = np.arange(1, n - k) / (n - k)
    knots = np.concatenate([np.zeros(k + 1), inner, np.ones(k + 1)])
    u = np.linspace(0.0, 1.0, samples * (n - 1) + 1)
    out = np.empty((len(u), 2))
    for index, x in enumerate(u):
        span = min(np.searchsorted(knots, x, side="right") - 1, n - 1)
        d = [p[j].copy() for j in range(span - k, span + 1)]
        for r in range(1, k + 1):  # de Boor
            for j in range(k, r - 1, -1):
                i = span - k + j
                denominator = knots[i + k + 1 - r] - knots[i]
                a = 0.0 if denominator == 0 else (x - knots[i]) / denominator
                d[j] = (1 - a) * d[j - 1] + a * d[j]
        out[index] = d[k]
    return out


def _straight(p: np.ndarray, samples: int) -> np.ndarray:
    if len(p) < 2:
        return p.copy()
    t = np.linspace(0.0, 1.0, samples * (len(p) - 1) + 1)[:, None]
    return p[0] + t * (p[-1] - p[0])


def lengths(line: np.ndarray) -> np.ndarray:
    """Distance along a polyline (m, 2) from its start, at each point."""
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample(line: np.ndarray, count: int) -> np.ndarray:
    """``count`` points spaced equally along a polyline, its ends included."""
    s = lengths(line)
    if s[-1] <= 0:
        return np.repeat(line[:1], count, axis=0)
    at = np.linspace(0.0, s[-1], count)
    return np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])


# ==================================================================== corridor
#: grids made lately: (banks, curve, cells, smoothing) -> (x, y)
_GRIDS: dict = {}


@dataclass
class Corridor:
    """A river corridor (module doc)."""

    left: np.ndarray
    right: np.ndarray
    curve: str = "through"
    cells_along: int = 100
    cells_across: int = 20
    smoothing: int = 0
    bed: str = "slope"
    z_start: float = 0.0
    slope: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.left = np.asarray(self.left, float).reshape(-1, 2)
        self.right = np.asarray(self.right, float).reshape(-1, 2)
        if self.curve not in CURVES:
            raise ValueError(f"curve must be one of {', '.join(CURVES)}, got {self.curve!r}")
        if self.bed not in BEDS:
            raise ValueError(f"bed must be one of {', '.join(BEDS)}, got {self.bed!r}")
        for name, bank in (("left", self.left), ("right", self.right)):
            if len(bank) < 2:
                raise ValueError(f"the {name} bank needs two points at least")
        self.cells_along = max(int(self.cells_along), 1)
        self.cells_across = max(int(self.cells_across), 1)
        self.smoothing = max(int(self.smoothing), 0)

    # ------------------------------------------------------------- the banks
    def bank(self, side: str) -> np.ndarray:
        """The control points of a bank ("left", "right")."""
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return self.left if side == "left" else self.right

    def curve_of(self, points) -> np.ndarray:
        return catmull_rom(points) if self.curve == "through" else bspline(points)

    def bank_line(self, side: str) -> np.ndarray:
        """The bank as a dense polyline (m, 2)."""
        return self.curve_of(self.bank(side))

    def moved(self, side: str, index: int, x: float, y: float) -> Corridor:
        """The corridor with one point of a bank moved."""
        bank = self.bank(side).copy()
        bank[index] = (float(x), float(y))
        return replace(self, **{side: bank})

    def with_points(self, count: int) -> Corridor:
        """The corridor with ``count`` points on each bank, spaced equally along the
        banks as they are now (their shape kept as well as that many points can)."""
        count = int(min(max(count, MIN_POINTS), MAX_POINTS))
        return replace(
            self,
            left=resample(self.bank_line("left"), count),
            right=resample(self.bank_line("right"), count),
        )

    def reversed(self) -> Corridor:
        """The river flowing the other way: upstream and downstream, and so left and
        right, swapped."""
        return replace(self, left=self.right[::-1].copy(), right=self.left[::-1].copy())

    def centreline(self, count: int = 200) -> np.ndarray:
        """Halfway between the banks (each cut into equal lengths): (count, 2)."""
        return 0.5 * (
            resample(self.bank_line("left"), count) + resample(self.bank_line("right"), count)
        )

    # ------------------------------------------------------------- the grid
    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """The nodes (x, y), each (nj + 1, ni + 1): i downstream, j across from the
        right bank (from the left one if the banks are the other way round, so that
        the cells always turn counter-clockwise)."""
        key = (
            self.left.tobytes(),
            self.right.tobytes(),
            self.curve,
            self.cells_along,
            self.cells_across,
            self.smoothing,
        )
        found = _GRIDS.get(key)
        if found is None:
            found = self._make_grid()
            _GRIDS[key] = found
            while len(_GRIDS) > 8:  # the last few (the set-up and the panel ask alike)
                _GRIDS.pop(next(iter(_GRIDS)))
        return found[0].copy(), found[1].copy()

    def _make_grid(self) -> tuple[np.ndarray, np.ndarray]:
        ni, nj = self.cells_along, self.cells_across
        right = resample(self.bank_line("right"), ni + 1)
        left = resample(self.bank_line("left"), ni + 1)
        s = np.linspace(0.0, 1.0, nj + 1)[:, None, None]
        nodes = (1.0 - s) * right[None] + s * left[None]  # (nj+1, ni+1, 2)
        x, y = nodes[..., 0].copy(), nodes[..., 1].copy()
        if cell_areas(x, y).sum() < 0:  # the banks the other way round: j from the left
            x, y = x[::-1].copy(), y[::-1].copy()
        if self.smoothing and ni > 1 and nj > 1:
            x, y = winslow(x, y, self.smoothing)
        return x, y

    def bed_heights(self, x: np.ndarray, y: np.ndarray, geo_points=None) -> np.ndarray:
        """zb at the nodes: geo.dat's points interpolated (``bed = "geo"``, when
        given), else the sloping flat bed."""
        if self.bed == "geo" and geo_points is not None and len(geo_points) >= 1:
            return interpolate_bed(np.asarray(geo_points, float), x, y)
        centre = 0.5 * (x[0] + x[-1]), 0.5 * (y[0] + y[-1])
        s = lengths(np.column_stack(centre))
        return np.broadcast_to(self.z_start - self.slope * s, x.shape).copy()

    # ------------------------------------------------------------- the file
    def to_dict(self) -> dict:
        data = {
            "left": [[round(float(a), 6), round(float(b), 6)] for a, b in self.left],
            "right": [[round(float(a), 6), round(float(b), 6)] for a, b in self.right],
            "curve": self.curve,
            "cells_along": self.cells_along,
            "cells_across": self.cells_across,
            "smoothing": self.smoothing,
            "bed": self.bed,
            "z_start": self.z_start,
            "slope": self.slope,
        }
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Corridor:
        known = {
            "left",
            "right",
            "curve",
            "cells_along",
            "cells_across",
            "smoothing",
            "bed",
            "z_start",
            "slope",
        }
        values = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**values, extra=extra)

    def text(self) -> str:
        return json.dumps(self.to_dict(), indent=1) + "\n"


def read(folder: str | Path) -> Corridor | None:
    """The corridor of a case folder (None: no corridor.json, or not readable)."""
    path = Path(folder) / FILE
    if not path.is_file():
        return None
    try:
        return Corridor.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def default(area, count: int = 12, along_x: bool | None = None, **options) -> Corridor:
    """A straight corridor filling a plan area (x0, x1, y0, y1), flowing along its
    longer side (or along x when ``along_x``), ``count`` points on each bank."""
    x0, x1 = sorted(float(v) for v in area[:2])
    y0, y1 = sorted(float(v) for v in area[2:4])
    if along_x is None:
        along_x = (x1 - x0) >= (y1 - y0)
    t = np.linspace(0.0, 1.0, count)
    if along_x:  # downstream +x: the left bank at y1
        xs = x0 + t * (x1 - x0)
        left, right = (
            np.column_stack([xs, np.full(count, y1)]),
            np.column_stack([xs, np.full(count, y0)]),
        )
        length, width = x1 - x0, y1 - y0
    else:  # downstream +y: the left bank at x0
        ys = y0 + t * (y1 - y0)
        left, right = (
            np.column_stack([np.full(count, x0), ys]),
            np.column_stack([np.full(count, x1), ys]),
        )
        length, width = y1 - y0, x1 - x0
    options.setdefault("cells_across", 20)
    step = width / options["cells_across"]
    options.setdefault("cells_along", max(int(round(length / step)), 1) if step > 0 else 100)
    return Corridor(left, right, **options)


# ==================================================================== the mesh
def winslow(x: np.ndarray, y: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Interior nodes moved by Winslow's elliptic grid equations (Jacobi sweeps),
    the boundary nodes kept: a smoother grid that does not fold."""
    x, y = x.copy(), y.copy()
    for _ in range(int(iterations)):
        x_xi = 0.5 * (x[1:-1, 2:] - x[1:-1, :-2])
        y_xi = 0.5 * (y[1:-1, 2:] - y[1:-1, :-2])
        x_eta = 0.5 * (x[2:, 1:-1] - x[:-2, 1:-1])
        y_eta = 0.5 * (y[2:, 1:-1] - y[:-2, 1:-1])
        a = x_eta**2 + y_eta**2
        b = x_xi * x_eta + y_xi * y_eta
        c = x_xi**2 + y_xi**2
        denominator = 2.0 * (a + c)
        denominator[denominator == 0] = 1.0

        x[1:-1, 1:-1], y[1:-1, 1:-1] = (_winslow_step(f, a, b, c, denominator) for f in (x, y))
    return x, y


def _winslow_step(f, a, b, c, denominator):
    cross = f[2:, 2:] - f[2:, :-2] - f[:-2, 2:] + f[:-2, :-2]
    return (
        a * (f[1:-1, 2:] + f[1:-1, :-2]) + c * (f[2:, 1:-1] + f[:-2, 1:-1]) - 0.5 * b * cross
    ) / denominator


def cell_areas(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed areas of the cells (nj, ni): positive when they turn counter-clockwise."""
    dx1, dy1 = x[1:, 1:] - x[:-1, :-1], y[1:, 1:] - y[:-1, :-1]
    dx2, dy2 = x[:-1, 1:] - x[1:, :-1], y[:-1, 1:] - y[1:, :-1]
    return 0.5 * (dx2 * dy1 - dy2 * dx1)


@dataclass(frozen=True)
class Quality:
    """What the grid is like: cells, sizes, angles, folds."""

    cells: tuple[int, int]
    along: tuple[float, float]
    across: tuple[float, float]
    min_angle: float
    folded: int
    area: tuple[float, float]

    def text(self) -> str:
        ni, nj = self.cells
        words = (
            f"{ni} x {nj} = {ni * nj:,} cells; along {self.along[0]:.3g} – "
            f"{self.along[1]:.3g} m, across {self.across[0]:.3g} – {self.across[1]:.3g} m; "
            f"smallest angle {self.min_angle:.0f}°"
        )
        if self.folded:
            words += f"; {self.folded} folded cells: REEF3D stops"
        return words


def quality(x: np.ndarray, y: np.ndarray) -> Quality:
    along = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    across = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    areas = cell_areas(x, y)
    # the angle at each cell corner between its edge along and its edge across
    ex, ey = np.diff(x, axis=1)[:-1], np.diff(y, axis=1)[:-1]
    fx, fy = np.diff(x, axis=0)[:, :-1], np.diff(y, axis=0)[:, :-1]
    norm = np.maximum(np.hypot(ex, ey) * np.hypot(fx, fy), 1e-30)
    cosine = np.clip((ex * fx + ey * fy) / norm, -1.0, 1.0)
    angle = np.degrees(np.arccos(np.abs(cosine)))
    return Quality(
        (x.shape[1] - 1, x.shape[0] - 1),
        (float(along.min()), float(along.max())),
        (float(across.min()), float(across.max())),
        float(angle.min()) if angle.size else 90.0,
        int((areas <= 0).sum()),
        (float(areas.min()), float(areas.max())),
    )


def interpolate_bed(points: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Heights of scattered points (n, 3) at the nodes: linear on their triangles,
    the nearest point's outside them."""
    from scipy.spatial import cKDTree

    points = points[np.isfinite(points).all(axis=1)]
    targets = np.column_stack([x.ravel(), y.ravel()])
    values = np.full(len(targets), np.nan)
    if len(points) >= 3:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import QhullError

        try:
            values = LinearNDInterpolator(points[:, :2], points[:, 2])(targets)
        except QhullError:
            pass
    holes = ~np.isfinite(values)
    if holes.any():
        values[holes] = points[cKDTree(points[:, :2]).query(targets[holes])[1], 2]
    return values.reshape(x.shape)


# ================================================================ curvgrid.dat
def grid_text(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> str:
    """curvgrid.dat: "ni nj", then x y zb of each node, i fastest."""
    nj, ni = x.shape[0] - 1, x.shape[1] - 1
    rows = [f"{ni} {nj}"]
    for j in range(nj + 1):
        for i in range(ni + 1):
            rows.append(f"{x[j, i]:.17g} {y[j, i]:.17g} {z[j, i]:.17g}")
    return "\n".join(rows) + "\n"


def read_grid(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """x, y, zb of a curvgrid.dat, each (nj + 1, ni + 1); None if not readable."""
    try:
        words = Path(path).read_text(encoding="utf-8").split()
        ni, nj = int(words[0]), int(words[1])
        values = np.asarray(words[2 : 2 + 3 * (ni + 1) * (nj + 1)], float)
        nodes = values.reshape(nj + 1, ni + 1, 3)
    except (OSError, ValueError, IndexError):
        return None
    return nodes[..., 0], nodes[..., 1], nodes[..., 2]


def logical_box(ni: int, nj: int, step: float) -> tuple[float, float, float, float]:
    """DIVEMesh's B 10 (x and y) for the logical box of an ni x nj grid of cells
    ``step`` wide: exactly ni x nj cells with B 1 ``step``."""
    step = float(step)
    return 0.0, round(ni * step, 9), 0.0, round(nj * step, 9)


def nice_step(size: float) -> float:
    """A round cell size near ``size`` for the logical box (1, 2 or 5 times 10^k)."""
    if not size > 0 or not math.isfinite(size):
        return 1.0
    k = math.floor(math.log10(size))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * 10**k >= size * 0.75:
            return m * 10**k
    return 10.0 ** (k + 1)
