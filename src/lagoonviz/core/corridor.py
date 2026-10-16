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

The corridor is kept in ``corridor.json`` in the case folder; a river drawn on the
map as a line (before the corridor is made along it) in ``river.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

FILE = "corridor.json"
#: the river drawn on the map: its line, upstream first
RIVER_FILE = "river.json"
GRID_FILE = "curvgrid.dat"
CURVES = ("through", "bspline")
BEDS = ("slope", "geo")
#: how the banks are paired into cross-sections: equal lengths along both banks, or
#: across the river (perpendicular to the line halfway between the banks)
SECTIONS = ("even", "normal")
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


def _sections(centre: np.ndarray, count: int, right_line: np.ndarray, left_line: np.ndarray):
    """The ends on the right and the left bank of ``count`` cross-sections square
    to a line along the river, from points spaced equally along it."""
    stations = resample(centre, count)
    tangent = np.gradient(resample(centre, 4 * (count - 1) + 1), axis=0)[::4]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    ends = []
    for line, sign in ((right_line, -1.0), (left_line, 1.0)):
        along = lengths(line)
        at = np.array(
            [_crossing(line, along, p, sign * n) for p, n in zip(stations, normal, strict=True)]
        )
        even = np.linspace(0.0, along[-1], count)
        missing = ~np.isfinite(at)
        at[missing] = even[missing]
        at[0], at[-1] = 0.0, along[-1]
        at = _even_out(_monotone(at, even))
        ends.append(
            np.column_stack([np.interp(at, along, line[:, 0]), np.interp(at, along, line[:, 1])])
        )
    return ends[0], ends[1]


#: the most a cell along a bank may differ in length from its neighbour
STRETCH = 1.2
#: the shortest and longest a cell along a bank may be, of the even length
SPREAD = (0.6, 1.6)


def _even_out(at: np.ndarray, stretch: float = STRETCH) -> np.ndarray:
    """Positions along a bank (increasing) whose spacing changes smoothly: no step
    shorter or longer than SPREAD of an even one, then the steps evened out
    (diffused, their sum kept) until none is more than ``stretch`` times its
    neighbour - square cross-sections where the banks allow, never cells squeezed
    or stretched."""
    steps = np.diff(at).astype(float)
    if len(steps) < 2:
        return at
    total = float(steps.sum())
    mean = total / len(steps)
    for _ in range(50):  # none much shorter or longer than an even step
        clipped = np.clip(steps, SPREAD[0] * mean, SPREAD[1] * mean)
        clipped *= total / clipped.sum()
        if np.allclose(clipped, steps):
            break
        steps = clipped
    for _ in range(20000):
        ratio = steps[1:] / steps[:-1]
        if np.maximum(ratio, 1.0 / ratio).max() <= stretch:
            break
        padded = np.concatenate([steps[:1], steps, steps[-1:]])
        steps = steps + 0.25 * (padded[:-2] - 2.0 * steps + padded[2:])
        steps *= total / steps.sum()
    return np.concatenate([[at[0]], at[0] + np.cumsum(steps)])


def _crossing(line: np.ndarray, along: np.ndarray, point, direction) -> float:
    """Where a ray from ``point`` along ``direction`` first meets a polyline: the
    distance along the polyline (NaN if it does not)."""
    a, b = line[:-1], line[1:]
    edge = b - a
    d = np.asarray(direction, float)
    offset = a - np.asarray(point, float)
    denominator = edge[:, 0] * d[1] - edge[:, 1] * d[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset[:, 1] * d[0] - offset[:, 0] * d[1]) / denominator  # along the edge
        ray = (offset[:, 0] * edge[:, 1] - offset[:, 1] * edge[:, 0]) / -denominator
    hit = (np.abs(denominator) > 1e-15) & (t >= 0) & (t <= 1) & (ray >= 0)
    if not hit.any():
        return float("nan")
    k = np.flatnonzero(hit)[np.argmin(ray[hit])]
    return float(along[k] + t[k] * (along[k + 1] - along[k]))


def _monotone(at: np.ndarray, even: np.ndarray) -> np.ndarray:
    """Positions along a bank made to increase: where they do not (cross-sections
    crossing on the inside of a tight bend), spaced evenly between the nearest
    positions round them that are in order - only there."""
    at = at.astype(float).copy()
    gap = 1e-6 * max(float(even[-1]), 1e-9)
    for _ in range(len(at)):
        bad = np.flatnonzero(np.diff(at) <= gap)
        if not len(bad):
            return at
        low, high = int(bad[0]), int(bad[0]) + 1
        while True:
            low, high = max(low - 1, 0), min(high + 1, len(at) - 1)
            if at[high] - at[low] > gap * (high - low) or (low == 0 and high == len(at) - 1):
                break
        if at[high] - at[low] <= gap * (high - low):
            return even.copy()
        at[low : high + 1] = np.linspace(at[low], at[high], high - low + 1)
    return at if (np.diff(at) > gap).all() else even.copy()


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
    sections: str = "even"
    bed: str = "slope"
    z_start: float = 0.0
    slope: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.left = np.asarray(self.left, float).reshape(-1, 2)
        self.right = np.asarray(self.right, float).reshape(-1, 2)
        if self.curve not in CURVES:
            raise ValueError(f"curve must be one of {', '.join(CURVES)}, got {self.curve!r}")
        if self.sections not in SECTIONS:
            raise ValueError(
                f"sections must be one of {', '.join(SECTIONS)}, got {self.sections!r}"
            )
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
            self.sections,
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
        if self.sections == "normal":
            right, left = self._normal_sections()
        else:
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

    def _normal_sections(self) -> tuple[np.ndarray, np.ndarray]:
        """The ends of ni + 1 cross-sections across the river: from points spaced
        equally along the line halfway between the banks, straight across it to
        both banks. That line is found again from the cross-sections twice (each
        bank's point straight across from it, not the point as far along). Where
        cross-sections would cross one another (the inside of a tight bend) they
        are spaced evenly there; the two ends stay the corridor's."""
        left_line, right_line = self.bank_line("left"), self.bank_line("right")
        count = max(400, 4 * self.cells_along)
        centre = 0.5 * (resample(left_line, count) + resample(right_line, count))
        width = float(
            np.median(
                np.linalg.norm(resample(left_line, count) - resample(right_line, count), axis=1)
            )
        )
        for _ in range(2):  # the line halfway across, from the cross-sections
            step = lengths(centre)[-1] / max(len(centre) - 1, 1)
            centre = _smooth_line(centre, 0.1 * width / max(step, 1e-12))
            right, left = _sections(centre, count // 4, right_line, left_line)
            centre = resample(0.5 * (right + left), count)
        step = lengths(centre)[-1] / max(len(centre) - 1, 1)
        centre = _smooth_line(centre, 0.1 * width / max(step, 1e-12))
        return _sections(centre, self.cells_along + 1, right_line, left_line)

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
            "sections": self.sections,
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
            "sections",
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


def read_river(folder: str | Path) -> np.ndarray | None:
    """The river drawn on the map of a case folder: its line (n, 2), upstream first
    (None: no river.json, or not readable)."""
    path = Path(folder) / RIVER_FILE
    if not path.is_file():
        return None
    try:
        line = np.asarray(json.loads(path.read_text(encoding="utf-8"))["line"], float)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return line.reshape(-1, 2) if line.size >= 4 and line.size % 2 == 0 else None


def river_text(line) -> str:
    """river.json of a line."""
    points = [[round(float(x), 4), round(float(y), 4)] for x, y in np.asarray(line, float)]
    return json.dumps({"line": points}, indent=1) + "\n"


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


# ================================================================ the thalweg
#: cells of the bed raster along the domain's longer side (thalweg search)
RASTER = 300


@dataclass(frozen=True)
class Thalweg:
    """What :func:`from_bed` found: the thalweg (n, 2), the bed along it, the wetted
    half-widths on the left and right (n each) and the level they were taken at."""

    line: np.ndarray
    bed: np.ndarray
    left: np.ndarray
    right: np.ndarray
    level: float


def bed_raster(points: np.ndarray, domain, cells: int = RASTER):
    """The bed on a regular raster over the domain: (xs, ys, z (ny, nx))."""
    xs0, xe, ys0, ye = (float(v) for v in domain[:4])
    step = max(xe - xs0, ye - ys0) / max(int(cells), 2)
    xs = np.arange(xs0, xe + 0.5 * step, step)
    ys = np.arange(ys0, ye + 0.5 * step, step)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, interpolate_bed(np.asarray(points, float), gx, gy)


def _side_cells(shape, side: int):
    """The raster cells (row, column) along a side of the domain: 1 x min, 4 x max,
    3 y min, 2 y max."""
    ny, nx = shape
    if side == 1:
        return [(r, 0) for r in range(ny)]
    if side == 4:
        return [(r, nx - 1) for r in range(ny)]
    if side == 3:
        return [(0, c) for c in range(nx)]
    return [(ny - 1, c) for c in range(nx)]


def thalweg_path(
    z: np.ndarray, start: tuple[int, int], end: tuple[int, int], open_=None
) -> np.ndarray:
    """The least-cost path (rows, columns) between two raster cells, the cost of a
    step rising steeply with the bed's height: it keeps to the deepest channel.
    ``open_``: the cells it may cross (a mask; default all)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    ny, nx = z.shape
    inside = z if open_ is None else z[open_]
    low, high = float(np.nanmin(inside)), float(np.nanmax(inside))
    weight = 1.0 + 1000.0 * ((z - low) / max(high - low, 1e-12)) ** 2
    if open_ is not None:
        weight = np.where(open_, weight, 1e9)
    index = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, costs = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = index[max(0, -dr) : ny - max(0, dr), max(0, -dc) : nx - max(0, dc)]
        b = index[max(0, dr) : ny + min(0, dr) or None, max(0, dc) : nx + min(0, dc) or None]
        wa = weight.ravel()[a.ravel()]
        wb = weight.ravel()[b.ravel()]
        cost = 0.5 * (wa + wb) * math.hypot(dr, dc)
        rows += [a.ravel(), b.ravel()]
        cols += [b.ravel(), a.ravel()]
        costs += [cost, cost]
    graph = coo_matrix(
        (np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ny * nx, ny * nx),
    ).tocsr()
    first, last = index[start], index[end]
    _, before = dijkstra(graph, indices=first, return_predecessors=True)
    path = [last]
    while path[-1] != first:
        previous = before[path[-1]]
        if previous < 0:
            raise ValueError("no path between the two ends of the river")
        path.append(previous)
    path = np.asarray(path[::-1])
    return np.column_stack([path // nx, path % nx])


def _smooth(values: np.ndarray, width: float) -> np.ndarray:
    """A Gaussian running mean over ``width`` samples (the ends kept in place)."""
    from scipy.ndimage import gaussian_filter1d

    if width <= 0 or len(values) < 3:
        return values
    return gaussian_filter1d(values, width, axis=0, mode="nearest")


def _smooth_line(line: np.ndarray, width: float) -> np.ndarray:
    """A polyline smoothed by a Gaussian running mean, its ends where they were (the
    line continued past them by point reflection)."""
    if width <= 0 or len(line) < 3:
        return line
    pad = min(len(line) - 1, int(4 * width) + 1)
    head = 2 * line[0] - line[pad:0:-1]
    tail = 2 * line[-1] - line[-2 : -pad - 2 : -1]
    smooth = _smooth(np.vstack([head, line, tail]), width)
    return smooth[pad : pad + len(line)]


#: the farthest a bank lies from the thalweg, in the river's usual half widths
WIDEST = 2.5


def default_level(z: np.ndarray, bed: np.ndarray) -> float:
    """The water level when none is given (no F 60): 0 where the bed reaches below
    and above 0 (heights to a sea level datum: a coast, an estuary), else halfway
    between the thalweg's bed and the top of the banks."""
    low, high = float(np.nanmin(z)), float(np.nanmax(z))
    if low < 0.0 < high and float(np.median(bed)) < 0.0:
        return 0.0
    return float(np.median(bed) + 0.5 * (np.nanpercentile(z, 75) - np.median(bed)))


def find_thalweg(
    points: np.ndarray,
    domain,
    level: float | None = None,
    start_side: int = 1,
    end_side: int = 4,
    cells: int = RASTER,
) -> Thalweg:
    """The thalweg of the bed (geo.dat's points) from the deepest cell of one side
    of the domain to the deepest of another, and how far the water (below
    ``level``; default: halfway between the thalweg's bed and the top of the banks)
    reaches to its left and right."""
    from scipy.ndimage import map_coordinates

    xs, ys, z = bed_raster(points, domain, cells)
    step = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0

    def deepest(side):
        cells_ = _side_cells(z.shape, side)
        return min(cells_, key=lambda rc: z[rc])

    path = thalweg_path(z, deepest(start_side), deepest(end_side))
    line = np.column_stack([xs[path[:, 1]], ys[path[:, 0]]])
    line = resample(_smooth_line(line, 3.0), max(len(line), 8))
    line = resample(_smooth_line(line, 0.02 * len(line)), max(len(line) // 2, 8))
    # the bed along it, and across it to both sides
    to_index = lambda p: np.array([(p[:, 1] - ys[0]) / step, (p[:, 0] - xs[0]) / step])  # noqa: E731
    bed = map_coordinates(z, to_index(line), order=1, mode="nearest")
    if level is None:
        level = default_level(z, bed)
    reach = 0.5 * max(domain[1] - domain[0], domain[3] - domain[2])
    left, right = _wetted(line, xs, ys, z, level, reach)
    return Thalweg(line, bed, left, right, float(level))


def _wetted(line, xs, ys, z, level, reach):
    """How far the water below ``level`` reaches to the left and right of a line on
    the raster (xs, ys, z): (left, right) half-widths, at most ``reach``."""
    from scipy.ndimage import map_coordinates

    step = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0

    def to_index(p):
        return np.array([(p[:, 1] - ys[0]) / step, (p[:, 0] - xs[0]) / step])

    tangent = np.gradient(line, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # to the left
    distances = np.arange(0.0, max(reach, step), 0.5 * step)
    widths = []
    for sign in (1.0, -1.0):
        found = np.full(len(line), np.nan)
        for k, d in enumerate(distances):
            probe = line + sign * d * normal
            inside = (
                (probe[:, 0] >= xs[0])
                & (probe[:, 0] <= xs[-1])
                & (probe[:, 1] >= ys[0])
                & (probe[:, 1] <= ys[-1])
            )
            height = map_coordinates(z, to_index(probe), order=1, mode="nearest")
            dry = (height > level) | ~inside
            new = np.isnan(found) & dry
            found[new] = d
            if not np.isnan(found).any() or k == len(distances) - 1:
                break
        found[np.isnan(found)] = distances[-1]
        widths.append(found)
    return widths[0], widths[1]


def from_line(line, width: float, count: int = 12, **options) -> Corridor:
    """A corridor along a line drawn on the map (upstream first), ``width`` m wide,
    ``count`` points on each bank: the banks parallel to it, the line smoothed a
    little so its corners do not fold the grid."""
    line = np.asarray(line, float)
    if len(line) < 2:
        raise ValueError("a river needs two points at least")
    keep = np.concatenate([[True], np.hypot(*np.diff(line, axis=0).T) > 1e-9])
    line = line[keep]
    if len(line) < 2:
        raise ValueError("the points of the river lie on one another")
    dense = resample(line, max(8 * len(line), 64))
    dense = _smooth_line(dense, 0.25 * len(dense) / max(len(line) - 1, 1))
    dense[0], dense[-1] = line[0], line[-1]
    n = len(dense)
    half = np.full(n, 0.5 * float(width))
    flat = Thalweg(dense, np.zeros(n), half, half, 0.0)
    options.setdefault("bed", "slope")
    return from_thalweg(flat, 0.0, count, width=float(width), **options)


def fit_to_bed(
    corridor: Corridor,
    points: np.ndarray,
    level: float | None = None,
    margin: float = 0.0,
    count: int | None = None,
    cells: int = RASTER,
    **options,
) -> tuple[Corridor, Thalweg]:
    """A corridor drawn by eye made to follow the river: the thalweg of the bed
    (geo.dat's points) within the corridor, from the deepest point of its upstream
    end to the deepest of its downstream end; its banks where the water below
    ``level`` reaches (default: halfway up the banks) plus ``margin``, at most half
    as far again as the corridor drawn. The banks' curve, the grid's cells and
    smoothing are kept. Returns (the corridor, the thalweg found)."""
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree

    centre = corridor.centreline(400)
    left = resample(corridor.bank_line("left"), 400)
    right = resample(corridor.bank_line("right"), 400)
    half = 0.5 * float(np.max(np.hypot(*(left - right).T)))
    pad = 0.75 * half
    box = (
        float(centre[:, 0].min() - half - pad),
        float(centre[:, 0].max() + half + pad),
        float(centre[:, 1].min() - half - pad),
        float(centre[:, 1].max() + half + pad),
    )
    xs, ys, z = bed_raster(points, box, cells)
    step = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    gx, gy = np.meshgrid(xs, ys)
    cells_xy = np.column_stack([gx.ravel(), gy.ravel()])
    near, _ = cKDTree(centre).query(cells_xy)
    band = (near <= half + 0.5 * step).reshape(z.shape) & np.isfinite(z)
    if band.sum() < 4:
        raise ValueError("the corridor is too narrow for the bed's raster")
    if not np.isfinite(z[band]).all() or np.nanmax(z[band]) - np.nanmin(z[band]) <= 0:
        raise ValueError("the bed is flat under the corridor: nothing to follow")

    def deepest_at(a, b):
        """The deepest cell of the band on the cross-section from a to b."""
        ab = b - a
        t = np.clip(((cells_xy - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d = np.hypot(*(cells_xy - (a + t[:, None] * ab)).T).reshape(z.shape)
        on = band & (d <= 0.75 * step)
        if not on.any():
            on = band & (d <= 2.0 * step)
        if not on.any():
            raise ValueError("an end of the corridor lies outside the bed")
        rc = np.unravel_index(np.argmin(np.where(on, z, np.inf)), z.shape)
        return int(rc[0]), int(rc[1])

    start = deepest_at(right[0], left[0])
    end = deepest_at(right[-1], left[-1])
    path = thalweg_path(np.where(band, z, 0.0), start, end, band)
    line = np.column_stack([xs[path[:, 1]], ys[path[:, 0]]])
    line = resample(_smooth_line(line, 3.0), max(len(line), 8))
    line = resample(_smooth_line(line, 0.02 * len(line)), max(len(line) // 2, 8))
    index = np.array([(line[:, 1] - ys[0]) / step, (line[:, 0] - xs[0]) / step])
    bed = map_coordinates(z, index, order=1, mode="nearest")
    if level is None:
        level = default_level(z[band], bed)
    widest = 1.5 * half
    wet_left, wet_right = _wetted(line, xs, ys, z, level, widest)
    thalweg = Thalweg(
        line, bed, np.minimum(wet_left, widest), np.minimum(wet_right, widest), float(level)
    )
    keep = {k: getattr(corridor, k) for k in ("curve", "cells_along", "cells_across", "smoothing")}
    keep.update(options)
    made = from_thalweg(thalweg, float(margin), int(count or len(corridor.left)), **keep)
    return made, thalweg


def from_thalweg(
    thalweg: Thalweg, margin: float = 0.0, count: int = 12, width: float | None = None, **options
) -> Corridor:
    """A corridor along the thalweg: its banks where the water reaches plus
    ``margin`` (m) on each side (or ``width`` / 2 on each side), smoothed along the
    river, ``count`` points on each bank."""
    from scipy.ndimage import median_filter

    line = thalweg.line
    n = len(line)
    if width is not None:
        left = right = np.full(n, 0.5 * float(width))
    else:
        size = max(3, n // 15) | 1
        # where the water spreads far over flat land (a marsh, a harbour basin) the
        # bank stays within WIDEST times the river's usual half width
        widest = WIDEST * float(np.median(np.concatenate([thalweg.left, thalweg.right])))
        left = median_filter(np.minimum(thalweg.left, widest), size=size, mode="nearest")
        right = median_filter(np.minimum(thalweg.right, widest), size=size, mode="nearest")
        left, right = left + float(margin), right + float(margin)
        left = _smooth(left, n / 30)
        right = _smooth(right, n / 30)
    tangent = np.gradient(line, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    left_bank = line + left[:, None] * normal
    right_bank = line - right[:, None] * normal
    mean_width = float(np.mean(left + right))
    options.setdefault("cells_across", 20)
    options.setdefault("bed", "geo")
    options.setdefault("sections", "normal")
    length = float(lengths(line)[-1])
    options.setdefault(
        "cells_along", max(int(round(length / max(mean_width / options["cells_across"], 1e-9))), 4)
    )
    count = int(min(max(count, MIN_POINTS), MAX_POINTS))
    return Corridor(resample(left_bank, count), resample(right_bank, count), **options)


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


#: scattered points beyond which the bed is not triangulated (slow): the nearest
#: points' heights, weighted by distance, instead
LINEAR_MOST = 150_000


def interpolate_bed(points: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Heights of geo.dat's points (n, 3) at the nodes: only the points round the
    nodes are used; points on a regular grid (as LAGOON writes geo.dat) bilinear on
    it, other points linear on their triangles (inverse distance of the nearest
    six when there are very many), the nearest point's height outside them."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(axis=1)]
    targets = np.column_stack([np.ravel(x), np.ravel(y)])
    if len(points) == 0:
        raise ValueError("no points with heights")
    low, high = targets.min(axis=0), targets.max(axis=0)
    pad = 0.05 * float(max(high - low)) + 1e-9
    inside = ((points[:, :2] >= low - pad) & (points[:, :2] <= high + pad)).all(axis=1)
    if inside.sum() >= 3:
        nearby = points[inside]
        step = _spacing(nearby)
        if step is not None:  # widen to at least two spacings round the nodes
            inside = (
                (points[:, :2] >= low - pad - 2 * step) & (points[:, :2] <= high + pad + 2 * step)
            ).all(axis=1)
            nearby = points[inside]
        points = nearby
    values = np.full(len(targets), np.nan)
    grid = _regular(points)
    if grid is not None:
        from scipy.interpolate import RegularGridInterpolator

        xs, ys, heights = grid
        values = RegularGridInterpolator((ys, xs), heights, bounds_error=False, fill_value=np.nan)(
            targets[:, ::-1]
        )
    elif len(points) > LINEAR_MOST:
        tree = cKDTree(points[:, :2])
        distance, index = tree.query(targets, k=min(6, len(points)))
        weight = 1.0 / np.maximum(distance, 1e-9) ** 2
        values = (weight * points[index, 2]).sum(axis=1) / weight.sum(axis=1)
    elif len(points) >= 3:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import QhullError

        try:
            values = LinearNDInterpolator(points[:, :2], points[:, 2])(targets)
        except QhullError:
            pass
    holes = ~np.isfinite(values)
    if holes.any():
        values[holes] = points[cKDTree(points[:, :2]).query(targets[holes])[1], 2]
    return values.reshape(np.shape(x))


def _regular(points: np.ndarray):
    """(xs, ys, heights (ny, nx)) when the points fill a regular grid, else None."""
    if len(points) < 4:
        return None
    span = float(np.ptp(points[:, :2])) or 1.0
    key = np.round(points[:, :2] / (1e-7 * span)).astype(np.int64)
    xs, xi = np.unique(key[:, 0], return_inverse=True)
    ys, yi = np.unique(key[:, 1], return_inverse=True)
    if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) != len(points):
        return None
    heights = np.full((len(ys), len(xs)), np.nan)
    heights[yi, xi] = points[:, 2]
    if np.isnan(heights).any():  # a point given twice, another missing
        return None
    return xs * (1e-7 * span), ys * (1e-7 * span), heights


def _spacing(points: np.ndarray) -> float | None:
    """A typical distance between neighbouring points (a sample's median)."""
    from scipy.spatial import cKDTree

    if len(points) < 2:
        return None
    sample = points[:: max(1, len(points) // 2000), :2]
    distance, _ = cKDTree(points[:, :2]).query(sample, k=2)
    found = float(np.median(distance[:, 1]))
    return found if found > 0 else None


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
