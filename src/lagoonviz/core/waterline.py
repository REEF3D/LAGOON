# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Edges of a multi-block grid laid along the water.

The faces of the blocks of a coast or an estuary do not lie on the waterline: they
lie on land a little beyond it, smooth, and the shore inside the blocks is wetted
and dried. :func:`waterline` finds such a line between two corners:

1. the bed (geo.dat's points) on a raster round the edge; water where it is at or
   below the ``level`` (F 60), land elsewhere (and where the bed is not known);
2. the water widened by ``margin`` onto the land (every point within ``margin`` of
   water), then closed by the bend ``radius`` (inlets of land narrower than twice
   the radius filled in, bends towards the land at least that round) and opened by
   the smaller of margin and radius (so the line keeps at least ``margin`` from
   the water everywhere: a disc of the margin round any water point stays inside);
3. the outline of that area, from the point nearest the edge's start to the point
   nearest its end, the way round that lies nearer to where the edge is now;
4. smoothed a little, ends kept.

The edge's corners move to the line's ends (or stay, the line joining the shore
from them). Moving a corner later lays the edge
again (the session does, with geo.dat), so a corner dragged along the coast keeps
the edge on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lagoonviz.core import corridor as curvi

#: the most raster cells along a side
MOST_CELLS = 1600


@dataclass(frozen=True)
class Waterline:
    """What :func:`waterline` found: the ``line`` (start to end), the ``level`` it
    was taken at, the raster ``step`` and how many of the line's points lie in the
    water (``wet``: none, unless the bed under it is coarser than the margin)."""

    line: np.ndarray
    level: float
    step: float
    wet: int


def waterline(
    points: np.ndarray,
    guide: np.ndarray,
    level: float,
    margin: float = 0.0,
    radius: float = 0.0,
    step: float | None = None,
    corners: bool = True,
) -> Waterline:
    """The line along the water between the two ends of ``guide`` (the edge as it is
    now, start to end: a dense polyline), on land ``margin`` m from the water below
    ``level`` (geo.dat's ``points``, (n, 3)), bending no tighter than ``radius`` m.
    The raster ``step`` defaults to a third of the smaller of margin and radius (at
    most MOST_CELLS a side). ``corners`` False: the line starts and ends at the
    guide's ends (the corners stay), joining the shore smoothly."""
    from scipy.ndimage import distance_transform_edt as edt

    guide = np.asarray(guide, float)
    if len(guide) < 2:
        raise ValueError("the edge has no length")
    margin, radius = max(float(margin), 0.0), max(float(radius), 0.0)
    low, high = guide.min(axis=0), guide.max(axis=0)
    size = float(max(high - low))
    reach = max(0.3 * size, 3.0 * (margin + radius), 1e-6)
    box = (low[0] - reach, high[0] + reach, low[1] - reach, high[1] + reach)
    fine = [v for v in (margin, radius) if v > 0]
    wanted = min(fine) / 3.0 if fine else size / 400.0
    least = max(box[1] - box[0], box[3] - box[2]) / MOST_CELLS
    step = float(step) if step else max(wanted, least)
    xs = np.arange(box[0], box[1] + 0.5 * step, step)
    ys = np.arange(box[2], box[3] + 0.5 * step, step)
    gx, gy = np.meshgrid(xs, ys)
    z = curvi.interpolate_bed(np.asarray(points, float), gx, gy)
    water = np.isfinite(z) & (z <= level)
    if not water.any():
        raise ValueError(f"no water below {level:g} m near the edge (the bed of geo.dat)")
    if water.all():
        raise ValueError(f"only water below {level:g} m near the edge: no shore to follow")
    to_water = edt(~water, sampling=step)  # how far each cell is from the water
    grown = to_water <= margin + radius  # widened by margin, then by the radius ...
    area = edt(grown, sampling=step) > radius  # ... and shrunk back: closed by it
    keep = min(margin, radius)
    if keep > 0:  # opened by the smaller: no corners sharper than it towards the water
        core = edt(area, sampling=step) > keep
        area = edt(~core, sampling=step) <= keep
    area |= water  # (never less than the water itself)
    signed = edt(area, sampling=step) - edt(~area, sampling=step)
    line = _path(xs, ys, signed, guide)
    if not corners:
        line = np.vstack([guide[0], line, guide[-1]])
    line = curvi.resample(line, max(int(curvi.lengths(line)[-1] / step), 8))
    if not corners:
        line[0], line[-1] = guide[0], guide[-1]
    shore = to_water - edt(water, sampling=step)  # signed: below 0 in the water
    line = _relax(line, xs, ys, shore, margin, max(radius, 2.0 * step), step)
    heights = curvi.interpolate_bed(np.asarray(points, float), line[:, 0], line[:, 1])
    wet = int(np.sum(np.isfinite(heights) & (heights <= level)))
    return Waterline(line, float(level), step, wet)


def _relax(line, xs, ys, shore, margin, radius, step, sweeps: int = 40) -> np.ndarray:
    """The line smoothed over about ``radius`` while it keeps ``margin`` from the water
    (``shore``: the signed distance to it on the raster): smoothed a little, pushed
    back out where it came too near, again and again; its ends stay."""
    from scipy.ndimage import map_coordinates

    gy, gx = np.gradient(shore, step)

    def at(field, p):
        index = np.array([(p[:, 1] - ys[0]) / step, (p[:, 0] - xs[0]) / step])
        return map_coordinates(field, index, order=1, mode="nearest")

    width = 0.25 * radius / step
    a, b = line[0].copy(), line[-1].copy()
    for _ in range(sweeps):
        line = curvi._smooth_line(line, width)
        d = at(shore, line)
        near = d < margin
        if near.any():
            g = np.column_stack([at(gx, line[near]), at(gy, line[near])])
            g /= np.maximum(np.hypot(*g.T), 1e-12)[:, None]
            line[near] += (margin - d[near])[:, None] * g
        line[0], line[-1] = a, b
        line = curvi.resample(line, len(line))
    return line


def _path(xs, ys, field, guide) -> np.ndarray:
    """The zero line of ``field`` between the points of it nearest the ends of the
    guide, the way nearer to the guide."""
    import contourpy
    from scipy.spatial import cKDTree

    lines = contourpy.contour_generator(xs, ys, field, line_type="Separate").lines(0.0)
    lines = [np.asarray(c, float) for c in lines if len(c) >= 2]
    if not lines:
        raise ValueError("no shore found near the edge")
    a, b = guide[0], guide[-1]
    best = None
    for line in lines:
        da = np.hypot(*(line - a).T)
        db = np.hypot(*(line - b).T)
        i, j = int(np.argmin(da)), int(np.argmin(db))
        cost = da[i] + db[j]
        if best is None or cost < best[0]:
            best = (cost, line, i, j)
    _, line, i, j = best
    closed = len(line) > 3 and np.hypot(*(line[0] - line[-1])) < 1e-9
    if closed:
        line = line[:-1]
        n = len(line)
        forward = line[[(i + k) % n for k in range((j - i) % n + 1)]]
        backward = line[[(i - k) % n for k in range((i - j) % n + 1)]]
        candidates = [forward, backward]
    else:
        candidates = [line[i : j + 1] if i <= j else line[j : i + 1][::-1]]
    sample = curvi.resample(guide, 200)

    def distance(c):
        if len(c) < 2:
            return np.inf
        return float(cKDTree(curvi.resample(c, 400)).query(sample)[0].mean())

    path = min(candidates, key=distance)
    if len(path) < 2:
        raise ValueError("the edge's two corners come to the same point of the shore")
    return path
