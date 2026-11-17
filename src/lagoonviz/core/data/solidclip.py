# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""A volume's surfaces without the solids in them: "Without solids" of the volume,
its slices and iso-surfaces.

REEF3D writes its volume over the whole domain, the cells in solids too (CFD's in
DIVEMesh's solids, NHFLOW's in its direct forcing solids, the cells a floating body
covers): there the flow is no flow. :class:`Solids` tells which points of a surface
lie in the solids (closed triangle surfaces: the winding of a vertical ray, so
solids that overlap stay inside), and :func:`without_solids` cuts the surface where
it enters them, along their faces (the signed distance to the nearest face at the
points round the cut, so the cut follows the solid, not the grid).
"""

from __future__ import annotations

import math

import numpy as np

from lagoonviz.core.data.mesh import SurfaceMesh, clip_surface

#: point-triangle pairs tested at once (memory: a few dozen bytes each)
CHUNK = 2_000_000
#: halvings of an edge to find where it enters a solid (to a 10⁻⁷ of it)
HALVINGS = 24


class Solids:
    """Closed triangle surfaces (m, 3, 3) facing out, for inside tests."""

    def __init__(self, triangles: np.ndarray) -> None:
        t = np.asarray(triangles, np.float64).reshape(-1, 3, 3)
        normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        size = np.linalg.norm(normal, axis=1)
        t, normal = t[size > 0], normal[size > 0]
        self.triangles = t
        self.low = t.reshape(-1, 3).min(axis=0) if len(t) else np.zeros(3)
        self.high = t.reshape(-1, 3).max(axis=0) if len(t) else np.zeros(3)
        extent = float(np.ptp(np.array([self.low, self.high]), axis=0).max()) if len(t) else 1.0
        self.extent = max(extent, 1e-12)
        # the ray goes up from a point a little aside (off edges and corners)
        self.shift = np.array([0.31830989, 0.57721566]) * 1e-7 * self.extent
        self.reach = 1e-7 * self.extent
        flat = np.abs(normal[:, 2]) > 1e-12 * np.linalg.norm(normal, axis=1)
        self.crossing = t[flat]
        self.facing = np.sign(normal[flat, 2]).astype(np.int64)  # +1 up (a top), -1 down
        self._bins()

    def _bins(self) -> None:
        """The faces a vertical ray can cross, binned by their plan (CSR by cell)."""
        t = self.crossing
        count = len(t)
        n = max(1, min(256, int(math.sqrt(max(count, 1)))))
        self.cells = n
        span = np.maximum(self.high[:2] - self.low[:2], 1e-12)
        self.cell = span / n
        if not count:
            self.order = np.zeros(0, np.int64)
            self.starts = np.zeros(n * n + 1, np.int64)
            return
        lo = np.floor((t[..., :2].min(axis=1) - self.low[:2]) / self.cell).astype(np.int64)
        hi = np.floor((t[..., :2].max(axis=1) - self.low[:2]) / self.cell).astype(np.int64)
        lo, hi = np.clip(lo, 0, n - 1), np.clip(hi, 0, n - 1)
        wide = hi - lo + 1
        each = wide[:, 0] * wide[:, 1]
        face = np.repeat(np.arange(count), each)
        k = np.arange(each.sum()) - np.repeat(np.cumsum(each) - each, each)
        i = lo[face, 0] + k % wide[face, 0]
        j = lo[face, 1] + k // wide[face, 0]
        cell = j * n + i
        order = np.argsort(cell, kind="stable")
        self.order = face[order]
        self.starts = np.searchsorted(cell[order], np.arange(n * n + 1))

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Whether each point (n, 3) lies in the solids (on their faces too): the
        faces crossed by a ray straight up, tops counted +1 and bottoms -1 (in a
        solid: 1; in two that overlap: 2; outside: 0)."""
        points = np.asarray(points, np.float64)
        found = np.zeros(len(points), bool)
        if not len(self.crossing) or not len(points):
            return found
        margin = 2.0 * self.reach  # outside the solids' box: outside them
        near = np.flatnonzero(
            ((points >= self.low - margin) & (points <= self.high + margin)).all(axis=1)
        )
        if not len(near):
            return found
        xy = points[near, :2] + self.shift
        z = points[near, 2]
        n = self.cells
        ij = np.clip(np.floor((xy - self.low[:2]) / self.cell).astype(np.int64), 0, n - 1)
        cell = ij[:, 1] * n + ij[:, 0]
        by = np.argsort(cell, kind="stable")
        bounds = np.searchsorted(cell[by], np.arange(n * n + 1))
        winding = np.zeros(len(near), np.int64)
        for c in np.flatnonzero(np.diff(bounds)):
            faces = self.order[self.starts[c] : self.starts[c + 1]]
            if not len(faces):
                continue
            who = by[bounds[c] : bounds[c + 1]]
            step = max(1, CHUNK // len(faces))
            for s in range(0, len(who), step):
                part = who[s : s + step]
                winding[part] = self._crossed(xy[part], z[part], faces)
        found[near] = winding > 0
        return found

    def _crossed(self, xy: np.ndarray, z: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """The winding of rays up from points (k, 2 and k) through some faces."""
        t = self.crossing[faces]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        px, py = xy[:, None, 0], xy[:, None, 1]

        def edge(p, q):
            return (q[:, 0] - p[:, 0]) * (py - p[:, 1]) - (q[:, 1] - p[:, 1]) * (px - p[:, 0])

        e0, e1, e2 = edge(a, b), edge(b, c), edge(c, a)
        area = e0 + e1 + e2
        hit = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        hit &= area != 0
        safe = np.where(area == 0, 1.0, area)
        height = (e1 * a[:, 2] + e2 * b[:, 2] + e0 * c[:, 2]) / safe
        # from a little below and a little above the point: on a top or a bottom
        # face (a solid up to the domain's top or bottom) it is in the solid
        below = ((hit & (height > z[:, None] - self.reach)) * self.facing[faces]).sum(axis=1)
        above = ((hit & (height > z[:, None] + self.reach)) * self.facing[faces]).sum(axis=1)
        return np.maximum(below, above)


def without_solids(mesh: SurfaceMesh, solids: Solids) -> SurfaceMesh:
    """``mesh`` without its parts in ``solids``: cut where its edges enter them (found
    by halving along each edge), its data interpolated to the cut, its outlines (a
    slice's grid lines) ending there."""
    if not mesh.n_points or not mesh.n_triangles or not len(solids.triangles):
        return mesh
    points = np.asarray(mesh.points, np.float64)
    inside = solids.inside(points)
    if not inside.any():
        return mesh

    def crossing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        start, end = points[a], points[b]
        first = inside[a]
        low, high = np.zeros(len(a)), np.ones(len(a))
        for _ in range(HALVINGS):
            middle = 0.5 * (low + high)
            same = solids.inside(start + middle[:, None] * (end - start)) == first
            low = np.where(same, middle, low)
            high = np.where(same, high, middle)
        return 0.5 * (low + high)

    return clip_surface(mesh, np.where(inside, -1.0, 1.0), 0.0, "above", crossing)
