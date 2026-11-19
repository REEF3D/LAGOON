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
from dataclasses import replace

import numpy as np

from lagoonviz.core.data.mesh import SurfaceMesh, clip_surface

#: point-triangle pairs tested at once (memory: a few dozen bytes each)
CHUNK = 2_000_000
#: halvings of an edge to find where it enters a solid (to a 10⁻⁷ of it)
HALVINGS = 24
#: points of each piece of a free surface looked at first for a sheet on a solid
SAMPLES = 16


class Solids:
    """Closed triangle surfaces (m, 3, 3), for inside tests.

    A solid that faces out as it should (each edge of its triangles run one way by
    one and the other way by the other, its volume positive; an STL, a body) counts
    by the faces' winding, so solids that overlap stay inside. One that does not
    (DIVEMesh's REEF3D_Solid.vtp: every flat face of a box turned up, its bottom as
    its top) counts by the number of its faces a ray crosses, odd inside: by the
    winding, under a deck was in the deck."""

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
        turn, odd = self._oriented(t)
        self.crossing = t[flat]
        # +1 up (a top), -1 down; 0 for a solid counted by its crossings
        self.facing = (np.sign(normal[flat, 2]) * turn[flat]).astype(np.int64)
        #: the solid counted by its crossings each face is of (-1: by the winding),
        #: as a random 64-bit tag (0: none): the tags of the faces a ray crosses,
        #: XORed, are 0 when it crosses each solid an even number of times
        self.parity = odd[flat]
        tags = np.random.default_rng(20261119).integers(1, 2**63, max(int(odd.max()) + 1, 1))
        self.tags = np.where(self.parity >= 0, tags.astype(np.uint64)[self.parity], 0).astype(
            np.uint64
        )
        self._bins(t[~flat])

    def _oriented(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per triangle: +1 or -1 to face out (its solid faces in: -1), 0 when its
        solid is not oriented; and that solid's number among those (-1 for the
        others). A solid: triangles joined by their corners."""
        count = len(t)
        if not count:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        corners = np.round(t.reshape(-1, 3) / (1e-9 * self.extent))
        _, ids = np.unique(corners, axis=0, return_inverse=True)
        ids = ids.reshape(-1, 3).astype(np.int64)
        n = int(ids.max()) + 1
        edges = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
        graph = coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])), (n, n))
        _, of_point = connected_components(graph, directed=False)
        solid = of_point[ids[:, 0]]
        solids = int(solid.max()) + 1
        # oriented: every edge run as often one way as the other within its solid
        key = np.minimum(edges[:, 0], edges[:, 1]) * n + np.maximum(edges[:, 0], edges[:, 1])
        way = np.where(edges[:, 0] < edges[:, 1], 1, -1)
        keys, where = np.unique(key, return_inverse=True)
        balance = np.bincount(where.reshape(-1), weights=way, minlength=len(keys))
        unbalanced_key = np.abs(balance) > 0.5
        edge_solid = np.tile(solid, 3)
        bad = np.zeros(solids, bool)
        bad[edge_solid[unbalanced_key[where.reshape(-1)]]] = True
        volume = np.bincount(
            solid,
            weights=np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])),
            minlength=solids,
        )
        turn = np.where(bad[solid], 0, np.where(volume[solid] < 0, -1, 1)).astype(np.int64)
        number = np.full(solids, -1, np.int64)
        number[bad] = np.arange(int(bad.sum()))
        return turn, number[solid]

    def _bins(self, upright: np.ndarray) -> None:
        """The faces a vertical ray can cross, binned by their plan (CSR by cell), and
        the cells where the solids' sides (``upright``, the faces that are not
        crossed) stand."""
        t = self.crossing
        count = len(t)
        n = max(1, min(256, int(math.sqrt(max(count, 1)))))
        self.cells = n
        span = np.maximum(self.high[:2] - self.low[:2], 1e-12)
        self.cell = span / n
        if not count:
            self.order = np.zeros(0, np.int64)
            self.starts = np.zeros(n * n + 1, np.int64)
            self.sides = np.zeros((n, n), bool)
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
        # the cells with a side of a solid (in plan, a line) and those round them:
        # only there the ray's start aside matters (see inside)
        sides = np.zeros((n + 2, n + 2), bool)
        edges = upright[..., :2]
        e_lo = np.floor((edges.min(axis=1) - self.low[:2]) / self.cell).astype(np.int64)
        e_hi = np.floor((edges.max(axis=1) - self.low[:2]) / self.cell).astype(np.int64)
        e_lo, e_hi = np.clip(e_lo, 0, n - 1), np.clip(e_hi, 0, n - 1)
        wide = e_hi - e_lo + 1
        each = wide[:, 0] * wide[:, 1]
        edge = np.repeat(np.arange(len(edges)), each)
        k = np.arange(each.sum()) - np.repeat(np.cumsum(each) - each, each)
        i = e_lo[edge, 0] + k % wide[edge, 0]
        j = e_lo[edge, 1] + k // wide[edge, 0]
        for di in (0, 1, 2):
            for dj in (0, 1, 2):
                sides[i + di, j + dj] = True
        self.sides = sides[1:-1, 1:-1]

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Whether each point (n, 3) lies in the solids (on their faces too): the
        faces crossed by a ray straight up, tops counted +1 and bottoms -1 (in a
        solid: 1; in two that overlap: 2; outside: 0).

        The ray starts a little aside, off edges and corners: once to one side and
        once to the other, and the point is in when either is. On a side face of a
        solid (a domain's side wall where a solid ends, the beach of a flume reaching
        both walls) one of them lies in it: with one side alone, the solid was cut
        from the volume's outside at one wall and left at the other."""
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
        inside = self._winding(points[near, :2] + self.shift, points[near, 2]) > 0
        found[near[inside]] = True
        # the other side only for those not in yet, near a solid's side in plan
        again = near[~inside]
        n = self.cells
        ij = np.clip(np.floor((points[again, :2] - self.low[:2]) / self.cell), 0, n - 1)
        ij = ij.astype(np.int64)
        again = again[self.sides[ij[:, 0], ij[:, 1]]]
        if len(again):
            found[again] = self._winding(points[again, :2] - self.shift, points[again, 2]) > 0
        return found

    def _winding(self, xy: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The winding of rays straight up from points (k, 2 and k)."""
        n = self.cells
        ij = np.clip(np.floor((xy - self.low[:2]) / self.cell).astype(np.int64), 0, n - 1)
        cell = ij[:, 1] * n + ij[:, 0]
        by = np.argsort(cell, kind="stable")
        bounds = np.searchsorted(cell[by], np.arange(n * n + 1))
        winding = np.zeros(len(xy), np.int64)
        for c in np.flatnonzero(np.diff(bounds)):
            faces = self.order[self.starts[c] : self.starts[c + 1]]
            if not len(faces):
                continue
            who = by[bounds[c] : bounds[c + 1]]
            step = max(1, CHUNK // len(faces))
            for s in range(0, len(who), step):
                part = who[s : s + step]
                winding[part] = self._crossed(xy[part], z[part], faces)
        return winding

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
        facing = self.facing[faces]
        tag = self.tags[faces]
        counted = bool(tag.any())
        found = []
        for offset in (-self.reach, self.reach):
            over = hit & (height > z[:, None] + offset)
            winding = (over * facing).sum(axis=1)
            if counted:  # solids that are not oriented: an odd number crossed
                odd = np.bitwise_xor.reduce(np.where(over, tag, np.uint64(0)), axis=1) != 0
                winding = np.where(odd, np.maximum(winding, 1), winding)
            found.append(winding)
        return np.maximum(found[0], found[1])


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


def without_films(mesh: SurfaceMesh, solids: Solids, band: float) -> SurfaceMesh:
    """A free surface (CFD's level set at its zero) without the sheets that lie on a
    solid's top and nowhere else: each piece of the surface (its triangles joined
    by their corners) all of whose points have the solid less than ``band`` below
    them. They are not water: REEF3D keeps the level set in the cells of a solid
    (a deck over the waves has the water of an earlier wave in it), and its zero
    lies in the solid or a cell above its top. Water that is joined to the rest of
    the surface stays (a wave over the deck, the swash on a beach), and so does a
    body of water thicker than ``band`` (the interface band at the smallest cells:
    a sheet thinner than that the level set cannot hold)."""
    if not mesh.n_points or not mesh.n_triangles or not len(solids.triangles) or band <= 0:
        return mesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    points = np.asarray(mesh.points, np.float64)
    t = np.asarray(mesh.triangles, np.int64).reshape(-1, 3)
    # the corners of different pieces of the volume at one place are one
    extent = max(float(np.ptp(points, axis=0).max()), 1e-12)
    _, same = np.unique(np.round(points / (1e-7 * extent)), axis=0, return_inverse=True)
    same = same.reshape(-1)
    joined = same[t]
    count = int(same.max()) + 1
    edges = np.concatenate([joined[:, [0, 1]], joined[:, [1, 2]]])
    graph = coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])), (count, count))
    _, piece = connected_components(graph, directed=False)
    piece_of = piece[same]  # each point's piece
    pieces = int(piece.max()) + 1
    used = np.zeros(len(points), bool)
    used[t.ravel()] = True
    down = np.array([0.0, 0.0, band])

    def away_of(which: np.ndarray) -> np.ndarray:
        """The pieces with a point of ``which`` away from the solids' tops."""
        off = which[~solids.inside(points[which] - down)]
        return np.bincount(piece_of[off], minlength=pieces) > 0

    # a piece is a sheet on a solid when none of its points is away from one: a few
    # points of each first (the sea is away at once), then all of the rest
    order = np.flatnonzero(used)
    order = order[np.argsort(piece_of[order], kind="stable")]
    starts = np.searchsorted(piece_of[order], np.arange(pieces + 1))
    size = np.diff(starts)
    some = starts[:-1, None] + (np.arange(SAMPLES)[None] * size[:, None]) // SAMPLES
    some = np.unique(order[some[size > 0].ravel()])
    away = away_of(some)
    rest = order[~away[piece_of[order]]]
    if len(rest):
        away |= away_of(rest)
    sheet = ~away & (size > 0)
    if not sheet.any():
        return mesh
    kept = t[~sheet[piece_of[t[:, 0]]]]
    used[:] = False
    used[kept.ravel()] = True
    new = np.full(len(points), -1, np.int64)
    new[used] = np.arange(int(used.sum()))

    def lines(pairs):
        if pairs is None:
            return None
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        pairs = pairs[used[pairs].all(axis=1)]
        return new[pairs].astype(np.uint32)

    return replace(
        mesh,
        points=mesh.points[used],
        triangles=new[kept].astype(np.asarray(mesh.triangles).dtype),
        point_data={name: np.asarray(values)[used] for name, values in mesh.point_data.items()},
        normals=None if mesh.normals is None else mesh.normals[used],
        segments=lines(mesh.segments),
        outlines=lines(mesh.outlines),
    )
