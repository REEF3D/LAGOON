# SPDX-License-Identifier: GPL-3.0-or-later
"""Plain-NumPy meshes and the geometry steps between files and the renderer.

* :class:`SurfaceMesh` is what every renderer draws: points, triangles and point
  data. It is independent of any GPU or plotting library.
* :func:`triangulate` turns VTK polygons (REEF3D writes triangles; ParaView may
  write quads or larger polygons) into triangles.
* :class:`PieceMerger` joins the MPI pieces of one time step into one mesh, merging
  the points that neighbouring ranks both write. The merge plan depends only on
  the topology, so it is computed once per dataset and reused for every time step.
* :func:`boundary_faces` finds the outer surface of a volume mesh.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from lagoonviz.core.data.vtkxml import (
    VTK_HEXAHEDRON,
    VTK_QUAD,
    VTK_TETRA,
    VTK_TRIANGLE,
    VTK_WEDGE,
    Piece,
)


@dataclass
class SurfaceMesh:
    """A triangle surface with point data. All renderers draw this."""

    points: np.ndarray  # (n, 3) float32
    triangles: np.ndarray  # (m, 3) int32 (pygfx and VTK both use 32 bit indices)
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    #: (n, 3) float32 unit vertex normals; computed by the reader (in a worker thread)
    normals: np.ndarray | None = None
    #: (k, 2) uint32 line segments (mooring lines, rods); drawn as lines
    segments: np.ndarray | None = None
    #: (k, 2) uint32 point pairs of lines drawn over the surface (AMR patch borders)
    outlines: np.ndarray | None = None

    def with_normals(self) -> SurfaceMesh:
        if self.normals is None:
            self.normals = vertex_normals(self.points, self.triangles)
        return self

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_points == 0:
            return np.zeros(3, np.float32), np.zeros(3, np.float32)
        return self.points.min(axis=0), self.points.max(axis=0)

    def sample(
        self, triangle: int, weights: tuple[float, float, float]
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Point and values inside a triangle (barycentric weights), e.g. a mouse pick.

        Vector arrays give their magnitude and components (``velocity``,
        ``velocity.x``, ...), like the colour-by choices.
        """
        nodes = self.triangles[int(triangle)].astype(np.int64)
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum() if w.sum() > 0 else np.full(3, 1 / 3)
        point = (self.points[nodes].astype(np.float64) * w[:, None]).sum(axis=0)
        values: dict[str, float] = {}
        for name, array in self.point_data.items():
            corner = array[nodes].astype(np.float64)
            if corner.ndim == 1:
                values[name] = float(corner @ w)
            else:
                vector = (corner * w[:, None]).sum(axis=0)
                values[name] = float(np.linalg.norm(vector))
                for axis, component in zip("xyz", vector, strict=False):
                    values[f"{name}.{axis}"] = float(component)
        return point, values

    def scalar(self, variable: str) -> np.ndarray | None:
        """Values to colour by: ``"eta"``, ``"velocity"`` (magnitude) or ``"velocity.x"``."""
        return scalar_values(self.point_data, variable)


def scalar_values(point_data: dict[str, np.ndarray], variable: str) -> np.ndarray | None:
    """One float32 value per point for a variable name, or ``None`` for ``"none"``."""
    if not variable or variable == "none":
        return None
    name, _, component = variable.partition(".")
    if name not in point_data:
        raise KeyError(f"no variable {name!r}; available: {', '.join(point_data) or 'none'}")
    values = point_data[name]
    if values.ndim == 1:
        return values.astype(np.float32, copy=False)
    if component:
        index = "xyz".find(component)
        if index < 0 or index >= values.shape[1]:
            raise KeyError(f"no component {component!r} of {name!r}")
        return np.ascontiguousarray(values[:, index], dtype=np.float32)
    return np.linalg.norm(values, axis=1).astype(np.float32)


def variable_names(point_data_components: dict[str, int]) -> list[str]:
    """Colour-by choices for arrays with the given number of components."""
    names: list[str] = []
    for name, components in point_data_components.items():
        names.append(name)
        if components == 3:
            names += [f"{name}.x", f"{name}.y", f"{name}.z"]
    return names


# ----------------------------------------------------------------- polygons
def triangulate(connectivity: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Fan-triangulate VTK polygons given as connectivity and end offsets."""
    if offsets.size == 0:
        return np.zeros((0, 3), np.int64)
    starts = np.concatenate(([0], offsets[:-1]))
    sizes = offsets - starts
    if np.all(sizes == 3):
        return connectivity.reshape(-1, 3)
    parts = []
    for size in np.unique(sizes):
        if size < 3:
            continue
        first = starts[sizes == size]
        for k in range(1, size - 1):
            parts.append(
                np.stack(
                    [connectivity[first], connectivity[first + k], connectivity[first + k + 1]],
                    axis=1,
                )
            )
    return np.concatenate(parts) if parts else np.zeros((0, 3), np.int64)


def polygon_edges(connectivity: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """The edges of VTK polygons (each once): the grid lines of a surface, (k, 2)."""
    if offsets.size == 0:
        return np.zeros((0, 2), np.uint32)
    starts = np.concatenate(([0], offsets[:-1])).astype(np.int64)
    sizes = offsets - starts
    parts = []
    for size in np.unique(sizes):
        if size < 2:
            continue
        first = starts[sizes == size]
        nodes = connectivity[first[:, None] + np.arange(size)]
        parts.append(np.stack([nodes, np.roll(nodes, -1, axis=1)], axis=2).reshape(-1, 2))
    return unique_edges(np.concatenate(parts) if parts else np.zeros((0, 2), np.int64))


def grid_edges(triangles: np.ndarray, points: np.ndarray, across: bool = True) -> np.ndarray:
    """The grid lines of a surface made of triangles, two per grid cell (as REEF3D
    writes them, as LAGOON splits a grid's cells): their edges without the cells'
    diagonals (an edge that is the longest of both triangles sharing it; measured
    across, in x and y, with ``across``, else in 3D), each once."""
    triangles = np.asarray(triangles, np.int64)
    if not len(triangles):
        return np.zeros((0, 2), np.uint32)
    p = np.asarray(points, np.float64)
    p = p[:, :2] if across else p
    edges = np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=2)  # (m, 3, 2)
    length = np.linalg.norm(p[edges[..., 0]] - p[edges[..., 1]], axis=2)
    span = int(triangles.max()) + 1
    longest = edges[np.arange(len(triangles)), length.argmax(axis=1)]
    keys = _edge_keys(longest, span)
    keys.sort()
    twice = keys[1:][keys[1:] == keys[:-1]]  # diagonals: the longest edge of two triangles
    every = np.unique(_edge_keys(edges.reshape(-1, 2), span))
    if len(twice):
        every = every[~np.isin(every, twice, assume_unique=False)]
    return np.stack([every // span, every % span], axis=1).astype(np.uint32)


def _edge_keys(edges: np.ndarray, span: int) -> np.ndarray:
    """One int64 per edge, whichever way round (sorting rows of pairs is slow)."""
    a, b = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    return np.minimum(a, b) * span + np.maximum(a, b)


def face_edges(faces: np.ndarray) -> np.ndarray:
    """The edges of (k, 4) faces (triangles repeat their last node), each once."""
    faces = np.asarray(faces, np.int64)
    if not len(faces):
        return np.zeros((0, 2), np.uint32)
    edges = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)
    return unique_edges(edges[edges[:, 0] != edges[:, 1]])


def unique_edges(edges: np.ndarray) -> np.ndarray:
    """Point pairs each once, whichever way round, (k, 2) uint32."""
    if not len(edges):
        return np.zeros((0, 2), np.uint32)
    edges = np.asarray(edges, np.int64)
    span = int(edges.max()) + 1
    keys = np.unique(_edge_keys(edges, span))
    return np.stack([keys // span, keys % span], axis=1).astype(np.uint32)


# ----------------------------------------------------------------- volumes
# Faces of the linear cells REEF3D writes, in VTK's node order. Faces are listed
# with outward normals (right-hand rule) so the extracted surface shades correctly.
CELL_FACES: dict[int, tuple[int, list[tuple[int, ...]]]] = {
    VTK_HEXAHEDRON: (
        8,
        [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    ),
    VTK_WEDGE: (6, [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]),
    VTK_TETRA: (4, [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)]),
    VTK_QUAD: (4, [(0, 1, 2, 3)]),
    VTK_TRIANGLE: (3, [(0, 1, 2)]),
}


def outer_faces(connectivity: np.ndarray, offsets: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Faces that belong to exactly one cell, as (k, 4) node ids.

    Triangular faces repeat their last node. Faces are compared by their sorted node
    ids packed into two 64-bit keys, which keeps memory low on large grids.
    """
    starts = np.concatenate(([0], offsets[:-1])).astype(np.int64)
    blocks: list[np.ndarray] = []
    for cell_type in np.unique(types):
        if int(cell_type) not in CELL_FACES:
            continue
        n_nodes, faces = CELL_FACES[int(cell_type)]
        first = starts[types == cell_type]
        nodes = connectivity[first[:, None] + np.arange(n_nodes)].astype(np.int64)
        for face in faces:
            padded = face if len(face) == 4 else (*face, face[-1])
            blocks.append(nodes[:, list(padded)])
    if not blocks:
        return np.zeros((0, 4), np.int64)
    faces = np.concatenate(blocks)
    del blocks
    key = np.sort(faces, axis=1)
    high = (key[:, 0] << 32) | key[:, 1]
    low = (key[:, 2] << 32) | key[:, 3]
    del key
    order = np.lexsort((low, high))
    high, low = high[order], low[order]
    same_as_next = (high[1:] == high[:-1]) & (low[1:] == low[:-1])
    shared = np.zeros(order.size, bool)
    shared[1:] |= same_as_next
    shared[:-1] |= same_as_next
    return faces[order[~shared]]


def faces_to_triangles(faces: np.ndarray) -> np.ndarray:
    """Split (k, 4) faces (triangles repeat their last node) into triangles."""
    is_triangle = faces[:, 2] == faces[:, 3]
    quad = faces[~is_triangle]
    return np.concatenate([faces[is_triangle][:, :3], quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])


def boundary_faces(connectivity: np.ndarray, offsets: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Triangles of the faces that belong to exactly one cell (the outer surface)."""
    return faces_to_triangles(outer_faces(connectivity, offsets, types))


@dataclass(frozen=True)
class VolumePlan:
    """Outer surface of a volume split into MPI pieces, planned piece by piece.

    Each piece's outer faces are found on its own (small memory), only their points
    are merged across pieces, and faces between two pieces (which appear twice
    after merging) are dropped. Applying the plan to a frame reads nothing but the
    points and arrays of the surface.
    """

    sizes: tuple[int, ...]
    take: tuple[np.ndarray, ...]  # per piece: local ids of the surface points
    first: np.ndarray  # stacked surface point -> merged point (unique, in order)
    triangles: np.ndarray  # (m, 3) uint32 on merged points

    @classmethod
    def build(cls, pieces: list[Piece], decimals: int = 5) -> VolumePlan:
        takes, local_faces = [], []
        for piece in pieces:
            faces = outer_faces(piece.connectivity, piece.offsets, piece.types)
            used, compact = np.unique(faces, return_inverse=True)
            takes.append(used)
            local_faces.append(compact.reshape(faces.shape))
        stacked = np.concatenate([p.points[t] for p, t in zip(pieces, takes, strict=True)])
        rounded = np.round(stacked.astype(np.float64), decimals)
        _, first, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        inverse = rank[inverse]
        shift = np.cumsum([0] + [t.size for t in takes[:-1]])
        faces = np.concatenate([inverse[f + s] for f, s in zip(local_faces, shift, strict=True)])
        # faces on the boundary between two pieces now appear twice: drop them
        key = np.sort(faces, axis=1)
        _, index, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
        faces = faces[np.sort(index[counts == 1])]
        # points that were only on faces between pieces are not needed any more
        kept, compact = np.unique(faces, return_inverse=True)
        faces = compact.reshape(faces.shape)
        triangles = np.ascontiguousarray(faces_to_triangles(faces), dtype=np.uint32)
        sizes = tuple(p.points.shape[0] for p in pieces)
        return cls(sizes, tuple(takes), first[order][kept], triangles)

    def accepts(self, pieces: list[Piece]) -> bool:
        return tuple(p.points.shape[0] for p in pieces) == self.sizes

    def apply(self, pieces: list[Piece]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        points = np.concatenate([p.points[t] for p, t in zip(pieces, self.take, strict=True)])[
            self.first
        ]
        data = {
            name: np.concatenate(
                [p.point_data[name][t] for p, t in zip(pieces, self.take, strict=True)]
            )[self.first]
            for name in pieces[0].point_data
        }
        return points, data


# ----------------------------------------------------------------- merging
@dataclass(frozen=True)
class MergePlan:
    """How the stacked points of the pieces map to merged points (topology only)."""

    sizes: tuple[int, ...]  # points per piece
    first: np.ndarray  # stacked index of each merged point
    connectivity: np.ndarray  # cells in merged point ids
    offsets: np.ndarray
    types: np.ndarray | None

    def apply(self, pieces: list[Piece]) -> Piece:
        """Merge pieces (read with or without cells) of the sizes this plan was made for."""
        points = np.concatenate([p.points for p in pieces])[self.first]
        data = {
            name: np.concatenate([p.point_data[name] for p in pieces])[self.first]
            for name in pieces[0].point_data
        }
        return Piece(pieces[0].kind, points, self.connectivity, self.offsets, self.types, data)


class PieceMerger:
    """Join the MPI pieces of one time step into a single mesh.

    Neighbouring ranks both write the points on their shared boundary. Merging them
    gives one connected surface (smooth shading, no seams, correct outer faces).
    The plan is keyed on the piece sizes and reused while they stay the same. Plans
    are immutable, so worker threads can apply one while another is being made.

    ``mode`` chooses how points are matched: ``"xy"`` for 2.5D surfaces whose height
    changes in time (free surface, bed), ``"xyz"`` for volumes and fixed geometry.
    """

    def __init__(self, mode: str = "xyz", decimals: int = 5):
        if mode not in ("xy", "xyz"):
            raise ValueError("mode must be 'xy' or 'xyz'")
        self.mode = mode
        self.decimals = decimals
        self.plan: MergePlan | None = None

    def make_plan(self, pieces: list[Piece]) -> MergePlan:
        if any(p.offsets.size == 0 and p.points.shape[0] for p in pieces):
            raise RuntimeError("a merge plan needs pieces read with their cells")
        stacked = np.concatenate([p.points for p in pieces])
        columns = stacked[:, :2] if self.mode == "xy" else stacked
        rounded = np.round(columns.astype(np.float64), self.decimals)
        _, first, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        # keep the merged points in their original order (pieces stay contiguous)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        inverse = rank[inverse]

        shift = np.cumsum([0] + [p.points.shape[0] for p in pieces[:-1]])
        connectivity = np.concatenate(
            [inverse[p.connectivity + s] for p, s in zip(pieces, shift, strict=True)]
        )
        cell_shift = np.cumsum([0] + [int(p.offsets[-1]) if p.offsets.size else 0 for p in pieces])
        offsets = np.concatenate(
            [p.offsets + s for p, s in zip(pieces, cell_shift[:-1], strict=True)]
        )
        types = None
        if pieces[0].types is not None:
            types = np.concatenate([p.types for p in pieces])
        plan = MergePlan(self._sizes(pieces), first[order], connectivity, offsets, types)
        self.plan = plan
        return plan

    def reset(self) -> None:
        """Forget the plan (e.g. when a run was restarted with another grid)."""
        self.plan = None

    @staticmethod
    def _sizes(pieces: list[Piece]) -> tuple[int, ...]:
        return tuple(p.points.shape[0] for p in pieces)

    def accepts(self, pieces: list[Piece]) -> bool:
        """True if a plan exists for pieces of these sizes (cells need not be read)."""
        plan = self.plan
        return plan is not None and self._sizes(pieces) == plan.sizes

    def merge(self, pieces: list[Piece]) -> Piece:
        """Merge one time step. Pieces need cells unless :meth:`accepts` them."""
        plan = self.plan if self.accepts(pieces) else self.make_plan(pieces)
        return plan.apply(pieces)


def exaggerate_normals(normals: np.ndarray, factor: float) -> np.ndarray:
    """Normals of a surface whose heights are scaled by ``factor`` about a level.

    Scaling z by s transforms normals by diag(s, s, 1) (the inverse transpose,
    up to a factor), so the normals need not be recomputed from the geometry.
    """
    scaled = normals * np.array([factor, factor, 1.0], np.float32)
    length = np.linalg.norm(scaled, axis=1, keepdims=True)
    return (scaled / np.where(length > 0, length, 1.0)).astype(np.float32)


def grid_triangles(ny: int, nx: int, missing: np.ndarray | None = None) -> np.ndarray:
    """Two triangles (m, 3) per cell of a structured grid of ny x nx points (x
    fastest), leaving out cells with a ``missing`` corner (a (ny, nx) mask)."""
    if ny < 2 or nx < 2:
        return np.zeros((0, 3), np.uint32)
    base = (
        np.arange(ny - 1, dtype=np.uint32)[:, None] * np.uint32(nx)
        + np.arange(nx - 1, dtype=np.uint32)[None, :]
    )
    if missing is not None:
        m = np.asarray(missing, bool)
        gone = m[:-1, :-1] | m[:-1, 1:] | m[1:, :-1] | m[1:, 1:]
        base = base[~gone] if gone.any() else base.ravel()
    else:
        base = base.ravel()
    n = len(base)
    out = np.empty((2 * n, 3), np.uint32)
    out[:n, 0] = base
    np.add(base, 1, out=out[:n, 1])
    np.add(base, nx + 1, out=out[:n, 2])
    out[n:, 0] = base
    out[n:, 1] = out[:n, 2]
    np.add(base, nx, out=out[n:, 2])
    return out


def grid_points(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Points (ny * nx, 3) float32 of a surface of heights z (ny, nx) over a
    rectilinear grid, filled in place (no coordinate grids made first)."""
    z = np.asarray(z)
    out = np.empty((*z.shape, 3), np.float32)
    out[..., 0] = np.asarray(x, np.float32)[None, :]
    out[..., 1] = np.asarray(y, np.float32)[:, None]
    out[..., 2] = z
    return out.reshape(-1, 3)


def grid_normals(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit normals (ny, nx, 3) of a surface given as heights z (ny, nx) on a
    rectilinear grid (x along the columns, y along the rows), pointing up: from the
    slopes, with central differences (one-sided at the edges). In single precision,
    a few passes over the grid, large grids in bands on several threads: a surface
    of 10 million points takes a fraction of a second."""
    z = np.asarray(z, np.float32)
    ny, nx = z.shape
    out = np.empty((ny, nx, 3), np.float32)
    if ny < 2 or nx < 2:
        out[...] = (0.0, 0.0, 1.0)
        return out
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    inv_x = np.empty(nx, np.float32)
    inv_x[1:-1] = 1.0 / (x[2:] - x[:-2])
    inv_x[0], inv_x[-1] = 1.0 / (x[1] - x[0]), 1.0 / (x[-1] - x[-2])
    inv_y = np.empty(ny, np.float32)
    inv_y[1:-1] = 1.0 / (y[2:] - y[:-2])
    inv_y[0], inv_y[-1] = 1.0 / (y[1] - y[0]), 1.0 / (y[-1] - y[-2])
    bands = _bands(ny, nx)
    if len(bands) > 1:
        list(_pool().map(lambda band: _normals_band(z, inv_x, inv_y, out, *band), bands))
    else:
        _normals_band(z, inv_x, inv_y, out, 0, ny)
    return out


def _normals_band(z, inv_x, inv_y, out, r0: int, r1: int) -> None:
    """:func:`grid_normals` for rows r0..r1 (the rows around read from ``z``)."""
    ny = z.shape[0]
    band = out[r0:r1]
    zb = z[r0:r1]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        sx = band[..., 0]  # dz/dx, then the normal's x
        np.subtract(zb[:, 2:], zb[:, :-2], out=sx[:, 1:-1])
        sx[:, 0] = zb[:, 1] - zb[:, 0]
        sx[:, -1] = zb[:, -1] - zb[:, -2]
        sx *= inv_x
        sy = band[..., 1]  # dz/dy, then the normal's y
        below = np.maximum(np.arange(r0, r1) - 1, 0)
        above = np.minimum(np.arange(r0, r1) + 1, ny - 1)
        lo, hi = max(r0 - 1, 0), min(r1 + 1, ny)
        rows = z[lo:hi]
        np.subtract(rows[above - lo], rows[below - lo], out=sy)
        sy *= inv_y[r0:r1, None]
        scale = band[..., 2]  # 1 / |(-dz/dx, -dz/dy, 1)|, then the normal's z
        np.multiply(sx, sx, out=scale)
        scale += sy * sy
        scale += 1.0
        np.sqrt(scale, out=scale)
        np.divide(1.0, scale, out=scale)
        sx *= -scale
        sy *= -scale
    bad = ~np.isfinite(scale)
    if bad.any():
        band[bad] = (0.0, 0.0, 1.0)  # dry points, holes


#: points from which a grid is worked on in bands on several threads
BAND_POINTS = 1_000_000
_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(min(8, os.cpu_count() or 2), thread_name_prefix="mesh")
    return _POOL


def _bands(ny: int, nx: int) -> list[tuple[int, int]]:
    """Row bands of a large grid, one per thread (one band for a small grid)."""
    count = min(8, os.cpu_count() or 2) if ny * nx >= BAND_POINTS else 1
    edges = np.linspace(0, ny, count + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def vertex_normals(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, (n, 3) float32 and unit length."""
    normals = np.zeros(positions.shape, np.float64)
    if triangles.size:
        tri = triangles.astype(np.intp, copy=False)
        p = positions.astype(np.float32, copy=False)
        p0 = p[tri[:, 0]]
        face = np.cross(p[tri[:, 1]] - p0, p[tri[:, 2]] - p0)
        n = positions.shape[0]
        for c in range(3):
            weights = face[:, c]
            normals[:, c] = (
                np.bincount(tri[:, 0], weights, n)
                + np.bincount(tri[:, 1], weights, n)
                + np.bincount(tri[:, 2], weights, n)
            )
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(length > 0, length, 1.0)
    normals[length[:, 0] == 0] = (0.0, 0.0, 1.0)
    return normals.astype(np.float32)


# ----------------------------------------------------------------- slicing
#: linear cells split into tetrahedra (VTK node order)
CELL_TETS: dict[int, tuple[int, list[tuple[int, int, int, int]]]] = {
    VTK_HEXAHEDRON: (
        8,
        [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)],
    ),
    VTK_WEDGE: (6, [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]),
    VTK_TETRA: (4, [(0, 1, 2, 3)]),
}
#: tetrahedron edges as vertex pairs
_TET_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
#: marching tetrahedra: for each sign case, the cut polygon as edge indices
_CUT_CASES: dict[int, tuple[int, ...]] = {}
for _case, _poly in (
    (1, (0, 1, 2)),
    (2, (0, 4, 3)),
    (4, (1, 3, 5)),
    (8, (2, 5, 4)),
    (3, (1, 3, 4, 2)),
    (5, (0, 3, 5, 2)),
    (6, (0, 1, 5, 4)),
):
    _CUT_CASES[_case] = _poly
    _CUT_CASES[15 - _case] = _poly


@dataclass(frozen=True)
class CutPlan:
    """Where a plane cuts a volume mesh: for every cut point, an edge and a weight.

    Planning finds the crossing cells; applying interpolates points and data. For a
    plane whose position along its axis does not change between frames (vertical
    slices of REEF3D's σ-grids, any slice of a fixed grid) one plan serves all
    frames, and a frame costs only the interpolation.

    ``line_a``, ``line_b``, ``line_t``: the cells' edges in the plane (the grid
    lines of a slice), two points per segment, given the same way; applying puts
    them after the surface's points and their pairs in the mesh's ``outlines``.
    """

    a: np.ndarray  # (k,) first node of the cut edge of each output point
    b: np.ndarray  # (k,) second node
    t: np.ndarray  # (k,) weight: point = a + t (b - a)
    triangles: np.ndarray  # (m, 3) uint32
    line_a: np.ndarray | None = None  # (2q,) the grid lines' points, as a, b, t
    line_b: np.ndarray | None = None
    line_t: np.ndarray | None = None

    @property
    def n_points(self) -> int:
        return int(self.a.size)

    def apply(self, points: np.ndarray, point_data: dict[str, np.ndarray]) -> SurfaceMesh:
        a, b, weight = self.a, self.b, self.t
        lines = self.line_a is not None and self.line_a.size > 0
        if lines:
            a = np.concatenate([a, self.line_a])
            b = np.concatenate([b, self.line_b])
            weight = np.concatenate([weight, self.line_t])
        t = weight[:, None]
        cut = points[a] + t * (points[b] - points[a])
        data = {}
        for name, values in point_data.items():
            va, vb = values[a], values[b]
            data[name] = (va + (t if va.ndim == 2 else weight) * (vb - va)).astype(np.float32)
        outlines = None
        if self.line_a is not None:  # a slice: its grid lines (none in a piece it misses)
            count = self.line_a.size
            outlines = (self.n_points + np.arange(count, dtype=np.uint32)).reshape(-1, 2)
        return SurfaceMesh(cut.astype(np.float32), self.triangles, data, outlines=outlines)


def plan_cell_edges(
    distance: np.ndarray, connectivity: np.ndarray, offsets: np.ndarray, types: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid lines where the zero level of ``distance`` (a plane) cuts the cells'
    faces: a segment per face crossed (a face two cells share once), as (a, b, t)
    with two points per segment (see :class:`CutPlan`)."""
    distance = np.asarray(distance, dtype=np.float64)
    starts = np.concatenate(([0], offsets[:-1])).astype(np.int64)
    by_size: dict[int, list[np.ndarray]] = {}
    for cell_type in np.unique(types):
        if int(cell_type) not in CELL_FACES or int(cell_type) in (VTK_QUAD, VTK_TRIANGLE):
            continue
        n_nodes, faces = CELL_FACES[int(cell_type)]  # (the cells' faces, as above)
        first = starts[types == cell_type]
        nodes = connectivity[first[:, None] + np.arange(n_nodes)]
        signs = distance[nodes]
        nodes = nodes[(signs.min(axis=1) <= 0) & (signs.max(axis=1) > 0)]
        for face in faces:
            by_size.setdefault(len(face), []).append(nodes[:, list(face)])
    found_a, found_b = [], []
    for size, groups in by_size.items():
        faces = np.concatenate(groups).astype(np.int64)
        if not len(faces):
            continue
        # a face shared by two cells once
        _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        faces = faces[np.sort(first)]
        above = distance[faces] > 0
        crossed = above != np.roll(above, -1, axis=1)  # edge e: node e to node e + 1
        two = crossed.sum(axis=1) >= 2
        faces, crossed = faces[two], crossed[two]
        if not len(faces):
            continue
        # the first two crossed edges of each face: the segment across it
        order = np.argsort(~crossed, axis=1, kind="stable")[:, :2]
        rows = np.arange(len(faces))[:, None]
        start = faces[rows, order]
        end = faces[rows, (order + 1) % size]
        found_a.append(start.ravel())
        found_b.append(end.ravel())
    if not found_a:
        none = np.zeros(0, np.int64)
        return none, none, np.zeros(0)
    a, b = np.concatenate(found_a), np.concatenate(found_b)
    da, db = distance[a], distance[b]
    return a, b, da / np.where(da == db, 1.0, da - db)


def plan_cut(
    points: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    types: np.ndarray,
    axis: int,
    position: float,
) -> CutPlan:
    """Plan the section of a volume mesh with the plane ``x[axis] = position``.

    Cells are split into tetrahedra and cut with marching tetrahedra. Only cells
    that cross the plane are processed, so a slice of a large grid is cheap.
    """
    distance = points[:, axis].astype(np.float64) - position
    plan = plan_level(distance, connectivity, offsets, types)
    line_a, line_b, line_t = plan_cell_edges(distance, connectivity, offsets, types)
    return replace(plan, line_a=line_a, line_b=line_b, line_t=line_t)


def plan_level(
    distance: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    types: np.ndarray,
    merge: bool = False,
) -> CutPlan:
    """Plan the surface where a field given at the points is zero (marching
    tetrahedra): a plane for slices, ``field - value`` for iso-surfaces.

    With ``merge``, a point shared by neighbouring tetrahedra (the same cut edge) is
    made once, so the surface is connected and shades smoothly.
    """
    distance = np.asarray(distance, dtype=np.float64)
    starts = np.concatenate(([0], offsets[:-1])).astype(np.int64)
    tets: list[np.ndarray] = []
    for cell_type in np.unique(types):
        if int(cell_type) not in CELL_TETS:
            continue
        n_nodes, split = CELL_TETS[int(cell_type)]
        first = starts[types == cell_type]
        nodes = connectivity[first[:, None] + np.arange(n_nodes)]
        signs = distance[nodes]
        crossing = (signs.min(axis=1) <= 0) & (signs.max(axis=1) > 0)
        nodes = nodes[crossing]
        for tet in split:
            tets.append(nodes[:, list(tet)])
    none = np.zeros(0, np.int64)
    if not tets:
        return CutPlan(none, none, np.zeros(0), np.zeros((0, 3), np.uint32))
    tet_nodes = np.concatenate(tets).astype(np.int64)
    d = distance[tet_nodes]
    case = ((d > 0) * np.array([1, 2, 4, 8])).sum(axis=1)
    edges_a, edges_b, triangles = [], [], []
    count = 0
    for code, polygon in _CUT_CASES.items():
        chosen = tet_nodes[case == code]
        n = chosen.shape[0]
        if n == 0:
            continue
        # output point of corner c of polygon i: count + c * n + i
        for edge in polygon:
            edges_a.append(chosen[:, _TET_EDGES[edge, 0]])
            edges_b.append(chosen[:, _TET_EDGES[edge, 1]])
        index = count + np.arange(n)
        fan = [(0, 1, 2)] if len(polygon) == 3 else [(0, 1, 2), (0, 2, 3)]
        for i, j, k in fan:
            triangles.append(np.stack([index + i * n, index + j * n, index + k * n], axis=1))
        count += len(polygon) * n
    if not edges_a:
        return CutPlan(none, none, np.zeros(0), np.zeros((0, 3), np.uint32))
    a = np.concatenate(edges_a)
    b = np.concatenate(edges_b)
    triangles = np.concatenate(triangles)
    if merge and a.size:
        low, high = np.minimum(a, b), np.maximum(a, b)
        keys = low * (int(max(high.max(), 0)) + 1) + high
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        a, b = a[first], b[first]
        triangles = inverse.ravel()[triangles]
        keep = (
            (triangles[:, 0] != triangles[:, 1])
            & (triangles[:, 1] != triangles[:, 2])
            & (triangles[:, 0] != triangles[:, 2])
        )
        triangles = triangles[keep]
    da, db = distance[a], distance[b]
    denominator = np.where(da == db, 1.0, da - db)
    t = da / denominator
    return CutPlan(a, b, t, np.ascontiguousarray(triangles, dtype=np.uint32))


def orient_by_field(plan: CutPlan, points: np.ndarray, field: np.ndarray) -> CutPlan:
    """Turn every triangle of an iso-surface to face the rising field.

    Marching tetrahedra makes triangles in either winding. Averaged vertex normals
    of mixed windings cancel, and the surface shaded in dark and light speckles and
    stripes. Facing one way, they shade smoothly.
    """
    triangles = plan.triangles.astype(np.int64)
    if not len(triangles):
        return plan
    pa, pb = points[plan.a].astype(np.float64), points[plan.b].astype(np.float64)
    rising = (field[plan.b] > field[plan.a])[:, None]
    up = np.where(rising, pb - pa, pa - pb)  # each cut edge, towards the higher value
    cut = pa + plan.t[:, None] * (pb - pa)
    normal = np.cross(
        cut[triangles[:, 1]] - cut[triangles[:, 0]], cut[triangles[:, 2]] - cut[triangles[:, 0]]
    )
    flip = (normal * up[triangles].sum(axis=1)).sum(axis=1) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return CutPlan(plan.a, plan.b, plan.t, np.ascontiguousarray(triangles, dtype=np.uint32))


def clip_surface(mesh: SurfaceMesh, field: np.ndarray, level: float, keep: str) -> SurfaceMesh:
    """The part of a triangle surface where ``field`` is at least (``keep`` "above")
    or at most ("below") ``level``: triangles across the level are cut on their
    edges (points and data interpolated, cut points shared by neighbours)."""
    d = np.asarray(field, np.float64) - float(level)
    if keep == "below":
        d = -d
    triangles = np.asarray(mesh.triangles, np.int64)
    inside = d[triangles] >= 0
    count = inside.sum(axis=1)
    whole = triangles[count == 3]
    partial = triangles[(count == 1) | (count == 2)]
    one = count[(count == 1) | (count == 2)] == 1
    if not len(partial):
        kept, compact = np.unique(whole, return_inverse=True)
        return SurfaceMesh(
            mesh.points[kept],
            compact.reshape(-1, 3).astype(np.uint32),
            {name: values[kept] for name, values in mesh.point_data.items()},
        ).with_normals()
    # the odd corner first (inside of one, outside of two), the winding kept
    flags = inside[(count == 1) | (count == 2)]
    odd = np.where(one, flags.argmax(axis=1), (~flags).argmax(axis=1))
    order = (odd[:, None] + np.arange(3)) % 3
    corners = np.take_along_axis(partial, order, axis=1)
    v0, v1, v2 = corners[:, 0], corners[:, 1], corners[:, 2]
    # the cut points on the edges v0-v1 and v0-v2, each edge once
    a = np.concatenate([v0, v0])
    b = np.concatenate([v1, v2])
    low, high = np.minimum(a, b), np.maximum(a, b)
    keys, first, inverse = np.unique(
        low * (int(high.max()) + 1) + high, return_index=True, return_inverse=True
    )
    a, b = a[first], b[first]
    t = d[a] / np.where(d[a] == d[b], 1.0, d[a] - d[b])
    n = mesh.n_points
    cut01 = n + inverse[: len(v0)]
    cut02 = n + inverse[len(v0) :]
    made = [whole]
    made.append(np.stack([v0, cut01, cut02], axis=1)[one])  # one corner inside
    two = ~one  # two corners inside: a quad, two triangles
    made.append(np.stack([cut01, v1, v2], axis=1)[two])
    made.append(np.stack([cut01, v2, cut02], axis=1)[two])
    triangles = np.concatenate(made)
    w = t[:, None]
    points = np.concatenate(
        [mesh.points, (mesh.points[a] + w * (mesh.points[b] - mesh.points[a])).astype(np.float32)]
    )
    data = {}
    for name, values in mesh.point_data.items():
        va, vb = values[a], values[b]
        mixed = va + (w if va.ndim == 2 else t) * (vb - va)
        data[name] = np.concatenate([values, mixed.astype(values.dtype)])
    kept, compact = np.unique(triangles, return_inverse=True)
    return SurfaceMesh(
        points[kept].astype(np.float32),
        compact.reshape(-1, 3).astype(np.uint32),
        {name: values[kept] for name, values in data.items()},
    ).with_normals()


def cut_cells(
    points: np.ndarray,
    connectivity: np.ndarray,
    offsets: np.ndarray,
    types: np.ndarray,
    point_data: dict[str, np.ndarray],
    axis: int,
    position: float,
) -> SurfaceMesh:
    """The section of a volume mesh with a plane (see :func:`plan_cut`)."""
    plan = plan_cut(points, connectivity, offsets, types, axis, position)
    return plan.apply(points, point_data)


def tubes(
    points: np.ndarray,
    segments: np.ndarray,
    radius: np.ndarray,
    point_data: dict[str, np.ndarray],
    sides: int = 8,
) -> SurfaceMesh:
    """Line segments as round tubes (rod trees, cylinders): a ring of ``sides``
    points at each end, of the radius given at that end; the end points' values
    carried to their rings."""
    segments = np.asarray(segments, dtype=np.int64).reshape(-1, 2)
    if not len(segments):
        return SurfaceMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), {})
    p = np.asarray(points, dtype=np.float64)
    a, b = p[segments[:, 0]], p[segments[:, 1]]
    axis = b - a
    length = np.linalg.norm(axis, axis=1)
    axis = axis / np.where(length > 0, length, 1.0)[:, None]
    # two directions across each segment
    helper = np.where(np.abs(axis[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    u = np.cross(axis, helper)
    u /= np.maximum(np.linalg.norm(u, axis=1), 1e-12)[:, None]
    v = np.cross(axis, u)
    angle = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    ring = (
        np.cos(angle)[None, :, None] * u[:, None, :] + np.sin(angle)[None, :, None] * v[:, None, :]
    )
    r = np.asarray(radius, dtype=np.float64)
    ends = []
    for end, centre in ((0, a), (1, b)):
        ends.append(centre[:, None, :] + r[segments[:, end]][:, None, None] * ring)
    vertices = np.concatenate([ends[0], ends[1]], axis=1).reshape(-1, 3)  # (n * 2 sides, 3)
    n = len(segments)
    base = (np.arange(n) * 2 * sides)[:, None]
    k = np.arange(sides)[None, :]
    k2 = (k + 1) % sides
    a0, a1 = base + k, base + k2
    b0, b1 = base + sides + k, base + sides + k2
    triangles = np.concatenate(
        [
            np.stack([a0, a1, b1], axis=-1).reshape(-1, 3),
            np.stack([a0, b1, b0], axis=-1).reshape(-1, 3),
        ]
    )
    data = {}
    for name, values in point_data.items():
        values = np.asarray(values)
        at_end = [
            np.repeat(values[segments[:, end]], sides, axis=0).reshape(n, sides, *values.shape[1:])
            for end in (0, 1)
        ]
        data[name] = np.concatenate(at_end, axis=1).reshape(n * 2 * sides, *values.shape[1:])
    mesh = SurfaceMesh(vertices.astype(np.float32), triangles.astype(np.uint32), data)
    return mesh.with_normals()
