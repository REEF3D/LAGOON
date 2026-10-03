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

from dataclasses import dataclass, field

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
    """

    a: np.ndarray  # (k,) first node of the cut edge of each output point
    b: np.ndarray  # (k,) second node
    t: np.ndarray  # (k,) weight: point = a + t (b - a)
    triangles: np.ndarray  # (m, 3) uint32

    @property
    def n_points(self) -> int:
        return int(self.a.size)

    def apply(self, points: np.ndarray, point_data: dict[str, np.ndarray]) -> SurfaceMesh:
        t = self.t[:, None]
        cut = points[self.a] + t * (points[self.b] - points[self.a])
        data = {}
        for name, values in point_data.items():
            va, vb = values[self.a], values[self.b]
            data[name] = (va + (t if va.ndim == 2 else self.t) * (vb - va)).astype(np.float32)
        return SurfaceMesh(cut.astype(np.float32), self.triangles, data)


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
    da, db = distance[a], distance[b]
    denominator = np.where(da == db, 1.0, da - db)
    t = da / denominator
    return CutPlan(a, b, t, np.ascontiguousarray(np.concatenate(triangles), dtype=np.uint32))


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
