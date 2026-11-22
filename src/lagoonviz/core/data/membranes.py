# LAGOON — the membranes of NHFLOW (X 330) and the collar of a flexible bag
# Architect: Hans Bihs
"""REEF3D's membranes (X 330, NHFLOW; membrane.dat in the case folder) are written
at their own print interval: REEF3D_NHFLOW_Membrane_VTP/REEF3D-NHFLOW-Membrane-
<membrane>-<output>.vtp, or with P 18 into the store as the particle set
nhflow_membrane (role "membrane"). Each output holds the fabric: its nodes with
velocity, displacement (from the start) and load, its triangles with dp, force,
tension and tag.

A flexible bag whose top edge is a floating pipe ring (membrane.dat
``collar D m EA EI [Cd [Ca]]``, net_membrane_collar.cpp) has that ring's centre
line on the fabric's top edge nodes, but REEF3D writes no pipe. LAGOON draws it: a
tube of diameter D through the nodes that started on the top edge (at or above
``attach z``, default z_top), in their order round the bag."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lagoonviz.core.data.mesh import SurfaceMesh

log = logging.getLogger(__name__)

#: points round the tube's section
SIDES = 16


@dataclass
class Membrane:
    """One membrane of membrane.dat, as far as LAGOON needs it."""

    number: int  # REEF3D's membrane number: its place in membrane.dat, from 0
    shape: str  # box, cylinder, cylcone
    values: tuple[float, ...]
    structure: str = "fixed"
    collar: float | None = None  # the collar's diameter D [m]
    attach: float | None = None  # 'attach z'
    name: str = ""

    @property
    def top(self) -> float:
        """z_top: the last number of every shape."""
        return self.values[-1]

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """x0, x1, y0, y1 of the bag in plan."""
        v = self.values
        if self.shape == "box":
            return v[0], v[1], v[2], v[3]
        xc, yc, r = v[0], v[1], v[2]
        return xc - r, xc + r, yc - r, yc + r

    @property
    def has_collar(self) -> bool:
        """A flexible bag with a collar (REEF3D: collar() is collar and structure 2)."""
        return self.collar is not None and self.collar > 0 and self.structure == "flexible"


@dataclass
class Collar:
    """A collar ring to draw: its diameter, and where its nodes start."""

    membrane: int
    diameter: float
    z: float  # nodes that start at or above this are the ring
    footprint: tuple[float, float, float, float]
    _rings: dict = field(default_factory=dict, repr=False, compare=False)

    def ring(self, start: np.ndarray, spacing: float) -> np.ndarray:
        """The nodes of the ring among the start positions, in order round it."""
        x0, x1, y0, y1 = self.footprint
        margin = 0.1 * max(x1 - x0, y1 - y0) + 2.0 * spacing
        tol = max(0.05 * spacing, 1e-4)
        inside = (
            (start[:, 0] >= x0 - margin)
            & (start[:, 0] <= x1 + margin)
            & (start[:, 1] >= y0 - margin)
            & (start[:, 1] <= y1 + margin)
            & (start[:, 2] >= self.z - tol)
        )
        nodes = np.flatnonzero(inside)
        if len(nodes) < 4:
            return np.zeros(0, np.int64)
        centre = start[nodes].mean(axis=0)
        angle = np.arctan2(start[nodes, 1] - centre[1], start[nodes, 0] - centre[0])
        return nodes[np.argsort(angle, kind="stable")]


def _numbers(words: list[str]) -> tuple[float, ...]:
    out = []
    for word in words:
        try:
            out.append(float(word))
        except ValueError:
            break
    return tuple(out)


def read_membrane_dat(path: str | Path) -> list[Membrane]:
    """The membranes of a membrane.dat, as REEF3D reads them
    (net_interface_membrane_nhflow.cpp): each starts with a 'membrane' line, its
    parameter lines follow; # starts a comment."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return []
    membranes: list[Membrane] = []
    for line in text.splitlines():
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        key, rest = words[0].lower(), words[1:]
        if key == "membrane":
            if not rest:
                continue
            shape = rest[0].lower()
            values = _numbers(rest[1:])
            need = {"box": 6, "cylinder": 5, "cylcone": 6}.get(shape)
            if need is None or len(values) < need:
                log.warning("%s: cannot read %r", path, line.strip())
                continue
            membranes.append(Membrane(len(membranes), shape, values[:need]))
        elif not membranes:
            continue
        elif key == "structure" and rest:
            membranes[-1].structure = rest[0].lower()
        elif key == "collar":
            values = _numbers(rest)
            if values:
                membranes[-1].collar = values[0]
        elif key == "attach":
            values = _numbers(rest)
            if values:
                membranes[-1].attach = values[0]
        elif key == "name" and rest:
            membranes[-1].name = " ".join(rest)
    return membranes


def find_membrane_dat(start: Path, levels: int = 4) -> Path | None:
    """membrane.dat of the case a dataset's folder belongs to (the case folder, or
    one up from an output folder or store)."""
    folder = Path(start)
    for _ in range(levels):
        candidate = folder / "membrane.dat"
        if candidate.is_file():
            return candidate
        if folder.parent == folder:
            break
        folder = folder.parent
    return None


def collars_of(membrane_dat: Path | None) -> list[Collar]:
    """The collar rings of a case's flexible bags."""
    if membrane_dat is None:
        return []
    collars = []
    for m in read_membrane_dat(membrane_dat):
        if m.has_collar:
            z = m.attach if m.attach is not None else m.top
            collars.append(Collar(m.number, float(m.collar), float(z), m.footprint))
    return collars


def ring_tube(
    centres: np.ndarray, radius: float, sides: int = SIDES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A closed tube through the points of a ring, in their order: a circle of
    ``sides`` points round each, square to the ring's direction there (the mean of
    the two segments that meet), so the tube bends without kinks. Its vertices,
    triangles and normals."""
    c = np.asarray(centres, np.float64)
    n = len(c)
    tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)[:, None]
    # across the ring: away from its centre in the plane of the ring, and up
    middle = c.mean(axis=0)
    out = c - middle
    out -= (out * tangent).sum(axis=1)[:, None] * tangent
    flat = np.linalg.norm(out, axis=1) < 1e-9
    if flat.any():  # a ring through its centre: any direction across
        helper = np.where(np.abs(tangent[flat, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
        out[flat] = np.cross(tangent[flat], helper)
    u = out / np.maximum(np.linalg.norm(out, axis=1), 1e-12)[:, None]
    v = np.cross(tangent, u)
    angle = np.linspace(0.0, 2 * np.pi, sides, endpoint=False)
    across = (
        np.cos(angle)[None, :, None] * u[:, None, :] + np.sin(angle)[None, :, None] * v[:, None, :]
    )
    vertices = (c[:, None, :] + radius * across).reshape(-1, 3)
    normals = across.reshape(-1, 3)
    i = np.arange(n)[:, None]
    k = np.arange(sides)[None, :]
    a0 = i * sides + k
    a1 = i * sides + (k + 1) % sides
    b0 = ((i + 1) % n) * sides + k
    b1 = ((i + 1) % n) * sides + (k + 1) % sides
    triangles = np.concatenate(
        [
            np.stack([a0, b1, b0], axis=-1).reshape(-1, 3),
            np.stack([a0, a1, b1], axis=-1).reshape(-1, 3),
        ]
    )
    return vertices.astype(np.float32), triangles.astype(np.uint32), normals.astype(np.float32)


def _spacing(points: np.ndarray, triangles: np.ndarray) -> float:
    """The fabric's typical edge length."""
    if not len(triangles):
        return 0.0
    t = triangles[: min(len(triangles), 2000)].astype(np.int64)
    edges = np.linalg.norm(points[t[:, 1]] - points[t[:, 0]], axis=1)
    edges = edges[edges > 0]
    return float(np.median(edges)) if edges.size else 0.0


def with_collars(mesh: SurfaceMesh, collars: list[Collar]) -> SurfaceMesh:
    """The fabric with the collar rings of its flexible bags as tubes. The rings'
    nodes are found from where they started (the positions less the displacement);
    each tube's vertices carry the values of its ring node."""
    if not collars or mesh.n_points == 0 or "displacement" not in mesh.point_data:
        return mesh
    points = np.asarray(mesh.points, np.float64)
    start = points - np.asarray(mesh.point_data["displacement"], np.float64).reshape(-1, 3)
    spacing = _spacing(start, np.asarray(mesh.triangles))
    parts_points, parts_triangles, parts_normals, rings = [], [], [], []
    offset = mesh.n_points
    for collar in collars:
        key = (mesh.n_points, round(spacing, 9))
        ring = collar._rings.get(key)
        if ring is None:
            ring = collar._rings[key] = collar.ring(start, spacing)
        if len(ring) < 4:
            continue
        vertices, triangles, normals = ring_tube(points[ring], 0.5 * collar.diameter)
        parts_points.append(vertices)
        parts_triangles.append(triangles + offset)
        parts_normals.append(normals)
        rings.append(ring)
        offset += len(vertices)
    if not rings:
        return mesh
    repeat = np.repeat(np.concatenate(rings), SIDES)
    data = {
        name: np.concatenate([np.asarray(values), np.asarray(values)[repeat]])
        for name, values in mesh.point_data.items()
    }
    normals = mesh.with_normals().normals
    return SurfaceMesh(
        points=np.ascontiguousarray(np.concatenate([mesh.points, *parts_points]), np.float32),
        triangles=np.ascontiguousarray(
            np.concatenate([mesh.triangles, *parts_triangles]), np.uint32
        ),
        point_data=data,
        normals=np.ascontiguousarray(np.concatenate([normals, *parts_normals]), np.float32),
        segments=mesh.segments,
        outlines=mesh.outlines,
    )
