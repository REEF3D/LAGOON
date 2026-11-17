# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""REEF3D::CFD's porous structures (VRANS, ctrl.txt) as surfaces, layer by layer.

REEF3D::CFD gives every porous structure its own material on its line
(vrans_f::structure_cell)::

    B 270 xs xe ys ye zs ze     n d50 α β    box
    B 274 xc yc zs ze r         n d50 α β    vertical cylinder
    B 281 xs xe ys ye zs ze     n d50 α β    wedge along x: the top from zs at xs to ze at xe
    B 282 xs xe ys ye zs ze     n d50 α β    wedge along y
    B 291 xs xe ys ye zs ze d   n d50 α β    plate along x: the band d thick above that line

and sets the cells of each in that order, the boxes first, then the cylinders, the
wedges and the plates, each command's lines one after the other: where structures
overlap, the later one's material holds. A rubble mound breakwater is set up so:
the armour layer's outline first, the filter inside it, the core inside that.

Each structure is drawn as what is left of it in the flow (:func:`porous_mesh`):
its faces without the parts inside a later structure, so the layers lie side by
side as REEF3D's cells have them (a flume's side shows armour, filter and core as
bands) instead of one inside the other. Its outline is its edges, without the
parts inside a later structure: where one layer ends and the next begins. The
points carry the material: ``porosity`` (n), ``d50`` (m) and ``layer`` (the
materials numbered in the order they first appear, from 1).

NHFLOW's porous structures (B 210 - B 219, one material for all, B 201) are the
file NHFLOW writes (REEF3D_NHFLOW_VRANS_VTP), shown as a structure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lagoonviz.core.data.mesh import SurfaceMesh

log = logging.getLogger(__name__)

#: segments around a cylinder
SEGMENTS = 48
#: edges between faces turned less than this are no edges (a cylinder's side)
SMOOTH_DEGREES = 25.0
#: REEF3D's order of the commands (vrans_f::structure_cell) and their shape values
COMMANDS = {270: 6, 274: 5, 281: 6, 282: 6, 291: 7}
#: the material values after the shape: porosity, d50, alpha, beta
MATERIAL = 4


@dataclass(frozen=True)
class Porous:
    """A porous structure of ctrl.txt: its command, shape values and material."""

    code: int
    shape: tuple[float, ...]
    porosity: float
    d50: float
    alpha: float
    beta: float

    @property
    def material(self) -> tuple[float, float, float, float]:
        return (self.porosity, self.d50, self.alpha, self.beta)

    def label(self) -> str:
        return f"B {self.code}: n {self.porosity:g}, d50 {self.d50:g} m"


def read_porous(ctrl: str | Path) -> list[Porous]:
    """The porous structures of a CFD ctrl.txt (B 270 - B 291), in REEF3D's order:
    by command (:data:`COMMANDS`), each command's lines in the file's order. An
    empty list for another module's ctrl.txt (A 10 not 6)."""
    try:
        text = Path(ctrl).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: dict[int, list[Porous]] = {code: [] for code in COMMANDS}
    module = None
    for line in text.splitlines():
        words = line.split("//")[0].split()
        if len(words) < 3:
            continue
        try:
            code = int(words[1])
            values = [float(w) for w in words[2:]]
        except ValueError:
            continue
        if words[0] == "A" and code == 10:
            module = int(values[0])
        if words[0] != "B" or code not in COMMANDS:
            continue
        n = COMMANDS[code]
        if len(values) < n + MATERIAL:
            log.info("%s: B %d with %d values, not drawn", ctrl, code, len(values))
            continue
        shape, material = values[:n], values[n : n + MATERIAL]
        found[code].append(Porous(code, tuple(shape), *material))
    if module is not None and module != 6:
        return []
    return [structure for code in COMMANDS for structure in found[code]]


def find_porous(case_dir: str | Path) -> Path | None:
    """A case's ctrl.txt if it sets up porous structures for CFD, else None."""
    ctrl = Path(case_dir) / "ctrl.txt"
    return ctrl if ctrl.is_file() and read_porous(ctrl) else None


# ======================================================================== shapes
def corners(structure: Porous) -> np.ndarray:
    """The corners of a structure (each is convex), as REEF3D fills it."""
    v = structure.shape
    if structure.code == 270:
        xs, xe, ys, ye, zs, ze = v
        return np.array([(x, y, z) for x in (xs, xe) for y in (ys, ye) for z in (zs, ze)])
    if structure.code == 274:
        xc, yc, zs, ze, r = v
        phi = np.linspace(0.0, 2.0 * math.pi, SEGMENTS, endpoint=False)
        ring = np.stack([xc + r * np.cos(phi), yc + r * np.sin(phi)], axis=1)
        return np.concatenate([np.column_stack([ring, np.full(SEGMENTS, z)]) for z in (zs, ze)])
    xs, xe, ys, ye, zs, ze = v[:6]
    if structure.code == 282:  # along y: the top from zs at ys to ze at ye
        low = min(zs, ze)
        section = [(ys, low), (ye, low), (ye, ze), (ys, zs)]
        return np.array([(x, a, b) for a, b in section for x in (xs, xe)])
    if structure.code == 281:
        low = min(zs, ze)
        section = [(xs, low), (xe, low), (xe, ze), (xs, zs)]
    else:  # 291: the band d thick above the line from (xs, zs) to (xe, ze)
        d = v[6]
        section = [(xs, zs), (xe, ze), (xe, ze + d), (xs, zs + d)]
    return np.array([(a, y, b) for a, b in section for y in (ys, ye)])


@dataclass
class Convex:
    """A convex solid: its faces (rings of corners, facing out) and the planes they
    lie in (outward normal n, offset d: inside where n · x <= d)."""

    faces: list[np.ndarray]
    normals: np.ndarray
    offsets: np.ndarray
    low: np.ndarray
    high: np.ndarray

    @classmethod
    def hull(cls, points: np.ndarray) -> Convex | None:
        """The convex hull of corners; None when they span no volume."""
        from scipy.spatial import ConvexHull, QhullError

        points = np.unique(np.round(np.asarray(points, np.float64), 12), axis=0)
        try:
            hull = ConvexHull(points)
        except (QhullError, ValueError):
            return None
        scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
        planes: dict[tuple, list[int]] = {}
        for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
            key = tuple(np.round(equation[:3], 6)) + (round(float(equation[3]) / scale, 6),)
            planes.setdefault(key, []).extend(int(i) for i in simplex)
        faces, normals, offsets = [], [], []
        for key, indices in planes.items():
            normal = np.array(key[:3], np.float64)
            normal /= np.linalg.norm(normal)
            ring = _ordered(points[sorted(set(indices))], normal)
            faces.append(ring)
            normals.append(normal)
            offsets.append(float(np.mean(ring @ normal)))
        return cls(faces, np.array(normals), np.array(offsets), points.min(0), points.max(0))

    def edges(self) -> list[np.ndarray]:
        """Its edges (pairs of points): where two faces meet at an angle."""
        found: dict[tuple, list] = {}
        for ring, normal in zip(self.faces, self.normals, strict=True):
            for a, b in zip(ring, np.roll(ring, -1, axis=0), strict=True):
                key = tuple(sorted((tuple(np.round(a, 9)), tuple(np.round(b, 9)))))
                found.setdefault(key, [a, b, []])[2].append(normal)
        smooth = math.cos(math.radians(SMOOTH_DEGREES))
        edges = []
        for a, b, normals in found.values():
            if len(normals) == 2 and float(normals[0] @ normals[1]) > smooth:
                continue
            edges.append(np.array([a, b]))
        return edges


def _ordered(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """The corners of a flat convex polygon in turn, facing ``normal``."""
    centre = points.mean(axis=0)
    u = points[0] - centre
    if np.linalg.norm(u) < 1e-15:
        u = points[-1] - centre
    u /= max(np.linalg.norm(u), 1e-300)
    v = np.cross(normal, u)
    angle = np.arctan2((points - centre) @ v, (points - centre) @ u)
    return points[np.argsort(angle)]


def _clip(ring: np.ndarray, distance: np.ndarray, keep_below: bool) -> np.ndarray:
    """A convex polygon cut where ``distance`` (at its corners) is 0: the part below
    (``keep_below``) or above."""
    s = distance if keep_below else -distance
    out = []
    count = len(ring)
    for k in range(count):
        a, b = ring[k], ring[(k + 1) % count]
        sa, sb = s[k], s[(k + 1) % count]
        if sa <= 0.0:
            out.append(a)
        if (sa < 0.0 < sb) or (sb < 0.0 < sa):
            out.append(a + (b - a) * (sa / (sa - sb)))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 3))


def _area(ring: np.ndarray) -> float:
    if len(ring) < 3:
        return 0.0
    total = np.zeros(3)
    for k in range(1, len(ring) - 1):
        total += np.cross(ring[k] - ring[0], ring[k + 1] - ring[0])
    return 0.5 * float(np.linalg.norm(total))


def without(ring: np.ndarray, solid: Convex, tol: float) -> list[np.ndarray]:
    """A flat convex polygon without its part in a convex solid (its faces count as
    inside): the convex pieces left."""
    if (ring.min(axis=0) > solid.high + tol).any() or (ring.max(axis=0) < solid.low - tol).any():
        return [ring]
    pieces = []
    rest = ring
    for normal, offset in zip(solid.normals, solid.offsets, strict=True):
        distance = rest @ normal - (offset + tol)
        if (distance <= 0.0).all():
            continue
        outside = _clip(rest, distance, keep_below=False)
        if _area(outside) > tol * tol:
            pieces.append(outside)
        rest = _clip(rest, distance, keep_below=True)
        if _area(rest) <= tol * tol:
            return pieces
    return pieces  # the rest lies in the solid


def _crossings(a: np.ndarray, b: np.ndarray, solid: Convex, tol: float) -> list[float]:
    """Where a segment (a + t (b - a), t in 0 - 1) enters and leaves a convex solid."""
    t0, t1 = 0.0, 1.0
    for normal, offset in zip(solid.normals, solid.offsets, strict=True):
        sa = float(a @ normal) - offset - tol
        sb = float(b @ normal) - offset - tol
        if sa > 0.0 and sb > 0.0:
            return []
        if sa > 0.0 or sb > 0.0:
            t = sa / (sa - sb)
            if sa > 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    return [t for t in (t0, t1) if 0.0 < t < 1.0] if t0 < t1 else []


class Layers:
    """Which material (layer, from 1; 0: none) REEF3D gives a point: the last of
    the structures holding it."""

    #: directions round an edge it is looked at in (see :meth:`is_edge`)
    AROUND = 24

    def __init__(self, solids: list[tuple[int, Convex]], reach: float) -> None:
        self.solids = solids
        self.reach = reach
        angle = (np.arange(self.AROUND) + 0.5) * 2.0 * math.pi / self.AROUND
        self.circle = np.stack([np.cos(angle), np.sin(angle)], axis=1)

    def at(self, points: np.ndarray) -> np.ndarray:
        found = np.zeros(len(points), np.int64)
        for layer, solid in self.solids:
            inside = ((points @ solid.normals.T) < solid.offsets).all(axis=1)
            found[inside] = layer
        return found

    def is_edge(self, point: np.ndarray, direction: np.ndarray) -> bool:
        """Whether a line through ``point`` along ``direction`` is an edge of the
        layers there: where the layers round it are not those of a plain face (one
        layer on one side of a flat face, another or none on the other) or of the
        inside of one layer."""
        direction = direction / max(float(np.linalg.norm(direction)), 1e-300)
        helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(direction, helper)
        u /= np.linalg.norm(u)
        w = np.cross(direction, u)
        around = point + self.reach * (self.circle[:, :1] * u + self.circle[:, 1:] * w)
        labels = self.at(around)
        changes = np.flatnonzero(labels != np.roll(labels, 1))
        if not len(changes):
            return False
        flat = len(changes) == 2 and changes[1] - changes[0] == self.AROUND // 2
        return not flat


# ======================================================================== mesh
def porous_mesh(structures: list[Porous]) -> SurfaceMesh:
    """The structures as REEF3D fills its cells with them (module doc): triangles,
    three points each; ``outlines`` their edges (points of their own, after the
    triangles'); ``porosity``, ``d50`` and ``layer`` at every point."""
    solids = [(s, Convex.hull(corners(s))) for s in structures]
    solids = [(s, c) for s, c in solids if c is not None]
    if not solids:
        return SurfaceMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), {})
    extent = np.ptp(np.concatenate([np.array([c.low, c.high]) for _, c in solids]), axis=0).max()
    tol = 1e-7 * max(float(extent), 1e-9)
    layers: dict[tuple, int] = {}
    numbered = [layers.setdefault(s.material, len(layers) + 1) for s, _ in solids]
    # each layer's edges: where it meets another, a corner of it (not where two
    # structures of the same material meet, e.g. a crest's box and a slope's wedge)
    where = Layers(list(zip(numbered, (c for _, c in solids), strict=True)), 20.0 * tol)
    triangles, lines, values = [], [], []
    for number, (structure, solid) in enumerate(solids):
        layer = numbered[number]
        later = [c for _, c in solids[number + 1 :]]
        value = (structure.porosity, structure.d50, float(layer))
        for ring in solid.faces:
            pieces = [ring]
            for other in later:
                pieces = [p for piece in pieces for p in without(piece, other, tol)]
            for piece in pieces:
                for k in range(1, len(piece) - 1):
                    triangles.append((piece[0], piece[k], piece[k + 1]))
                    values.append(value)
        for a, b in solid.edges():
            cuts = sorted({0.0, 1.0, *(t for _, c in solids for t in _crossings(a, b, c, tol))})
            for t0, t1 in zip(cuts[:-1], cuts[1:], strict=False):
                if (t1 - t0) * float(np.linalg.norm(b - a)) <= 100.0 * tol:
                    continue  # (a sliver where two cuts lie a rounding apart)
                p, q = a + t0 * (b - a), a + t1 * (b - a)
                if where.is_edge(0.5 * (p + q), b - a):
                    lines.append((p, q, value))
    lines = _merged(lines, tol)
    faces = np.asarray(triangles, np.float64).reshape(-1, 3)
    ends = np.array([[p, q] for p, q, _ in lines], np.float64).reshape(-1, 3)
    per_point = np.concatenate(
        [np.repeat(np.asarray(values, np.float64).reshape(-1, 3), 3, axis=0),
         np.repeat(np.array([v for _, _, v in lines], np.float64).reshape(-1, 3), 2, axis=0)]
    )  # fmt: skip
    points = np.concatenate([faces, ends]).astype(np.float32)
    first = len(faces)
    data = {
        name: per_point[:, k].astype(np.float32)
        for k, name in enumerate(("porosity", "d50", "layer"))
    }
    mesh = SurfaceMesh(
        points=points,
        triangles=np.arange(first, dtype=np.uint32).reshape(-1, 3),
        point_data=data,
        outlines=(first + np.arange(len(ends), dtype=np.uint32)).reshape(-1, 2),
    )
    return mesh.with_normals()


def _merged(lines: list[tuple], tol: float) -> list[tuple]:
    """Segments on one line that overlap as one (an edge of two structures, each
    cut elsewhere: drawn once)."""
    groups: dict[tuple, list] = {}
    for p, q, value in lines:
        d = q - p
        d = d / np.linalg.norm(d)
        k = int(np.argmax(np.abs(d)))
        if d[k] < 0:
            d = -d
        foot = p - (p @ d) * d
        digits = max(int(-math.log10(max(tol, 1e-300))) - 1, 0)
        key = (*np.round(d, 6), *np.round(foot, digits))
        groups.setdefault(key, []).append((float(p @ d), float(q @ d), d, foot, value))
    merged = []
    for parts in groups.values():
        spans = sorted(
            ((min(a, b), max(a, b), d, foot, v) for a, b, d, foot, v in parts),
            key=lambda span: span[:2],
        )
        start, end, d, foot, value = spans[0]
        for a, b, *_ in spans[1:]:
            if a <= end + tol:
                end = max(end, b)
                continue
            merged.append((foot + start * d, foot + end * d, value))
            start, end = a, b
        merged.append((foot + start * d, foot + end * d, value))
    return merged


def porous_from_ctrl(ctrl: str | Path) -> tuple[SurfaceMesh, list[Porous]]:
    """The porous structures of a ctrl.txt and their surface (:func:`porous_mesh`)."""
    structures = read_porous(ctrl)
    return porous_mesh(structures), structures
