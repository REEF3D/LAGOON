# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""REEF3D's porous structures (VRANS, ctrl.txt) as surfaces, layer by layer: CFD's and NHFLOW's.

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

**NHFLOW** (:func:`nhflow_mesh`) makes one porous body of its shapes (B 210 box,
B 212 cylinder along y, B 213 vertical cylinder, B 214 jacket member, B 215 sphere,
B 217 - B 219 wedges; nhflow_geometry::objects_vrans) and gives it layers by the
depth below its exposed surface (nhflow_geometry_layer.cpp, vrans_nhflow.cpp)::

    B 201 n d50 α β            the core
    B 202 t n d50 α β          a layer t thick, the first the outermost (armour)
    B 203 1                    faces resting on the bed expose nothing (default)

The exposed surface is the body's outside without its faces on the domain's sides,
without the faces between its shapes, in 2D without the y-ends, and (B 203 1)
without the faces facing down onto the bed. A body that is the same at every y (a
breakwater's trunk: boxes, wedges and cylinders across y over one y-range, its ends
not exposed) shows its layers on its ends; another, only its outside (the armour:
depth 0). REEF3D blends the materials over B 209 cells at each layer's boundary;
LAGOON draws the boundary where the blend is half way.
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
    """A case's ctrl.txt if it sets up porous structures (CFD or NHFLOW), else None."""
    ctrl = Path(case_dir) / "ctrl.txt"
    if not ctrl.is_file():
        return None
    return ctrl if read_porous(ctrl) or read_nhflow_porous(ctrl) is not None else None


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
        return _assembled([], [], [])
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
    return _assembled(triangles, values, _merged(lines, tol))


def _assembled(triangles: list, values: list, lines: list) -> SurfaceMesh:
    """Triangles (three corners each) with their (porosity, d50, layer), and
    outlines (two ends and their values) as one surface."""
    if not triangles:
        return SurfaceMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32), {})
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
        if float(np.linalg.norm(q - p)) <= tol:
            continue
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


def porous_from_ctrl(ctrl: str | Path) -> SurfaceMesh:
    """The porous structures of a case's ctrl.txt as a surface: CFD's
    (:func:`porous_mesh`) or NHFLOW's (:func:`nhflow_mesh`, with the domain, the
    cell size and 2D or not of the case's control.txt)."""
    structures = read_porous(ctrl)
    if structures:
        return porous_mesh(structures)
    nhflow = read_nhflow_porous(ctrl)
    if nhflow is None:
        return _assembled([], [], [])
    from lagoonviz.core.data.setup import read_setup

    setup = read_setup(Path(ctrl).parent)
    spacing = setup.spacing
    return nhflow_mesh(nhflow, setup.domain, setup.two_d, spacing[0] if spacing else None)


# ======================================================================== NHFLOW
#: NHFLOW's porous shapes in its order (objects_vrans) and their values
NHFLOW_COMMANDS = {210: 6, 212: 5, 213: 5, 214: 8, 215: 4, 217: 6, 218: 6, 219: 6}
#: the same shapes as NHFLOW's direct forcing solids (forcing.SHAPES)
_AS_FORCING = {210: 581, 212: 583, 213: 584, 214: 585, 215: 586, 217: 587, 218: 588, 219: 589}
#: shapes the same at every y (their section across x and z): box, cylinder along y,
#: wedges across x; and where their y-range is in their values
_ALONG_Y = {210: 2, 212: 2, 217: 2, 219: 2}


@dataclass(frozen=True)
class NhflowPorous:
    """NHFLOW's porous body (module doc): its shapes, its core's material (B 201),
    its layers from the outside in (B 202: t, n, d50, α, β), B 203 and B 209."""

    shapes: tuple[tuple[int, tuple[float, ...]], ...]
    core: tuple[float, float, float, float]
    layers: tuple[tuple[float, float, float, float, float], ...] = ()
    on_bed: bool = True
    heaviside: float = 1.1

    def materials(self) -> list[tuple[float, float, float]]:
        """(porosity, d50, layer) of the layers from the outside in, the core last."""
        found = [(n, d50, float(k + 1)) for k, (_, n, d50, _, _) in enumerate(self.layers)]
        return [*found, (self.core[0], self.core[1], float(len(self.layers) + 1))]

    def depths(self) -> list[float]:
        """The depth of each layer's inner boundary below the exposed surface."""
        total, found = 0.0, []
        for layer in self.layers:
            total += max(layer[0], 0.0)
            found.append(total)
        return found

    def layer_at(self, depth: np.ndarray) -> np.ndarray:
        """The number of the layer (from 1; the core last) at depths below the surface."""
        bounds = [t for t, (thick, *_) in zip(self.depths(), self.layers, strict=True) if thick > 0]
        number = [k + 1 for k, (thick, *_) in enumerate(self.layers) if thick > 0]
        found = np.full(np.shape(depth), len(self.layers) + 1, np.int64)
        for bound, k in reversed(list(zip(bounds, number, strict=True))):
            found[np.asarray(depth) < bound] = k
        return found


def read_nhflow_porous(ctrl: str | Path) -> NhflowPorous | None:
    """NHFLOW's porous body of a ctrl.txt (A 10 5, shapes B 210 - B 219 and B 201);
    None for another module or without them."""
    try:
        text = Path(ctrl).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    module, core, on_bed, heaviside = None, None, True, 1.1
    shapes: dict[int, list] = {code: [] for code in NHFLOW_COMMANDS}
    layers = []
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
        if words[0] != "B":
            continue
        if code == 201 and len(values) >= 4:
            core = tuple(values[:4])
        elif code == 202 and len(values) >= 5:
            layers.append(tuple(values[:5]))
        elif code == 203:
            on_bed = int(values[0]) == 1
        elif code == 209:
            heaviside = values[0]
        elif code in NHFLOW_COMMANDS and len(values) >= NHFLOW_COMMANDS[code]:
            shapes[code].append(tuple(values[: NHFLOW_COMMANDS[code]]))
    ordered = tuple((code, v) for code in NHFLOW_COMMANDS for v in shapes[code])
    if module != 5 or not ordered:
        return None
    if core is None:  # REEF3D without B 201: n = 1, no porous media
        core = (1.0, 0.01, 0.0, 0.0)
    return NhflowPorous(ordered, core, tuple(layers), on_bed, heaviside)


def _nhflow_corners(code: int, values: tuple[float, ...]) -> np.ndarray:
    from lagoonviz.core.data.forcing import SHAPES

    triangles = SHAPES[_AS_FORCING[code]][1](list(values))
    return np.asarray(triangles, np.float64).reshape(-1, 3)


def _inside(solid: Convex, point: np.ndarray, tol: float) -> bool:
    return bool((solid.normals @ point <= solid.offsets + tol).all())


def distance_to_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The distance of each point (n, 3) to the nearest of the triangles (m, 3, 3)."""
    points = np.asarray(points, np.float64)
    best = np.full(len(points), np.inf)
    for a, b, c in np.asarray(triangles, np.float64):
        best = np.minimum(best, _to_triangle(points, a, b, c))
    return best


def _to_triangle(p: np.ndarray, a, b, c) -> np.ndarray:
    """Distances of points to one triangle (the closest point by its regions)."""
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    area = float(normal @ normal)
    if area < 1e-30:  # degenerate: its longest edge
        return np.min([_to_segment(p, a, b), _to_segment(p, b, c), _to_segment(p, a, c)], axis=0)
    # the point projected onto the plane, in barycentric terms
    w = p - a
    d00, d01, d11 = ab @ ab, ab @ ac, ac @ ac
    d20, d21 = w @ ab, w @ ac
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    u = (d00 * d21 - d01 * d20) / denom
    inside = (v >= 0) & (u >= 0) & (v + u <= 1)
    plane = np.abs(w @ normal) / math.sqrt(area)
    edges = np.min([_to_segment(p, a, b), _to_segment(p, b, c), _to_segment(p, a, c)], axis=0)
    return np.where(inside, plane, edges)


def _to_segment(p: np.ndarray, a, b) -> np.ndarray:
    d = b - a
    t = np.clip((p - a) @ d / max(float(d @ d), 1e-300), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * d), axis=1)


def nhflow_mesh(
    porous: NhflowPorous,
    domain: tuple | None = None,
    two_d: bool = False,
    dx: float | None = None,
) -> SurfaceMesh:
    """NHFLOW's porous body as a surface, coloured by its layers (module doc):
    ``domain`` (xs, xe, ys, ye, zs, ze) and ``dx`` the case's (the faces on the
    domain's sides and on the bed expose nothing), ``two_d`` a 2D case (no y-ends)."""
    shapes = [(code, v, Convex.hull(_nhflow_corners(code, v))) for code, v in porous.shapes]
    shapes = [(code, v, c) for code, v, c in shapes if c is not None]
    if not shapes:
        return _assembled([], [], [])
    solids = [c for *_, c in shapes]
    low = np.min([c.low for c in solids], axis=0)
    high = np.max([c.high for c in solids], axis=0)
    extent = max(float(np.ptp(np.array([low, high]), axis=0).max()), 1e-9)
    tol = 1e-7 * extent
    step = 1e-4 * extent  # off a face, to see what is beyond it
    # the body's outside: each shape's faces without what a later shape covers and
    # without those between two shapes
    outside = []  # (ring, outward normal)
    for i, solid in enumerate(solids):
        for ring, normal in zip(solid.faces, solid.normals, strict=True):
            parts = [ring]
            for other in solids[i + 1 :]:
                parts = [q for part in parts for q in without(part, other, tol)]
            for part in parts:
                beyond = part.mean(axis=0) + step * normal
                if any(_inside(o, beyond, tol) for j, o in enumerate(solids) if j != i):
                    continue
                outside.append((part, normal))
    bed = domain[4] if domain is not None else float(low[2])
    bed_tol = max(porous.heaviside * (dx or 0.0), 1e3 * tol)

    def exposed(ring: np.ndarray, normal: np.ndarray) -> bool:
        beyond = ring.mean(axis=0) + step * normal
        if domain is not None:
            if beyond[0] < domain[0] or beyond[0] > domain[1]:
                return False
            if not two_d and (beyond[1] < domain[2] or beyond[1] > domain[3]):
                return False
        if two_d and abs(normal[1]) > 0.9:
            return False
        down = normal[2] < -math.sqrt(0.5)
        return not (porous.on_bed and down and float(ring[:, 2].min()) <= bed + bed_tol)

    open_ = [(ring, normal) for ring, normal in outside if exposed(ring, normal)]
    surface = np.array(
        [(r[0], r[k], r[k + 1]) for r, _ in open_ for k in range(1, len(r) - 1)], np.float64
    ).reshape(-1, 3, 3)

    def depth(points: np.ndarray) -> np.ndarray:
        if not len(surface):  # nothing exposed: REEF3D gives the body its core
            return np.full(len(points), np.inf)
        return distance_to_triangles(points, surface)

    materials = porous.materials()
    triangles, values, lines = [], [], []

    def add(ring: np.ndarray, layer: int) -> None:
        value = materials[layer - 1]
        for k in range(1, len(ring) - 1):
            triangles.append((ring[0], ring[k], ring[k + 1]))
            values.append(value)

    ys = {v[_ALONG_Y[code]] for code, v, _ in shapes if code in _ALONG_Y}
    ye = {v[_ALONG_Y[code] + 1] for code, v, _ in shapes if code in _ALONG_Y}
    trunk = (
        all(code in _ALONG_Y for code, *_ in shapes)
        and len(ys) == 1
        and len(ye) == 1
        and not any(abs(n[1]) > 0.9 for _, n in open_)  # its ends expose nothing
    )
    for ring, normal in outside:
        if trunk and abs(normal[1]) > 0.9:
            continue  # its ends: layered below
        centre = ring.mean(axis=0)
        add(ring, int(porous.layer_at(depth(centre[None, :] - 1e3 * tol * normal))[0]))
    # the body's edges (where its outside bends, not where two shapes meet)
    where = Layers([(1, c) for c in solids], 20.0 * tol)
    for solid in solids:
        for a, b in solid.edges():
            cuts = sorted({0.0, 1.0, *(t for c in solids for t in _crossings(a, b, c, tol))})
            for t0, t1 in zip(cuts[:-1], cuts[1:], strict=False):
                if (t1 - t0) * float(np.linalg.norm(b - a)) <= 100.0 * tol:
                    continue
                p, q = a + t0 * (b - a), a + t1 * (b - a)
                if where.is_edge(0.5 * (p + q), b - a):
                    lines.append((p, q, materials[0]))
    if trunk:
        y0, y1 = float(next(iter(ys))), float(next(iter(ye)))
        for ring, layer, contour in _trunk_ends(porous, solids, low, high, depth, y0, y1):
            if ring is not None:
                add(ring, layer)
            if contour is not None:
                lines.append((*contour, materials[min(layer, len(materials)) - 1]))
    return _assembled(triangles, values, _merged(lines, tol))


#: points across a trunk's end at most (its layers drawn on a grid this fine)
END_POINTS = 250_000


def _trunk_ends(porous, solids, low, high, depth, y0: float, y1: float):
    """A trunk's two ends (y0, y1) with its layers: (triangle, layer, None) for each
    triangle, (None, layer, (p, q)) for each piece of a layer's inner boundary."""
    from lagoonviz.core.data.mesh import grid_triangles, iso_lines

    thick = [t for t, *_ in porous.layers if t > 0]
    size = np.array([high[0] - low[0], high[2] - low[2]])
    step = min(min(thick) / 6.0 if thick else size.max() / 100.0, size.max() / 100.0)
    step = max(step, math.sqrt(size[0] * size[1] / END_POINTS))
    pad = 2.0 * step
    x = np.arange(low[0] - pad, high[0] + pad + step, step)
    z = np.arange(low[2] - pad, high[2] + pad + step, step)
    gx, gz = np.meshgrid(x, z)
    middle = 0.5 * (y0 + y1)
    points = np.stack([gx.ravel(), np.full(gx.size, middle), gz.ravel()], axis=1)
    inside = np.zeros(len(points), bool)
    for solid in solids:
        inside |= ((points @ solid.normals.T) <= solid.offsets).all(axis=1)
    # the signed distance to the body's outline across x and z (inside positive)
    edges = []
    for solid in solids:
        for ring, normal in zip(solid.faces, solid.normals, strict=True):
            if abs(normal[1]) > 0.9:
                xz = ring[:, [0, 2]]
                edges += [(xz[k], xz[(k + 1) % len(xz)]) for k in range(len(xz))]
    flat = points[:, [0, 2]]
    outline = np.min([_to_segment(flat, a, b) for a, b in edges], axis=0)
    signed = np.where(inside, outline, -outline)
    deep = depth(points)
    grid = SurfaceMesh(
        points.astype(np.float32),
        grid_triangles(len(z), len(x)),
        {"signed": signed.astype(np.float32), "depth": np.minimum(deep, 1e6).astype(np.float32)},
    )
    body = clip_surface_of(grid, "signed", 0.0, "above")
    bounds = [0.0, *porous.depths()]
    for k in range(len(porous.layers) + 1):
        part = body
        if k > 0:
            part = clip_surface_of(part, "depth", bounds[k], "above")
        if k < len(porous.layers):
            if porous.layers[k][0] <= 0:
                continue
            part = clip_surface_of(part, "depth", bounds[k + 1], "below")
            line = iso_lines(body, body.point_data["depth"], bounds[k + 1])
            for y in (y0, y1):
                ends = np.asarray(line.points, np.float64)
                ends[:, 1] = y
                for a, b in np.asarray(line.segments, np.int64):
                    yield None, k + 1, (ends[a], ends[b])
        for y, flip in ((y0, False), (y1, True)):
            corners = np.asarray(part.points, np.float64)[np.asarray(part.triangles, np.int64)]
            corners[..., 1] = y
            facing = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 1]
            turn = (facing > 0) if not flip else (facing < 0)  # y0 faces -y, y1 +y
            corners[turn] = corners[turn][:, ::-1]
            for triangle in corners:
                yield triangle, k + 1, None


def clip_surface_of(mesh: SurfaceMesh, name: str, level: float, keep: str) -> SurfaceMesh:
    """:func:`~lagoonviz.core.data.mesh.clip_surface` by one of the mesh's arrays."""
    from lagoonviz.core.data.mesh import clip_surface

    if not mesh.n_triangles:
        return mesh
    return clip_surface(mesh, mesh.point_data[name], level, keep)
