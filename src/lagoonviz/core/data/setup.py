# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The set-up of a case from its input files, before (or beside) any results.

DIVEMesh's ``control.txt`` gives the domain (B 10), the grid (B 1, B 2), the kinds of
the domain's sides (C 11 - C 16), the solids (S) and the bed (T, ``geo.dat``).
REEF3D's ``ctrl.txt`` gives the still water level (F 60), the waves (B 90 - B 108),
the gauges and probes (P), floating bodies and their moorings (X), porous and
direct forcing structures (A 581 - A 589, B 210 - B 270).

:func:`read_setup` reads both into a :class:`CaseSetup`; :meth:`CaseSetup.marks`
turns them into :class:`Mark` s, lines, points and labels in groups (domain, sides,
zones, waves, gauges, ...) that the "Case set-up" item draws in the 3D view and the
set-up sketch (``lagoon-setup``) draws in plan and side view::

    setup = read_setup("~/runs/flume")
    print(setup.describe())
    for mark in setup.marks():
        print(mark.group, mark.legend, len(mark.segments))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lagoonviz.core.data.controlfile import (
    CONTROL,
    CTRL,
    ControlFile,
    read_commands,
)
from lagoonviz.core.kinds import SETUP_GROUPS

log = logging.getLogger(__name__)

__all__ = [
    "CONTROL",
    "CTRL",
    "GEO",
    "GROUPS",
    "CaseSetup",
    "Commands",
    "Level",
    "Mark",
    "Refinement",
    "has_inputs",
    "read_commands",
    "read_setup",
]

#: the control files, as the set-up reads them (a ControlFile)
Commands = ControlFile

GEO = "geo.dat"
#: geo.dat points drawn at most (a random sample of larger sets)
GEO_POINTS = 200_000

#: groups of marks: name -> label (each is a property "show_<name>" of the item)
GROUPS = SETUP_GROUPS

#: the sides of the domain (C 11 - C 16): number -> (axis, at the end, name)
SIDES = {
    1: (0, False, "x min"),
    4: (0, True, "x max"),
    3: (1, False, "y min"),
    2: (1, True, "y max"),
    5: (2, False, "bottom"),
    6: (2, True, "top"),
}
#: kinds of sides: C 1x value -> (name, colour)
SIDE_KINDS = {
    1: ("inflow", "#22b5c9"),
    2: ("outflow", "#e0892b"),
    3: ("symmetry", "#b18be0"),
    6: ("wave generation", "#2f7fd8"),
    7: ("numerical beach", "#2aa876"),
    21: ("wall", "#7d8796"),
}

COLORS = {
    "domain": "#8a94a6",
    "water": "#3b8fd9",
    "generation": "#2f7fd8",
    "beach": "#2aa876",
    "waves": "#2f7fd8",
    "gauges": "#f2b134",
    "probes": "#e8706a",
    "sections": "#c58af0",
    "forces": "#ff8c42",
    "bodies": "#ff5d5d",
    "moorings": "#d9a066",
    "structures": "#9aa3ad",
    "regions": "#4fb3a9",
    "solids": "#a0a8b4",
    "bed": "#b08d57",
    "refinement": "#e07b39",
}

SOLVERS = {2: "SFLOW", 3: "FNPF", 5: "NHFLOW", 6: "CFD", 7: "SEASTATE"}


# ======================================================================== files
def has_inputs(case_dir: str | Path) -> bool:
    """True if a folder holds REEF3D or DIVEMesh input (ctrl.txt or control.txt, or
    the copies of ctrl.txt REEF3D keeps in REEF3D_Case)."""
    case_dir = Path(case_dir)
    return (case_dir / CONTROL).is_file() or ctrl_file(case_dir) is not None


def ctrl_file(case_dir: str | Path) -> Path | None:
    """The case's ctrl.txt; without one, the newest copy REEF3D kept of it in
    REEF3D_Case (``ctrl-<run>.txt``, one per run)."""
    case_dir = Path(case_dir)
    if (case_dir / CTRL).is_file():
        return case_dir / CTRL
    copies = sorted((case_dir / "REEF3D_Case").glob("ctrl-*.txt"))
    return copies[-1] if copies else None


# ======================================================================== marks
@dataclass
class Mark:
    """One sign of the set-up: lines, points and labels of one command (or of the
    domain), in one colour."""

    group: str
    #: what it is, for the legend of the sketch: "wave gauge (P 51)"
    legend: str
    color: str
    #: (n, 2, 3): start and end of each line
    segments: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3)))
    #: (m, 3): dots
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    #: (text, (x, y, z))
    labels: list[tuple[str, tuple[float, float, float]]] = field(default_factory=list)
    #: the command it shows ("P 51"; "" for the domain and the sides) and which line
    code: str = ""
    occurrence: int = 0
    #: filled areas (relaxation zones, the still water): (k, 3, 3) triangles
    faces: np.ndarray | None = None
    #: the file of the command: "control.txt" (DIVEMesh) or "ctrl.txt" (REEF3D)
    file: str = "ctrl.txt"


def _segments(points: list) -> np.ndarray:
    return np.asarray(points, float).reshape(-1, 2, 3)


def _box_edges(xs, xe, ys, ye, zs, ze) -> np.ndarray:
    c = [(x, y, z) for z in (zs, ze) for y in (ys, ye) for x in (xs, xe)]
    pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return _segments([(c[a], c[b]) for a, b in pairs])


def _rect(xs, xe, ys, ye, z) -> np.ndarray:
    c = [(xs, ys, z), (xe, ys, z), (xe, ye, z), (xs, ye, z)]
    return _segments([(c[i], c[(i + 1) % 4]) for i in range(4)])


def _rect_faces(xs, xe, ys, ye, z) -> np.ndarray:
    a, b, c, d = (xs, ys, z), (xe, ys, z), (xe, ye, z), (xs, ye, z)
    return np.asarray([(a, b, c), (a, c, d)], float)


def _circle(centre, radius: float, axis: int, n: int = 48) -> np.ndarray:
    others = [i for i in range(3) if i != axis]
    phi = np.linspace(0.0, 2 * math.pi, n + 1)
    ring = np.zeros((n + 1, 3))
    ring[:, axis] = centre[axis]
    ring[:, others[0]] = centre[others[0]] + radius * np.cos(phi)
    ring[:, others[1]] = centre[others[1]] + radius * np.sin(phi)
    return np.stack([ring[:-1], ring[1:]], axis=1)


def _cylinder(axis: int, centre2, start: float, end: float, r1: float, r2: float | None = None):
    """Edges of a cylinder (cone) along an axis: the two rings and four lines."""
    r2 = r1 if r2 is None else r2
    others = [i for i in range(3) if i != axis]

    def point(t, a, b):
        p = [0.0, 0.0, 0.0]
        p[axis], p[others[0]], p[others[1]] = t, a, b
        return p

    rings = [
        _circle(point(start, *centre2), r1, axis),
        _circle(point(end, *centre2), r2, axis),
    ]
    lines = []
    for angle in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
        ca, sa = math.cos(angle), math.sin(angle)
        lines.append(
            (
                point(start, centre2[0] + r1 * ca, centre2[1] + r1 * sa),
                point(end, centre2[0] + r2 * ca, centre2[1] + r2 * sa),
            )
        )
    return np.concatenate([*rings, _segments(lines)])


def _sphere(centre, radius: float) -> np.ndarray:
    return np.concatenate([_circle(centre, radius, axis) for axis in range(3)])


def _tilted_cylinder(start, end, r1: float, r2: float) -> np.ndarray:
    """A cylinder (cone) between two points: rings at both ends and four lines."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    axis = b - a
    length = np.linalg.norm(axis)
    if length < 1e-12:
        return _sphere(a, max(r1, r2))
    axis /= length
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    phi = np.linspace(0.0, 2 * math.pi, 49)
    rings = []
    for centre, r in ((a, r1), (b, r2)):
        ring = centre + r * (np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v))
        rings.append(np.stack([ring[:-1], ring[1:]], axis=1))
    lines = [(a + r1 * d, b + r2 * d) for d in (u, v, -u, -v)]
    return np.concatenate([*rings, _segments(lines)])


def _hull_edges(corners) -> np.ndarray:
    """Edges of a convex solid from its corners (the edges of its hull's faces)."""
    from lagoonviz.core.data.divemesh import _convex

    triangles = np.asarray(_convex(np.asarray(corners, float)), float)
    return _feature_edges(triangles)


def _feature_edges(triangles: np.ndarray, angle: float = 20.0) -> np.ndarray:
    """The edges of a triangle soup where the faces meet at a crease (or at a border):
    the outline of a solid without the lines across its flat faces."""
    t = np.asarray(triangles, float).reshape(-1, 3, 3)
    if not len(t):
        return np.zeros((0, 2, 3))
    scale = max(float(np.ptp(t.reshape(-1, 3), axis=0).max()), 1e-9)
    keys = np.round(t / (1e-6 * scale)).astype(np.int64)
    normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    size = np.linalg.norm(normal, axis=1)
    normal = normal / np.where(size > 0, size, 1.0)[:, None]
    edges: dict[tuple, list[int]] = {}
    for index in range(len(t)):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ka, kb = tuple(keys[index, a]), tuple(keys[index, b])
            key = (ka, kb) if ka <= kb else (kb, ka)
            edges.setdefault(key, []).append(index)
    cos_limit = math.cos(math.radians(angle))
    out = []
    for (ka, kb), faces in edges.items():
        if len(faces) == 2 and float(normal[faces[0]] @ normal[faces[1]]) > cos_limit:
            continue
        index = faces[0]
        pa = t[index][[tuple(k) for k in keys[index]].index(ka)]
        pb = t[index][[tuple(k) for k in keys[index]].index(kb)]
        out.append((pa, pb))
    return _segments(out) if out else np.zeros((0, 2, 3))


# ======================================================================== the set-up
@dataclass
class CaseSetup:
    """What the input files of a case say (see the module docstring)."""

    path: Path
    control: ControlFile
    ctrl: ControlFile
    #: the domain: xs, xe, ys, ye, zs, ze (None: not known)
    domain: tuple[float, float, float, float, float, float] | None = None
    #: cells in x, y and z (None: not known)
    cells: tuple[int, int, int] | None = None
    #: the kinds of the sides: side number (1 - 6) -> C 1x value
    sides: dict[int, int] = field(default_factory=dict)
    #: still water level (F 60), None if not given
    water_level: float | None = None
    #: what could not be shown, for the user
    notes: list[str] = field(default_factory=list)
    #: where the domain came from when control.txt does not give it ("the results")
    domain_from: str = ""
    _bed: tuple | None = None
    _marks: list[Mark] | None = None
    _meshes: dict = field(default_factory=dict)
    _revisions: tuple[int, int] = (-1, -1)

    # ------------------------------------------------------------- facts
    @property
    def solver(self) -> str:
        """The REEF3D module of ctrl.txt (A 10): "CFD", "NHFLOW", ... ("" without ctrl.txt)."""
        if not self.ctrl:
            return ""
        return SOLVERS.get(self.ctrl.int("A 10", 6), "REEF3D")

    @property
    def files(self) -> list[str]:
        """The input files found in the case folder."""
        names = [CONTROL, CTRL, GEO, "solid.stl", "floating.stl"]
        found = [name for name in names if (self.path / name).is_file()]
        if CTRL not in found and self.ctrl.path is not None and self.ctrl.path.is_file():
            found.insert(1, f"{self.ctrl.path.parent.name}/{self.ctrl.path.name}")
        if self.domain_from:
            found.append(f"domain from {self.domain_from}")
        return found

    @property
    def two_d(self) -> bool:
        return self.cells is not None and self.cells[1] == 1

    @property
    def waves(self) -> bool:
        """REEF3D makes waves (B 90 1, or a wave generation method B 98)."""
        return self.ctrl.int("B 90", 0) > 0 or self.ctrl.int("B 98", 0) > 0

    @property
    def spacing(self) -> tuple[float, float, float] | None:
        """The cell size in x, y and z of a uniform grid (None: not known)."""
        if self.domain is None or self.cells is None:
            return None
        xs, xe, ys, ye, zs, ze = self.domain
        nx, ny, nz = self.cells
        return ((xe - xs) / max(nx, 1), (ye - ys) / max(ny, 1), (ze - zs) / max(nz, 1))

    @property
    def stretched(self) -> bool:
        """The grid is stretched (B 101 - B 103): only DIVEMesh knows its lines."""
        return any(self.control.int(f"B {c}", 0) > 0 for c in (101, 102, 103))

    def refresh(self) -> bool:
        """Read the facts again after the files were edited (True if they changed)."""
        revisions = (self.control.revision, self.ctrl.revision)
        if revisions == self._revisions:
            return False
        if not self.domain_from or "B 10" in self.control:  # the results' extent stays
            self.domain = _domain(self)
            if self.domain is not None:
                self.domain_from = ""
        self.cells = _cells(self)
        self.sides = _sides(self)
        self.water_level = _water_level(self)
        self._marks = None
        self._meshes.clear()
        self._revisions = revisions
        return True

    # ------------------------------------------------------------- the bed
    def geo_points(self, limit: int = GEO_POINTS) -> np.ndarray:
        """The points of geo.dat (a random sample of ``limit`` at most), (n, 3)."""
        if self._bed is None:
            self._bed = (_read_geo(self.path / GEO, self.control),)
        points = self._bed[0]
        if len(points) > limit:
            pick = np.random.default_rng(0).choice(len(points), limit, replace=False)
            points = points[np.sort(pick)]
        return points

    def use_domain(self, low, high, source: str) -> None:
        """Take the domain from elsewhere (the extent of a run's results) when the
        input files do not give it: no control.txt, or no B 10 in it."""
        if self.domain is not None:
            return
        self.domain = (
            float(low[0]),
            float(high[0]),
            float(low[1]),
            float(high[1]),
            float(low[2]),
            float(high[2]),
        )
        self.domain_from = source
        self.cells = _cells(self)
        self.sides = _sides(self)
        self._marks = None
        self._meshes.clear()

    # ------------------------------------------------------------- the sketch
    def refinement(self) -> Refinement | None:
        """The mesh refinement of SFLOW, FNPF and NHFLOW (G 1 levels, G 10 boxes),
        with the columns each level has about; None without refinement."""
        self.refresh()
        return _refinement(self)

    def facts(self) -> list[tuple[str, str]]:
        """The main facts of the input as (label, value) rows, for the sketch's table."""
        self.refresh()
        return _facts(self)

    def bed_raster(self, points: np.ndarray | None = None, size: int | None = None):
        """The bed as a regular raster over the domain: (x, y, z), z (ny, nx), filled
        between the points as DIVEMesh does. From ``points`` (the bed of the results)
        or geo.dat; None without either."""
        self.refresh()
        if points is None:
            if not (self.path / GEO).is_file():
                return None
            self.geo_points()
            points = self._bed[0]
        return _raster(np.asarray(points, float), self.domain, size)

    def solid_mesh(self) -> tuple[np.ndarray, np.ndarray] | None:
        """DIVEMesh's solids as (points, triangles): REEF3D_Solid.vtp, or built from
        control.txt; None without solids."""
        from lagoonviz.core.data.divemesh import find_solids
        from lagoonviz.core.data.reef3d import SolidDataset

        source = find_solids(self.path)
        if source is None:
            return None
        try:
            mesh = SolidDataset(source, self.path).read(0)
        except (OSError, ValueError, KeyError, IndexError):
            return None
        if not len(mesh.triangles):
            return None
        return mesh.points, mesh.triangles

    # ------------------------------------------------------------- signs
    def marks(self) -> list[Mark]:
        """The signs of the set-up: domain, sides, zones, waves, gauges and probes."""
        self.refresh()
        if self._marks is None:
            self._marks = _marks(self)
        return self._marks

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The box all marks fill (the domain, or what the commands place)."""
        if self.domain is not None:
            xs, xe, ys, ye, zs, ze = self.domain
            return np.array([xs, ys, zs]), np.array([xe, ye, ze])
        points = [m.segments.reshape(-1, 3) for m in self.marks() if len(m.segments)]
        points += [m.points for m in self.marks() if len(m.points)]
        if not points:
            return None
        every = np.concatenate(points)
        return every.min(axis=0), every.max(axis=0)

    def describe(self) -> str:
        """The main facts of the input, as text."""
        self.refresh()
        lines = [f"Case set-up of {self.path}"]
        lines.append("  files: " + (", ".join(self.files) or "none"))
        if self.ctrl:
            lines.append(f"  REEF3D module: {self.solver} (A 10)")
        if self.domain is not None:
            xs, xe, ys, ye, zs, ze = self.domain
            lines.append(
                f"  domain: x {xs:g} .. {xe:g}, y {ys:g} .. {ye:g}, z {zs:g} .. {ze:g} m"
                + (f" (from {self.domain_from})" if self.domain_from else "")
            )
        if self.cells is not None:
            nx, ny, nz = self.cells
            spacing = self.spacing
            total = nx * ny * nz
            text = f"  grid: {nx} x {ny} x {nz} = {total:,} cells"
            if spacing is not None and not self.stretched:
                sizes = spacing if ny > 1 else (spacing[0], spacing[2])
                text += " of " + " x ".join(f"{d:.4g}" for d in sizes) + " m"
                text += " (2D)" if ny == 1 else ""
            if self.stretched:
                text += ", stretched (B 101 - B 103)"
            lines.append(text)
        from lagoonviz.core.data.divemesh_grid import read_preview

        grid = read_preview(self.path)
        if grid is not None:
            nx, ny, nz = grid.cells
            sizes = ", ".join(
                f"d{axis} {low:.4g}" + (f" - {high:.4g}" if high > low * 1.0001 else "")
                for axis, (low, high) in zip("xyz", grid.spacing(), strict=True)
            )
            lines.append(
                f"  DIVEMesh's grid: {nx} x {ny} x {nz} = {nx * ny * nz:,} cells, {sizes} m"
            )
        if self.sides:
            parts = []
            for side in (1, 4, 3, 2, 5, 6):
                kind = SIDE_KINDS.get(self.sides.get(side, 21), (str(self.sides.get(side)), ""))[0]
                parts.append(f"{SIDES[side][2]} {kind}")
            lines.append("  sides: " + ", ".join(parts))
        if self.water_level is not None:
            lines.append(f"  still water level: {self.water_level:g} m (F 60)")
        if self.waves:
            from lagoonviz.core.data.controlfile import catalog_for

            kind = self.ctrl.int("B 92", 0)
            info = catalog_for(CTRL).get("B 92")
            name = info.options.get(str(kind), "").rstrip(" *") if info else ""
            text = f"  waves: {name or 'type'} (B 92 {kind})"
            wave = self.ctrl.values("B 93") or self.ctrl.values("B 91")
            if wave and len(wave) >= 2:
                text += f", H {wave[0]:g} m, " + (
                    f"T {wave[1]:g} s" if "B 93" in self.ctrl else f"L {wave[1]:g} m"
                )
            lines.append(text)
        counts = {}
        for mark in self.marks():
            if mark.code:
                counts[mark.legend] = counts.get(mark.legend, 0) + 1
        for legend, n in counts.items():
            lines.append(f"  {legend}: {n}")
        processes = (self.control.int("M 10", 0), self.ctrl.int("M 10", 0))
        if all(processes) and processes[0] != processes[1]:
            lines.append(
                f"  M 10 differs: {processes[0]} in control.txt, {processes[1]} in ctrl.txt "
                "(REEF3D stops)"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


@dataclass
class Level:
    """One level of a refined grid, for the sketch's table."""

    level: int
    size: float  # cell size [m]
    columns: int  # about how many columns (cells in plan) it has
    where: str  # "whole domain", "inside the boxes", ...


@dataclass
class Refinement:
    """The mesh refinement of a case (ctrl.txt G 1, G 3, G 10, G 11)."""

    levels: list[Level]
    boxes: list[tuple[float, float, float, float]]  # G 10: xs, xe, ys, ye
    #: the run refines more where it flags cells (G 12 body, G 20 - G 23 surface)
    flagged: list[str]
    #: columns of a uniform grid as fine as the finest level
    uniform: int
    uniform_cells: tuple[int, int]


def _union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """The area the boxes cover together (overlaps once)."""
    boxes = [b for b in boxes if b[1] > b[0] and b[3] > b[2]]
    if not boxes:
        return 0.0
    xs = np.unique([v for b in boxes for v in b[:2]])
    ys = np.unique([v for b in boxes for v in b[2:]])
    covered = np.zeros((len(ys) - 1, len(xs) - 1), bool)
    for x0, x1, y0, y1 in boxes:
        i0, i1 = np.searchsorted(xs, [x0, x1])
        j0, j1 = np.searchsorted(ys, [y0, y1])
        covered[j0:j1, i0:i1] = True
    areas = np.outer(np.diff(ys), np.diff(xs))
    return float(areas[covered].sum())


def _refinement(setup: CaseSetup) -> Refinement | None:
    ctrl = setup.ctrl
    levels = ctrl.int("G 1", 0)
    if levels <= 0 or setup.solver not in ("SFLOW", "FNPF", "NHFLOW"):
        return None
    if setup.domain is None or setup.cells is None:
        return None
    xs, xe, ys, ye, _, _ = setup.domain
    nx, ny, _ = setup.cells
    dx = (xe - xs) / max(nx, 1)
    buffer = ctrl.int("G 3", 2)
    boxes = []
    for values in ctrl.all("G 10"):
        if len(values) >= 4:
            x0, x1, y0, y1 = (float(v) for v in values[:4])
            boxes.append((max(x0, xs), min(x1, xe), max(y0, ys), min(y1, ye)))
    rows = [Level(0, dx, nx * ny, "whole domain")]
    for level in range(1, levels + 1):
        size = dx / 2**level
        # the finest level fills the boxes; a coarser one holds the next finer with a
        # buffer of G 3 of its cells around it
        margin = 0.0 if level == levels else (levels - level) * buffer * size
        grown = [
            (max(a - margin, xs), min(b + margin, xe), max(c - margin, ys), min(d + margin, ye))
            for a, b, c, d in boxes
        ]
        area = _union_area(grown)
        where = "inside the boxes" if level == levels else "the boxes and a nesting buffer"
        rows.append(Level(level, size, round(area / size**2), where))
    flagged = []
    if ctrl.int("G 12", 0) > 0:
        flagged.append("around the floating body (G 12)")
    for code, what in (
        ("G 20", "surface jumps"),
        ("G 21", "surface curvature"),
        ("G 22", "the shoreline"),
        ("G 23", "breaking"),
    ):
        if (ctrl.value(code) or 0) > 0:
            flagged.append(f"{what} ({code})")
    finest = dx / 2**levels
    ux, uy = round((xe - xs) / finest), round((ye - ys) / finest)
    return Refinement(rows, boxes, flagged, ux * uy, (ux, uy))


def _option(code: str, value: int) -> str:
    from lagoonviz.core.data.controlfile import catalog_for

    info = catalog_for(CTRL).get(code)
    name = info.options.get(str(value), "") if info else ""
    return name.rstrip(" *").replace('\\"', "")


def _number(value: float) -> str:
    """A number for the table: thin spaces between thousands, 4 significant digits."""
    if abs(value) >= 1e4 and float(value).is_integer():
        return f"{int(value):,}".replace(",", "\u2009")
    return f"{value:.4g}"


def _facts(setup: CaseSetup) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    ctrl, control = setup.ctrl, setup.control
    if ctrl:
        rows.append(("Module", f"REEF3D::{setup.solver} (A 10)"))
    if setup.domain is not None:
        xs, xe, ys, ye, zs, ze = setup.domain
        plan = (
            f"{_number(xe - xs)} × {_number(ye - ys)} m"
            if not setup.two_d
            else f"{_number(xe - xs)} m (2D)"
        )
        if setup.solver in ("FNPF", "NHFLOW", "SFLOW"):
            rows.append(("Domain", f"{plan}, bed to free surface (σ grid)"))
        else:
            rows.append(("Domain", f"{plan}, z {zs:g} – {ze:g} m"))
    if setup.cells is not None:
        nx, ny, nz = setup.cells
        text = f"{nx} × {ny} × {nz} = {_number(nx * ny * nz)}"
        spacing = setup.spacing
        if setup.stretched:
            text += ", stretched"
        elif spacing is not None:
            text += f", dx = {spacing[0]:.4g} m"
        rows.append(("Cells", text))
    if setup.water_level is not None:
        rows.append(("Still water", f"z = {setup.water_level:g} m"))
    if setup.waves:
        kind = ctrl.int("B 92", 0)
        name = _option("B 92", kind) or f"type {kind}"
        text = name[0].upper() + name[1:] if name else name
        wave = ctrl.values("B 93") or ctrl.values("B 91")
        if wave and len(wave) >= 2:
            second = f"T = {wave[1]:g} s" if "B 93" in ctrl else f"L = {wave[1]:g} m"
            text += f", H = {wave[0]:g} m, {second}"
        rows.append(("Waves", text))
        generation = ctrl.int("B 98", 0)
        if generation:
            rows.append(("Generation", f"{_option('B 98', generation) or generation} (B 98)"))
        absorption = ctrl.int("B 99", 0)
        if absorption:
            rows.append(("Absorption", f"{_option('B 99', absorption) or absorption} (B 99)"))
    if 1 in setup.sides.values() or ctrl.int("B 60", 0) > 0:
        discharge = ctrl.value("W 10")
        if discharge:
            rows.append(("Inflow", f"{discharge:g} m³/s (W 10)"))
    end = ctrl.value("N 41")
    if end and end < 1e18:
        rows.append(("Simulated time", f"{end:g} s (N 41)"))
    every = ctrl.value("P 30")
    if every and every > 0:
        rows.append(("Output every", f"{every:g} s (P 30)"))
    elif (ctrl.value("P 20") or 0) > 0:
        rows.append(("Output every", f"{ctrl.int('P 20', 0)} iterations (P 20)"))
    refinement = _refinement(setup)
    if refinement is not None:
        text = (
            f"{len(refinement.levels) - 1} levels (G 1), finest {refinement.levels[-1].size:.3g} m"
        )
        if refinement.boxes:
            text += f", {len(refinement.boxes)} boxes (G 10)"
        rows.append(("Refinement", text))
    processes = ctrl.int("M 10", 0) or control.int("M 10", 0)
    if processes:
        rows.append(("Processes", f"{processes} (M 10)"))
    return rows


def _raster(points: np.ndarray, domain, size: int | None = None):
    """Bed points averaged into a regular raster over the domain (or the points'
    box); cells without points are filled from their neighbours, as DIVEMesh
    extends the bed over the whole domain."""
    points = points[np.isfinite(points).all(axis=1)] if len(points) else points
    if len(points) < 3:
        return None
    if domain is not None:
        x0, x1, y0, y1 = domain[:4]
    else:
        (x0, y0), (x1, y1) = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    lx, ly = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    if size is None:  # about as many cells as points, 400 at most along the longer side
        size = int(np.clip(math.sqrt(len(points)) * 1.2, 40, 400))
    nx = max(2, round(size if lx >= ly else size * lx / ly))
    ny = max(2, round(size if ly > lx else size * ly / lx))
    i = np.clip(((points[:, 0] - x0) / lx * nx).astype(int), 0, nx - 1)
    j = np.clip(((points[:, 1] - y0) / ly * ny).astype(int), 0, ny - 1)
    inside = (
        (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    )
    cell = (j * nx + i)[inside]
    sums = np.bincount(cell, points[inside, 2], nx * ny)
    counts = np.bincount(cell, None, nx * ny)
    z = np.full(nx * ny, np.nan)
    z[counts > 0] = sums[counts > 0] / counts[counts > 0]
    z = z.reshape(ny, nx)
    for _ in range(nx + ny):  # fill the gaps from the neighbours, as DIVEMesh extends the bed
        missing = np.isnan(z)
        if not missing.any():
            break
        padded = np.pad(z, 1, constant_values=np.nan)
        near = np.stack(
            [padded[1 + a : 1 + a + ny, 1 + b : 1 + b + nx] for a in (-1, 0, 1) for b in (-1, 0, 1)]
        )
        valid = np.isfinite(near)
        total = np.where(valid, near, 0.0).sum(axis=0)
        number = valid.sum(axis=0)
        fill = missing & (number >= 1)
        if not fill.any():
            break
        z[fill] = total[fill] / number[fill]
    x = x0 + (np.arange(nx) + 0.5) * lx / nx
    y = y0 + (np.arange(ny) + 0.5) * ly / ny
    return x, y, z


def read_setup(case_dir: str | Path) -> CaseSetup:
    """The set-up of a case folder (or of the folder of a file in it)."""
    case_dir = Path(case_dir).expanduser()
    if case_dir.is_file():
        case_dir = case_dir.parent
    control = read_commands(case_dir / CONTROL)
    ctrl_path = ctrl_file(case_dir)
    ctrl = read_commands(ctrl_path) if ctrl_path is not None else Commands(case_dir / CTRL)
    setup = CaseSetup(case_dir, control, ctrl)
    setup.domain = _domain(setup)
    setup.cells = _cells(setup)
    setup.sides = _sides(setup)
    setup.water_level = _water_level(setup)
    setup._revisions = (control.revision, ctrl.revision)
    return setup


# ------------------------------------------------------------------ facts
def _domain(setup: CaseSetup) -> tuple | None:
    """B 10 (several boxes: the box around them); else from geo.dat (G 20 with its
    margins G 21, G 22) or the STL (S 2 with S 3).

    FNPF, NHFLOW and SFLOW have σ-grids from the bed to the free surface: when B 10's
    heights do not reach the still water (F 60, e.g. 0 - 1 m for a bed of geo.dat
    from 50 to 300 m), the domain reaches from the bed's lowest point (geo.dat, else
    B 10's bottom) to it."""
    boxes = [v[:6] for v in setup.control.all("B 10") if len(v) >= 6]
    if boxes:
        b = np.asarray(boxes, float)
        box = [
            float(b[:, 0].min()),
            float(b[:, 1].max()),
            float(b[:, 2].min()),
            float(b[:, 3].max()),
            float(b[:, 4].min()),
            float(b[:, 5].max()),
        ]
        water = setup.ctrl.value("F 60")
        sigma = setup.ctrl.int("A 10", 6) in (2, 3, 5)
        if sigma and water is not None and abs(water) < 1e18 and not box[4] <= water <= box[5]:
            if (setup.path / GEO).is_file() and setup.control.int("G 10", 0) > 0:
                points = setup.geo_points()
                if len(points):
                    box[4] = float(points[:, 2].min())
            box[4] = min(box[4], water)
            box[5] = max(water, box[4] + 1e-6)
        return tuple(box)
    control = setup.control
    if control.int("G 20", 0) > 0 and (setup.path / GEO).is_file():
        points = setup.geo_points()
        if len(points):
            low, high = points.min(axis=0), points.max(axis=0)
            start = control.values("G 21") or [0.0, 0.0, 0.0]
            end = control.values("G 22") or [0.0, 0.0, 0.0]
            start, end = (start + [0.0] * 3)[:3], (end + [0.0] * 3)[:3]
            return (
                float(low[0] - start[0]),
                float(high[0] + end[0]),
                float(low[1] - start[1]),
                float(high[1] + end[1]),
                float(low[2] - start[2]),
                float(high[2] + end[2]),
            )
    return None


def _cells(setup: CaseSetup) -> tuple[int, int, int] | None:
    """B 2 (counts), else from B 1 (the cell size) and the domain."""
    counts = setup.control.values("B 2")
    if counts and len(counts) >= 3 and min(counts[:3]) > 0:
        return int(counts[0]), int(counts[1]), int(counts[2])
    dx = setup.control.value("B 1")
    if dx and dx > 0 and setup.domain is not None:
        xs, xe, ys, ye, zs, ze = setup.domain
        return (
            max(1, round((xe - xs) / dx)),
            max(1, round((ye - ys) / dx)),
            max(1, round((ze - zs) / dx)),
        )
    return None


def _sides(setup: CaseSetup) -> dict[int, int]:
    sides = {side: setup.control.int(f"C 1{side}", 21) for side in range(1, 7)}
    if setup.cells is not None and setup.cells[1] == 1:  # 2D: the y sides are symmetry planes
        sides[2] = sides[3] = 3
    return sides


def _water_level(setup: CaseSetup) -> float | None:
    level = setup.ctrl.value("F 60")
    if level is None or abs(level) > 1e18:
        return None
    return level


def _read_geo(path: Path, control: ControlFile) -> np.ndarray:
    """geo.dat: x y z per line (a letter before them with G 19 1), scaled (G 12) and
    with the sign of z reversed with G 23 1."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return np.zeros((0, 3))
    if control.int("G 19", 0) > 0:
        text = "\n".join(
            line.split(None, 1)[1] if len(line.split()) > 3 else line for line in text.splitlines()
        )
    values = np.array(text.split(), dtype=object)
    try:
        numbers = values.astype(float)
    except ValueError:
        numbers = np.array([v for v in values if _is_number(v)], float)
    numbers = numbers[: len(numbers) // 3 * 3].reshape(-1, 3)
    factors = control.values("G 12")
    if factors and len(factors) >= 3:
        numbers = numbers * np.asarray(factors[:3], float)
    if control.int("G 23", 0) > 0:
        numbers[:, 2] *= -1
    return numbers


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


# ------------------------------------------------------------------ the marks
#: ctrl.txt commands drawn as marks: code -> (group, shape, legend, colour)
#:
#: shapes (the values they read):
#:   "staff" x, y: a vertical line over the water column
#:   "line_x" y: a line along x;  "line_y" x: a line along y
#:   "plane_x" x: a section across the domain;  "window_x" x, zs, ze
#:   "point" x, y, z;  "point_xy" x, y: at the water
#:   "segment" xs, xe, ys, ye, zs, ze;  "segment_xy" xs, xe, ys, ye: at the water
#:   "box" xs, xe, ys, ye, zs, ze;  "box_after2": two values, then a box
#:   "box7", "box8": a box with hinge heights (flap wavemakers)
#:   "rect" xs, xe, ys, ye: at the water;  "rect_h": at the level given after them
#:   "zone" xs, xe, ys, ye, d: a line and its band of +- d
#:   "ycyl" xc, zc, ys, ye, r;  "zcyl" xc, yc, zs, ze, r;  "zcyl_full" xc, yc, r
#:   "cyl_tilted" xs, ys, zs, rs, xe, ye, ze, re;  "sphere" xc, yc, zc, r
#:   "body_cyl_x/y/z" r, h, xc, yc, zc;  "corners6", "corners8": x, y, z of each
#:   "net" D, L, x0, y0, z0: a cylinder hanging down from z0
MARKS = {
    # gauges
    "P 50": ("gauges", "staff", "theoretical wave gauge", COLORS["gauges"]),
    "P 51": ("gauges", "staff", "wave gauge", COLORS["gauges"]),
    "P 58": ("gauges", "staff", "theoretical wave time series", COLORS["gauges"]),
    "P 121": ("gauges", "staff", "bed level gauge", COLORS["bed"]),
    "P 125": ("gauges", "staff", "bed shear stress gauge", COLORS["bed"]),
    "P 140": ("gauges", "staff", "run-up on a cylinder", COLORS["gauges"]),
    "P 351": ("gauges", "staff", "wave gauge of level set 1", COLORS["gauges"]),
    "P 352": ("gauges", "staff", "wave gauge of level set 2", COLORS["gauges"]),
    "P 52": ("gauges", "line_x", "free surface line in x", COLORS["gauges"]),
    "P 56": ("gauges", "line_y", "free surface line in y", COLORS["gauges"]),
    "P 123": ("gauges", "line_x", "bed profile in x", COLORS["bed"]),
    "P 124": ("gauges", "line_y", "bed profile in y", COLORS["bed"]),
    "P 133": ("gauges", "line_x", "run-up gauge in x", COLORS["gauges"]),
    "P 134": ("gauges", "line_x", "maximum run-up gauge in x", COLORS["gauges"]),
    "P 146": ("gauges", "line_x", "time-averaged free surface line in x", COLORS["gauges"]),
    "P 147": ("gauges", "line_y", "time-averaged free surface line in y", COLORS["gauges"]),
    # probes
    "P 61": ("probes", "point", "point probe", COLORS["probes"]),
    "P 64": ("probes", "point", "pressure probe", COLORS["probes"]),
    "P 65": ("probes", "point", "velocity probe", COLORS["probes"]),
    "P 66": ("probes", "point", "velocity probe of the theoretical wave", COLORS["probes"]),
    "B 32": ("probes", "point", "pressure reference point", COLORS["probes"]),
    "X 324": ("probes", "point", "net knot probe", COLORS["probes"]),
    "P 62": ("probes", "segment", "line probe", COLORS["probes"]),
    "P 63": ("probes", "staff", "depth-averaged point probe", COLORS["probes"]),
    "P 67": ("probes", "staff", "vertical velocity profile", COLORS["probes"]),
    "P 88": ("probes", "staff", "wave kinematics", COLORS["probes"]),
    "P 148": ("probes", "staff", "time-averaged velocity profile", COLORS["probes"]),
    "P 144": ("probes", "segment_xy", "depth-averaged velocity line probe", COLORS["probes"]),
    "P 145": ("probes", "segment_xy", "time-averaged velocity line probe", COLORS["probes"]),
    # sections
    "P 167": ("sections", "plane_x", "discharge gauge", COLORS["sections"]),
    "P 168": ("sections", "window_x", "discharge window", COLORS["sections"]),
    "P 230": ("sections", "plane_x", "FlowFile section", COLORS["sections"]),
    "P 240": ("sections", "plane_x", "PotentialFile section", COLORS["sections"]),
    # forces
    "P 81": ("forces", "box", "force box", COLORS["forces"]),
    "P 85": ("forces", "zcyl_full", "Morison cylinder", COLORS["forces"]),
    # waves
    "B 81": ("waves", "point_xy", "focus point", COLORS["waves"]),
    "X 170": ("waves", "box", "piston wavemaker", COLORS["waves"]),
    "X 171": ("waves", "box7", "flap wavemaker", COLORS["waves"]),
    "X 172": ("waves", "box8", "double-hinged flap wavemaker", COLORS["waves"]),
    # relaxation zones
    "B 107": ("zones", "zone", "numerical beach zone", COLORS["beach"]),
    "B 108": ("zones", "zone", "wave generation zone", COLORS["generation"]),
    # floating bodies
    "X 110": ("bodies", "box", "floating body", COLORS["bodies"]),
    "X 131": ("bodies", "body_cyl_x", "floating cylinder", COLORS["bodies"]),
    "X 132": ("bodies", "body_cyl_y", "floating cylinder", COLORS["bodies"]),
    "X 133": ("bodies", "body_cyl_z", "floating cylinder", COLORS["bodies"]),
    "X 153": ("bodies", "box", "floating wedge", COLORS["bodies"]),
    "X 163": ("bodies", "corners6", "floating wedge", COLORS["bodies"]),
    "X 164": ("bodies", "corners8", "floating hexahedron", COLORS["bodies"]),
    "X 23": ("bodies", "point", "centre of gravity", COLORS["bodies"]),
    "X 311": ("bodies", "segment", "mooring line", COLORS["moorings"]),
    "X 312": ("bodies", "segment", "mooring spring", COLORS["moorings"]),
    "X 322": ("bodies", "net", "net", COLORS["moorings"]),
    # structures: direct forcing (NHFLOW) and porous media (VRANS)
    "A 581": ("structures", "box", "direct forcing box", COLORS["structures"]),
    "A 583": ("structures", "ycyl", "direct forcing cylinder", COLORS["structures"]),
    "A 584": ("structures", "zcyl", "direct forcing cylinder", COLORS["structures"]),
    "A 585": ("structures", "cyl_tilted", "direct forcing cylinder", COLORS["structures"]),
    "A 586": ("structures", "sphere", "direct forcing sphere", COLORS["structures"]),
    "A 587": ("structures", "box", "direct forcing wedge", COLORS["structures"]),
    "A 588": ("structures", "box", "direct forcing wedge", COLORS["structures"]),
    "A 589": ("structures", "box", "direct forcing wedge", COLORS["structures"]),
    "B 210": ("structures", "box", "porous box (VRANS)", COLORS["structures"]),
    "B 212": ("structures", "ycyl", "porous cylinder (VRANS)", COLORS["structures"]),
    "B 213": ("structures", "zcyl", "porous cylinder (VRANS)", COLORS["structures"]),
    "B 215": ("structures", "sphere", "porous sphere (VRANS)", COLORS["structures"]),
    "B 217": ("structures", "box", "porous wedge (VRANS)", COLORS["structures"]),
    "B 218": ("structures", "box", "porous wedge (VRANS)", COLORS["structures"]),
    "B 219": ("structures", "box", "porous wedge (VRANS)", COLORS["structures"]),
    "B 240": ("structures", "box_after2", "Darcy porous box", COLORS["structures"]),
    "B 270": ("structures", "box", "porous box (VRANS)", COLORS["structures"]),
    # regions
    "A 276": ("regions", "rect", "mesh refinement box", COLORS["regions"]),
    "A 277": ("regions", "rect", "box without mesh refinement", COLORS["regions"]),
    "A 372": ("regions", "rect", "wind forcing area", COLORS["regions"]),
    "A 572": ("regions", "rect", "wind forcing area", COLORS["regions"]),
    "F 72": ("regions", "rect_h", "prescribed water level region", COLORS["water"]),
    "S 74": ("regions", "rect", "non-erodible zone", COLORS["bed"]),
    "P 43": ("regions", "rect", "state file area", COLORS["regions"]),
    # mesh refinement of SFLOW, FNPF and NHFLOW (G 1 levels)
    "G 10": ("regions", "rect", "refinement box", COLORS["refinement"]),
    "G 11": ("regions", "rect", "box without refinement", COLORS["domain"]),
}

#: control.txt topo objects (T) drawn as outlines of the bed: the S builders
TOPO = (10, 32, 33, 41, 42, 43, 51, 61, 62, 63, 81, 82, 83, 84)


def _marks(setup: CaseSetup) -> list[Mark]:
    marks: list[Mark] = []
    setup.notes = [n for n in setup.notes if not n.startswith("not drawn:")]
    domain = setup.domain
    low_z, high_z = (domain[4], domain[5]) if domain is not None else (None, None)
    water = setup.water_level
    if water is None and domain is not None:
        water_z = high_z
    else:
        water_z = water if water is not None else 0.0

    # ---- the domain
    boxes = [v[:6] for v in setup.control.all("B 10") if len(v) >= 6]
    if not boxes and domain is not None:
        boxes = [list(domain)]
    if boxes and domain is not None:
        raw = (min(b[4] for b in boxes), max(b[5] for b in boxes))
        if raw != (domain[4], domain[5]):  # a σ-grid from the bed to the still water
            boxes = [list(b[:4]) + [domain[4], domain[5]] for b in boxes]
    for index, box in enumerate(boxes):
        marks.append(
            Mark(
                "domain",
                f"domain (from {setup.domain_from})" if setup.domain_from else "domain (B 10)",
                COLORS["domain"],
                segments=_box_edges(*box),
                code="" if setup.domain_from else "B 10",
                file=CONTROL,
                occurrence=index,
            )
        )
    if domain is not None:
        xs, xe, ys, ye, zs, ze = domain
        # ---- the sides: the kind of each vertical side, as a label and a coloured
        # outline just inside the domain's edges
        for side in (1, 4, 3, 2):
            kind = setup.sides.get(side, 21)
            if kind == 21:
                continue  # walls: the domain's own lines
            name, color = SIDE_KINDS.get(kind, (f"C 1{side} {kind}", COLORS["domain"]))
            axis, at_end, _ = SIDES[side]
            if axis == 1 and setup.two_d:
                continue
            if axis == 0:
                x = xe if at_end else xs
                segments = _segments(
                    [
                        ((x, ys, zs), (x, ye, zs)),
                        ((x, ys, ze), (x, ye, ze)),
                        ((x, ys, zs), (x, ys, ze)),
                        ((x, ye, zs), (x, ye, ze)),
                    ]
                )
                where = (x, 0.5 * (ys + ye), ze)
            else:
                y = ye if at_end else ys
                segments = _segments(
                    [
                        ((xs, y, zs), (xe, y, zs)),
                        ((xs, y, ze), (xe, y, ze)),
                        ((xs, y, zs), (xs, y, ze)),
                        ((xe, y, zs), (xe, y, ze)),
                    ]
                )
                where = (0.5 * (xs + xe), y, ze)
            marks.append(
                Mark(
                    "sides",
                    f"{name} side (C 1{side})",
                    color,
                    segments=segments,
                    labels=[(name, where)],
                    code=f"C 1{side}",
                    file=CONTROL,
                )
            )
        # ---- the still water
        if water is not None and zs <= water <= ze:
            marks.append(
                Mark(
                    "water",
                    "still water level (F 60)",
                    COLORS["water"],
                    segments=_rect(xs, xe, ys, ye, water),
                    faces=_rect_faces(xs, xe, ys, ye, water),
                    code="F 60",
                )
            )
        # ---- relaxation zones of B 96 (when B 107, B 108 do not replace them)
        zones = setup.ctrl.values("B 96")
        if zones and len(zones) >= 2:
            relaxation = setup.ctrl.int("B 98", 0) == 2, setup.ctrl.int("B 99", 0) in (1, 2)
            for length, active, at_end, legend, color, replaced in (
                (
                    zones[0],
                    relaxation[0],
                    False,
                    "wave generation zone (B 96)",
                    COLORS["generation"],
                    "B 108",
                ),
                (
                    zones[1],
                    relaxation[1],
                    True,
                    "numerical beach zone (B 96)",
                    COLORS["beach"],
                    "B 107",
                ),
            ):
                if length <= 0 or not active or replaced in setup.ctrl:
                    continue
                a, b = (xe - length, xe) if at_end else (xs, xs + length)
                marks.append(
                    Mark(
                        "zones",
                        legend,
                        color,
                        segments=_rect(a, b, ys, ye, water_z),
                        faces=_rect_faces(a, b, ys, ye, water_z),
                        labels=[
                            (
                                legend.split(" (")[0].replace(" zone", ""),
                                (0.5 * (a + b), 0.5 * (ys + ye), water_z),
                            )
                        ],
                        code="B 96",
                    )
                )
        # ---- wave direction (B 105) and the generated wave's origin
        if setup.waves:
            values = setup.ctrl.values("B 105") or [0.0, xs, ys]
            angle = values[0] if values else 0.0
            x0 = values[1] if len(values) > 1 else xs
            y0 = values[2] if len(values) > 2 else ys
            if "B 105" not in setup.ctrl:
                y0 = 0.5 * (ys + ye)  # the arrow along the middle of the domain
            length = 0.15 * max(xe - xs, ye - ys)
            marks.append(
                Mark(
                    "waves",
                    "wave direction (B 105)",
                    COLORS["waves"],
                    segments=_arrow((x0, y0, water_z), angle, length),
                    labels=[("waves", (x0, y0, water_z))],
                    code="B 105" if "B 105" in setup.ctrl else "",
                )
            )

    # ---- ctrl.txt commands
    span_z = (low_z, high_z) if low_z is not None else (water_z - 1.0, water_z + 0.5)
    span_x = (domain[0], domain[1]) if domain is not None else None
    span_y = (domain[2], domain[3]) if domain is not None else None
    skipped = []
    for code in setup.ctrl.codes():
        entry = MARKS.get(code)
        if entry is None:
            continue
        group, shape, legend, color = entry
        for occurrence, line in enumerate(setup.ctrl.entries(code)):
            values = line.numbers()
            try:
                mark = _shaped(shape, values or [], span_x, span_y, span_z, water_z)
            except (IndexError, ValueError, TypeError, ZeroDivisionError):
                mark = None
            if mark is None:
                skipped.append(code)
                continue
            segments, points, where = mark
            label = (
                str(occurrence + 1) if group in ("gauges", "probes", "forces", "sections") else ""
            )
            if code == "B 502" and values:
                label = f"{int(values[0])}"
            faces = None
            if shape == "zone" and len(segments) > 3:  # the band, filled
                faces = _fan(segments[1:, 0, :])
            marks.append(
                Mark(
                    group,
                    f"{legend} ({code})",
                    color,
                    segments=segments,
                    points=points,
                    labels=[(label, where)] if label else [],
                    code=code,
                    occurrence=occurrence,
                    faces=faces,
                )
            )
    # ---- control.txt: solids and the bed's topo objects
    from lagoonviz.core.data.divemesh import BUILDERS

    for letter, group, legend, color in (
        ("S", "solids", "solid", COLORS["solids"]),
        ("T", "bed", "topo object", COLORS["bed"]),
    ):
        for code in setup.control.codes():
            if not code.startswith(letter + " "):
                continue
            number = int(code.split()[1])
            builder = BUILDERS.get(number) if number in TOPO else None
            if builder is None:
                continue
            for occurrence, values in enumerate(setup.control.all(code)):
                try:
                    triangles = builder(values, domain)
                except (IndexError, ValueError, ZeroDivisionError):
                    triangles = None
                if not triangles:
                    skipped.append(code)
                    continue
                marks.append(
                    Mark(
                        group,
                        f"{legend} ({code})",
                        color,
                        segments=_feature_edges(np.asarray(triangles, float)),
                        code=code,
                        occurrence=occurrence,
                        file=CONTROL,
                    )
                )
    if (setup.path / GEO).is_file() and setup.control.int("G 10", 0) > 0:
        points = setup.geo_points()
        if len(points):
            marks.append(
                Mark(
                    "bed",
                    "geo.dat points (G 10)",
                    COLORS["bed"],
                    points=points,
                    code="G 10",
                    file=CONTROL,
                )
            )
    if skipped:
        setup.notes.append("not drawn: " + ", ".join(sorted(set(skipped))))
    return marks


def _arrow(start, angle_deg: float, length: float) -> np.ndarray:
    a = math.radians(angle_deg)
    d = np.array([math.cos(a), math.sin(a), 0.0])
    side = np.array([-d[1], d[0], 0.0])
    p0 = np.asarray(start, float)
    p1 = p0 + length * d
    head = 0.18 * length
    return _segments(
        [(p0, p1), (p1, p1 - head * d + 0.5 * head * side), (p1, p1 - head * d - 0.5 * head * side)]
    )


def _shaped(shape: str, v: list[float], span_x, span_y, span_z, water: float):
    """Lines, points and the label position of one command line (None: not enough
    to place it, e.g. a line across a domain that is not known)."""
    zs, ze = span_z
    none = np.zeros((0, 3))
    if shape == "staff":
        x, y = v[0], v[1]
        return _segments([((x, y, zs), (x, y, ze))]), np.array([(x, y, water)]), (x, y, ze)
    if shape == "line_x":
        if span_x is None:
            return None
        y = v[0]
        return (
            _segments([((span_x[0], y, water), (span_x[1], y, water))]),
            none,
            (span_x[0], y, water),
        )
    if shape == "line_y":
        if span_y is None:
            return None
        x = v[0]
        return (
            _segments([((x, span_y[0], water), (x, span_y[1], water))]),
            none,
            (x, span_y[0], water),
        )
    if shape in ("plane_x", "window_x"):
        if span_y is None:
            return None
        x = v[0]
        z0, z1 = (v[1], v[2]) if shape == "window_x" else (zs, ze)
        return _rect_x(x, span_y[0], span_y[1], z0, z1), none, (x, span_y[0], z1)
    if shape == "point":
        p = (v[0], v[1], v[2])
        return np.zeros((0, 2, 3)), np.array([p]), p
    if shape == "point_xy":
        p = (v[0], v[1], water)
        return np.zeros((0, 2, 3)), np.array([p]), p
    if shape == "segment":
        a, b = (v[0], v[2], v[4]), (v[1], v[3], v[5])
        return _segments([(a, b)]), np.array([a, b]), b
    if shape == "segment_xy":
        a, b = (v[0], v[2], water), (v[1], v[3], water)
        return _segments([(a, b)]), none, a
    if shape == "box":
        return _box_edges(*v[:6]), none, (v[0], v[2], v[5])
    if shape == "box_after2":
        return _box_edges(*v[2:8]), none, (v[2], v[4], v[7])
    if shape == "box7":  # xs, xe, ys, ye, zs, z_hinge, ze
        return _box_edges(v[0], v[1], v[2], v[3], v[4], v[6]), none, (v[0], v[2], v[6])
    if shape == "box8":  # xs, xe, ys, ye, zs, z_hinge1, z_hinge2, ze
        return _box_edges(v[0], v[1], v[2], v[3], v[4], v[7]), none, (v[0], v[2], v[7])
    if shape in ("rect", "rect_h"):  # rect_h: at the water level given (F 72)
        z = v[4] if shape == "rect_h" else water
        return _rect(v[0], v[1], v[2], v[3], z), none, (v[0], v[2], z)
    if shape == "zone":  # the line where the zone acts fully, and its band of +- d
        xs, xe, ys, ye, d = v[:5]
        line = np.array([xe - xs, ye - ys], float)
        length = float(np.hypot(*line))
        if length < 1e-12:
            return None
        normal = np.array([-line[1], line[0]]) / length * d
        a, b = np.array([xs, ys]), np.array([xe, ye])
        ring = [a + normal, b + normal, b - normal, a - normal]
        if span_x is not None and span_y is not None:  # the zone acts inside the domain
            ring = _clip_to_box(ring, span_x, span_y)
            if len(ring) < 3:
                return None
        corners = [(p[0], p[1], water) for p in ring]
        segments = [((xs, ys, water), (xe, ye, water))]
        segments += [(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))]
        centre = np.mean(ring, axis=0)
        return _segments(segments), none, (float(centre[0]), float(centre[1]), water)
    if shape == "ycyl":  # xc, zc, ys, ye, r
        return _cylinder(1, (v[0], v[1]), v[2], v[3], v[4]), none, (v[0], v[3], v[1] + v[4])
    if shape == "zcyl":  # xc, yc, zs, ze, r
        return _cylinder(2, (v[0], v[1]), v[2], v[3], v[4]), none, (v[0], v[1], v[3])
    if shape == "zcyl_full":  # xc, yc, r over the water column
        return _cylinder(2, (v[0], v[1]), zs, ze, v[2]), none, (v[0], v[1], ze)
    if shape == "cyl_tilted":  # xs, ys, zs, rs, xe, ye, ze, re
        return _tilted_cylinder(v[0:3], v[4:7], v[3], v[7]), none, (v[4], v[5], v[6])
    if shape == "sphere":
        return _sphere(v[:3], v[3]), none, (v[0], v[1], v[2] + v[3])
    if shape.startswith("body_cyl_"):  # r, h, xc, yc, zc
        axis = "xyz".index(shape[-1])
        r, h, c = v[0], v[1], (v[2], v[3], v[4])
        others = [c[i] for i in range(3) if i != axis]
        return _cylinder(axis, others, c[axis] - 0.5 * h, c[axis] + 0.5 * h, r), none, c
    if shape in ("corners6", "corners8"):
        n = 6 if shape == "corners6" else 8
        corners = np.asarray(v[: 3 * n], float).reshape(n, 3)
        return _hull_edges(corners), none, tuple(corners.max(axis=0))
    if shape == "net":  # D, L, x0, y0, z0: a cylinder of the net hanging down from z0
        diameter, length, x0, y0, z0 = v[:5]
        return _cylinder(2, (x0, y0), z0 - length, z0, 0.5 * diameter), none, (x0, y0, z0)
    raise ValueError(shape)


def _clip_to_box(ring, span_x, span_y) -> list[np.ndarray]:
    """A convex polygon (2D points) clipped to the box span_x x span_y."""
    points = [np.asarray(p, float) for p in ring]
    for axis, bound, keep_above in (
        (0, span_x[0], True),
        (0, span_x[1], False),
        (1, span_y[0], True),
        (1, span_y[1], False),
    ):
        if not points:
            break

        def inside(p, axis=axis, bound=bound, keep_above=keep_above):
            return p[axis] >= bound - 1e-9 if keep_above else p[axis] <= bound + 1e-9

        clipped = []
        for i, current in enumerate(points):
            previous = points[i - 1]
            if inside(current) != inside(previous):
                t = (bound - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(previous + t * (current - previous))
            if inside(current):
                clipped.append(current)
        points = clipped
    return points


def _fan(ring: np.ndarray) -> np.ndarray:
    """Triangles (k, 3, 3) of a convex polygon (m, 3)."""
    return np.array([(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)])


def _rect_x(x, ys, ye, zs, ze) -> np.ndarray:
    c = [(x, ys, zs), (x, ye, zs), (x, ye, ze), (x, ys, ze)]
    return _segments([(c[i], c[(i + 1) % 4]) for i in range(4)])
