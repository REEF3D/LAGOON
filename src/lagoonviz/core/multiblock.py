# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""A multi-block grid: several structured blocks of curvilinear cells that meet
along their faces, the level-0 grids of REEF3D's SFLOW and NHFLOW with ``G 50 1``
(DIVEMesh ``R 20 1`` reads them from ``blocks.dat``).

The grid is kept as its **topology** in ``blocks.json`` in the case folder:

- **vertices**: named points (x, y), the corners of the blocks;
- **edges**: a line from one vertex to another, shaped by ``points`` between them
  (``curve``: ``"straight"``, ``"through"`` the points (centripetal Catmull-Rom) or a
  ``"bspline"`` they pull on), cut into ``cells`` cells, equal along the line or
  growing by ``ratio`` (last cell / first cell);
- **blocks**: each with its four faces in its own index frame, ``i0`` (-i), ``i1``
  (+i), ``j0`` (-j), ``j1`` (+j), every face a chain of edges (``"e3"``, or
  ``"-e3"`` for the edge run backwards). ``i0`` and ``i1`` run along j, ``j0`` and
  ``j1`` along i; ``j`` lies to the left of ``i`` (right-handed, as REEF3D wants).

Two blocks that use the same edge meet there: they share its nodes, so the grid is
point matching by construction. The first block is **block 0**, the grid of the run
(DIVEMesh's box with a ring of ``R 21`` cells on its faces with an interface).

A block's nodes are the transfinite interpolation of its four faces (Coons, the
faces parametrised by arc length), then made smoother by Winslow's elliptic grid
equations (``smoothing`` sweeps, the faces kept).

The **boundaries** of the faces without an interface (``bc``): ``wall`` (default),
``inflow`` (a discharge ``value`` in m3/s) or ``outflow`` (a water level ``value`` in
m). On block 0 they are DIVEMesh's sides (C 11 inflow, C 14 outflow: REEF3D's B 60
1, W 10, F 62); on the blocks 1.. REEF3D's ``G 54 b 1 Q`` (the -i face) and
``G 54 b 2 WL`` (the +i face). An ``open`` face of block 0 has a condition of its own
in ctrl.txt (a tide edge, B 520): DIVEMesh's side gets the code ``value`` (2 when
none).

Edges are spaced by a ``spacing`` (the cells at their ends, a largest cell; the
cells follow from the length), made circular ``arc``s, laid along the water
(:mod:`lagoonviz.core.waterline`), and a blocks.dat made elsewhere is read back into
a topology by :func:`from_blocks` (its faces' nodes kept as ``"nodes"`` edges).

:func:`check` lists what DIVEMesh and REEF3D would refuse or warn about (the
**rules**), :func:`quality` how good the cells are, :func:`blocks_text` writes
``blocks.dat`` (the number of blocks, then per block "ni nj" and x y zb of every node,
i fastest).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from lagoonviz.core import corridor as curvi

FILE = "blocks.json"
GRID_FILE = "blocks.dat"
#: the faces of a block in its index frame: -i, +i, -j, +j (DIVEMesh's faces 0 .. 3)
FACES = ("i0", "i1", "j0", "j1")
FACE_NAMES = {"i0": "-i", "i1": "+i", "j0": "-j", "j1": "+j"}
CURVES = ("straight", "through", "bspline", "arc", "nodes")
BCS = ("wall", "inflow", "outflow", "open")
#: an open face's DIVEMesh side code when none is given (ctrl.txt sets the condition)
OPEN_CODE = 2
#: the growth of the cells from a fine end when an edge's spacing has a largest cell
GROWTH = 1.1
BEDS = ("geo", "flat")
#: DIVEMesh's ring of block 0 (R 21's default): the blocks behind its interfaces must
#: be at least this many cells deep
RING = 8
#: REEF3D's own need behind any interface (EXT + the ghost cells): refused below
DEEP_MIN = 5
#: Winslow sweeps between two slides of the wall nodes along their edges
SLIDE_EVERY = 10
#: the wall nodes slide during this many sweeps on each grid (they keep drifting
#: slowly after that, without making the cells better)
SLIDE_SWEEPS = 200
#: the classes of the cells' quality (colours in the set-up), worst last
CLASSES = ("good", "fair", "poor", "folded")


# ====================================================================== edges
@dataclass
class Edge:
    """A line from vertex ``start`` to vertex ``end`` cut into ``cells`` cells.

    ``curve``: ``"straight"``; ``"through"`` the points; a ``"bspline"`` they pull on;
    a circular ``"arc"`` through its first point; ``"nodes"``: straight from point to
    point, the points being the nodes of an imported grid (kept exactly while the
    cells are as many; more or fewer keep their grading).

    How the nodes are spaced along it: ``spacing`` (``{"start": m, "end": m}``, and a
    largest cell ``"max"`` with the ``"growth"`` from each end) or ``ratio`` (last
    cell / first). ``water``: the edge follows the water (``{"level", "margin",
    "radius"}``, :mod:`lagoonviz.core.waterline`): its points are where it was put."""

    start: str
    end: str
    cells: int = 10
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    curve: str = "through"
    ratio: float = 1.0
    spacing: dict | None = None
    water: dict | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, float).reshape(-1, 2)
        self.cells = max(int(self.cells), 1)
        self.ratio = float(self.ratio) if float(self.ratio) > 0 else 1.0
        if self.curve not in CURVES:
            raise ValueError(f"curve must be one of {', '.join(CURVES)}, got {self.curve!r}")
        self.spacing = check_spacing(self.spacing)
        self.water = dict(self.water) if self.water else None

    def controls(self, vertices: dict) -> np.ndarray:
        """Its start, shaping points and end (k + 2, 2)."""
        a, b = vertices[self.start], vertices[self.end]
        return np.vstack([np.asarray(a, float), self.points, np.asarray(b, float)])

    def line(self, vertices: dict) -> np.ndarray:
        """The edge as a dense polyline."""
        p = self.controls(vertices)
        if self.curve == "straight" or len(p) == 2:
            return curvi._straight(p, 1)
        if self.curve == "arc":
            return arc_through(p[0], p[1], p[-1])
        if self.curve == "nodes":
            return p
        return curvi.catmull_rom(p) if self.curve == "through" else curvi.bspline(p)

    def length(self, vertices: dict) -> float:
        return float(curvi.lengths(self.line(vertices))[-1])

    def fractions(self) -> np.ndarray:
        """Where its ``cells + 1`` nodes lie, as fractions of its length (0 .. 1):
        by ``spacing``, or graded by ``ratio`` (an imported edge's: :meth:`nodes`)."""
        n = self.cells
        if n == 1:
            return np.array([0.0, 1.0])
        if self.spacing:
            w = spacing_sizes(n, self.spacing)
        elif abs(self.ratio - 1.0) < 1e-12:
            return np.linspace(0.0, 1.0, n + 1)
        else:
            w = (self.ratio ** (1.0 / (n - 1))) ** np.arange(n)
        return np.concatenate([[0.0], np.cumsum(w)]) / w.sum()

    def nodes(self, vertices: dict) -> np.ndarray:
        """Its ``cells + 1`` nodes along the line (spaced by ``spacing`` or graded by
        ``ratio``; an imported edge's own nodes), (n + 1, 2)."""
        line = self.line(vertices)
        s = curvi.lengths(line)
        n = self.cells
        if self.curve == "nodes" and len(self.points) and not self.spacing and self.ratio == 1.0:
            if n == len(line) - 1:
                t = s / s[-1] if s[-1] > 0 else np.linspace(0.0, 1.0, n + 1)
            else:  # as many cells as asked, graded as imported
                old = s / s[-1] if s[-1] > 0 else np.linspace(0.0, 1.0, len(line))
                t = np.interp(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, len(line)), old)
        else:
            t = self.fractions()
        at = t * s[-1]
        out = np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])
        out[0], out[-1] = vertices[self.start], vertices[self.end]  # exactly on the vertices
        return out

    def end_sizes(self, vertices: dict) -> tuple[float, float]:
        """The lengths of its first and last cell as they are (m)."""
        nodes = self.nodes(vertices)
        d = np.hypot(*np.diff(nodes, axis=0).T)
        return float(d[0]), float(d[-1])

    def wanted_cells(self, vertices: dict) -> int | None:
        """The cells its ``spacing`` asks for over its length (None: no spacing)."""
        if not self.spacing:
            return None
        return cells_for(self.length(vertices), self.spacing)

    def to_dict(self) -> dict:
        data = {"start": self.start, "end": self.end, "cells": self.cells}
        if len(self.points):
            data["points"] = [[round(float(a), 6), round(float(b), 6)] for a, b in self.points]
        if self.curve != "through":
            data["curve"] = self.curve
        if self.ratio != 1.0:
            data["ratio"] = self.ratio
        if self.spacing:
            data["spacing"] = dict(self.spacing)
        if self.water:
            data["water"] = dict(self.water)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Edge:
        known = ("start", "end", "cells", "points", "curve", "ratio", "spacing", "water")
        return cls(**{k: data[k] for k in known if k in data})


def check_spacing(spacing) -> dict | None:
    """An edge's spacing made sure of: start and end (m, > 0), a largest cell
    ``max`` (m, or none) with ``growth`` (> 1)."""
    if not spacing:
        return None
    spacing = dict(spacing)
    try:
        start, end = float(spacing["start"]), float(spacing.get("end", spacing["start"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"spacing needs start (and end) in m: {spacing!r}") from error
    if not (start > 0 and end > 0):
        raise ValueError("the spacing at the ends must be more than 0 m")
    out = {"start": start, "end": end}
    if spacing.get("max") not in (None, 0, 0.0):
        largest = float(spacing["max"])
        if largest < max(start, end):
            raise ValueError("the largest cell must be at least as long as those at the ends")
        out["max"] = largest
        growth = float(spacing.get("growth") or GROWTH)
        if not growth > 1.0:
            raise ValueError("the growth must be more than 1 (1.1: 10 % a cell)")
        out["growth"] = growth
    return out


def spacing_sizes(n: int, spacing: dict) -> np.ndarray:
    """The relative lengths of ``n`` cells by a spacing: from ``start`` to ``end``
    growing evenly (geometric), or, with a largest cell ``max``, growing by
    ``growth`` from each end up to it."""
    a, b = spacing["start"], spacing["end"]
    k = np.arange(n, dtype=float)
    if "max" not in spacing:
        if n == 1:
            return np.ones(1)
        return a * (b / a) ** (k / (n - 1))
    g, m = spacing["growth"], spacing["max"]
    return np.minimum(np.minimum(a * g**k, b * g ** (n - 1 - k)), m)


def cells_for(length: float, spacing: dict) -> int:
    """The cells a spacing asks for over ``length`` m."""
    a, b = spacing["start"], spacing["end"]
    length = float(length)
    if length <= max(a, b):
        return 1
    if "max" not in spacing:
        if abs(b - a) <= 1e-9 * max(a, b):
            return max(1, int(round(length / a)))
        g = (length - a) / (length - b)  # the growth of a geometric series from a to b
        if g <= 0 or abs(g - 1.0) < 1e-12:
            return max(1, int(round(2 * length / (a + b))))
        return max(1, int(round(1 + math.log(b / a) / math.log(g))))
    low, high = 1, 2
    while spacing_sizes(high, spacing).sum() < length:
        high *= 2
        if high > 10**6:
            raise ValueError("the spacing asks for more than a million cells")
    while high - low > 1:  # the fewest cells that reach the length
        mid = (low + high) // 2
        if spacing_sizes(mid, spacing).sum() >= length:
            high = mid
        else:
            low = mid
    fewer = spacing_sizes(low, spacing).sum()
    return low if length - fewer < spacing_sizes(high, spacing).sum() - length else high


def arc_through(a, m, b, samples: int = 400) -> np.ndarray:
    """The circular arc from ``a`` through ``m`` to ``b`` as a dense polyline (a
    straight line when the three lie on one)."""
    a, m, b = (np.asarray(p, float) for p in (a, m, b))
    d = 2.0 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]))
    scale = max(np.hypot(*(b - a)), np.hypot(*(m - a)), 1e-12)
    if abs(d) < 1e-9 * scale**2:
        return curvi._straight(np.vstack([a, b]), 1)
    sa, sm, sb = a @ a, m @ m, b @ b
    centre = np.array(
        [
            (sa * (m[1] - b[1]) + sm * (b[1] - a[1]) + sb * (a[1] - m[1])) / d,
            (sa * (b[0] - m[0]) + sm * (a[0] - b[0]) + sb * (m[0] - a[0])) / d,
        ]
    )
    radius = np.hypot(*(a - centre))
    ta, tm, tb = (math.atan2(p[1] - centre[1], p[0] - centre[0]) for p in (a, m, b))
    # the way round from a to b that passes m
    ccw = lambda t: (t - ta) % (2 * math.pi)  # noqa: E731
    sweep = ccw(tb) if ccw(tm) < ccw(tb) else ccw(tb) - 2 * math.pi
    t = ta + sweep * np.linspace(0.0, 1.0, max(int(samples), 8))
    out = np.column_stack([centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)])
    out[0], out[-1] = a, b
    return out


def shape_points(line: np.ndarray, count: int = 0) -> np.ndarray:
    """Points of a polyline (its ends left out) that a Catmull-Rom curve through
    them follows it by: ``count``, or as many as it needs (more where it bends)."""
    line = np.asarray(line, float)
    s = curvi.lengths(line)
    if s[-1] <= 0 or len(line) < 3:
        return np.zeros((0, 2))
    if count <= 0:  # by its turning: about one point per 15° of turn, 3 .. 120
        d = np.diff(line, axis=0)
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        turn = float(np.abs(np.diff(heading)).sum())
        count = int(min(max(round(math.degrees(turn) / 15.0) + 2, 3), 120))
    at = np.linspace(0.0, s[-1], count + 2)[1:-1]
    return np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])


def _ref(name: str) -> tuple[str, bool]:
    """An edge of a face: its name and whether it runs backwards ("-e3")."""
    name = str(name).strip()
    return (name[1:], True) if name.startswith("-") else (name, False)


# ====================================================================== blocks
@dataclass
class Block:
    """A block: the edge chain of each face (``FACES``), the boundary of its faces
    without an interface (``bc``: face -> {"type": ..., "value": ...}), its
    smoothing sweeps, and the ``pieces`` it is written as (cut along ``split``,
    "i" or "j", into equal parts: blocks.dat's blocks, each on one process)."""

    name: str
    faces: dict
    bc: dict = field(default_factory=dict)
    smoothing: int = 100
    slide: bool = True
    pieces: int = 1
    split: str = "i"

    def __post_init__(self) -> None:
        self.faces = {f: [str(e) for e in (self.faces.get(f) or [])] for f in FACES}
        self.bc = {f: dict(v) for f, v in (self.bc or {}).items() if f in FACES}
        self.smoothing = max(int(self.smoothing), 0)
        self.pieces = max(int(self.pieces), 1)
        if self.split not in ("i", "j"):
            raise ValueError(f"split must be 'i' or 'j', got {self.split!r}")

    def bc_type(self, face: str) -> str:
        return str(self.bc.get(face, {}).get("type", "wall"))

    def bc_value(self, face: str) -> float | None:
        value = self.bc.get(face, {}).get("value")
        return None if value is None else float(value)

    def to_dict(self) -> dict:
        data = {"name": self.name, "faces": {f: list(self.faces[f]) for f in FACES}}
        bc = {f: v for f, v in self.bc.items() if v.get("type", "wall") != "wall"}
        if bc:
            data["bc"] = bc
        data["smoothing"] = self.smoothing
        if not self.slide:
            data["slide"] = False
        if self.pieces > 1:
            data["pieces"] = self.pieces
            data["split"] = self.split
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Block:
        return cls(
            str(data.get("name", "")),
            dict(data.get("faces") or {}),
            dict(data.get("bc") or {}),
            int(data.get("smoothing", 100)),
            bool(data.get("slide", True)),
            int(data.get("pieces", 1)),
            str(data.get("split", "i")),
        )


# ==================================================================== topology
@dataclass
class Topology:
    """The multi-block grid of a case (module doc)."""

    vertices: dict
    edges: dict
    blocks: list
    bed: str = "geo"
    z: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = {str(k): (float(v[0]), float(v[1])) for k, v in dict(self.vertices).items()}
        self.edges = {
            str(k): (v if isinstance(v, Edge) else Edge.from_dict(v))
            for k, v in dict(self.edges).items()
        }
        self.blocks = [b if isinstance(b, Block) else Block.from_dict(b) for b in self.blocks]
        if self.bed not in BEDS:
            raise ValueError(f"bed must be one of {', '.join(BEDS)}, got {self.bed!r}")

    # ------------------------------------------------------------- faces
    def face_edges(self, block: int, face: str) -> list[tuple[str, bool]]:
        return [_ref(e) for e in self.blocks[block].faces[face]]

    def face_nodes(self, block: int, face: str) -> np.ndarray:
        """The nodes along a face in the block's index direction (n + 1, 2)."""
        parts = []
        for name, back in self.face_edges(block, face):
            nodes = self.edges[name].nodes(self.vertices)
            parts.append(nodes[::-1] if back else nodes)
        if not parts:
            return np.zeros((0, 2))
        out = [parts[0]]
        for p in parts[1:]:
            out.append(p[1:])
        return np.vstack(out)

    def face_cells(self, block: int, face: str) -> int:
        return sum(self.edges[name].cells for name, _ in self.face_edges(block, face))

    def face_ends(self, block: int, face: str) -> tuple[str | None, str | None]:
        """The vertices a face starts and ends at (None: no edges)."""
        chain = self.face_edges(block, face)
        if not chain:
            return None, None
        first, back = chain[0]
        a = self.edges[first].end if back else self.edges[first].start
        last, back = chain[-1]
        b = self.edges[last].start if back else self.edges[last].end
        return a, b

    def size(self, block: int) -> tuple[int, int]:
        """Cells (ni, nj) of a block: along its j faces and along its i faces."""
        return self.face_cells(block, "j0"), self.face_cells(block, "i0")

    # ------------------------------------------------------------- the nodes
    def _block_key(self, block: int) -> str:
        """What a block's nodes depend on: its faces' edges and their corners, whether
        other blocks share them (sliding), its smoothing (not the other blocks)."""
        uses = self.edge_uses()
        b = self.blocks[block]
        parts = [b.smoothing, b.slide]
        for face in FACES:
            for name, back in self.face_edges(block, face):
                e = self.edges.get(name)
                if e is None:
                    parts.append([name, None])
                    continue
                corners = [self.vertices.get(e.start), self.vertices.get(e.end)]
                parts.append([name, back, e.to_dict(), corners, len(uses.get(name, []))])
        return json.dumps(parts, default=float)

    def block_nodes(self, block: int) -> tuple[np.ndarray, np.ndarray]:
        """The nodes (x, y) of a block, each (nj + 1, ni + 1) (i along a row)."""
        key = self._block_key(block)
        found = _GRIDS.get(key)
        if found is None:
            found = self._make_block(block)
            _GRIDS[key] = found
            while len(_GRIDS) > 32:
                _GRIDS.pop(next(iter(_GRIDS)))
        return found[0].copy(), found[1].copy()

    def _make_block(self, block: int) -> tuple[np.ndarray, np.ndarray]:
        problems = self.face_problems(block)
        if problems:
            raise ValueError(f"block {block}: " + "; ".join(problems))
        south, north = self.face_nodes(block, "j0"), self.face_nodes(block, "j1")
        west, east = self.face_nodes(block, "i0"), self.face_nodes(block, "i1")
        x, y = tfi(south, north, west, east)
        smoothing = self.blocks[block].smoothing
        if not (smoothing and x.shape[0] > 2 and x.shape[1] > 2):
            return x, y
        slides = self._sliding(block) if self.blocks[block].slide else []
        return smooth(x, y, smoothing, slides)

    def _sliding(self, block: int) -> list:
        """The walls whose nodes slide: each edge of the block's outline that no other
        block shares (on a face that is no interface anywhere), as (face, first node,
        cells, line) for :func:`_slider`."""
        uses = self.edge_uses()
        out = []
        for face in FACES:
            first = 0
            chain = self.face_edges(block, face)
            # a face that is partly an interface (a river joining it) keeps its nodes:
            # sliding would crowd them against the fixed part
            partly = any(len(uses.get(name, [])) > 1 for name, _ in chain)
            for name, back in chain:
                n = self.edges[name].cells
                if len(uses.get(name, [])) == 1 and n > 1 and not partly:
                    line = self.edges[name].line(self.vertices)
                    if back:
                        line = line[::-1]
                    out.append((face, first, n, line))
                first += n
        return out

    def face_problems(self, block: int) -> list[str]:
        """Faces that do not join up (broken chains, corners that differ, opposite
        faces with different numbers of cells)."""
        out = []
        for face in FACES:
            chain = self.face_edges(block, face)
            if not chain:
                out.append(f"face {FACE_NAMES[face]} has no edges")
                continue
            for name, _ in chain:
                if name not in self.edges:
                    out.append(f"face {FACE_NAMES[face]}: no edge {name!r}")
            if any(name not in self.edges for name, _ in chain):
                continue
            for (a, ba), (b, bb) in zip(chain[:-1], chain[1:], strict=True):
                end = self.edges[a].start if ba else self.edges[a].end
                start = self.edges[b].end if bb else self.edges[b].start
                if end != start:
                    out.append(f"face {FACE_NAMES[face]}: {a} and {b} do not meet")
        if out:
            return out
        ends = {f: self.face_ends(block, f) for f in FACES}
        corners = (
            ("i0", 0, "j0", 0, "(-i, -j)"),
            ("i1", 0, "j0", 1, "(+i, -j)"),
            ("i0", 1, "j1", 0, "(-i, +j)"),
            ("i1", 1, "j1", 1, "(+i, +j)"),
        )
        for fa, ka, fb, kb, where in corners:
            if ends[fa][ka] != ends[fb][kb]:
                out.append(
                    f"corner {where}: face {FACE_NAMES[fa]} has {ends[fa][ka]}, "
                    f"{FACE_NAMES[fb]} {ends[fb][kb]}"
                )
        ni0, ni1 = self.face_cells(block, "j0"), self.face_cells(block, "j1")
        nj0, nj1 = self.face_cells(block, "i0"), self.face_cells(block, "i1")
        if ni0 != ni1:
            out.append(f"faces -j and +j have {ni0} and {ni1} cells (must be equal)")
        if nj0 != nj1:
            out.append(f"faces -i and +i have {nj0} and {nj1} cells (must be equal)")
        return out

    def cuts(self, block: int) -> list[int]:
        """Where a block is cut into its pieces: node indices along its ``split``
        direction, from 0 to its cells that way."""
        b = self.blocks[block]
        ni, nj = self.size(block)
        n = ni if b.split == "i" else nj
        k = max(1, min(b.pieces, n))
        return [int(round(v)) for v in np.linspace(0.0, n, k + 1)]

    def piece_lengths(self, block: int) -> list[int]:
        c = self.cuts(block)
        return [b - a for a, b in zip(c[:-1], c[1:], strict=True)]

    def output(self) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
        """blocks.dat's blocks in order: (block, piece, x, y), each block's pieces one
        after the other (the blocks that are not cut: one piece)."""
        out = []
        for b, g in enumerate(self.grids()):
            if g is None:
                raise ValueError(f"block {b} cannot be made: " + "; ".join(self.face_problems(b)))
            x, y = g
            c = self.cuts(b)
            for k, (lo, hi) in enumerate(zip(c[:-1], c[1:], strict=True)):
                if self.blocks[b].split == "i":
                    out.append((b, k, x[:, lo : hi + 1], y[:, lo : hi + 1]))
                else:
                    out.append((b, k, x[lo : hi + 1], y[lo : hi + 1]))
        return out

    def output_index(self) -> dict[tuple[int, int], int]:
        """(block, piece) -> its number in blocks.dat."""
        out, n = {}, 0
        for b in range(len(self.blocks)):
            for k in range(len(self.cuts(b)) - 1):
                out[(b, k)] = n
                n += 1
        return out

    def depth(self, block: int, face: str) -> int:
        """How many cells deep the block (its piece at that face) is behind a face."""
        ni, nj = self.size(block)
        across = face in ("i0", "i1")
        b = self.blocks[block]
        if b.pieces > 1 and (b.split == "i") == across:
            lengths = self.piece_lengths(block)
            return lengths[0] if face in ("i0", "j0") else lengths[-1]
        return ni if across else nj

    def grids(self) -> list[tuple[np.ndarray, np.ndarray] | None]:
        """The nodes of every block (None for a block that cannot be made)."""
        out = []
        for b in range(len(self.blocks)):
            try:
                out.append(self.block_nodes(b))
            except (ValueError, KeyError, FloatingPointError):
                out.append(None)
        return out

    # ------------------------------------------------------------- uses
    def edge_uses(self) -> dict[str, list[tuple[int, str]]]:
        """Which block faces use each edge: name -> [(block, face), ...]."""
        uses: dict[str, list[tuple[int, str]]] = {name: [] for name in self.edges}
        for b in range(len(self.blocks)):
            for face in FACES:
                for name, _ in self.face_edges(b, face):
                    uses.setdefault(name, []).append((b, face))
        return uses

    def interface_faces(self, block: int) -> dict[str, list[str]]:
        """Per face the edges it shares with another block (its interfaces)."""
        uses = self.edge_uses()
        return {
            face: [n for n, _ in self.face_edges(block, face) if len(uses.get(n, [])) > 1]
            for face in FACES
        }

    # ------------------------------------------------------------- edits
    def moved_vertex(self, name: str, x: float, y: float) -> Topology:
        """A corner moved; the points of the edges from it go along, the more the
        nearer they are to it (an edge keeps its shape, stretched), so the edge does
        not kink where the corner left its points behind."""
        vertices = dict(self.vertices)
        if name not in vertices:
            raise ValueError(f"no vertex {name!r}")
        shift = np.array([float(x), float(y)]) - np.asarray(vertices[name], float)
        edges = dict(self.edges)
        for key, e in self.edges.items():
            if not len(e.points) or name not in (e.start, e.end):
                continue
            s = curvi.lengths(e.controls(self.vertices))
            t = (s / s[-1] if s[-1] > 0 else np.linspace(0.0, 1.0, len(s)))[1:-1, None]
            weight = (1.0 - t if e.start == name else 0.0) + (t if e.end == name else 0.0)
            edges[key] = replace(e, points=e.points + weight * shift)
        vertices[name] = (float(x), float(y))
        return replace(self, vertices=vertices, edges=edges, blocks=list(self.blocks))

    def moved_point(self, edge: str, index: int, x: float, y: float) -> Topology:
        """A point of an edge moved (an edge that followed the water is shaped by hand
        from then on)."""
        if edge not in self.edges:
            raise ValueError(f"no edge {edge!r}")
        e = self.edges[edge]
        points = e.points.copy()
        points[int(index)] = (float(x), float(y))
        edges = dict(self.edges)
        edges[edge] = replace(e, points=points, water=None)
        return replace(self, vertices=dict(self.vertices), edges=edges, blocks=list(self.blocks))

    def with_edge(self, edge: str, **options) -> Topology:
        """One edge changed (``cells``, ``curve``, ``ratio``, ``points``, ``spacing``,
        ``water``); a new number of cells goes on across the blocks to the opposite
        faces (single edges there), so that opposite faces keep the same number of
        cells. A spacing sets the cells of the edge and of those across the blocks
        from it (the most any of their spacings asks for). An edge made an ``"arc"``
        gets one point, the middle of its line (bulged out of its block when
        straight)."""
        if edge not in self.edges:
            raise ValueError(f"no edge {edge!r}")
        edges = dict(self.edges)
        old = edges[edge]
        if options.get("curve") == "arc" and old.curve != "arc" and "points" not in options:
            options["points"] = [self._arc_middle(edge)]
        edges[edge] = replace(old, **options)
        result = replace(self, vertices=dict(self.vertices), edges=edges, blocks=list(self.blocks))
        if "cells" in options:
            result = result._propagate(edge)
        if options.get("spacing"):
            result = result.fitted_cells([edge])
        return result

    def _arc_middle(self, edge: str) -> tuple[float, float]:
        """Where an edge's arc passes: the middle of its line, or, for a straight one,
        a quarter of its length out of its block (to its left without a block)."""
        e = self.edges[edge]
        line = e.line(self.vertices)
        s = curvi.lengths(line)
        mid = np.array([np.interp(0.5 * s[-1], s, line[:, k]) for k in (0, 1)])
        a, c = np.asarray(self.vertices[e.start]), np.asarray(self.vertices[e.end])
        chord = c - a
        size = float(np.hypot(*chord))
        off = abs((mid - a)[0] * chord[1] - (mid - a)[1] * chord[0]) / max(size, 1e-12)
        if off > 1e-3 * size:
            return float(mid[0]), float(mid[1])
        normal = np.array([-chord[1], chord[0]]) / max(size, 1e-12)
        uses = self.edge_uses().get(edge, [])
        if uses:
            try:
                x, y = self.block_nodes(uses[0][0])
                if np.dot(normal, 0.5 * (a + c) - (x.mean(), y.mean())) < 0:
                    normal = -normal
            except (ValueError, KeyError):
                pass
        p = 0.5 * (a + c) + 0.25 * size * normal
        return float(p[0]), float(p[1])

    def cell_groups(self) -> list[set[str]]:
        """Edges whose cells go together: those on opposite faces of a block (faces of
        one edge each), and on through the blocks beyond."""
        parent = {name: name for name in self.edges}

        def root(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        for b in range(len(self.blocks)):
            for fa, fb in (("i0", "i1"), ("j0", "j1")):
                ca = [n for n, _ in self.face_edges(b, fa)]
                cb = [n for n, _ in self.face_edges(b, fb)]
                if len(ca) == 1 and len(cb) == 1 and ca[0] in parent and cb[0] in parent:
                    parent[root(ca[0])] = root(cb[0])
        groups: dict[str, set[str]] = {}
        for name in self.edges:
            groups.setdefault(root(name), set()).add(name)
        return list(groups.values())

    def across(self, edge: str) -> dict[str, bool]:
        """The edges whose cells go with ``edge``'s (itself included) and whether each
        runs the same way (its start across from ``edge``'s start)."""
        if edge not in self.edges:
            raise ValueError(f"no edge {edge!r}")
        found = {edge: True}
        todo = [edge]
        while todo:
            name = todo.pop()
            for b in range(len(self.blocks)):
                for fa, fb in (("i0", "i1"), ("i1", "i0"), ("j0", "j1"), ("j1", "j0")):
                    ca, cb = self.face_edges(b, fa), self.face_edges(b, fb)
                    if len(ca) != 1 or len(cb) != 1 or ca[0][0] != name:
                        continue
                    other, back = cb[0]
                    if other in found or other not in self.edges:
                        continue
                    found[other] = found[name] == (ca[0][1] == back)
                    todo.append(other)
        return found

    def with_spacing(self, edge: str, spacing: dict | None, across: bool = True) -> Topology:
        """An edge's spacing (None: none), with ``across`` also on the edges across the
        blocks from it that have none (turned to run their way); the cells from it."""
        spacing = check_spacing(spacing)
        names = self.across(edge) if across and spacing else {edge: True}
        edges = dict(self.edges)
        for name, same in names.items():
            if name != edge and edges[name].spacing:
                continue
            own = spacing
            if spacing and not same:
                own = dict(spacing, start=spacing["end"], end=spacing["start"])
            edges[name] = replace(edges[name], spacing=own)
        result = replace(self, vertices=dict(self.vertices), edges=edges, blocks=list(self.blocks))
        return result.fitted_cells([edge]) if spacing else result

    def fitted_cells(self, edges=None) -> Topology:
        """The cells from the spacings: each group of edges whose cells go together
        (:meth:`cell_groups`) gets the most cells any spacing in it asks for over its
        edge's length. ``edges``: only the groups of these (default all)."""
        chosen = None if edges is None else set(edges)
        new = dict(self.edges)
        for group in self.cell_groups():
            if chosen is not None and not group & chosen:
                continue
            wanted = [self.edges[n].wanted_cells(self.vertices) for n in group]
            wanted = [w for w in wanted if w]
            if not wanted:
                continue
            for name in group:
                new[name] = replace(new[name], cells=max(wanted))
        return replace(self, vertices=dict(self.vertices), edges=new, blocks=list(self.blocks))

    def followed(self, edge: str, line, water: dict | None = None, points: int = 0) -> Topology:
        """An edge laid along ``line`` (from its start to its end): its two corners
        moved to the line's ends, its shape through ``points`` points of it (0: as
        many as the line has bends, :func:`shape_points`), ``water`` kept with it."""
        if edge not in self.edges:
            raise ValueError(f"no edge {edge!r}")
        line = np.asarray(line, float)
        e = self.edges[edge]
        vertices = dict(self.vertices)
        vertices[e.start] = (float(line[0, 0]), float(line[0, 1]))
        vertices[e.end] = (float(line[-1, 0]), float(line[-1, 1]))
        inner = shape_points(line, points)
        edges = dict(self.edges)
        edges[edge] = replace(e, points=inner, curve="through", water=water)
        result = replace(self, vertices=vertices, edges=edges, blocks=list(self.blocks))
        return result.fitted_cells([edge]) if e.spacing else result

    def _propagate(self, start: str) -> Topology:
        """Cells of edge ``start`` carried to the opposite faces (single edges)."""
        opposite = {"i0": "i1", "i1": "i0", "j0": "j1", "j1": "j0"}
        cells = self.edges[start].cells
        edges = dict(self.edges)
        todo, seen = [start], {start}
        while todo:
            name = todo.pop()
            for b in range(len(self.blocks)):
                for face in FACES:
                    chain = [n for n, _ in self.face_edges(b, face)]
                    if chain != [name]:
                        continue
                    other = [n for n, _ in self.face_edges(b, opposite[face])]
                    if len(other) == 1 and other[0] not in seen and other[0] in edges:
                        edges[other[0]] = replace(edges[other[0]], cells=cells)
                        seen.add(other[0])
                        todo.append(other[0])
        return replace(self, vertices=dict(self.vertices), edges=edges, blocks=list(self.blocks))

    def scaled(self, factor: float) -> Topology:
        """Every edge with ``factor`` times its cells (at least one): a finer or
        coarser grid with the same blocks (opposite faces stay equal for factors
        that keep the counts whole, e.g. 2 or 0.5 of even counts)."""
        edges = {
            k: replace(e, cells=max(1, int(round(e.cells * float(factor)))))
            for k, e in self.edges.items()
        }
        return replace(self, vertices=dict(self.vertices), edges=edges, blocks=list(self.blocks))

    def with_bc(self, block: int, face: str, kind: str, value: float | None = None) -> Topology:
        if face not in FACES:
            raise ValueError(f"face must be one of {', '.join(FACES)}, got {face!r}")
        if kind not in BCS:
            raise ValueError(f"boundary must be one of {', '.join(BCS)}, got {kind!r}")
        blocks = list(self.blocks)
        b = blocks[int(block)]
        bc = dict(b.bc)
        if kind == "wall":
            bc.pop(face, None)
        else:
            bc[face] = {"type": kind} | ({"value": float(value)} if value is not None else {})
        blocks[int(block)] = replace(b, bc=bc)
        return replace(self, vertices=dict(self.vertices), edges=dict(self.edges), blocks=blocks)

    def with_block(self, block: int, **options) -> Topology:
        blocks = list(self.blocks)
        blocks[int(block)] = replace(blocks[int(block)], **options)
        return replace(self, vertices=dict(self.vertices), edges=dict(self.edges), blocks=blocks)

    def added_box(self, area, cells: tuple[int, int] = (10, 10), name: str = "") -> Topology:
        """One more block: a rectangle (x0, x1, y0, y1), i along x, its own four
        straight edges and corners (join it to the others by sharing edges in
        blocks.json)."""
        x0, x1 = sorted(float(v) for v in area[:2])
        y0, y1 = sorted(float(v) for v in area[2:4])
        k = len(self.blocks)
        names = [f"b{k}_{c}" for c in ("sw", "se", "ne", "nw")]
        vertices = dict(self.vertices)
        for n, p in zip(names, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)), strict=True):
            vertices[_fresh(n, vertices)] = p
        sw, se, ne, nw = list(vertices)[-4:]
        edges = dict(self.edges)
        ni, nj = int(cells[0]), int(cells[1])
        made = {}
        sides = (("s", sw, se, ni), ("n", nw, ne, ni), ("w", sw, nw, nj), ("e", se, ne, nj))
        for key, a, b, n in sides:
            e = _fresh(f"b{k}_{key}", edges)
            edges[e] = Edge(a, b, n, curve="straight")
            made[key] = e
        block = Block(
            name or f"block {k}",
            {"j0": [made["s"]], "j1": [made["n"]], "i0": [made["w"]], "i1": [made["e"]]},
        )
        return replace(self, vertices=vertices, edges=edges, blocks=[*self.blocks, block])

    def boundary_edges(self) -> list[str]:
        """The edges only one block face uses (the outline of the grid)."""
        return [n for n, use in self.edge_uses().items() if len(use) == 1]

    def extruded(self, edge: str, length: float, cells: int = 10, points: int = 4) -> Topology:
        """One more block on the outside of a boundary edge: the edge one of its faces,
        the opposite face the edge moved ``length`` m outwards (its shape kept), the two
        faces between them straight with ``points`` points each to drag; ``cells``
        cells outwards. Its i runs along the edge, j outwards."""
        uses = self.edge_uses()
        if edge not in self.edges or len(uses.get(edge, [])) != 1:
            raise ValueError(f"{edge!r} is not an edge on the outline of the grid")
        b, face = uses[edge][0]
        e = self.edges[edge]
        a, c = np.asarray(self.vertices[e.start]), np.asarray(self.vertices[e.end])
        chord = c - a
        if not np.hypot(*chord) > 0:
            raise ValueError(f"edge {edge} has no length")
        # outwards: away from the middle of its block
        x, y = self.block_nodes(b)
        centre = np.array([x.mean(), y.mean()])
        normal = np.array([-chord[1], chord[0]]) / np.hypot(*chord)  # left of start -> end
        middle = 0.5 * (a + c)
        if np.dot(normal, middle - centre) < 0:
            normal = -normal
        shift = float(length) * normal
        left = chord[0] * normal[1] - chord[1] * normal[0] > 0  # outwards to the left of e
        vertices = dict(self.vertices)
        a2 = _fresh(f"{e.start}_x", vertices)
        vertices[a2] = tuple(a + shift)
        c2 = _fresh(f"{e.end}_x", vertices)
        vertices[c2] = tuple(c + shift)
        edges = dict(self.edges)
        far = _fresh(f"{edge}_x", edges)
        edges[far] = replace(e, start=a2, end=c2, points=e.points + shift)
        t = np.linspace(0.0, 1.0, int(points) + 2)[1:-1, None]
        side_a = _fresh(f"{edge}_xa", edges)
        edges[side_a] = Edge(e.start, a2, int(cells), a + t * shift, "through")
        side_c = _fresh(f"{edge}_xc", edges)
        edges[side_c] = Edge(e.end, c2, int(cells), c + t * shift, "through")
        k = len(self.blocks)
        if left:  # i along the edge, j outwards on its left
            faces = {"j0": [edge], "j1": [far], "i0": [side_a], "i1": [side_c]}
        else:  # i along the edge backwards
            faces = {"j0": [f"-{edge}"], "j1": [f"-{far}"], "i0": [side_c], "i1": [side_a]}
        block = Block(f"block {k}", faces)
        return replace(self, vertices=vertices, edges=edges, blocks=[*self.blocks, block])

    def split_edge(self, edge: str, x: float, y: float) -> Topology:
        """An edge cut in two at the point of it nearest to (x, y): a new vertex
        there, every face that used the edge uses both halves, the cells shared out
        by length (each half one at least)."""
        if edge not in self.edges:
            raise ValueError(f"no edge {edge!r}")
        e = self.edges[edge]
        if e.cells < 2:
            raise ValueError(f"edge {edge} has one cell: give it more before cutting it")
        nodes = e.nodes(self.vertices)
        k = int(np.argmin(np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)))
        k = min(max(k, 1), e.cells - 1)  # at a node of the edge: the grid stays as it is
        where = nodes[k]
        line = e.line(self.vertices)
        s = curvi.lengths(line)

        def along(p):  # distance along the line to the point of it nearest to p
            return s[int(np.argmin(np.hypot(line[:, 0] - p[0], line[:, 1] - p[1])))]

        cut = along(where)
        before = np.array([p for p in e.points if along(p) < cut]).reshape(-1, 2)
        after = np.array([p for p in e.points if along(p) > cut]).reshape(-1, 2)
        vertices = dict(self.vertices)
        v = _fresh(f"{edge}_v", vertices)
        vertices[v] = (float(where[0]), float(where[1]))
        edges = {name: value for name, value in self.edges.items() if name != edge}
        ea, eb = _fresh(f"{edge}a", self.edges), _fresh(f"{edge}b", self.edges)
        edges[ea] = Edge(e.start, v, k, before, e.curve)
        edges[eb] = Edge(v, e.end, e.cells - k, after, e.curve)
        blocks = []
        for block in self.blocks:
            faces = {}
            for face, chain in block.faces.items():
                out = []
                for ref in chain:
                    name, back = _ref(ref)
                    if name != edge:
                        out.append(ref)
                    elif back:
                        out += [f"-{eb}", f"-{ea}"]
                    else:
                        out += [ea, eb]
                faces[face] = out
            blocks.append(replace(block, faces=faces))
        return replace(self, vertices=vertices, edges=edges, blocks=blocks)

    def _vertex_at(self, p, tol: float = 1e-6) -> str | None:
        """The vertex nearest to p within ``tol`` (None: none)."""
        best = None
        for name, v in self.vertices.items():
            d = math.hypot(v[0] - p[0], v[1] - p[1])
            if d <= tol and (best is None or d < best[0]):
                best = (d, name)
        return None if best is None else best[1]

    def _edge_between(self, a: str, b: str) -> str | None:
        for name, e in self.edges.items():
            if {e.start, e.end} == {a, b}:
                return name
        return None

    def attached_river(
        self,
        edge: str,
        line,
        cells: int | None = None,
        width: float | None = None,
        step: float | None = None,
        discharge: float | None = None,
        smoothing: int = 500,
        name: str = "river",
    ) -> Topology:
        """A river block on a part of an edge on the outline (a river into a bay): the
        edge cut where the river ``line`` (upstream first) meets it, ``cells`` of its
        cells wide (or about ``width`` m), the block along the line upstream from
        there, its banks parallel to the line and ending on the cut, cells about
        ``step`` m long (default twice those across). Its i runs downstream: the
        upstream face -i takes the ``discharge`` (G 54), the face on the edge +i."""
        uses = self.edge_uses()
        if edge not in self.edges or len(uses.get(edge, [])) != 1:
            raise ValueError(f"{edge!r} is not an edge on the outline of the grid")
        line = np.asarray(line, float).reshape(-1, 2)
        if len(line) < 2:
            raise ValueError("the river needs a line of two points at least")
        e = self.edges[edge]
        nodes = e.nodes(self.vertices)
        n = e.cells
        dense = e.line(self.vertices)
        # where the river meets the edge: its first crossing from upstream, else the
        # point of the edge nearest the river's end
        at, upto = None, len(line) - 1
        for i in range(len(line) - 1):
            p0, p1 = line[i], line[i + 1]
            for q0, q1 in zip(dense[:-1], dense[1:], strict=True):
                hit = _crossing(p0, p1, q0, q1)
                if hit is not None:
                    at, upto = hit, i
                    break
            if at is not None:
                break
        if at is None:
            at = dense[int(np.argmin(np.hypot(*(dense - line[-1]).T)))]
        m = int(np.argmin(np.hypot(*(nodes - at).T)))
        reach = 0.5 * float(np.min(np.hypot(*np.diff(nodes, axis=0).T)))
        if cells is None:
            if width is None:
                raise ValueError("give the river's width or its cells across")
            local = np.hypot(*(nodes[min(m + 1, n)] - nodes[max(m - 1, 0)]))
            cells = max(2, int(round(float(width) / max(0.5 * local, 1e-9))))
        k = int(cells)
        if k >= n:
            raise ValueError(f"the river is as wide as edge {edge} or wider ({k} of {n} cells)")
        k0 = min(max(m - k // 2, 0), n - k)
        a_xy, b_xy = nodes[k0].copy(), nodes[k0 + k].copy()
        t = self
        if 0 < k0:
            t = t.split_edge(edge, *a_xy)
        va = t._vertex_at(a_xy, reach) if k0 > 0 else e.start
        if k0 + k < n:
            t = t.split_edge(t._edge_between(va, e.end) if k0 > 0 else edge, *b_xy)
        vb = t._vertex_at(b_xy, reach) if k0 + k < n else e.end
        piece = t._edge_between(va, vb) if va and vb else None
        if piece is None:
            raise ValueError(f"edge {edge} could not be cut for the river")
        a_xy, b_xy = np.asarray(t.vertices[va], float), np.asarray(t.vertices[vb], float)
        mid = t.edges[piece].nodes(t.vertices)
        middle = mid[len(mid) // 2] if k % 2 == 0 else 0.5 * (mid[k // 2] + mid[k // 2 + 1])
        # the river's line, upstream to the middle of the cut
        river = np.vstack([line[: upto + 1], middle])
        keep = np.concatenate([[True], np.hypot(*np.diff(river, axis=0).T) > 1e-9])
        centre = curvi.resample(river[keep], 2000)
        length = float(curvi.lengths(centre)[-1])
        half = 0.5 * float(np.hypot(*(b_xy - a_xy)))
        tangent = np.gradient(centre, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        tangent = curvi._smooth(tangent, 10.0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # to the left
        end_t = tangent[-1]
        right_is_a = end_t[0] * (a_xy - middle)[1] - end_t[1] * (a_xy - middle)[0] < 0
        right_end, left_end = (a_xy, b_xy) if right_is_a else (b_xy, a_xy)
        right = centre - half * normal
        left = centre + half * normal
        s = curvi.lengths(centre)
        blend = min(0.5 * length, 4.0 * 2.0 * half)
        w = np.clip((s - (length - blend)) / max(blend, 1e-9), 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)  # smooth onto the cut's ends
        right += w[:, None] * (right_end - right[-1])
        left += w[:, None] * (left_end - left[-1])
        across = 2.0 * half / k
        along = int(max(round(length / float(step or 2.0 * across)), RING))
        vertices = dict(t.vertices)
        up_r = _fresh(f"{name}_up_r", vertices)
        vertices[up_r] = (float(right[0, 0]), float(right[0, 1]))
        up_l = _fresh(f"{name}_up_l", vertices)
        vertices[up_l] = (float(left[0, 0]), float(left[0, 1]))
        r_end, l_end = (va, vb) if right_is_a else (vb, va)
        count = int(min(max(round(length / (4.0 * half)), 4), 150))
        edges = dict(t.edges)
        rb, lb, up = (_fresh(f"{name}_{k_}", edges) for k_ in ("right", "left", "up"))
        edges[rb] = Edge(up_r, r_end, along, shape_points(right, count), "through")
        edges[lb] = Edge(up_l, l_end, along, shape_points(left, count), "through")
        edges[up] = Edge(up_r, up_l, k, curve="straight")
        faces = {
            "i0": [up],
            "i1": [piece if t.edges[piece].start == r_end else f"-{piece}"],
            "j0": [rb],
            "j1": [lb],
        }
        bc = {"i0": {"type": "inflow", "value": float(discharge)}} if discharge else {}
        block = Block(_fresh_block(name, t.blocks), faces, bc, int(smoothing))
        return replace(t, vertices=vertices, edges=edges, blocks=[*t.blocks, block])

    def with_native(self, block: int) -> Topology:
        """Block ``block`` first: block 0, the grid of the run."""
        k = int(block)
        blocks = [self.blocks[k]] + [b for i, b in enumerate(self.blocks) if i != k]
        return replace(self, vertices=dict(self.vertices), edges=dict(self.edges), blocks=blocks)

    def removed_block(self, block: int) -> Topology:
        """A block taken out with the edges and vertices only it used."""
        blocks = [b for k, b in enumerate(self.blocks) if k != int(block)]
        kept = replace(self, vertices=dict(self.vertices), edges=dict(self.edges), blocks=blocks)
        used = {n for n, u in kept.edge_uses().items() if u}
        edges = {k: e for k, e in self.edges.items() if k in used}
        corners = {e.start for e in edges.values()} | {e.end for e in edges.values()}
        vertices = {k: v for k, v in self.vertices.items() if k in corners}
        return replace(kept, vertices=vertices, edges=edges)

    # ------------------------------------------------------------- the file
    def to_dict(self) -> dict:
        data = {
            "version": 1,
            "vertices": {
                k: [round(float(v[0]), 6), round(float(v[1]), 6)] for k, v in self.vertices.items()
            },
            "edges": {k: e.to_dict() for k, e in self.edges.items()},
            "blocks": [b.to_dict() for b in self.blocks],
            "bed": self.bed,
            "z": self.z,
        }
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Topology:
        known = {"version", "vertices", "edges", "blocks", "bed", "z"}
        return cls(
            dict(data.get("vertices") or {}),
            dict(data.get("edges") or {}),
            list(data.get("blocks") or []),
            str(data.get("bed", "geo")),
            float(data.get("z", 0.0)),
            {k: v for k, v in data.items() if k not in known},
        )

    def text(self) -> str:
        return json.dumps(self.to_dict(), indent=1) + "\n"


#: grids made lately: what a block depends on (Topology._block_key) -> (x, y)
_GRIDS: dict = {}


def _slider(face: str, first: int, cells: int, line: np.ndarray):
    """Moves the inner nodes first + 1 .. first + cells - 1 of a face along ``line`` to
    the points of it nearest to their neighbours inside the block."""
    s = curvi.lengths(line)
    dense = curvi.resample(line, max(len(line), 8 * cells + 1))
    ds = curvi.lengths(dense)

    def slide(x: np.ndarray, y: np.ndarray) -> None:
        if face in ("j0", "j1"):
            row = 0 if face == "j0" else -1
            inner = 1 if face == "j0" else -2
            bx, by = x[row], y[row]
            px, py = x[inner], y[inner]
        else:
            col = 0 if face == "i0" else -1
            inner = 1 if face == "i0" else -2
            bx, by = x[:, col], y[:, col]
            px, py = x[:, inner], y[:, inner]
        k = np.arange(first + 1, first + cells)
        d2 = (dense[None, :, 0] - px[k, None]) ** 2 + (dense[None, :, 1] - py[k, None]) ** 2
        at = ds[np.argmin(d2, axis=1)]
        # keep the order and some room between the nodes (a tenth of a mean cell)
        gap = 0.1 * s[-1] / cells
        lo, hi = 0.0, s[-1]
        for m in range(len(at)):
            lo_m = (at[m - 1] if m else lo) + gap
            at[m] = min(max(at[m], lo_m), hi - gap * (len(at) - m))
        bx[k] = np.interp(at, ds, dense[:, 0])
        by[k] = np.interp(at, ds, dense[:, 1])

    return slide


def _crossing(p0, p1, q0, q1):
    """Where segment p0-p1 crosses segment q0-q1 (None: they do not)."""
    r, d = p1 - p0, q1 - q0
    den = r[0] * d[1] - r[1] * d[0]
    if abs(den) < 1e-12:
        return None
    w = q0 - p0
    t = (w[0] * d[1] - w[1] * d[0]) / den
    u = (w[0] * r[1] - w[1] * r[0]) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p0 + t * r
    return None


def _fresh_block(name: str, blocks) -> str:
    taken = {b.name for b in blocks}
    return _fresh(name, taken)


def _fresh(name: str, taken) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name}_{k}" in taken:
        k += 1
    return f"{name}_{k}"


def read(folder: str | Path) -> Topology | None:
    """The multi-block grid of a case folder (None: no blocks.json, or not readable)."""
    path = Path(folder) / FILE
    if not path.is_file():
        return None
    try:
        return Topology.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


# ===================================================================== meshing
#: smoothing stops when no node moves more than this part of the block's typical
#: cell in a sweep
SETTLED = 1e-4
#: over-relaxation of the red-black sweeps (more is faster but can fold bends)
OMEGA = 1.2
#: blocks are smoothed on a grid half as fine first while both sides keep this many
#: cells
COARSEST = 12


def _sweeps(x, y, sweeps: int, tol: float, omega: float = OMEGA) -> tuple[int, float]:
    """Red-black Gauss-Seidel sweeps of Winslow's equations, over-relaxed, the
    boundary kept: in place, until no node moves more than ``tol`` (or ``sweeps``).
    Returns the sweeps done and the last largest move."""
    nj, ni = x.shape
    jj, ii = np.mgrid[1 : nj - 1, 1 : ni - 1]
    red = (jj + ii) % 2 == 0
    move, n = 0.0, 0
    while n < int(sweeps):
        n += 1
        move = 0.0
        for colour in (red, ~red):
            x_xi = 0.5 * (x[1:-1, 2:] - x[1:-1, :-2])
            y_xi = 0.5 * (y[1:-1, 2:] - y[1:-1, :-2])
            x_eta = 0.5 * (x[2:, 1:-1] - x[:-2, 1:-1])
            y_eta = 0.5 * (y[2:, 1:-1] - y[:-2, 1:-1])
            a = x_eta**2 + y_eta**2
            b = x_xi * x_eta + y_xi * y_eta
            c = x_xi**2 + y_xi**2
            denominator = 2.0 * (a + c)
            denominator[denominator == 0] = 1.0
            for f in (x, y):
                inner = f[1:-1, 1:-1]
                step = np.where(
                    colour, omega * (curvi._winslow_step(f, a, b, c, denominator) - inner), 0.0
                )
                inner += step
                move = max(move, float(np.abs(step).max()))
        if move < tol:
            break
    return n, move


def _coarse_guess(x: np.ndarray, y: np.ndarray, sweeps: int, walls=()) -> None:
    """The inside of a block from a grid half as fine (made the same way, its faces
    sampled from these, its walls sliding too), interpolated: a start near the end,
    so the fine sweeps only smooth out the small wiggles. In place."""
    from scipy.interpolate import RegularGridInterpolator

    nj, ni = x.shape[0] - 1, x.shape[1] - 1
    if min(nj, ni) < 2 * COARSEST:
        return
    shape = (nj // 2 + 1, ni // 2 + 1)
    fine = (np.arange(nj + 1), np.arange(ni + 1))
    cj, ci = np.meshgrid(np.linspace(0, nj, shape[0]), np.linspace(0, ni, shape[1]), indexing="ij")
    at = np.stack([cj, ci], -1)
    xc = RegularGridInterpolator(fine, x)(at)
    yc = RegularGridInterpolator(fine, y)(at)
    xc, yc = tfi(
        np.c_[xc[0], yc[0]],
        np.c_[xc[-1], yc[-1]],
        np.c_[xc[:, 0], yc[:, 0]],
        np.c_[xc[:, -1], yc[:, -1]],
    )
    coarse_walls = []
    for face, first, cells, line in walls:  # the walls on the coarse grid's nodes
        along = ni if face in ("j0", "j1") else nj
        scale = (shape[1] - 1 if face in ("j0", "j1") else shape[0] - 1) / along
        a, b = int(round(first * scale)), int(round((first + cells) * scale))
        if b - a > 1:
            coarse_walls.append((face, a, b - a, line))
    _smooth_in_place(xc, yc, sweeps, coarse_walls)
    coarse = (np.arange(shape[0]), np.arange(shape[1]))
    fj, fi = np.meshgrid(
        np.linspace(0, shape[0] - 1, nj + 1), np.linspace(0, shape[1] - 1, ni + 1), indexing="ij"
    )
    at = np.stack([fj, fi], -1)
    gx = RegularGridInterpolator(coarse, xc)(at)
    gy = RegularGridInterpolator(coarse, yc)(at)
    x[1:-1, 1:-1], y[1:-1, 1:-1] = gx[1:-1, 1:-1], gy[1:-1, 1:-1]
    for face, first, cells, line in walls:  # the walls' nodes where the coarse ones slid
        idx = np.arange(first + 1, first + cells)
        if face in ("j0", "j1"):
            row = 0 if face == "j0" else -1
            x[row, idx], y[row, idx] = _onto(line, gx[row, idx], gy[row, idx])
        else:
            col = 0 if face == "i0" else -1
            x[idx, col], y[idx, col] = _onto(line, gx[idx, col], gy[idx, col])


def _onto(line: np.ndarray, px: np.ndarray, py: np.ndarray):
    """Points put onto a polyline (its nearest points), their order kept."""
    dense = curvi.resample(line, max(len(line), 8 * len(px) + 1))
    s = curvi.lengths(dense)
    d2 = (dense[None, :, 0] - px[:, None]) ** 2 + (dense[None, :, 1] - py[:, None]) ** 2
    at = np.maximum.accumulate(s[np.argmin(d2, axis=1)])
    return np.interp(at, s, dense[:, 0]), np.interp(at, s, dense[:, 1])


def _typical(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.median(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)))) or 1.0


def _smooth_in_place(x, y, sweeps: int, walls=()) -> None:
    _coarse_guess(x, y, sweeps, walls)
    tol = SETTLED * _typical(x, y)
    slides = [_slider(*w) for w in walls]
    if not slides:
        _sweeps(x, y, int(sweeps), tol)
        return
    done = 0
    sliding = min(int(sweeps), SLIDE_SWEEPS)
    while done < sliding:  # the nodes of the walls slide along them now and then
        before = (x.copy(), y.copy())
        n, _move = _sweeps(x, y, min(SLIDE_EVERY, sliding - done), tol)
        for slide in slides:
            slide(x, y)
        done += n
        if np.hypot(x - before[0], y - before[1]).max() < SLIDE_EVERY * tol:
            break
    # then the walls stay and the inside settles (a while: what is left is slow drift)
    _sweeps(x, y, min(int(sweeps) - done, SLIDE_SWEEPS), tol)


def smooth(x, y, sweeps: int, walls=()) -> tuple[np.ndarray, np.ndarray]:
    """A block's inside smoothed by Winslow's elliptic grid equations, its faces kept
    (the nodes of the ``walls``, (face, first node, cells, line), sliding along
    them during the first SLIDE_SWEEPS sweeps, then as many with them fixed): first
    on coarser grids, then red-black over-relaxed sweeps until it settles (no node
    moving more than SETTLED of a cell a sweep), ``sweeps`` at most on each grid."""
    x, y = np.array(x, float), np.array(y, float)
    if x.shape[0] < 3 or x.shape[1] < 3 or sweeps <= 0:
        return x, y
    _smooth_in_place(x, y, int(sweeps), walls)
    return x, y


def tfi(south, north, west, east) -> tuple[np.ndarray, np.ndarray]:
    """Transfinite (Coons) interpolation of four faces: ``south`` (j = 0) and
    ``north`` (j = nj) with ni + 1 nodes along i, ``west`` (i = 0) and ``east``
    (i = ni) with nj + 1 nodes along j; their corners shared. The nodes (x, y), each
    (nj + 1, ni + 1), the faces kept exactly."""
    S, N, W, E = (np.asarray(a, float) for a in (south, north, west, east))

    def arc(line):
        s = curvi.lengths(line)
        return s / s[-1] if s[-1] > 0 else np.linspace(0.0, 1.0, len(line))

    us, un = arc(S), arc(N)
    vw, ve = arc(W), arc(E)
    nj, ni = len(W) - 1, len(S) - 1
    # parameters of the inner nodes: blended between the opposite faces
    vv = np.linspace(0.0, 1.0, nj + 1)[:, None]
    uu = np.linspace(0.0, 1.0, ni + 1)[None, :]
    u = (1 - vv) * us[None, :] + vv * un[None, :]
    v = (1 - uu) * vw[:, None] + uu * ve[:, None]
    out = []
    for k in range(2):
        s, n, w, e = S[:, k], N[:, k], W[:, k], E[:, k]
        c00, c10, c01, c11 = s[0], s[-1], n[0], n[-1]
        f = (
            (1 - v) * s[None, :]
            + v * n[None, :]
            + (1 - u) * w[:, None]
            + u * e[:, None]
            - ((1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11)
        )
        f[0, :], f[-1, :], f[:, 0], f[:, -1] = s, n, w, e
        out.append(f)
    return out[0], out[1]


# ===================================================================== quality
def count(n: int, word: str) -> str:
    """'1 block', '3 blocks', '1,200 cells'."""
    return f"{n:,} {word}" + ("" if n == 1 else "s")


@dataclass(frozen=True)
class Quality:
    """What the cells of a multi-block grid are like."""

    cells: int
    blocks: list
    min_angle: float
    max_jump: float
    folded: int
    counts: dict

    def text(self) -> str:
        words = (
            f"{count(len(self.blocks), 'block')}, {count(self.cells, 'cell')} "
            f"({', '.join(f'{ni} x {nj}' for ni, nj in self.blocks)}); smallest angle "
            f"{self.min_angle:.0f}°, size jump up to {self.max_jump:.2f}"
        )
        bad = [f"{self.counts[c]:,} {c}" for c in ("poor", "folded") if self.counts.get(c)]
        if bad:
            words += "; " + ", ".join(bad)
        if self.folded:
            words += ": REEF3D stops on folded cells"
        return words


def cell_angles(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The smallest angle (degrees) at the corners of each cell (nj, ni)."""
    corners = [
        (x[:-1, :-1], y[:-1, :-1]),
        (x[:-1, 1:], y[:-1, 1:]),
        (x[1:, 1:], y[1:, 1:]),
        (x[1:, :-1], y[1:, :-1]),
    ]
    smallest = np.full(x[:-1, :-1].shape, 180.0)
    for k in range(4):
        px, py = corners[k]
        ax, ay = corners[(k + 1) % 4]
        bx, by = corners[(k - 1) % 4]
        ux, uy, vx, vy = ax - px, ay - py, bx - px, by - py
        norm = np.maximum(np.hypot(ux, uy) * np.hypot(vx, vy), 1e-30)
        angle = np.degrees(np.arccos(np.clip((ux * vx + uy * vy) / norm, -1.0, 1.0)))
        smallest = np.minimum(smallest, angle)
    return smallest


def size_jump(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The largest ratio of a cell's area to a neighbour's (along i or j), per cell."""
    a = np.abs(curvi.cell_areas(x, y))
    a = np.maximum(a, 1e-30)
    jump = np.ones_like(a)
    for axis in (0, 1):
        r = a.take(range(1, a.shape[axis]), axis=axis) / a.take(range(a.shape[axis] - 1), axis=axis)
        r = np.maximum(r, 1.0 / r)
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        jump = np.maximum(jump, np.pad(r, pad, constant_values=1.0))
        pad[axis] = (0, 1)
        jump = np.maximum(jump, np.pad(r, pad, constant_values=1.0))
    return jump


def cell_classes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The class of each cell (index into ``CLASSES``): good (angles of 45° and
    more, neighbours at most 1.3 times larger), fair (30°, 2 times), poor, folded."""
    angle = cell_angles(x, y)
    jump = size_jump(x, y)
    out = np.full(angle.shape, 2, dtype=np.int8)
    out[(angle >= 30) & (jump <= 2.0)] = 1
    out[(angle >= 45) & (jump <= 1.3)] = 0
    out[curvi.cell_areas(x, y) <= 0] = 3
    return out


def quality(topology: Topology) -> Quality:
    grids = topology.grids()
    angles, jumps, sizes = [], [], []
    counts = {c: 0 for c in CLASSES}
    folded = 0
    for b, g in enumerate(grids):
        sizes.append(topology.size(b) if g is None else (g[0].shape[1] - 1, g[0].shape[0] - 1))
        if g is None:
            continue
        x, y = g
        angles.append(cell_angles(x, y).min())
        jumps.append(size_jump(x, y).max())
        classes = cell_classes(x, y)
        for k, c in enumerate(CLASSES):
            counts[c] += int((classes == k).sum())
        folded += int((curvi.cell_areas(x, y) <= 0).sum())
    return Quality(
        int(sum(ni * nj for ni, nj in sizes)),
        sizes,
        float(min(angles)) if angles else 90.0,
        float(max(jumps)) if jumps else 1.0,
        folded,
        counts,
    )


# ====================================================================== rules
@dataclass(frozen=True)
class Issue:
    """Something DIVEMesh or REEF3D would refuse (``error``) or warn about."""

    level: str  # "error" or "warning"
    text: str
    where: tuple[float, float] | None = None
    block: int | None = None


def check(topology: Topology) -> list[Issue]:
    """The rules of multi-block grids (module doc), errors first."""
    issues: list[Issue] = []
    t = topology
    if not t.blocks:
        return [Issue("error", "no blocks")]
    uses = t.edge_uses()
    for name, use in uses.items():
        if len(use) > 2:
            mid = _edge_middle(t, name)
            issues.append(
                Issue("error", f"edge {name} is used by {len(use)} faces (two at most)", mid)
            )
    for name, edge in t.edges.items():
        for v in (edge.start, edge.end):
            if v not in t.vertices:
                issues.append(Issue("error", f"edge {name}: no vertex {v!r}"))
    if any(i.level == "error" for i in issues):
        return issues
    grids = []
    for b in range(len(t.blocks)):
        problems = t.face_problems(b)
        for p in problems:
            issues.append(Issue("error", f"block {b} ({t.blocks[b].name}): {p}", block=b))
        grids.append(None if problems else t.block_nodes(b))
    for b, g in enumerate(grids):
        if g is None:
            continue
        x, y = g
        areas = curvi.cell_areas(x, y)
        if (areas <= 0).all():
            issues.append(
                Issue(
                    "error",
                    f"block {b} ({t.blocks[b].name}) is left-handed: j must lie to the left of i "
                    "(swap its i faces or its j faces)",
                    (float(x.mean()), float(y.mean())),
                    b,
                )
            )
        elif (areas <= 0).any():
            jj, ii = np.unravel_index(int(np.argmin(areas)), areas.shape)
            issues.append(
                Issue(
                    "error",
                    f"block {b} ({t.blocks[b].name}): {int((areas <= 0).sum())} folded cells "
                    "(move the vertices or points, more smoothing)",
                    _cell_centre(x, y, jj, ii),
                    b,
                )
            )
    # interfaces: block 0 over whole faces; deep enough behind them
    for face in FACES:
        chain = [n for n, _ in t.face_edges(0, face)]
        shared = [n for n in chain if len(uses.get(n, [])) > 1]
        if shared and len(shared) != len(chain):
            mid = _edge_middle(t, [n for n in chain if n not in shared][0])
            issues.append(
                Issue(
                    "error",
                    f"block 0, face {FACE_NAMES[face]}: partly an interface; the interfaces of "
                    "block 0 (the grid of the run) have to cover whole faces",
                    mid,
                    0,
                )
            )
    for name, use in uses.items():
        if len(use) != 2:
            continue
        for (a, _), (b, fb) in (use, use[::-1]):
            depth = t.depth(b, fb)
            if depth < DEEP_MIN or (a == 0 and depth < RING):
                level = "error"
            elif depth < RING:
                level = "warning"
            else:
                continue
            need = RING if a == 0 else DEEP_MIN
            issues.append(
                Issue(
                    level,
                    f"block {b} is {depth} cells deep behind its face {FACE_NAMES[fb]} "
                    f"(interface with block {a}): at least {need}"
                    + (f", better {RING} (R 21)" if level == "warning" else ""),
                    _edge_middle(t, name),
                    b,
                )
            )
    # pieces: block 0 is cut by DIVEMesh itself; each piece deep enough
    for b, block in enumerate(t.blocks):
        if block.pieces <= 1:
            continue
        where = _face_middle(t, b, "j0")
        if b == 0:
            issues.append(
                Issue(
                    "error",
                    "block 0 is not cut into pieces: DIVEMesh shares it out over the "
                    "processes itself (M 10)",
                    where,
                    0,
                )
            )
            continue
        lengths = t.piece_lengths(b)
        if min(lengths) < DEEP_MIN:
            issues.append(
                Issue(
                    "error",
                    f"block {b} ({block.name}): pieces of {min(lengths)} cells along "
                    f"{block.split}; at least {DEEP_MIN} (fewer pieces)",
                    where,
                    b,
                )
            )
        valued = [f for f in ("i0", "i1") if block.bc_type(f) in ("inflow", "outflow")]
        if block.split == "j" and valued:
            issues.append(
                Issue(
                    "error",
                    f"block {b} ({block.name}): its {block.bc_type(valued[0])} face would be "
                    "shared by the pieces: cut it along i",
                    where,
                    b,
                )
            )
        shared = t.interface_faces(b)
        ends = ("j0", "j1") if block.split == "i" else ("i0", "i1")
        for face in ends:
            others = {o for name in shared[face] for o, _f in uses.get(name, []) if o != b}
            if 0 in others:
                issues.append(
                    Issue(
                        "warning",
                        f"block {b} ({block.name}): its pieces meet block 0 side by side on "
                        f"face {FACE_NAMES[face]}: see that DIVEMesh finds every interface "
                        "(multiblock.log)",
                        _face_middle(t, b, face),
                        b,
                    )
                )
    # boundaries
    native_in = native_out = 0
    for b, block in enumerate(t.blocks):
        shared = t.interface_faces(b)
        for face in FACES:
            kind = block.bc_type(face)
            if kind == "wall":
                continue
            where = _face_middle(t, b, face)
            if shared[face]:
                issues.append(
                    Issue(
                        "error",
                        f"block {b}, face {FACE_NAMES[face]}: {kind} on a face with an interface",
                        where,
                        b,
                    )
                )
            if kind == "open":
                if b != 0:
                    issues.append(
                        Issue(
                            "error",
                            f"block {b}, face {FACE_NAMES[face]}: open faces (a condition of "
                            "their own in ctrl.txt) only on block 0; the others take an inflow "
                            "or an outflow (G 54)",
                            where,
                            b,
                        )
                    )
                continue
            value = block.bc_value(face)
            if value is None:
                what = "discharge (m3/s)" if kind == "inflow" else "water level (m)"
                issues.append(
                    Issue("error", f"block {b}, face {FACE_NAMES[face]}: no {what}", where, b)
                )
            right = "i0" if kind == "inflow" else "i1"
            if face != right:
                issues.append(
                    Issue(
                        "error",
                        f"block {b}, face {FACE_NAMES[face]}: {kind} only on the "
                        f"{FACE_NAMES[right]} face (turn the block's index frame)",
                        where,
                        b,
                    )
                )
            if b == 0:
                native_in += kind == "inflow"
                native_out += kind == "outflow"
    if native_in > 1 or native_out > 1:
        issues.append(
            Issue(
                "error",
                "block 0: one inflow and one outflow face at most (B 60, W 10, F 62)",
                block=0,
            )
        )
    # corners: three or five blocks round a vertex inside the grid
    count: dict[str, int] = {}
    for b in range(len(t.blocks)):
        corners = {t.face_ends(b, "j0")[0], t.face_ends(b, "j0")[1]}
        corners |= {t.face_ends(b, "j1")[0], t.face_ends(b, "j1")[1]}
        for v in corners:
            if v is not None:
                count[v] = count.get(v, 0) + 1
    for v, n in count.items():
        touching = [e for e, edge in t.edges.items() if v in (edge.start, edge.end)]
        inner = all(len(uses.get(e, [])) > 1 for e in touching)
        if inner and n not in (4,) and n > 1:
            issues.append(
                Issue(
                    "warning",
                    f"vertex {v}: {n} blocks round it (not 4): no refinement around it",
                    t.vertices[v],
                )
            )
    if grids and all(g is not None for g in grids):
        q = quality(t)
        if q.counts.get("poor"):
            issues.append(
                Issue(
                    "warning",
                    f"{q.counts['poor']:,} poor cells (angles below 30° or size jumps over 2)",
                )
            )
    issues.sort(key=lambda i: i.level != "error")
    return issues


def _cell_centre(x, y, j, i) -> tuple[float, float]:
    return (
        float(0.25 * (x[j, i] + x[j + 1, i] + x[j, i + 1] + x[j + 1, i + 1])),
        float(0.25 * (y[j, i] + y[j + 1, i] + y[j, i + 1] + y[j + 1, i + 1])),
    )


def _edge_middle(t: Topology, name: str) -> tuple[float, float] | None:
    try:
        line = t.edges[name].line(t.vertices)
    except (KeyError, ValueError):
        return None
    s = curvi.lengths(line)
    k = int(np.searchsorted(s, 0.5 * s[-1]))
    k = min(k, len(line) - 1)
    return float(line[k, 0]), float(line[k, 1])


def _face_middle(t: Topology, block: int, face: str) -> tuple[float, float] | None:
    nodes = t.face_nodes(block, face)
    if not len(nodes):
        return None
    k = len(nodes) // 2
    return float(nodes[k, 0]), float(nodes[k, 1])


# ====================================================================== output
def bed_heights(topology: Topology, x, y, geo_points=None) -> np.ndarray:
    """zb at the nodes: geo.dat's points interpolated (``bed = "geo"``), else the
    flat bed ``z``."""
    if topology.bed == "geo" and geo_points is not None and len(geo_points):
        return curvi.interpolate_bed(np.asarray(geo_points, float), x, y)
    return np.full(np.shape(x), float(topology.z))


def blocks_text(nodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> str:
    """blocks.dat: the number of blocks, then per block "ni nj" and x y zb of every
    node, i fastest."""
    rows = [str(len(nodes))]
    for x, y, z in nodes:
        nj, ni = x.shape[0] - 1, x.shape[1] - 1
        rows.append(f"{ni} {nj}")
        for j in range(nj + 1):
            for i in range(ni + 1):
                rows.append(f"{x[j, i]:.10g} {y[j, i]:.10g} {z[j, i]:.6g}")
    return "\n".join(rows) + "\n"


def read_blocks(path: str | Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """The blocks of a blocks.dat: (x, y, zb) each (nj + 1, ni + 1); None if not
    readable."""
    try:
        words = Path(path).read_text(encoding="utf-8").split()
        count = int(words[0])
        k, out = 1, []
        for _ in range(count):
            ni, nj = int(words[k]), int(words[k + 1])
            k += 2
            n = 3 * (ni + 1) * (nj + 1)
            values = np.asarray(words[k : k + n], float).reshape(nj + 1, ni + 1, 3)
            k += n
            out.append((values[..., 0], values[..., 1], values[..., 2]))
    except (OSError, ValueError, IndexError):
        return None
    return out


@dataclass(frozen=True)
class Settings:
    """What the input files get from the grid: DIVEMesh's sides of block 0 (C 11 -
    C 14), the native in- and outflow (B 60, W 10, F 62) and the open faces of the
    blocks 1.. (G 54 lines)."""

    sides: dict
    inflow: float | None
    level: float | None
    g54: list
    box: tuple[float, float, float, float]
    spacing: float


def settings(topology: Topology) -> Settings:
    t = topology
    side_of = {"i0": "C 11", "i1": "C 14", "j0": "C 13", "j1": "C 12"}
    code = {"wall": 21, "inflow": 1, "outflow": 2}
    native = t.blocks[0]
    sides = {}
    for f in FACES:
        kind = native.bc_type(f)
        if kind == "open":  # its own condition in ctrl.txt (a tide edge): the code given
            sides[side_of[f]] = int(native.bc_value(f) or OPEN_CODE)
        else:
            sides[side_of[f]] = code[kind]
    inflow = next((native.bc_value(f) for f in FACES if native.bc_type(f) == "inflow"), None)
    level = next((native.bc_value(f) for f in FACES if native.bc_type(f) == "outflow"), None)
    g54 = []
    index = t.output_index()
    for b, block in enumerate(t.blocks[1:], start=1):
        last = len(t.cuts(b)) - 2
        for f in FACES:
            kind = block.bc_type(f)
            if kind in ("inflow", "outflow") and block.bc_value(f) is not None:
                piece = 0 if f == "i0" or block.split == "j" else last
                g54.append([index[(b, piece)], 1 if kind == "inflow" else 2, block.bc_value(f)])
    xs, ys, steps = [], [], []
    for g in t.grids():
        if g is None:
            continue
        x, y = g
        xs += [float(x.min()), float(x.max())]
        ys += [float(y.min()), float(y.max())]
        steps.append(float(np.mean(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)))))
    box = (min(xs), max(xs), min(ys), max(ys)) if xs else (0.0, 1.0, 0.0, 1.0)
    return Settings(sides, inflow, level, g54, box, float(np.mean(steps)) if steps else 1.0)


# ================================================================== start-ups
def from_box(area, cells: tuple[int, int] = (20, 10)) -> Topology:
    """One block over a rectangle (x0, x1, y0, y1), i along x."""
    return Topology({}, {}, []).added_box(area, cells, name="block 0")


def from_corridor(corridor) -> Topology:
    """One block from a river corridor (``lagoonviz.core.corridor.Corridor``): its
    banks the j faces (the right bank j = 0), its ends the i faces; the cells and
    smoothing of the corridor's grid."""
    curve = corridor.curve
    left, right = corridor.left, corridor.right
    vertices = {"up_r": right[0], "down_r": right[-1], "up_l": left[0], "down_l": left[-1]}
    edges = {
        "right": Edge("up_r", "down_r", corridor.cells_along, right[1:-1], curve),
        "left": Edge("up_l", "down_l", corridor.cells_along, left[1:-1], curve),
        "up": Edge("up_r", "up_l", corridor.cells_across, curve="straight"),
        "down": Edge("down_r", "down_l", corridor.cells_across, curve="straight"),
    }
    block = Block(
        "river",
        {"j0": ["right"], "j1": ["left"], "i0": ["up"], "i1": ["down"]},
        smoothing=max(corridor.smoothing, 50),
    )
    t = Topology(vertices, edges, [block], bed=corridor.bed if corridor.bed == "geo" else "flat")
    x, y = t.block_nodes(0)
    if curvi.cell_areas(x, y).sum() < 0:  # the banks the other way round: j from the left
        block = Block(
            "river",
            {"j0": ["left"], "j1": ["right"], "i0": ["-up"], "i1": ["-down"]},
            smoothing=block.smoothing,
        )
        t = Topology(vertices, edges, [block], bed=t.bed)
    return t


# ================================================================== processes
#: the shortest piece the balancing cuts (cells along the cut direction)
PIECE_MIN = RING


@dataclass(frozen=True)
class Balance:
    """How a run of ``processes`` shares out the grid, as REEF3D does: block 0 (with
    its ring) evenly over all processes by DIVEMesh, then the pieces of the other
    blocks, the largest first, each onto the process with the fewest cells.
    ``loads``: the cells of each process; ``ranks``: blocks.dat's block -> process."""

    processes: int
    loads: list
    ranks: dict
    native: int

    @property
    def mean(self) -> float:
        return float(sum(self.loads)) / max(len(self.loads), 1)

    @property
    def imbalance(self) -> float:
        """The busiest process over the mean (1: even)."""
        return max(self.loads) / self.mean if self.mean > 0 else 1.0

    def text(self) -> str:
        return (
            f"{self.processes} processes: {self.mean:,.0f} cells each on average, the "
            f"busiest {max(self.loads):,.0f} ({100 * (self.imbalance - 1):.0f} % more)"
        )


def native_cells(topology: Topology) -> int:
    """Block 0's cells with DIVEMesh's ring on its faces with an interface."""
    ni, nj = topology.size(0)
    shared = topology.interface_faces(0)
    ring = sum(RING * (nj if f in ("i0", "i1") else ni) for f in FACES if shared[f])
    return int(ni * nj + ring)


def balance(topology: Topology, processes: int) -> Balance:
    """The cells each process gets (:class:`Balance`)."""
    n = max(int(processes), 1)
    native = native_cells(topology)
    loads = [native / n] * n
    index = topology.output_index()
    pieces = []
    for b in range(1, len(topology.blocks)):
        ni, nj = topology.size(b)
        lengths = topology.piece_lengths(b)
        other = nj if topology.blocks[b].split == "i" else ni
        pieces += [(index[(b, k)], length * other) for k, length in enumerate(lengths)]
    pieces.sort(key=lambda p: -p[1])  # (stable: in block order among equals)
    ranks = {}
    for out, cells in pieces:
        r = int(np.argmin(loads))
        ranks[out] = r
        loads[r] += cells
    return Balance(n, [float(v) for v in loads], ranks, native)


def balanced(topology: Topology, processes: int, tolerance: float = 1.05) -> Topology:
    """The blocks 1.. cut into pieces so that ``processes`` share the cells evenly:
    each block first into pieces of about what a process has room for beside its
    share of block 0, then, while the busiest process has more than ``tolerance``
    times the mean, the one more cut that helps most, up to about two pieces a
    process. The cuts with the least cost win: the busiest process's cells plus
    0.4 % of the mean for every piece (each piece has interfaces to exchange).
    Pieces have PIECE_MIN cells at least; a block is cut along its longer
    direction, along i if it has an inflow or outflow."""
    n = max(int(processes), 1)
    native = native_cells(topology)
    total = native + sum(
        topology.size(k)[0] * topology.size(k)[1] for k in range(1, len(topology.blocks))
    )
    room = max(total / n - native / n, 1.0)  # what a process takes beside block 0
    blocks = []
    for k, b in enumerate(topology.blocks):
        if k == 0:
            blocks.append(replace(b, pieces=1))
            continue
        ni, nj = topology.size(k)
        valued = any(b.bc_type(f) in ("inflow", "outflow") for f in ("i0", "i1"))
        split = "i" if valued or ni >= nj else "j"
        along = ni if split == "i" else nj
        most = max(along // PIECE_MIN, 1)
        pieces = int(min(max(round(ni * nj / room), 1), most))
        blocks.append(replace(b, pieces=pieces, split=split))

    def with_blocks(new):
        return replace(
            topology, vertices=dict(topology.vertices), edges=dict(topology.edges), blocks=new
        )

    t = with_blocks(blocks)

    mean = total / n

    def score(top):  # the busiest process, and a little for every piece
        busiest = max(balance(top, n).loads)
        return busiest + 0.004 * mean * sum(b.pieces for b in top.blocks)

    best = (score(t), t)
    for _ in range(2 * n + len(t.blocks)):
        if balance(t, n).imbalance <= tolerance:
            break
        choice = None
        for k in range(1, len(t.blocks)):
            b = t.blocks[k]
            ni, nj = t.size(k)
            along = ni if b.split == "i" else nj
            if along // (b.pieces + 1) < PIECE_MIN:
                continue
            new = list(t.blocks)
            new[k] = replace(b, pieces=b.pieces + 1)
            trial = with_blocks(new)
            load = max(balance(trial, n).loads)
            if choice is None or load < choice[0]:
                choice = (load, trial)
        if choice is None:
            break
        t = choice[1]  # (even when it does not help yet: the next cut may)
        if score(t) < best[0]:
            best = (score(t), t)
    return best[1]


def mean_spacing(topology: Topology) -> float:
    return settings(topology).spacing


def interface_count(topology: Topology) -> int:
    return sum(1 for use in topology.edge_uses().values() if len(use) == 2)


def summary(topology: Topology) -> str:
    """One line: blocks, cells, interfaces."""
    q = quality(topology)
    return f"{q.text()}; {count(interface_count(topology), 'interface edge')}"


# ===================================================================== import
def _face_lines(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """A block's four faces as node lines in its index directions."""
    return {
        "j0": np.column_stack([x[0], y[0]]),
        "j1": np.column_stack([x[-1], y[-1]]),
        "i0": np.column_stack([x[:, 0], y[:, 0]]),
        "i1": np.column_stack([x[:, -1], y[:, -1]]),
    }


def from_blocks(
    nodes: list,
    names: list[str] | None = None,
    bcs: dict | None = None,
    smoothing: int = 0,
) -> Topology:
    """A topology from the nodes of a multi-block grid (as :func:`read_blocks` gives
    them, a blocks.dat made elsewhere): its faces cut where another block's face
    starts or ends on them, the pieces edges of their own nodes (``"nodes"``), shared
    where two faces have the same nodes: the interfaces. The blocks keep their nodes
    on the faces (no sliding) and get ``smoothing`` sweeps inside. ``bcs``: (block,
    face) -> {"type", "value"}."""
    from scipy.spatial import cKDTree

    grids = [(np.asarray(g[0], float), np.asarray(g[1], float)) for g in nodes]
    if not grids:
        raise ValueError("no blocks")
    faces = [_face_lines(x, y) for x, y in grids]
    every = np.vstack([line for f in faces for line in f.values()])
    extent = float(max(every.max(axis=0) - every.min(axis=0)))
    tol = max(1e-6 * extent, 1e-6)
    tree = cKDTree(every)
    owner = []  # (block, face, index) of each row of every
    for b, f in enumerate(faces):
        for face, line in f.items():
            owner += [(b, face, k) for k in range(len(line))]
    # where the faces are cut: their ends, and where a run of nodes shared with
    # another block's face starts or ends
    cuts = {(b, face): {0, len(line) - 1} for b, f in enumerate(faces) for face, line in f.items()}
    for b, f in enumerate(faces):
        for face, line in f.items():
            shared = np.zeros(len(line), bool)
            for k, p in enumerate(line):
                for row in tree.query_ball_point(p, tol):
                    if owner[row][0] != b:
                        shared[k] = True
                        break
            for k in range(len(line)):
                before = shared[k - 1] if k else False
                after = shared[k + 1] if k + 1 < len(line) else False
                if shared[k] and (not before or not after):
                    cuts[(b, face)].add(k)
    # the corners: named in the order found
    vertices: dict[str, tuple[float, float]] = {}
    spots: list[np.ndarray] = []

    def vertex(p) -> str:
        for name, q in zip(vertices, spots, strict=True):
            if np.hypot(*(p - q)) <= tol:
                return name
        name = f"v{len(vertices)}"
        vertices[name] = (float(p[0]), float(p[1]))
        spots.append(np.asarray(p, float))
        return name

    edges: dict[str, Edge] = {}
    known: dict[tuple, str] = {}  # (start, middle, end) rounded -> edge name

    def key(piece):
        mid = piece[len(piece) // 2] if len(piece) > 2 else 0.5 * (piece[0] + piece[-1])
        return tuple(np.round(np.concatenate([piece[0], mid, piece[-1]]) / tol).astype(np.int64))

    blocks = []
    for b, f in enumerate(faces):
        chains = {}
        for face in FACES:
            line = f[face]
            marks = sorted(cuts[(b, face)])
            chain = []
            for k0, k1 in zip(marks[:-1], marks[1:], strict=True):
                piece = line[k0 : k1 + 1]
                ahead, back = key(piece), key(piece[::-1])
                if ahead in known:
                    chain.append(known[ahead])
                    continue
                if back in known:
                    chain.append("-" + known[back])
                    continue
                name = f"b{b}_{face}" + (f"_{len(chain) + 1}" if len(marks) > 2 else "")
                name = _fresh(name, edges)
                edges[name] = Edge(
                    vertex(piece[0]), vertex(piece[-1]), len(piece) - 1, piece[1:-1], "nodes"
                )
                known[ahead] = name
                chain.append(name)
            chains[face] = chain
        name = names[b] if names and b < len(names) else f"block {b}"
        bc = {face: dict(v) for (k, face), v in (bcs or {}).items() if k == b}
        blocks.append(Block(name, chains, bc, int(smoothing), slide=False))
    return Topology(vertices, edges, blocks)


def bcs_from_inputs(control, ctrl) -> dict:
    """The boundaries of the faces from the input files: block 0's sides (C 11 -
    C 14 of control.txt: 21 a wall, 1 on -i the inflow of W 10, 2 on +i the outflow
    of F 62, others open with their code) and the G 54 lines of ctrl.txt (b 1 Q:
    inflow on -i, b 2 WL: outflow on +i). ``control``, ``ctrl``: their ControlFiles
    (or None)."""
    out: dict = {}
    face_of = {"C 11": "i0", "C 14": "i1", "C 13": "j0", "C 12": "j1"}
    for code, face in face_of.items():
        values = control.values(code) if control is not None else None
        if not values:
            continue
        kind = int(values[0])
        if kind == 21:
            continue
        if kind == 1 and face == "i0":
            w10 = ctrl.values("W 10") if ctrl is not None else None
            out[(0, face)] = {"type": "inflow"} | ({"value": float(w10[0])} if w10 else {})
        elif kind == 2 and face == "i1":
            f62 = ctrl.values("F 62") if ctrl is not None else None
            out[(0, face)] = {"type": "outflow"} | ({"value": float(f62[0])} if f62 else {})
        else:
            out[(0, face)] = {"type": "open", "value": float(kind)}
    for row in ctrl.all("G 54") if ctrl is not None else []:
        if len(row) >= 3:
            b, kind, value = int(row[0]), int(row[1]), float(row[2])
            face = "i0" if kind == 1 else "i1"
            out[(b, face)] = {"type": "inflow" if kind == 1 else "outflow", "value": value}
    return out
