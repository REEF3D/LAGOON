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
``G 54 b 2 WL`` (the +i face).

:func:`check` lists what DIVEMesh and REEF3D would refuse or warn about (the
**rules**), :func:`quality` how good the cells are, :func:`blocks_text` writes
``blocks.dat`` (the number of blocks, then per block "ni nj" and x y zb of every node,
i fastest).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from lagoonviz.core import corridor as curvi

FILE = "blocks.json"
GRID_FILE = "blocks.dat"
#: the faces of a block in its index frame: -i, +i, -j, +j (DIVEMesh's faces 0 .. 3)
FACES = ("i0", "i1", "j0", "j1")
FACE_NAMES = {"i0": "-i", "i1": "+i", "j0": "-j", "j1": "+j"}
CURVES = ("straight", "through", "bspline")
BCS = ("wall", "inflow", "outflow")
BEDS = ("geo", "flat")
#: DIVEMesh's ring of block 0 (R 21's default): the blocks behind its interfaces must
#: be at least this many cells deep
RING = 8
#: REEF3D's own need behind any interface (EXT + the ghost cells): refused below
DEEP_MIN = 5
#: Winslow sweeps between two slides of the wall nodes along their edges
SLIDE_EVERY = 10
#: the classes of the cells' quality (colours in the set-up), worst last
CLASSES = ("good", "fair", "poor", "folded")


# ====================================================================== edges
@dataclass
class Edge:
    """A line from vertex ``start`` to vertex ``end`` cut into ``cells`` cells."""

    start: str
    end: str
    cells: int = 10
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    curve: str = "through"
    ratio: float = 1.0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, float).reshape(-1, 2)
        self.cells = max(int(self.cells), 1)
        self.ratio = float(self.ratio) if float(self.ratio) > 0 else 1.0
        if self.curve not in CURVES:
            raise ValueError(f"curve must be one of {', '.join(CURVES)}, got {self.curve!r}")

    def controls(self, vertices: dict) -> np.ndarray:
        """Its start, shaping points and end (k + 2, 2)."""
        a, b = vertices[self.start], vertices[self.end]
        return np.vstack([np.asarray(a, float), self.points, np.asarray(b, float)])

    def line(self, vertices: dict) -> np.ndarray:
        """The edge as a dense polyline."""
        p = self.controls(vertices)
        if self.curve == "straight" or len(p) == 2:
            return curvi._straight(p, 1)
        return curvi.catmull_rom(p) if self.curve == "through" else curvi.bspline(p)

    def nodes(self, vertices: dict) -> np.ndarray:
        """Its ``cells + 1`` nodes along the line (graded by ``ratio``), (n + 1, 2)."""
        line = self.line(vertices)
        s = curvi.lengths(line)
        n = self.cells
        if abs(self.ratio - 1.0) < 1e-12 or n == 1:
            t = np.linspace(0.0, 1.0, n + 1)
        else:
            g = self.ratio ** (1.0 / (n - 1))
            w = g ** np.arange(n)
            t = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
        at = t * s[-1]
        out = np.column_stack([np.interp(at, s, line[:, 0]), np.interp(at, s, line[:, 1])])
        out[0], out[-1] = vertices[self.start], vertices[self.end]  # exactly on the vertices
        return out

    def to_dict(self) -> dict:
        data = {"start": self.start, "end": self.end, "cells": self.cells}
        if len(self.points):
            data["points"] = [[round(float(a), 6), round(float(b), 6)] for a, b in self.points]
        if self.curve != "through":
            data["curve"] = self.curve
        if self.ratio != 1.0:
            data["ratio"] = self.ratio
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Edge:
        known = ("start", "end", "cells", "points", "curve", "ratio")
        return cls(**{k: data[k] for k in known if k in data})


def _ref(name: str) -> tuple[str, bool]:
    """An edge of a face: its name and whether it runs backwards ("-e3")."""
    name = str(name).strip()
    return (name[1:], True) if name.startswith("-") else (name, False)


# ====================================================================== blocks
@dataclass
class Block:
    """A block: the edge chain of each face (``FACES``), the boundary of its faces
    without an interface (``bc``: face -> {"type": ..., "value": ...}) and its
    smoothing sweeps."""

    name: str
    faces: dict
    bc: dict = field(default_factory=dict)
    smoothing: int = 100
    slide: bool = True

    def __post_init__(self) -> None:
        self.faces = {f: [str(e) for e in (self.faces.get(f) or [])] for f in FACES}
        self.bc = {f: dict(v) for f, v in (self.bc or {}).items() if f in FACES}
        self.smoothing = max(int(self.smoothing), 0)

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
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Block:
        return cls(
            str(data.get("name", "")),
            dict(data.get("faces") or {}),
            dict(data.get("bc") or {}),
            int(data.get("smoothing", 100)),
            bool(data.get("slide", True)),
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
    def block_nodes(self, block: int) -> tuple[np.ndarray, np.ndarray]:
        """The nodes (x, y) of a block, each (nj + 1, ni + 1) (i along a row)."""
        key = (self.text(), block)
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
        if not slides:
            return curvi.winslow(x, y, smoothing)
        done = 0
        while done < smoothing:  # Winslow, the nodes of the walls sliding along them
            step = min(SLIDE_EVERY, smoothing - done)
            x, y = curvi.winslow(x, y, step)
            for slide in slides:
                slide(x, y)
            done += step
        return x, y

    def _sliding(self, block: int) -> list:
        """For each edge of the block's outline that no other block shares: a function
        that moves its inner nodes along the edge to where the grid line from the
        inside meets it (the nodes keep their order, the edge's ends stay)."""
        uses = self.edge_uses()
        out = []
        for face in FACES:
            first = 0
            for name, back in self.face_edges(block, face):
                n = self.edges[name].cells
                if len(uses.get(name, [])) == 1 and n > 1:
                    line = self.edges[name].line(self.vertices)
                    if back:
                        line = line[::-1]
                    out.append(_slider(face, first, n, line))
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
        vertices = dict(self.vertices)
        if name not in vertices:
            raise ValueError(f"no vertex {name!r}")
        vertices[name] = (float(x), float(y))
        return replace(self, vertices=vertices, edges=dict(self.edges), blocks=list(self.blocks))

    def moved_point(self, edge: str, index: int, x: float, y: float) -> Topology:
        if edge not in self.edges:
            raise ValueError(f"no edge {edge!r}")
        e = self.edges[edge]
        points = e.points.copy()
        points[int(index)] = (float(x), float(y))
        edges = dict(self.edges)
        edges[edge] = replace(e, points=points)
        return replace(self, vertices=dict(self.vertices), edges=edges, blocks=list(self.blocks))

    def with_edge(self, edge: str, **options) -> Topology:
        """One edge changed (``cells``, ``curve``, ``ratio``, ``points``); a new
        number of cells goes on across the blocks to the opposite faces (single
        edges there), so that opposite faces keep the same number of cells."""
        if edge not in self.edges:
            raise ValueError(f"no edge {edge!r}")
        edges = dict(self.edges)
        edges[edge] = replace(edges[edge], **options)
        result = replace(self, vertices=dict(self.vertices), edges=edges, blocks=list(self.blocks))
        if "cells" in options:
            result = result._propagate(edge)
        return result

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


#: grids made lately: (topology text, block) -> (x, y)
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
            depth = t.size(b)[0 if fb in ("i0", "i1") else 1]
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
    sides = {side_of[f]: code[native.bc_type(f)] for f in FACES}
    inflow = next((native.bc_value(f) for f in FACES if native.bc_type(f) == "inflow"), None)
    level = next((native.bc_value(f) for f in FACES if native.bc_type(f) == "outflow"), None)
    g54 = []
    for b, block in enumerate(t.blocks[1:], start=1):
        for f in FACES:
            kind = block.bc_type(f)
            if kind != "wall" and block.bc_value(f) is not None:
                g54.append([b, 1 if kind == "inflow" else 2, block.bc_value(f)])
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


def mean_spacing(topology: Topology) -> float:
    return settings(topology).spacing


def interface_count(topology: Topology) -> int:
    return sum(1 for use in topology.edge_uses().values() if len(use) == 2)


def summary(topology: Topology) -> str:
    """One line: blocks, cells, interfaces."""
    q = quality(topology)
    return f"{q.text()}; {count(interface_count(topology), 'interface edge')}"
