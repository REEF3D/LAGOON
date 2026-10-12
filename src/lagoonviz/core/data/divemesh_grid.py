# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""DIVEMesh's grid: its lines on the faces of the domain, read from the grid preview
DIVEMesh writes (``DIVEMesh_Paraview/DIVEMesh_grid-preview.vts``), or, before
DIVEMesh has run, a uniform grid from ``control.txt`` (B 1 or B 2 and B 10).

The preview holds every node of the grid; only the nodes on the faces that are
drawn are read (the file is mapped, not loaded), so a grid of many millions of
cells shows at once. A fine grid is drawn with every n-th line (:func:`grid_lines`
picks n so a face has a few hundred lines at most), the domain's edges always.

    grid = read_preview(case)              # None before DIVEMesh has run
    lines = grid_lines(grid, faces="bottom and back")
    print(grid.cells, grid.spacing())      # (nx, ny, nz), (min, max) per axis
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PREVIEW = Path("DIVEMesh_Paraview") / "DIVEMesh_grid-preview.vts"
PARTITION = Path("DIVEMesh_Paraview") / "DIVEMesh_Partition.vtp"
SOLIDS = Path("DIVEMesh_Paraview") / "REEF3D_Solid.vtp"
TOPO = Path("DIVEMesh_Paraview") / "REEF3D_Topo.vtp"
GRID_FOLDER = Path("DIVEMesh_Grid")

#: lines drawn per direction of a face at most (every n-th line beyond)
MAX_LINES = 160

#: the faces of the domain: name -> (axis across it, at its end)
FACES = {
    "x min": (0, False),
    "x max": (0, True),
    "y min": (1, False),
    "y max": (1, True),
    "bottom": (2, False),
    "top": (2, True),
}
#: choices of the grid item's "Faces" property
FACE_SETS = {
    "bottom and back": ("bottom", "x min", "y max"),
    "bottom": ("bottom",),
    "outside": tuple(FACES),
    "back": ("x min", "y max"),
}


@dataclass
class Grid:
    """The nodes of a structured grid: ``shape`` (nodes in x, y, z) and a function
    giving the nodes of any index box."""

    shape: tuple[int, int, int]
    #: (n, 3) nodes, x fastest, then y, then z (a memory map of the preview file)
    points: np.ndarray
    source: str  # "DIVEMesh" (the preview) or "control.txt" (uniform, not run yet)
    path: Path | None = None

    @property
    def cells(self) -> tuple[int, int, int]:
        return tuple(max(n - 1, 1) for n in self.shape)  # type: ignore[return-value]

    def nodes(self, i, j, k) -> np.ndarray:
        """The nodes at index arrays (broadcast) i, j, k."""
        nx, ny, _ = self.shape
        index = np.asarray(i) + nx * (np.asarray(j) + ny * np.asarray(k))
        return np.asarray(self.points[index.ravel()], np.float64).reshape(*np.shape(index), 3)

    def axis(self, axis: int) -> np.ndarray:
        """The node coordinates along one axis (at the other indices 0)."""
        n = self.shape[axis]
        index = [0, 0, 0]
        index[axis] = np.arange(n)
        return self.nodes(*index)[:, axis]

    def spacing(self) -> list[tuple[float, float]]:
        """(smallest, largest) cell size along x, y and z."""
        out = []
        for axis in range(3):
            d = np.diff(self.axis(axis))
            out.append((float(np.abs(d).min()), float(np.abs(d).max())) if d.size else (0.0, 0.0))
        return out

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny, nz = self.shape
        corners = self.nodes(
            np.array([0, nx - 1, 0, nx - 1, 0, nx - 1, 0, nx - 1]),
            np.array([0, 0, ny - 1, ny - 1, 0, 0, ny - 1, ny - 1]),
            np.array([0, 0, 0, 0, nz - 1, nz - 1, nz - 1, nz - 1]),
        )
        return corners.min(axis=0), corners.max(axis=0)


def read_preview(case_dir: str | Path) -> Grid | None:
    """The grid DIVEMesh wrote for a case (None: DIVEMesh has not run there)."""
    path = Path(case_dir) / PREVIEW
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        head = handle.read(8192)
    text = head.decode("latin-1")
    extent = re.search(r'WholeExtent="\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', text)
    points = re.search(
        r"<Points>\s*<DataArray[^>]*type=\"Float(32|64)\"[^>]*offset=\"(\d+)\"", text, re.S
    )
    start = text.find("<AppendedData")
    if extent is None or points is None or start < 0:
        return None
    data = text.find("_", start) + 1
    a = [int(v) for v in extent.groups()]
    shape = (a[1] - a[0] + 1, a[3] - a[2] + 1, a[5] - a[4] + 1)
    dtype = np.dtype("<f4" if points.group(1) == "32" else "<f8")
    count = shape[0] * shape[1] * shape[2]
    offset = data + int(points.group(2)) + 4  # each array starts with its size (UInt32)
    if path.stat().st_size < offset + count * 3 * dtype.itemsize:
        return None  # not written to the end (DIVEMesh still running)
    nodes = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count, 3))
    return Grid(shape, nodes, "DIVEMesh", path)


def uniform_grid(domain, cells) -> Grid | None:
    """The uniform grid control.txt sets up (B 10 and B 1 or B 2), before DIVEMesh
    has run (None without a domain or cells)."""
    if domain is None or cells is None:
        return None
    xs, xe, ys, ye, zs, ze = domain
    axes = [
        np.linspace(xs, xe, cells[0] + 1),
        np.linspace(ys, ye, cells[1] + 1),
        np.linspace(zs, ze, cells[2] + 1),
    ]
    shape = tuple(len(a) for a in axes)

    class _Lazy:
        """Nodes computed on demand (a uniform grid of millions of nodes is not
        stored)."""

        def __getitem__(self, index):
            index = np.asarray(index)
            i = index % shape[0]
            j = (index // shape[0]) % shape[1]
            k = index // (shape[0] * shape[1])
            return np.stack([axes[0][i], axes[1][j], axes[2][k]], axis=-1)

    return Grid(shape, _Lazy(), "control.txt")  # type: ignore[arg-type]


def _picked(n: int, every: int) -> np.ndarray:
    """Every ``every``-th of n lines, the first and the last always."""
    picked = np.arange(0, n, max(every, 1))
    if picked[-1] != n - 1:
        picked = np.append(picked, n - 1)
    return picked


def line_step(grid: Grid, every: int = 0) -> int:
    """How many lines to step: ``every``, or (0) so no face has more than MAX_LINES."""
    if every > 0:
        return every
    return max(1, int(np.ceil(max(grid.shape) / MAX_LINES)))


def grid_lines(grid: Grid, faces: str | tuple[str, ...] = "bottom and back", every: int = 0):
    """Line segments (n, 2, 3) of the grid on the faces of the domain: every node line
    (every ``every``-th, see :func:`line_step`) along both directions of each face."""
    names = FACE_SETS.get(faces, ()) if isinstance(faces, str) else faces
    step = line_step(grid, every)
    out = []
    shape = grid.shape
    for name in names:
        axis, at_end = FACES[name]
        across = shape[axis] - 1 if at_end else 0
        a, b = [x for x in range(3) if x != axis]
        if shape[a] < 2 and shape[b] < 2:
            continue
        for along, other in ((a, b), (b, a)):
            if shape[along] < 2:
                continue
            for value in _picked(shape[other], step):
                index = [None, None, None]
                index[axis] = np.full(shape[along], across)
                index[other] = np.full(shape[along], value)
                index[along] = np.arange(shape[along])
                nodes = grid.nodes(*index)
                out.append(np.stack([nodes[:-1], nodes[1:]], axis=1))
    if not out:
        return np.zeros((0, 2, 3))
    return np.concatenate(out)


def read_partition(case_dir: str | Path) -> np.ndarray:
    """The planes between the MPI ranks DIVEMesh chose (M 10), as outline segments
    (n, 2, 3); none if there is no partition file or only one rank."""
    path = Path(case_dir) / PARTITION
    if not path.is_file():
        return np.zeros((0, 2, 3))
    try:
        from lagoonviz.core.data.vtkxml import read_piece

        piece = read_piece(path)
    except (OSError, ValueError, KeyError):
        return np.zeros((0, 2, 3))
    points = np.asarray(piece.points, float)
    ends = np.asarray(piece.offsets, np.int64)
    corners = np.asarray(piece.connectivity, np.int64)
    triangles = []
    begin = 0
    for end in ends:  # each polygon as a fan of triangles
        ring = corners[begin:end]
        begin = end
        triangles += [(ring[0], ring[n], ring[n + 1]) for n in range(1, len(ring) - 1)]
    if not triangles:
        return np.zeros((0, 2, 3))
    from lagoonviz.core.data.setup import _feature_edges

    # the outlines of the planes, without the lines between their triangles
    return _feature_edges(points[np.asarray(triangles)])


#: DIVEMesh's grid geometry (grid format v2): its sections, the georeference among them
GEOMETRY = GRID_FOLDER / "grid-geometry.dat"


def grid_sections(path: str | Path) -> dict[str, tuple[int, int]]:
    """The sections of a DIVEMesh grid v2 file: tag -> (offset, size) of its payload.
    Empty for a file that is not one (or not there)."""
    import struct

    found: dict[str, tuple[int, int]] = {}
    try:
        with open(path, "rb") as file:
            head = file.read(16)
            if len(head) < 16 or head[:6] not in (b"DMGRID", b"DMGEOM"):
                return {}
            if struct.unpack("<i", head[12:16])[0] != 0x01020304:
                return {}  # written on a big-endian machine: not read here
            while True:
                tag = file.read(4)
                size_bytes = file.read(8)
                if len(tag) < 4 or len(size_bytes) < 8:
                    break
                size = struct.unpack("<q", size_bytes)[0]
                name = tag.decode("ascii", "replace")
                if name == "END " or size < 0:
                    break
                found[name] = (file.tell(), size)
                file.seek(size, 1)
    except OSError:
        return {}
    return found


def read_grid_georef(case_dir: str | Path):
    """The georeference DIVEMesh kept in its grid (the GREF section of
    grid-geometry.dat, from the case's georef.json when DIVEMesh ran), or None."""
    from lagoon_format.geo import Georef, GeorefError

    path = Path(case_dir) / GEOMETRY
    sections = grid_sections(path)
    if "GREF" not in sections:
        return None
    offset, size = sections["GREF"]
    try:
        with open(path, "rb") as file:
            file.seek(offset)
            text = file.read(size).decode("utf-8")
        return Georef.from_json(text)
    except (OSError, UnicodeDecodeError, GeorefError):
        return None
