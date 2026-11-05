# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Adaptive mesh refinement output of REEF3D (FNPF, NHFLOW and SFLOW with A 270).

The AMR solvers write their free surface to ``REEF3D_<SOLVER>_AMR``: per output a
``REEF3D-<SOLVER>-AMR-<output>.vtm`` that lists the level-0 grid of every MPI rank
(``...-AMR-L0-<output>-<rank>.vtr``) and every refined patch
(``...-AMR-<output>-<rank>-<patch>.vtr``). Each grid is a 2D VTK rectilinear grid
with cell data (eta, elevation, bed, ...) and its refinement ``level`` and the
``TimeValue`` in the field data.

LAGOON draws the leaf surface: every grid on its own node lattice (cell values
averaged to the nodes), without the cells a finer patch covers. The outlines of
the patches are kept with the mesh (``SurfaceMesh.outlines``), and every point has
its grid's ``level``, so the surface can be coloured by refinement level.

    grids = read_frame("REEF3D_NHFLOW_AMR/REEF3D-NHFLOW-AMR-00000012.vtm")
    mesh = leaf_mesh(grids)
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from lagoon_format.amr import Boxes, covered

from lagoonviz.core.data.mesh import SurfaceMesh, grid_normals, grid_points
from lagoonviz.core.data.vtkxml import (
    VTK_TYPES,
    PieceHeader,
    VTKFormatError,
    _array_info,
    _read_xml_head,
    read_array,
)

log = logging.getLogger(__name__)

#: ``REEF3D_NHFLOW_AMR``
AMR_FOLDER = re.compile(r"^REEF3D_(?P<solver>[A-Z0-9]+)_AMR$")
#: ``REEF3D-NHFLOW-AMR-00000012.vtm``
AMR_INDEX = re.compile(r"^REEF3D-(?P<solver>[A-Z0-9]+)-AMR-(?P<step>\d+)\.vtm$")
#: arrays read with every frame: the height of the surface and where it is wet (to
#: hide dry land: SFLOW's ``wetdry``; ``wet``, ``WL``, else ``elevation`` above ``bed``)
#: (``covered``: which cells another grid shows, of a curvilinear grid, G 50)
SURFACE_ARRAYS = ("elevation", "eta", "WL", "wet", "wetdry", "bed", "covered")

#: water shallower than this [m] is dry land: REEF3D keeps a film of about a millimetre
#: (its wetting and drying depth) on the dry cells
DRY_DEPTH = 0.005

#: the velocity components an AMR output writes as scalars: SFLOW u, v, w; NHFLOW
#: the top layer's u_top, v_top (the velocity at the free surface). LAGOON joins them
#: into the vector ``velocity`` (its magnitude, components and arrows)
VELOCITY_SETS = (("u", "v", "w"), ("u_top", "v_top", "w_top"))
VELOCITY_PARTS = VELOCITY_SETS[0]


def velocity_parts(present) -> tuple[str, str, str] | None:
    """The scalars x, y, z of ``velocity`` among the arrays ``present`` (z need not
    be written), or None without x and y."""
    for parts in VELOCITY_SETS:
        if parts[0] in present and parts[1] in present:
            return parts
    return None


def with_velocity(components: dict[str, int]) -> dict[str, int]:
    """The variables of an AMR output with ``velocity`` (3 components) added when it
    writes u and v (and w), or u_top and v_top, as scalars but no velocity vector."""
    parts = velocity_parts(components)
    if "velocity" not in components and parts and all(components[n] == 1 for n in parts[:2]):
        components = {**components, "velocity": 3}
    return components


def wanted_arrays(names: set[str], present) -> list[str]:
    """The arrays to read for these variables: ``velocity`` made of u, v, w (or
    u_top, v_top) when the output has no velocity array of its own."""
    names = set(names)
    if "velocity" in names and "velocity" not in present:
        names.discard("velocity")
        names.update(n for n in velocity_parts(present) or () if n in present)
    return sorted(n for n in names if n in present and n != "level")


def series_name(variable: str, present) -> str:
    """The array a point series of a variable reads: ``velocity.x`` is u (u_top)
    when the output writes the components as scalars; else the variable's array."""
    name, _, component = variable.partition(".")
    parts = velocity_parts(present)
    if name == "velocity" and "velocity" not in present and parts and component in ("x", "y", "z"):
        return parts["xyz".index(component)]
    return name


def _join_velocity(data: dict[str, np.ndarray]) -> None:
    """``velocity`` (n, 3) of the scalar components in ``data``, if it has none."""
    parts = velocity_parts(data)
    if "velocity" in data or parts is None:
        return
    u = data[parts[0]]
    w = data.get(parts[2], np.zeros_like(u))
    data["velocity"] = np.ascontiguousarray(np.stack([u, data[parts[1]], w], axis=-1))


@dataclass
class AmrGrid:
    """One grid of an AMR output: level 0 of an MPI rank, or a refined patch."""

    level: int
    x: np.ndarray  # (nx + 1,) node coordinates; curvilinear: (ny + 1, nx + 1), every node's
    y: np.ndarray  # (ny + 1,)
    z: float = 0.0
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)  # (ny, nx), x fastest
    point_data: dict[str, np.ndarray] = field(default_factory=dict)  # (ny + 1, nx + 1)
    time: float | None = None
    name: str = ""

    @property
    def curvilinear(self) -> bool:
        """A grid of a curvilinear grid (REEF3D's G 50): x and y of every node."""
        return np.ndim(self.x) == 2

    @property
    def nx(self) -> int:
        return int(self.x.shape[1] if self.curvilinear else self.x.size) - 1

    @property
    def ny(self) -> int:
        return int(self.y.shape[0] if self.curvilinear else self.y.size) - 1

    @property
    def box(self) -> tuple[float, float, float, float]:
        if self.curvilinear:
            return (
                float(self.x.min()),
                float(self.x.max()),
                float(self.y.min()),
                float(self.y.max()),
            )
        return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])

    def contains(self, x: float, y: float) -> bool:
        x0, x1, y0, y1 = self.box
        return x0 <= x <= x1 and y0 <= y <= y1


# ======================================================================== files
def index_files(path: str | Path) -> list[Path]:
    """The grid files an AMR index (``.vtm``) lists, in its order: level 0, then the
    patches (any nesting of blocks)."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise VTKFormatError(f"{path}: not a VTK multiblock file ({error})") from None
    if root.tag != "VTKFile" or root.get("type") != "vtkMultiBlockDataSet":
        raise VTKFormatError(f"{path}: not a VTK multiblock file")
    return [path.parent / e.get("file", "") for e in root.iter("DataSet") if e.get("file")]


@dataclass
class _GridHeader:
    header: PieceHeader
    extent: tuple[int, ...]
    coordinates: dict[str, object]
    level: int
    time: float | None


def _grid_header(path: Path, volume: bool = False) -> _GridHeader:
    """The header of a grid file: a 2D grid (an AMR surface's), or with ``volume`` a
    3D block (CFD's AMR, :mod:`lagoonviz.core.data.amr3d`)."""
    text, data_start = _read_xml_head(path)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise VTKFormatError(f"{path}: not a VTK XML file ({error})") from None
    kind = root.get("type")
    # a structured grid: a grid of a curvilinear grid (REEF3D's G 50), every node's place
    if root.tag != "VTKFile" or kind not in ("RectilinearGrid", "StructuredGrid"):
        raise VTKFormatError(f"{path}: not a VTK rectilinear or structured grid")
    if kind == "StructuredGrid" and volume:
        raise VTKFormatError(f"{path}: a structured grid, not a block of a volume")
    byte_order = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    header_type = root.get("header_type", "UInt32")
    piece = root.find(f"{kind}/Piece")
    if piece is None:
        raise VTKFormatError(f"{path}: no Piece")
    extent = tuple(int(v) for v in piece.get("Extent", "0 0 0 0 0 0").split())
    if len(extent) != 6:
        raise VTKFormatError(f"{path}: bad Extent")
    nx, ny, nz = extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4]
    if nz != 0 and not volume:
        raise VTKFormatError(f"{path}: a 3D grid; AMR surfaces are 2D")
    if nz == 0 and volume:
        raise VTKFormatError(f"{path}: a 2D grid, not a block of a volume")
    layers = nz + 1 if volume else 1
    header = PieceHeader(
        path=path,
        kind=kind,
        n_points=(nx + 1) * (ny + 1) * layers,
        n_cells=nx * ny * max(nz, 1),
        points=None,
        header_dtype=np.dtype(VTK_TYPES[header_type]).newbyteorder(byte_order),
        byte_order=byte_order,
        compressed=bool(root.get("compressor")),
        data_start=data_start,
    )
    appended = root.find("AppendedData")
    if appended is not None:
        header.appended_encoding = appended.get("encoding", "raw")
    for element in piece.findall("PointData/DataArray"):
        info = _array_info(element, byte_order)
        header.point_data[info.name] = info
    for element in piece.findall("CellData/DataArray"):
        info = _array_info(element, byte_order)
        header.cell_data[info.name] = info
    coordinates = {}
    for axis, element in zip("xyz", piece.findall("Coordinates/DataArray"), strict=False):
        coordinates[axis] = _array_info(element, byte_order)
    points = piece.find("Points/DataArray")
    if points is not None:
        coordinates["points"] = _array_info(points, byte_order)
    level, time = 0, None
    # (SEASTATE writes its FieldData in the Piece, with the time as TIME)
    for element in [
        *root.findall(f"{kind}/FieldData/DataArray"),
        *root.findall(f"{kind}/Piece/FieldData/DataArray"),
    ]:
        values = (element.text or "").split()
        if not values:
            continue
        if element.get("Name") == "level":
            level = int(float(values[0]))
        elif element.get("Name") in ("TimeValue", "TIME"):
            time = float(values[0])
    return _GridHeader(header, extent, coordinates, level, time)


def grid_arrays(path: str | Path) -> dict[str, int]:
    """The arrays of a grid file: name -> components (cell and point data)."""
    head = _grid_header(Path(path)).header
    found = {name: info.components for name, info in head.cell_data.items()}
    for name, info in head.point_data.items():
        found.setdefault(name, info.components)
    return found


def read_grid(path: str | Path, arrays: list[str] | None = None) -> AmrGrid:
    """One grid file; ``arrays=None`` reads every array."""
    path = Path(path)
    grid = _grid_header(path)
    head = grid.header
    nx = grid.extent[1] - grid.extent[0]
    ny = grid.extent[3] - grid.extent[2]
    sizes = {"x": nx + 1, "y": ny + 1, "z": 1}
    axes = {}
    if "points" in grid.coordinates:  # a structured grid: every node's place
        nodes = read_array(head, grid.coordinates["points"], (nx + 1) * (ny + 1))
        nodes = np.asarray(nodes, np.float64).reshape(ny + 1, nx + 1, -1)
        axes = {
            "x": nodes[..., 0].copy(),
            "y": nodes[..., 1].copy(),
            "z": nodes[:1, :1, -1].ravel(),
        }
    for axis in "xyz" if not axes else ():
        info = grid.coordinates.get(axis)
        if info is None:
            raise VTKFormatError(f"{path}: no {axis} coordinates")
        axes[axis] = read_array(head, info, sizes[axis]).astype(np.float64)
    wanted = None if arrays is None else set(arrays)
    cells = {
        name: read_array(head, info, nx * ny).reshape(
            (ny, nx, -1) if info.components > 1 else (ny, nx)
        )
        for name, info in head.cell_data.items()
        if wanted is None or name in wanted
    }
    points = {
        name: read_array(head, info, (nx + 1) * (ny + 1)).reshape(
            (ny + 1, nx + 1, -1) if info.components > 1 else (ny + 1, nx + 1)
        )
        for name, info in head.point_data.items()
        if wanted is None or name in wanted
    }
    return AmrGrid(
        level=grid.level,
        x=axes["x"],
        y=axes["y"],
        z=float(axes["z"][0]),
        cell_data=cells,
        point_data=points,
        time=grid.time,
        name=path.name,
    )


def read_frame(index: str | Path, arrays: list[str] | None = None) -> list[AmrGrid]:
    """All grids of one output (its ``.vtm``)."""
    return [read_grid(path, arrays) for path in index_files(index)]


# ======================================================================== surface
def _nodes(cells: np.ndarray) -> np.ndarray:
    """Cell values (ny, nx[, c]) averaged to the nodes (ny + 1, nx + 1[, c])."""
    cells = np.asarray(cells, np.float32)
    ny, nx = cells.shape[:2]
    padded = np.empty((ny + 2, nx + 2, *cells.shape[2:]), np.float32)  # edges repeated
    padded[1:-1, 1:-1] = cells
    padded[0, 1:-1], padded[-1, 1:-1] = cells[0], cells[-1]
    padded[:, 0], padded[:, -1] = padded[:, 1], padded[:, -2]
    out = padded[:-1, :-1] + padded[1:, :-1]
    out += padded[:-1, 1:]
    out += padded[1:, 1:]
    out *= 0.25
    return out


def _outline(nx: int, ny: int) -> np.ndarray:
    """Node index pairs around the border of an (nx, ny) cell lattice."""
    width = nx + 1
    ring = (
        [i for i in range(nx + 1)]  # bottom, left to right
        + [j * width + nx for j in range(1, ny + 1)]  # right, upwards
        + [ny * width + i for i in range(nx - 1, -1, -1)]  # top, right to left
        + [j * width for j in range(ny - 1, 0, -1)]  # left, downwards
    )
    ring = np.asarray(ring + [ring[0]], np.int64)
    return np.stack([ring[:-1], ring[1:]], axis=1)


#: ghost cells round a grid, from its neighbours: 2 layers (the nodes on its border
#: and one ring of nodes beyond it, for the normals there)
GHOST = 2


def _neighbours(k: int, grids: list[AmrGrid], boxes: Boxes) -> list[int]:
    """The grids round grid ``k`` that may hold its ghost cells: those of its level,
    then the coarser ones, finest first (each overlapping its box widened by
    ``GHOST`` cells)."""
    g = grids[k]
    dx = GHOST * max(float(g.x[1] - g.x[0]), float(g.x[-1] - g.x[-2]))
    dy = GHOST * max(float(g.y[1] - g.y[0]), float(g.y[-1] - g.y[-2]))
    b = boxes.box
    hit = (b[:, 1] >= g.x[0] - dx) & (b[:, 0] <= g.x[-1] + dx)
    hit &= (b[:, 3] >= g.y[0] - dy) & (b[:, 2] <= g.y[-1] + dy)
    hit &= boxes.level <= g.level
    hit[k] = False
    found = np.flatnonzero(hit)
    return sorted(found.tolist(), key=lambda n: -int(boxes.level[n]))


def _widened(edges: np.ndarray) -> np.ndarray:
    """Cell centres of a lattice and of ``GHOST`` cells beyond each end."""
    e = np.asarray(edges, np.float64)
    centres = 0.5 * (e[:-1] + e[1:])
    first, last = e[1] - e[0], e[-1] - e[-2]
    before = centres[0] - first * np.arange(GHOST, 0, -1)
    after = centres[-1] + last * np.arange(1, GHOST + 1)
    return np.concatenate([before, centres, after])


def _ghosted(k: int, grids: list[AmrGrid], names, neighbours: list[int]):
    """The cells of grid ``k`` with ``GHOST`` layers round them taken from the
    neighbours' cells (the same level first, else the finest coarser grid there),
    else its own border cells repeated: name -> (ny + 4, nx + 4[, c]); and where each
    ghost cell came from (the grid's index, -1: its own)."""
    g = grids[k]
    xc, yc = _widened(g.x), _widened(g.y)
    ny, nx = g.ny, g.nx
    source = np.full((ny + 2 * GHOST, nx + 2 * GHOST), -1, np.int64)
    inner = (slice(GHOST, GHOST + ny), slice(GHOST, GHOST + nx))
    source[inner] = k
    out = {}
    for name in names:
        cells = np.asarray(g.cell_data[name], np.float32)
        ext = np.empty((ny + 2 * GHOST, nx + 2 * GHOST, *cells.shape[2:]), np.float32)
        ext[inner] = cells
        out[name] = ext
    open_ = source < 0
    for n in neighbours:
        if not open_.any():
            break
        other = grids[n]
        if any(name not in other.cell_data for name in names):
            continue
        ox, oy = np.asarray(other.x, np.float64), np.asarray(other.y, np.float64)
        cols = (xc > ox[0]) & (xc < ox[-1])
        rows = (yc > oy[0]) & (yc < oy[-1])
        take = open_ & rows[:, None] & cols[None, :]
        if not take.any():
            continue
        jj, ii = np.nonzero(take)
        oi = np.clip(np.searchsorted(ox, xc[ii], "right") - 1, 0, other.nx - 1)
        oj = np.clip(np.searchsorted(oy, yc[jj], "right") - 1, 0, other.ny - 1)
        for name in names:
            out[name][jj, ii] = np.asarray(other.cell_data[name])[oj, oi]
        source[jj, ii] = n
        open_[jj, ii] = False
    if open_.any():
        # the domain's border: the cells next to it again, across it (those of a
        # neighbour too, so that two grids' corners on the border agree)
        rows_c = np.clip(np.arange(ny + 2 * GHOST), GHOST, GHOST + ny - 1)
        cols_c = np.clip(np.arange(nx + 2 * GHOST), GHOST, GHOST + nx - 1)
        filled = ~open_
        for across_rows in (True, False):
            jj, ii = np.nonzero(open_)
            fj = rows_c[jj] if across_rows else jj
            fi = ii if across_rows else cols_c[ii]
            ok = filled[fj, fi] & ((fj != jj) | (fi != ii))
            jj, ii, fj, fi = jj[ok], ii[ok], fj[ok], fi[ok]
            for name in names:
                out[name][jj, ii] = out[name][fj, fi]
            open_[jj, ii] = False
            filled[jj, ii] = True
        jj, ii = np.nonzero(open_)
        for name in names:
            out[name][jj, ii] = out[name][rows_c[jj], cols_c[ii]]
    return out, source


def _node_average(cells: np.ndarray) -> np.ndarray:
    """Cells (m, n[, c]) averaged to the nodes between them (m - 1, n - 1[, c])."""
    out = cells[:-1, :-1] + cells[1:, :-1]
    out += cells[:-1, 1:]
    out += cells[1:, 1:]
    out *= 0.25
    return out


def _widened_edges(edges: np.ndarray) -> np.ndarray:
    """Node coordinates with one more node beyond each end."""
    e = np.asarray(edges, np.float64)
    return np.concatenate([[2 * e[0] - e[1]], e, [2 * e[-1] - e[-2]]])


def _bilinear(x: np.ndarray, y: np.ndarray, values: np.ndarray, px, py) -> np.ndarray:
    """Values on a node lattice (len(y), len(x)[, c]) at points, bilinear."""
    i = np.clip(np.searchsorted(x, px, "right") - 1, 0, len(x) - 2)
    j = np.clip(np.searchsorted(y, py, "right") - 1, 0, len(y) - 2)
    fx = np.clip((px - x[i]) / (x[i + 1] - x[i]), 0.0, 1.0)
    fy = np.clip((py - y[j]) / (y[j + 1] - y[j]), 0.0, 1.0)
    if values.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    low = values[j, i] * (1 - fx) + values[j, i + 1] * fx
    high = values[j + 1, i] * (1 - fx) + values[j + 1, i + 1] * fx
    return (low * (1 - fy) + high * fy).astype(np.float32)


def _border_nodes(ny: int, nx: int):
    """The nodes on the border of a lattice of (ny, nx) cells, side by side: (j, i)
    of each node, and the layer-1 ghost cells outside it (two per node, in the
    ``GHOST``-widened cell indices)."""
    j_all, i_all = np.arange(ny + 1), np.arange(nx + 1)
    g = GHOST
    sides = (
        (
            j_all,
            np.zeros_like(j_all),
            (j_all + g - 1, np.full_like(j_all, g - 1)),
            (j_all + g, np.full_like(j_all, g - 1)),
        ),
        (
            j_all,
            np.full_like(j_all, nx),
            (j_all + g - 1, np.full_like(j_all, nx + g)),
            (j_all + g, np.full_like(j_all, nx + g)),
        ),
        (
            np.zeros_like(i_all),
            i_all,
            (np.full_like(i_all, g - 1), i_all + g - 1),
            (np.full_like(i_all, g - 1), i_all + g),
        ),
        (
            np.full_like(i_all, ny),
            i_all,
            (np.full_like(i_all, ny + g), i_all + g - 1),
            (np.full_like(i_all, ny + g), i_all + g),
        ),
    )
    return sides


def _curvilinear_leaf_mesh(grids: list[AmrGrid], max_level: int | None) -> SurfaceMesh:
    """The surface of the leaf cells of the grids of a curvilinear grid (REEF3D's G 50,
    also the blocks of a multi-block grid, G 50 1): each grid's cells are the
    quadrilaterals of its nodes; which cells another grid shows says the cell field
    ``covered`` (1: a finer patch over it, 2: the ring of block 0, whose cells the
    blocks 1 .. have). A node's values are the mean of the cells round it, of every
    grid that has the node (two ranks, two blocks: the same place, one point), so
    the grids meet without a crack and shade as one."""
    names = set.intersection(*(set(g.cell_data) for g in grids)) - {"covered"}
    names = sorted(names)
    if max_level is not None:
        grids = [g for g in grids if g.level <= max_level]
    grids = [g for g in grids if g.nx >= 1 and g.ny >= 1]
    if not grids:
        return SurfaceMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))
    xy = np.concatenate([np.column_stack([np.ravel(g.x), np.ravel(g.y)]) for g in grids]).astype(
        np.float64
    )
    span = float(np.ptp(xy, axis=0).max()) or 1.0
    key = np.round(xy / (1e-7 * span)).astype(np.int64)
    unique, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    count = len(unique)
    sums = {name: None for name in names}
    weight = np.zeros(count)
    levels = np.full(count, -1.0)
    triangles, outlines = [], []
    offset = 0
    for g in grids:
        nx, ny = g.nx, g.ny
        ids = inverse[offset : offset + (nx + 1) * (ny + 1)].reshape(ny + 1, nx + 1)
        corners = (ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1])
        for c in corners:
            np.add.at(weight, c.ravel(), 1.0)
        for name in names:
            cells = np.asarray(g.cell_data[name], np.float64).reshape(ny, nx, -1)
            if sums[name] is None:
                sums[name] = np.zeros((count, cells.shape[-1]))
            for c in corners:
                np.add.at(sums[name], c.ravel(), cells.reshape(-1, cells.shape[-1]))
        np.maximum.at(levels, ids.ravel(), float(g.level))
        flag = g.cell_data.get("covered")
        hidden = np.zeros((ny, nx), bool)
        if flag is not None:
            flag = np.asarray(flag).reshape(ny, nx)
            hidden = flag > 1.5  # the ring: always another grid's
            if max_level is None or g.level < max_level:
                hidden |= flag > 0.5  # a finer patch over it, shown
        a, b, c, d = (k[~hidden] for k in corners)
        triangles.append(np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])]))
        if g.level > 0:
            ring = np.concatenate([ids[0, :], ids[1:, -1], ids[-1, -2::-1], ids[-2::-1, 0]])
            outlines.append(np.column_stack([ring[:-1], ring[1:]]))
        offset += (nx + 1) * (ny + 1)
    weight = np.maximum(weight, 1.0)[:, None]
    point_data = {}
    for name in names:
        values = (sums[name] / weight).astype(np.float32)
        point_data[name] = np.ascontiguousarray(values[:, 0] if values.shape[1] == 1 else values)
    point_data["level"] = levels.astype(np.float32)
    height = point_data.get("elevation", point_data.get("eta"))
    z = height if height is not None else np.zeros(count, np.float32)
    points = np.column_stack([xy[first], z]).astype(np.float32)
    _join_velocity(point_data)
    if "WL" in point_data and "depth" not in point_data and "wetdry" not in point_data:
        point_data["depth"] = point_data["WL"]
    return SurfaceMesh(
        points=points,
        triangles=np.ascontiguousarray(np.concatenate(triangles), dtype=np.uint32),
        point_data=point_data,
        outlines=np.concatenate(outlines).astype(np.uint32) if outlines else None,
    )


def _with_wetdry(grids: list[AmrGrid]) -> list[AmrGrid]:
    """The grids with the cell field ``wetdry`` (1 wet, 0 dry; averaged to the nodes,
    the share of the cells around that are wet, as SFLOW writes it), which hides dry
    land, when they say where it is wet in another way: NHFLOW's ``wet`` (G 30), its
    water depth ``WL``, or the ``elevation`` above the ``bed`` (NHFLOW on a
    curvilinear grid writes neither), wet deeper than :data:`DRY_DEPTH`. As they are
    when they have ``wetdry`` or none of these."""
    if not grids or any("wetdry" in g.cell_data or "wetdry" in g.point_data for g in grids):
        return grids

    def wet(cells: dict[str, np.ndarray]) -> np.ndarray | None:
        if "wet" in cells:
            return np.asarray(cells["wet"]) > 0.5
        if "WL" in cells:
            return np.asarray(cells["WL"]) > DRY_DEPTH
        if "elevation" in cells and "bed" in cells:
            return np.asarray(cells["elevation"]) - np.asarray(cells["bed"]) > DRY_DEPTH
        return None

    flags = [wet(g.cell_data) for g in grids]
    if any(f is None for f in flags):
        return grids
    return [
        replace(g, cell_data={**g.cell_data, "wetdry": f.astype(np.float32)})
        for g, f in zip(grids, flags, strict=True)
    ]


def leaf_mesh(grids: list[AmrGrid], max_level: int | None = None) -> SurfaceMesh:
    """The surface of the leaf cells of all grids: each grid on its node lattice,
    without the cells a finer grid covers (grids finer than ``max_level`` left out:
    0 gives the base grid alone). The height is ``elevation`` (else
    ``eta``); point data are the cell data averaged to the nodes, the grid's
    ``level``, and where it is wet, ``wetdry`` (:func:`_with_wetdry`, to hide dry
    land). The borders of the refined patches are the mesh's ``outlines``.

    The grids meet without a crack: a node on the border between two grids of a
    level (two MPI ranks) averages the cells on both sides, so both grids give it
    the same value, and its normal sees both sides; a node of a finer grid on the
    border to a coarser one takes the coarser surface's value there (linear along
    the coarse cell's edge, which the coarse triangles have too)."""
    grids = _with_wetdry(grids)
    if any(g.curvilinear for g in grids):
        return _curvilinear_leaf_mesh(grids, max_level)
    names = (
        set.intersection(*(set(g.cell_data) | set(g.point_data) for g in grids)) if grids else set()
    )
    if max_level is not None:
        grids = [g for g in grids if g.level <= max_level]
    grids = [g for g in grids if g.nx >= 1 and g.ny >= 1]
    boxes = Boxes(grids)  # the finer grids over each found at once (thousands of patches)
    nodes_of: dict[int, dict[str, np.ndarray]] = {}
    normals_of: dict[int, np.ndarray] = {}
    # coarse grids first: a finer grid's border takes the coarser grid's final nodes
    for k in sorted(range(len(grids)), key=lambda n: grids[n].level):
        grid = grids[k]
        nx, ny = grid.nx, grid.ny
        cell_names = sorted(n for n in names if n not in grid.point_data and n in grid.cell_data)
        ext, source = _ghosted(k, grids, cell_names, _neighbours(k, grids, boxes))
        wide = {name: _node_average(values) for name, values in ext.items()}  # ny + 3, nx + 3
        nodes = {name: values[1:-1, 1:-1] for name, values in wide.items()}
        for name in names:
            if name in grid.point_data:
                nodes[name] = grid.point_data[name].astype(np.float32, copy=False)
        # a finer grid's border on a coarser grid: the coarser surface's values there
        if grid.level > 0:
            for jn, in_, (ja, ia), (jb, ib) in _border_nodes(ny, nx):
                # the coarser grid outside the node (of its two cells outside)
                coarse = np.full(jn.size, -1, np.int64)
                for s in (source[jb, ib], source[ja, ia]):
                    lower = (s >= 0) & (boxes.level[np.maximum(s, 0)] < grid.level)
                    coarse = np.where(lower, s, coarse)
                for c in np.unique(coarse[coarse >= 0]):
                    at = coarse == c
                    cx, cy = np.asarray(grids[c].x), np.asarray(grids[c].y)
                    px, py = np.asarray(grid.x)[in_[at]], np.asarray(grid.y)[jn[at]]
                    for name in cell_names:
                        if name in nodes_of[c]:
                            nodes[name][jn[at], in_[at]] = _bilinear(
                                cx, cy, nodes_of[c][name], px, py
                            )
        if "elevation" in nodes:
            z = nodes["elevation"]
        elif "eta" in nodes:
            z = nodes["eta"]
        else:
            z = np.full((ny + 1, nx + 1), grid.z, np.float32)
        height = "elevation" if "elevation" in wide else "eta" if "eta" in wide else None
        if height is not None:
            # each grid a height field: normals from its slopes, one ring of nodes
            # beyond its border from the cells round it (seamless across the grids;
            # from all triangles at once they took seconds)
            normals = grid_normals(wide[height], _widened_edges(grid.x), _widened_edges(grid.y))
            normals = normals[1:-1, 1:-1]
            # the domain's border (no grid beyond): one-sided slopes across it, as
            # before (along it, the ring's nodes: two grids' corners agree)
            w = wide[height]
            x, y = np.asarray(grid.x), np.asarray(grid.y)
            xw, yw = _widened_edges(x), _widened_edges(y)
            if (source[:, GHOST - 1] < 0).all():
                normals[:, 0] = grid_normals(w[:, 1:3], x[:2], yw)[1:-1, 0]
            if (source[:, nx + GHOST] < 0).all():
                normals[:, -1] = grid_normals(w[:, -3:-1], x[-2:], yw)[1:-1, -1]
            if (source[GHOST - 1, :] < 0).all():
                normals[0, :] = grid_normals(w[1:3], xw, y[:2])[0, 1:-1]
            if (source[ny + GHOST, :] < 0).all():
                normals[-1, :] = grid_normals(w[-3:-1], xw, y[-2:])[-1, 1:-1]
        else:
            normals = grid_normals(z, grid.x, grid.y)
        nodes_of[k] = nodes
        normals_of[k] = normals
    points, triangles, outlines, normals = [], [], [], []
    data: dict[str, list[np.ndarray]] = {name: [] for name in names}
    levels = []
    offset = 0
    for k, grid in enumerate(grids):
        nx, ny = grid.nx, grid.ny
        hidden = covered(grid, grids, boxes)
        nodes = nodes_of[k]
        if "elevation" in nodes:
            z = nodes["elevation"]
        elif "eta" in nodes:
            z = nodes["eta"]
        else:
            z = np.full((ny + 1, nx + 1), grid.z, np.float32)
        points.append(grid_points(grid.x, grid.y, z))
        normals.append(np.ascontiguousarray(normals_of[k]).reshape(-1, 3))
        width = nx + 1
        jj, ii = np.nonzero(~hidden)
        n00 = (jj * width + ii + offset).astype(np.uint32)
        quads = np.stack([n00, n00 + 1, n00 + width + 1, n00 + width], axis=1)
        triangles.append(np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]))
        for name in names:
            values = nodes[name]
            data[name].append(values.reshape((ny + 1) * (nx + 1), *values.shape[2:]))
        levels.append(np.full((ny + 1) * (nx + 1), grid.level, np.float32))
        if grid.level > 0:
            outlines.append(_outline(nx, ny) + offset)
        offset += (ny + 1) * (nx + 1)
    if not points:
        return SurfaceMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))
    point_data = {
        name: np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
        for name, parts in data.items()
    }
    point_data["level"] = np.concatenate(levels)
    _join_velocity(point_data)
    if "WL" in point_data and "depth" not in point_data and "wetdry" not in point_data:
        # the water depth hides dry land (NHFLOW); SFLOW says which cells are wet
        # (wetdry), its WL is no depth to go by on dry land
        point_data["depth"] = point_data["WL"]
    return SurfaceMesh(
        points=np.concatenate(points),
        triangles=np.ascontiguousarray(np.concatenate(triangles), dtype=np.uint32),
        point_data=point_data,
        outlines=np.concatenate(outlines).astype(np.uint32) if outlines else None,
        normals=np.concatenate(normals),
    )


def _cell_values(grid: AmrGrid, name: str) -> np.ndarray:
    """A grid's cell values of a variable; ``velocity`` made of u, v, w if need be."""
    if name in grid.cell_data:
        return grid.cell_data[name]
    x, y, z = velocity_parts(grid.cell_data)
    u = grid.cell_data[x]
    return np.stack([u, grid.cell_data[y], grid.cell_data.get(z, np.zeros_like(u))], axis=-1)


def _has(grid: AmrGrid, name: str) -> bool:
    if name in grid.cell_data:
        return True
    return name == "velocity" and velocity_parts(grid.cell_data) is not None


def value_at(grids: list[AmrGrid], name: str, x: float, y: float) -> tuple[float, int] | None:
    """A cell value at (x, y) from the finest grid there: (value, level), or None."""
    best = None
    for grid in grids:
        if not grid.contains(x, y) or not _has(grid, name):
            continue
        if best is not None and grid.level <= best.level:
            continue
        best = grid
    if best is None:
        return None
    if best.curvilinear:  # the cell whose centre is nearest, of those it shows
        return _curvilinear_value(grids, name, x, y)
    i = int(np.clip(np.searchsorted(best.x, x, "right") - 1, 0, best.nx - 1))
    j = int(np.clip(np.searchsorted(best.y, y, "right") - 1, 0, best.ny - 1))
    values = _cell_values(best, name)
    value = values[j, i]
    return float(np.linalg.norm(value) if np.ndim(value) else value), best.level


def _curvilinear_value(grids: list[AmrGrid], name: str, x: float, y: float):
    """A cell value at (x, y) of the grids of a curvilinear grid: the leaf cell
    whose centre is nearest (value, level), or None."""
    best = None
    for grid in grids:
        if not grid.contains(x, y) or not _has(grid, name):
            continue
        gx, gy = np.asarray(grid.x, np.float64), np.asarray(grid.y, np.float64)
        cx = 0.25 * (gx[:-1, :-1] + gx[:-1, 1:] + gx[1:, :-1] + gx[1:, 1:])
        cy = 0.25 * (gy[:-1, :-1] + gy[:-1, 1:] + gy[1:, :-1] + gy[1:, 1:])
        distance = (cx - x) ** 2 + (cy - y) ** 2
        flag = grid.cell_data.get("covered")
        if flag is not None:
            distance = np.where(np.asarray(flag).reshape(distance.shape) > 0.5, np.inf, distance)
        j, i = np.unravel_index(int(np.argmin(distance)), distance.shape)
        if not np.isfinite(distance[j, i]):
            continue
        if best is None or distance[j, i] < best[0]:
            value = _cell_values(grid, name)[j, i]
            best = (float(distance[j, i]), value, grid.level)
    if best is None:
        return None
    value = best[1]
    return float(np.linalg.norm(value) if np.ndim(value) else value), best[2]


# ======================================================================== datasets
from lagoonviz.core.data.reef3d import Dataset, Frame, _times_from_files  # noqa: E402


class AmrDataset(Dataset):
    """The AMR surface of a run: one frame per index file (``.vtm``)."""

    def __init__(self, key: str, solver: str, frames: list[Frame], folder: Path):
        super().__init__(key, solver, "amr", frames, folder)
        self._files: dict[int, list[Path]] = {}

    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        arrays = grid_arrays(first.pieces[0])
        self._native_velocity = "velocity" in arrays
        components = with_velocity(arrays)
        components["level"] = 1
        return "RectilinearGrid", components

    @property
    def label(self) -> str:
        return f"{self.solver} free surface AMR"

    @property
    def curvilinear(self) -> bool:
        """Grids of a curvilinear grid (G 50): structured grids (``.vts``)."""
        return self.frames[0].pieces[0].suffix == ".vts"

    def default_variable(self) -> str:
        for name in ("eta", "elevation", "Hs"):  # (Hs: SEASTATE's AMR maps)
            if name in self.components:
                return name
        return next(iter(self.components), "none")

    def grid_files(self, index: int) -> list[Path]:
        """The grid files of a frame (read from its index once)."""
        files = self._files.get(index)
        if files is None:
            files = index_files(self.frames[index].path)
            if len(self._files) > 4096:
                self._files.clear()
            self._files[index] = files
        return files

    def _arrays(self, variables: list[str] | None) -> list[str] | None:
        if variables is None:
            return None
        names = {v.partition(".")[0] for v in variables if v and v != "none"}
        names.update(SURFACE_ARRAYS)
        present = {n for n in self.components if n != "velocity" or self._own_velocity}
        return wanted_arrays(names, present)

    @property
    def _own_velocity(self) -> bool:
        """The output writes a velocity array (not only u, v, w)."""
        native = getattr(self, "_native_velocity", None)
        if native is None:
            native = self._native_velocity = "velocity" in grid_arrays(self.frames[0].pieces[0])
        return native

    def read(
        self, index: int, variables: list[str] | None = None, max_level: int | None = None
    ) -> SurfaceMesh:
        """A frame's surface; with ``max_level`` without the finer patches (their
        files, named by level, are not read)."""
        arrays = self._arrays(variables)
        files = self.grid_files(index)
        if max_level is not None:
            files = [path for path in files if _file_level(path) <= max_level]
        return leaf_mesh([read_grid(path, arrays) for path in files], max_level)

    def read_slice(self, index, axis, position, variables=None) -> SurfaceMesh:
        raise ValueError(f"{self.label}: a surface cannot be sliced")

    def point_series(self, variable: str, x: float, y: float) -> tuple[tuple, np.ndarray]:
        """A variable over all frames at (x, y), from the finest grid there in each."""
        if variable.partition(".")[0] not in self.components or variable == "level":
            raise KeyError(f"no variable {variable!r} in {self.label}")
        present = {n for n in self.components if n != "velocity" or self._own_velocity}
        name = series_name(variable, present)
        arrays = wanted_arrays({name}, present)
        values = np.full(len(self.frames), np.nan)
        for index in range(len(self.frames)):
            grids = []
            for path in self.grid_files(index):
                grid = read_grid(path, arrays)
                if grid.contains(x, y):
                    grids.append(grid)
            found = value_at(grids, name, x, y)
            if found is not None:
                values[index] = found[0]
        return (float(x), float(y), 0.0), values


_LEVEL = re.compile(r"-L(\d+)-")


def _file_level(path: Path) -> int:
    """A grid file's refinement level from its name (``...-AMR-L2-...``); 0 when the
    name does not say (it is then read, and left out by its level inside)."""
    found = _LEVEL.search(path.name)
    return int(found.group(1)) if found else 0


def amr_datasets(folder: Path) -> list[Dataset]:
    """The AMR surface in a ``REEF3D_<SOLVER>_AMR`` folder (none without index files)."""
    match = AMR_FOLDER.match(folder.name)
    if not match or not folder.is_dir():
        return []
    solver = match.group("solver")
    frames = []
    for path in folder.iterdir():
        found = AMR_INDEX.match(path.name)
        if not found or found.group("solver") != solver:
            continue
        step = found.group("step")
        first = folder / f"REEF3D-{solver}-AMR-L0-{step}-0001.vtr"
        if not first.is_file():
            try:
                listed = index_files(path)
            except VTKFormatError:
                continue
            if not listed:
                continue
            first = listed[0]
        frames.append(Frame(int(step), path, [first]))
    if not frames:
        return []
    frames.sort(key=lambda frame: frame.step)
    try:
        dataset = AmrDataset(f"{solver.lower()}_amr", solver, frames, folder)
    except VTKFormatError as error:  # e.g. CFD's 3D blocks: not a surface
        log.warning("%s: not read as an AMR surface: %s", folder, error)
        return []
    _times_from_files(dataset)
    return [dataset]


# ======================================================================== the bed
class AmrBedDataset(Dataset):
    """The bed under an AMR surface on a curvilinear grid, from its grids' cell field
    ``bed``: with blocks (G 50 1) the run's bed output holds block 0 alone, the
    blocks 1 … are only among the AMR grids (and the patches have the finer bed).
    The first output (a fixed bed, written once), or every output when the run's
    bed moves (sediment: more than one bed output)."""

    #: made from another output, not written by the run: not converted into a store
    derived = True

    def __init__(self, amr: Dataset, moving: bool = False):
        self.amr = amr
        frames = list(amr.frames) if moving else list(amr.frames[:1])
        super().__init__(f"{amr.key}_bed", amr.solver, "bed", frames, amr.folder)
        if amr.times is not None:
            self.times = np.asarray(amr.times[: len(frames)], np.float64)
        self.run_id = amr.run_id
        self.run_started = amr.run_started
        self.earlier_run = amr.earlier_run

    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        return "RectilinearGrid", {"elevation": 1, "level": 1}

    @property
    def label(self) -> str:
        return f"{self.amr.label.replace('free surface', 'bed')} (blocks and patches)"

    def written_span(self) -> tuple[float, float]:
        return self.amr.written_span()

    def read(self, index: int, variables: list[str] | None = None, **_options) -> SurfaceMesh:
        """The leaf surface of the grids at the height of the bed, every level."""
        mesh = self.amr.read(index, ["bed"])
        bed = mesh.point_data.get("bed")
        if bed is None:
            raise ValueError(f"{self.amr.label}: no bed in output {self.frames[index].step}")
        points = np.array(mesh.points, np.float32, copy=True)
        points[:, 2] = bed
        data = {"elevation": np.asarray(bed, np.float32)}
        if "level" in mesh.point_data:
            data["level"] = mesh.point_data["level"]
        return SurfaceMesh(points, mesh.triangles, data).with_normals()


def amr_beds(datasets: list[Dataset]) -> list[Dataset]:
    """The beds of the AMR surfaces on a curvilinear grid that write the bed
    (:class:`AmrBedDataset`), shown in place of the run's bed output."""
    beds = []
    for dataset in datasets:
        if dataset.role != "amr" or "bed" not in dataset.components:
            continue
        if not getattr(dataset, "curvilinear", False):
            continue
        written = [d for d in datasets if d.role == "bed" and d.solver == dataset.solver]
        moving = any(len(d) > 1 and d.same_run_as(dataset) for d in written)
        try:
            beds.append(AmrBedDataset(dataset, moving))
        except ValueError as error:
            log.warning("%s: no bed from its grids: %s", dataset.key, error)
    return beds
