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

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
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

#: ``REEF3D_NHFLOW_AMR``
AMR_FOLDER = re.compile(r"^REEF3D_(?P<solver>[A-Z0-9]+)_AMR$")
#: ``REEF3D-NHFLOW-AMR-00000012.vtm``
AMR_INDEX = re.compile(r"^REEF3D-(?P<solver>[A-Z0-9]+)-AMR-(?P<step>\d+)\.vtm$")
#: arrays read with every frame: the height of the surface, its water level, the depth
SURFACE_ARRAYS = ("elevation", "eta", "WL")


@dataclass
class AmrGrid:
    """One grid of an AMR output: level 0 of an MPI rank, or a refined patch."""

    level: int
    x: np.ndarray  # (nx + 1,) node coordinates
    y: np.ndarray  # (ny + 1,)
    z: float = 0.0
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)  # (ny, nx), x fastest
    point_data: dict[str, np.ndarray] = field(default_factory=dict)  # (ny + 1, nx + 1)
    time: float | None = None
    name: str = ""

    @property
    def nx(self) -> int:
        return int(self.x.size) - 1

    @property
    def ny(self) -> int:
        return int(self.y.size) - 1

    @property
    def box(self) -> tuple[float, float, float, float]:
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


def _grid_header(path: Path) -> _GridHeader:
    text, data_start = _read_xml_head(path)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise VTKFormatError(f"{path}: not a VTK XML file ({error})") from None
    if root.tag != "VTKFile" or root.get("type") != "RectilinearGrid":
        raise VTKFormatError(f"{path}: not a VTK rectilinear grid")
    byte_order = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    header_type = root.get("header_type", "UInt32")
    piece = root.find("RectilinearGrid/Piece")
    if piece is None:
        raise VTKFormatError(f"{path}: no Piece")
    extent = tuple(int(v) for v in piece.get("Extent", "0 0 0 0 0 0").split())
    if len(extent) != 6:
        raise VTKFormatError(f"{path}: bad Extent")
    nx, ny, nz = extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4]
    if nz != 0:
        raise VTKFormatError(f"{path}: a 3D grid; AMR surfaces are 2D")
    header = PieceHeader(
        path=path,
        kind="RectilinearGrid",
        n_points=(nx + 1) * (ny + 1),
        n_cells=nx * ny,
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
    level, time = 0, None
    # (SEASTATE writes its FieldData in the Piece, with the time as TIME)
    for element in [
        *root.findall("RectilinearGrid/FieldData/DataArray"),
        *root.findall("RectilinearGrid/Piece/FieldData/DataArray"),
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
    for axis in "xyz":
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


def leaf_mesh(grids: list[AmrGrid], max_level: int | None = None) -> SurfaceMesh:
    """The surface of the leaf cells of all grids: each grid on its node lattice,
    without the cells a finer grid covers (grids finer than ``max_level`` left out:
    0 gives the base grid alone). The height is ``elevation`` (else
    ``eta``); point data are the cell data averaged to the nodes, the grid's
    ``level``, and the water ``depth`` from ``WL`` (to hide dry land). The
    borders of the refined patches are the mesh's ``outlines``."""
    names = (
        set.intersection(*(set(g.cell_data) | set(g.point_data) for g in grids)) if grids else set()
    )
    if max_level is not None:
        grids = [g for g in grids if g.level <= max_level]
    points, triangles, outlines, normals = [], [], [], []
    data: dict[str, list[np.ndarray]] = {name: [] for name in names}
    levels = []
    offset = 0
    boxes = Boxes(grids)  # the finer grids over each found at once (thousands of patches)
    for grid in grids:
        nx, ny = grid.nx, grid.ny
        if nx < 1 or ny < 1:
            continue
        hidden = covered(grid, grids, boxes)
        nodes = {}
        for name in names:
            if name in grid.point_data:
                nodes[name] = grid.point_data[name].astype(np.float32, copy=False)
            else:
                nodes[name] = _nodes(grid.cell_data[name])
        if "elevation" in nodes:
            z = nodes["elevation"]
        elif "eta" in nodes:
            z = nodes["eta"]
        else:
            z = np.full((ny + 1, nx + 1), grid.z, np.float32)
        points.append(grid_points(grid.x, grid.y, z))
        # each grid a height field: normals from its slopes (from all triangles at
        # once they took seconds)
        normals.append(grid_normals(z, grid.x, grid.y).reshape(-1, 3))
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
    if "WL" in point_data and "depth" not in point_data:
        point_data["depth"] = point_data["WL"]
    return SurfaceMesh(
        points=np.concatenate(points),
        triangles=np.ascontiguousarray(np.concatenate(triangles), dtype=np.uint32),
        point_data=point_data,
        outlines=np.concatenate(outlines).astype(np.uint32) if outlines else None,
        normals=np.concatenate(normals),
    )


def value_at(grids: list[AmrGrid], name: str, x: float, y: float) -> tuple[float, int] | None:
    """A cell value at (x, y) from the finest grid there: (value, level), or None."""
    best = None
    for grid in grids:
        if not grid.contains(x, y) or name not in grid.cell_data:
            continue
        if best is not None and grid.level <= best.level:
            continue
        best = grid
    if best is None:
        return None
    i = int(np.clip(np.searchsorted(best.x, x, "right") - 1, 0, best.nx - 1))
    j = int(np.clip(np.searchsorted(best.y, y, "right") - 1, 0, best.ny - 1))
    values = best.cell_data[name]
    value = values[j, i]
    return float(np.linalg.norm(value) if np.ndim(value) else value), best.level


# ======================================================================== datasets
from lagoonviz.core.data.reef3d import Dataset, Frame, _times_from_files  # noqa: E402


class AmrDataset(Dataset):
    """The AMR surface of a run: one frame per index file (``.vtm``)."""

    def __init__(self, key: str, solver: str, frames: list[Frame], folder: Path):
        super().__init__(key, solver, "amr", frames, folder)
        self._files: dict[int, list[Path]] = {}

    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        components = grid_arrays(first.pieces[0])
        components["level"] = 1
        return "RectilinearGrid", components

    @property
    def label(self) -> str:
        return f"{self.solver} AMR surface"

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
        return sorted(n for n in names if n in self.components and n != "level")

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
        name = variable.partition(".")[0]
        if name not in self.components or name == "level":
            raise KeyError(f"no variable {name!r} in {self.label}")
        values = np.full(len(self.frames), np.nan)
        for index in range(len(self.frames)):
            grids = []
            for path in self.grid_files(index):
                grid = read_grid(path, [name])
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
    dataset = AmrDataset(f"{solver.lower()}_amr", solver, frames, folder)
    _times_from_files(dataset)
    return [dataset]
