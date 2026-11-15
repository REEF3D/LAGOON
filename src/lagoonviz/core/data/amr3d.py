# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""CFD's adaptive mesh refinement: the refined blocks of a run's volume.

REEF3D::CFD with AMR writes, next to its volume (VTU, or the LAGOON store), the
refined blocks of every output in ``REEF3D_CFD_AMR``: an index per output
(``REEF3D-CFD-AMR-<output>.vtm``) listing its blocks, 3D rectilinear grids
(``.vtr``) with cell data: ``phi``, ``pressure``, ``velocity``, the block's ``level``
(1: half the volume's cell size, 2: a quarter, ...) and ``covered`` (1 where a finer
block lies over the cell). Only the index says which blocks belong to an output: a
block file left over from an earlier run is not read. With ``P 18`` the blocks go
into the LAGOON store too, as an AMR set of 3D grids (``amr/cfd_amr``, ``dims`` 3):
:class:`FolderBlocks` and :class:`StoreBlocks` hand them out alike.

:class:`RefinedVolume` is the volume with its blocks: the volume's cells under a
block are left out, each block's cells not covered by a finer one are added, their
cell values averaged to the block's nodes. Every tool of a volume works on it as
on any other (the outer surface, slices, iso-surfaces, iso-volumes, colouring by
``level``): the free surface (phi at 0) is as fine as the refinement around a
floating body. :class:`AmrBlocks` draws the blocks' outlines, coloured by level.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from lagoonviz.core.data.amr import AMR_FOLDER, AMR_INDEX, _grid_header, index_files
from lagoonviz.core.data.mesh import SurfaceMesh, VolumePlan
from lagoonviz.core.data.reef3d import Dataset, Frame
from lagoonviz.core.data.vtkxml import Piece, VTKFormatError, read_array

#: arrays of a block that are not the flow's: its level and whether a finer block covers it
BLOCK_ARRAYS = ("level", "covered")


# ======================================================================== blocks
@dataclass(frozen=True)
class Block:
    """One refined block: its node coordinates and cell arrays (nz, ny, nx[, c])."""

    path: Path
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    cells: dict[str, np.ndarray]

    @property
    def box(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.x[0]),
            float(self.x[-1]),
            float(self.y[0]),
            float(self.y[-1]),
            float(self.z[0]),
            float(self.z[-1]),
        )

    @cached_property
    def level(self) -> int:
        # (once: asked for many times per block and frame, each time over its cells)
        level = self.cells.get("level")
        return int(round(float(np.nanmax(level)))) if level is not None and level.size else 0


def block_arrays(path: str | Path) -> dict[str, int]:
    """The cell arrays of a block file: name -> components."""
    head = _grid_header(Path(path), volume=True).header
    return {name: info.components for name, info in head.cell_data.items()}


def read_block(path: str | Path, arrays: list[str] | None = None) -> Block:
    """One block file; ``arrays=None`` reads every cell array (the level and
    whether a cell is covered always)."""
    path = Path(path)
    grid = _grid_header(path, volume=True)
    head = grid.header
    e = grid.extent
    nx, ny, nz = e[1] - e[0], e[3] - e[2], e[5] - e[4]
    axes = {}
    for axis, count in zip("xyz", (nx + 1, ny + 1, nz + 1), strict=True):
        info = grid.coordinates.get(axis)
        if info is None:
            raise VTKFormatError(f"{path}: no {axis} coordinates")
        axes[axis] = read_array(head, info, count).astype(np.float64)
    wanted = None if arrays is None else {*arrays, *BLOCK_ARRAYS}
    cells = {}
    for name, info in head.cell_data.items():
        if wanted is not None and name not in wanted:
            continue
        values = read_array(head, info, nx * ny * nz)
        shape = (nz, ny, nx, info.components) if info.components > 1 else (nz, ny, nx)
        cells[name] = values.reshape(shape)
    return Block(path, axes["x"], axes["y"], axes["z"], cells)


def block_box(path: str | Path) -> tuple[float, float, float, float, float, float]:
    """Where a block lies (x0, x1, y0, y1, z0, z1), from its coordinates alone."""
    path = Path(path)
    grid = _grid_header(path, volume=True)
    e = grid.extent
    counts = (e[1] - e[0] + 1, e[3] - e[2] + 1, e[5] - e[4] + 1)
    box = []
    for axis, count in zip("xyz", counts, strict=True):
        values = read_array(grid.header, grid.coordinates[axis], count)
        box += [float(values[0]), float(values[-1])]
    return tuple(box)  # type: ignore[return-value]


def _nodes(cells: np.ndarray) -> np.ndarray:
    """Cell values (nz, ny, nx[, c]) at the nodes (nz+1, ny+1, nx+1[, c]): each node
    the mean of the eight cells around it, those beyond the block's faces carried on
    linearly from the two inside (so a level set stays a distance up to its faces)."""
    cells = np.asarray(cells, np.float32)
    padded = cells
    for axis in range(3):
        n = padded.shape[axis]
        first = np.take(padded, [0], axis=axis)
        last = np.take(padded, [n - 1], axis=axis)
        if n > 1:
            before = 2.0 * first - np.take(padded, [1], axis=axis)
            after = 2.0 * last - np.take(padded, [n - 2], axis=axis)
        else:
            before, after = first, last
        padded = np.concatenate([before, padded, after], axis=axis)
    nz, ny, nx = cells.shape[:3]
    out = np.zeros((nz + 1, ny + 1, nx + 1, *cells.shape[3:]), np.float32)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                out += padded[dz : dz + nz + 1, dy : dy + ny + 1, dx : dx + nx + 1]
    return out / 8.0


#: the cells of blocks by their size and which of them a finer block covers:
#: (connectivity, offsets, types), read-only. A run has blocks of a few sizes, the
#: same from output to output: they were made again for every block of every frame,
#: twice (once to cut it, once for its points).
_BLOCK_CELLS: dict[tuple, tuple] = {}
_BLOCK_CELLS_LOCK = threading.Lock()
BLOCK_CELLS_BYTES = 256 << 20


def block_cells(nz: int, ny: int, nx: int, covered: np.ndarray | None = None) -> tuple:
    """The cells of a block of nz x ny x nx cells no finer block covers, as a
    piece's (connectivity, offsets, types); kept (see ``_BLOCK_CELLS``)."""
    hidden = None
    if covered is not None:
        hidden = np.asarray(covered).ravel() >= 0.5
        if not hidden.any():
            hidden = None
    key = (nz, ny, nx, None if hidden is None else hash(np.packbits(hidden).tobytes()))
    with _BLOCK_CELLS_LOCK:
        found = _BLOCK_CELLS.get(key)
    if found is not None:
        return found[:3]
    cells = _hexahedra(nz, ny, nx)
    if hidden is not None:
        cells = cells[~hidden]
    connectivity = np.ascontiguousarray(cells, dtype=np.int64).reshape(-1)
    offsets = (np.arange(1, len(cells) + 1) * 8).astype(np.int64)
    types = np.full(len(cells), 12, np.uint8)  # VTK_HEXAHEDRON
    for array in (connectivity, offsets, types):
        array.flags.writeable = False
    size = connectivity.nbytes + offsets.nbytes + types.nbytes
    with _BLOCK_CELLS_LOCK:
        total = sum(v[3] for v in _BLOCK_CELLS.values())
        while _BLOCK_CELLS and total + size > BLOCK_CELLS_BYTES:
            total -= _BLOCK_CELLS.pop(next(iter(_BLOCK_CELLS)))[3]
        _BLOCK_CELLS[key] = (connectivity, offsets, types, size)
    return connectivity, offsets, types


def _hexahedra(nz: int, ny: int, nx: int) -> np.ndarray:
    """The cells (nz * ny * nx, 8) of a block, its nodes ordered z, y, x (x fastest),
    in VTK's cell order (x fastest)."""
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    px, py = nx + 1, ny + 1
    base = ((k * py + j) * px + i).ravel()
    up = py * px
    return np.stack(
        [
            base,
            base + 1,
            base + px + 1,
            base + px,
            base + up,
            base + up + 1,
            base + up + px + 1,
            base + up + px,
        ],
        axis=1,
    )


def block_piece(
    block: Block,
    names: list[str],
    with_cells: bool = True,
    nodes_of: dict[str, np.ndarray] | None = None,
) -> Piece:
    """A block as the piece of a volume: its nodes, the arrays ``names`` at them
    (cell values averaged to the nodes, or those of ``nodes_of``: see
    :func:`seamless_nodes`; ``elevation`` their height; one the block lacks: NaN)
    and, ``with_cells``, its cells no finer block covers."""
    nz, ny, nx = len(block.z) - 1, len(block.y) - 1, len(block.x) - 1
    z, y, x = np.meshgrid(block.z, block.y, block.x, indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
    data = {}
    for name in names:
        values = block.cells.get(name)
        if values is not None:
            given = nodes_of.get(name) if nodes_of else None
            nodes = given if given is not None else _nodes(values)
            data[name] = nodes.reshape(len(points), -1) if nodes.ndim == 4 else nodes.ravel()
        elif name == "elevation":
            data[name] = points[:, 2].copy()
        elif name == "level":
            data[name] = np.full(len(points), float(block.level), np.float32)
        else:
            data[name] = np.full(len(points), np.nan, np.float32)
    if not with_cells:
        empty = np.zeros(0, np.int64)
        return Piece("UnstructuredGrid", points, empty, empty, np.zeros(0, np.uint8), data)
    connectivity, offsets, types = block_cells(nz, ny, nx, block.cells.get("covered"))
    return Piece("UnstructuredGrid", points, connectivity, offsets, types, data)


# ======================================================================== seams
#: the eight cells round a node, as offsets (dz, dy, dx)
_EIGHT = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]


def _simplex_plan(x, y, z, px, py, pz) -> tuple:
    """Where points lie in a node lattice (len(z), len(y), len(x)) for
    :func:`_simplex`: the four corners of each point's tetrahedron (k, j, i) and
    their weights. The same for every array on the lattice: found once."""
    index, fraction = [], []
    for axis, p in zip((x, y, z), (px, py, pz), strict=True):
        axis = np.asarray(axis, np.float64)
        i = np.minimum(np.maximum(np.searchsorted(axis, p, "right") - 1, 0), len(axis) - 2)
        f = np.minimum(np.maximum((p - axis[i]) / (axis[i + 1] - axis[i]), 0.0), 1.0)
        index.append(i)
        fraction.append(f)
    f = np.stack(fraction, axis=1)
    order = np.argsort(-f, axis=1, kind="stable")  # the axes from the largest step
    fs = np.take_along_axis(f, order, axis=1)
    weights = (1.0 - fs[:, 0], fs[:, 0] - fs[:, 1], fs[:, 1] - fs[:, 2], fs[:, 2])
    corner = np.zeros((len(f), 3), np.int64)
    rows = np.arange(len(f))
    corners = []
    for step in range(4):
        if step:
            corner[rows, order[:, step - 1]] = 1
        corners.append((index[2] + corner[:, 2], index[1] + corner[:, 1], index[0] + corner[:, 0]))
    return corners, weights


def _simplex_values(plan: tuple, values: np.ndarray) -> np.ndarray:
    """The values of a lattice (nz, ny, nx[, c]) at the points of a plan."""
    out = None
    for (k, j, i), w in zip(*plan, strict=True):
        v = values[k, j, i]
        part = (w[:, None] if v.ndim == 2 else w) * v.astype(np.float64)
        out = part if out is None else out + part
    return out.astype(np.float32)


def _simplex(x, y, z, values: np.ndarray, px, py, pz) -> np.ndarray:
    """Values on a node lattice (len(z), len(y), len(x)[, c]) at points, linear in
    each of the six tetrahedra a cell is cut into (as :func:`plan_level` cuts a
    hexahedron: round its diagonal from the lowest to the highest corner). A finer
    block's face takes so exactly the values the coarser cells' cut has there, and
    the two meet without a crack."""
    return _simplex_values(_simplex_plan(x, y, z, px, py, pz), values)


#: where a run's volume nodes lie on its lattice, by the pieces' points (see
#: :func:`volume_lattice`)
_LATTICES: dict[tuple, tuple] = {}
_LATTICE_LOCK = threading.Lock()


def volume_lattice(pieces: list[Piece], names: list[str]):
    """The volume's nodes as a lattice: (x, y, z, {name: (nz, ny, nx[, c])}), NaN
    where no piece has a node, or the piece no values of that name; None if its
    points are no lattice."""
    pieces = [p for p in pieces if len(p.points)]
    if not pieces:
        return None
    # where the nodes lie on the lattice: the same in every output of a run (found
    # once: rounding, sorting and searching the points took a sixth of an iso-surface)
    key = tuple((len(p.points), hash(np.ascontiguousarray(p.points).tobytes())) for p in pieces)
    with _LATTICE_LOCK:
        found = _LATTICES.get(key)
    if found is None:
        points = np.concatenate([p.points for p in pieces]).astype(np.float64)
        axes = [np.unique(np.round(points[:, d], 5)) for d in range(3)]
        if (
            np.prod([a.size for a in axes], dtype=np.float64) > 8.0 * len(points)
            or min(a.size for a in axes) < 2
        ):
            found = (None, None)
        else:
            ijk = tuple(np.searchsorted(axes[d], np.round(points[:, d], 5)) for d in range(3))
            found = (axes, ijk)
        with _LATTICE_LOCK:
            if len(_LATTICES) > 8:
                _LATTICES.clear()
            _LATTICES[key] = found
    axes, ijk = found
    if axes is None:
        return None
    i, j, k = ijk
    starts = np.cumsum([0] + [len(p.points) for p in pieces])
    arrays = {}
    for name in names:
        parts = [p.point_data.get(name) for p in pieces]
        if all(v is None for v in parts):
            continue
        lattice = None
        for n, values in enumerate(parts):  # a piece without it (far from every
            if values is None:  # block, not read): NaN
                continue
            values = np.asarray(values, np.float32)
            if lattice is None:
                shape = (axes[2].size, axes[1].size, axes[0].size, *values.shape[1:])
                lattice = np.full(shape, np.nan, np.float32)
            at = slice(starts[n], starts[n + 1])
            lattice[k[at], j[at], i[at]] = values
        arrays[name] = lattice
    return axes[0], axes[1], axes[2], arrays


def _centres(edges: np.ndarray) -> np.ndarray:
    """Cell centres of a lattice and of one cell beyond each end."""
    e = np.asarray(edges, np.float64)
    c = 0.5 * (e[:-1] + e[1:])
    return np.concatenate([[c[0] - (e[1] - e[0])], c, [c[-1] + (e[-1] - e[-2])]])


def _padded(cells: np.ndarray) -> np.ndarray:
    """Cells (nz, ny, nx[, c]) with a layer beyond each face, carried on linearly."""
    cells = np.asarray(cells, np.float32)
    nz, ny, nx = cells.shape[:3]
    padded = np.empty((nz + 2, ny + 2, nx + 2, *cells.shape[3:]), np.float32)
    padded[1:-1, 1:-1, 1:-1] = cells
    # axis by axis, each over what the axes before it added (one array filled in
    # place: taking and joining slices was a tenth of the time of the blocks' seams)
    for axis, n in enumerate((nz, ny, nx)):
        inner = tuple(slice(None) if d < axis else slice(1, -1) for d in range(axis + 1, 3))
        before = (slice(None),) * axis

        def at(i, before=before, inner=inner):
            return (*before, i, *inner)

        if n > 1:
            padded[at(0)] = 2.0 * padded[at(1)] - padded[at(2)]
            padded[at(n + 1)] = 2.0 * padded[at(n)] - padded[at(n - 1)]
        else:
            padded[at(0)] = padded[at(1)]
            padded[at(n + 1)] = padded[at(n)]
    return padded


def _cell_of(edges: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The cell of a lattice (its ``edges``) each of ``at`` lies in, the first or
    the last for those beyond."""
    cell = np.searchsorted(edges, at, "right") - 1
    return np.minimum(np.maximum(cell, 0), len(edges) - 2)


def seamless_nodes(blocks: list[Block], names: list[str], lattice=None) -> dict[Path, dict]:
    """The blocks' cell values at their nodes, the same on both sides of every face:

    - a node between blocks of one level (two MPI ranks) the mean of the eight
      cells round it, of both blocks;
    - a node on a face to something coarser (a coarser block, else the volume,
      ``lattice``: :func:`volume_lattice`) the coarser values there, as the coarser
      cells' cut has them (:func:`_simplex`);
    - inside, the mean of the eight cells round it.

    Each block's faces were carried on from its own cells before: a block and its
    neighbours disagreed at their faces, and the free surface had a crack there
    and contour lines a gap. Coarse blocks first: a finer block takes their final
    values. Returns block path -> {name: (nz + 1, ny + 1, nx + 1[, c])}."""
    order = sorted(range(len(blocks)), key=lambda n: blocks[n].level)
    boxes = [b.box for b in blocks]
    levels = np.array([b.level for b in blocks], np.int64)
    lows = np.array([b[0::2] for b in boxes], np.float64).reshape(-1, 3)
    highs = np.array([b[1::2] for b in boxes], np.float64).reshape(-1, 3)
    done: dict[int, dict[str, np.ndarray]] = {}
    for n in order:
        block = blocks[n]
        level = block.level
        reach = np.array(
            [float(max(a[1] - a[0], a[-1] - a[-2])) for a in (block.x, block.y, block.z)]
        )
        # the blocks that reach into this one widened by a cell (all at once: asking
        # block by block made a thousand blocks a million questions per frame)
        touching = ((highs >= lows[n] - reach) & (lows <= highs[n] + reach)).all(axis=1)
        touching[n] = False
        near = np.flatnonzero(touching)
        same = [int(m) for m in near[levels[near] == level]]
        coarser = [int(m) for m in near[levels[near] < level]]
        coarser.sort(key=lambda m: -levels[m])
        own = [name for name in names if name in block.cells]
        nz, ny, nx = (len(a) - 1 for a in (block.z, block.y, block.x))
        xc, yc, zc = _centres(block.x), _centres(block.y), _centres(block.z)
        alone = np.ones((nz + 2, ny + 2, nx + 2), bool)  # no cell of this level there
        alone[1:-1, 1:-1, 1:-1] = False
        padded = {name: _padded(block.cells[name]) for name in own}
        for m in same:
            other = blocks[m]
            ox, oy, oz = (np.asarray(a, np.float64) for a in (other.x, other.y, other.z))
            cols = (xc > ox[0]) & (xc < ox[-1])
            rows = (yc > oy[0]) & (yc < oy[-1])
            layers = (zc > oz[0]) & (zc < oz[-1])
            take = alone & layers[:, None, None] & rows[None, :, None] & cols[None, None, :]
            if not take.any():
                continue
            kk, jj, ii = np.nonzero(take)
            # (the other block's cell along each axis first, then for each cell)
            oi = _cell_of(ox, xc)[ii]
            oj = _cell_of(oy, yc)[jj]
            ok = _cell_of(oz, zc)[kk]
            for name in own:
                if name in other.cells:
                    padded[name][kk, jj, ii] = np.asarray(other.cells[name])[ok, oj, oi]
            alone[kk, jj, ii] = False
        nodes = {}
        for name, values in padded.items():
            total = np.zeros((nz + 1, ny + 1, nx + 1, *values.shape[3:]), np.float32)
            for dz, dy, dx in _EIGHT:
                total += values[dz : dz + nz + 1, dy : dy + ny + 1, dx : dx + nx + 1]
            nodes[name] = total / 8.0
        border = np.zeros((nz + 1, ny + 1, nx + 1), bool)
        for dz, dy, dx in _EIGHT:
            border |= alone[dz : dz + nz + 1, dy : dy + ny + 1, dx : dx + nx + 1]
        if own and border.any():
            kk, jj, ii = np.nonzero(border)
            px, py, pz = block.x[ii], block.y[jj], block.z[kk]
            open_ = np.ones(len(kk), bool)
            for m in coarser:  # the finest coarser block round each node
                x0, x1, y0, y1, z0, z1 = boxes[m]
                eps = 1e-6 * max(1.0, abs(x1), abs(y1), abs(z1))
                at = open_ & (px >= x0 - eps) & (px <= x1 + eps) & (py >= y0 - eps)
                at &= (py <= y1 + eps) & (pz >= z0 - eps) & (pz <= z1 + eps)
                if not at.any():
                    continue
                other = blocks[m]
                # (where the nodes lie in the coarser block: once for all arrays)
                plan = _simplex_plan(other.x, other.y, other.z, px[at], py[at], pz[at])
                ka, ja, ia = kk[at], jj[at], ii[at]
                for name in own:
                    if name in done[m]:
                        nodes[name][ka, ja, ia] = _simplex_values(plan, done[m][name])
                open_ &= ~at
            if lattice is not None and open_.any():
                lx, ly, lz, arrays = lattice
                plan = None
                for name in own:
                    if name not in arrays:
                        continue
                    if plan is None:
                        plan = _simplex_plan(lx, ly, lz, px[open_], py[open_], pz[open_])
                    found = _simplex_values(plan, arrays[name])
                    good = np.isfinite(found).reshape(len(found), -1).all(axis=1)
                    sel = np.flatnonzero(open_)[good]
                    nodes[name][kk[sel], jj[sel], ii[sel]] = found[good]
        done[n] = nodes
    return {blocks[n].path: done[n] for n in range(len(blocks))}


# ======================================================================== outputs
def is_volume_amr(folder: Path) -> bool:
    """Whether an AMR folder holds the blocks of a volume (CFD), not the patches of
    a surface (SFLOW, NHFLOW): its first index's first file is a 3D grid."""
    if not AMR_FOLDER.match(folder.name) or not folder.is_dir():
        return False
    for path in sorted(folder.iterdir()):
        if not AMR_INDEX.match(path.name):
            continue
        try:
            files = index_files(path)
            if not files:
                continue
            _grid_header(files[0], volume=True)
        except (VTKFormatError, OSError):
            return False
        return True
    return False


def amr_outputs(folder: Path) -> dict[int, Path]:
    """The index of every output in a CFD AMR folder: output number -> ``.vtm``."""
    found = {}
    for path in folder.iterdir():
        match = AMR_INDEX.match(path.name)
        if match:
            found[int(match.group("step"))] = path
    return dict(sorted(found.items()))


# ======================================================================== sources
class FolderBlocks:
    """The blocks of a ``REEF3D_CFD_AMR`` folder: each output's index and files."""

    def __init__(self, outputs: dict[int, Path]):
        self.outputs = outputs
        self._keys: dict[int, list[Path]] = {}
        self._boxes: dict[Path, tuple] = {}
        self._cache: dict[Path, Block] = {}
        self._lock = threading.Lock()

    @property
    def steps(self) -> list[int]:
        return list(self.outputs)

    def keys(self, step: int) -> list[Path]:
        """The block files of an output (from its index; none without one)."""
        keys = self._keys.get(step)
        if keys is None:
            vtm = self.outputs.get(step)
            try:
                keys = [p for p in index_files(vtm) if p.is_file()] if vtm is not None else []
            except VTKFormatError:
                keys = []
            self._keys[step] = keys
        return keys

    def box(self, key: Path) -> tuple:
        box = self._boxes.get(key)
        if box is None:
            box = self._boxes[key] = block_box(key)
        return box

    def block(self, key: Path) -> Block:
        """A block with all its arrays (small: read once, kept a while)."""
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        block = read_block(key, None)
        with self._lock:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = block
        return block

    def level(self, key: Path) -> int:
        return self.block(key).level

    def stamp(self, step: int) -> Path:
        """A file whose modification time says when the output was written."""
        return self.outputs[step]


class StoreBlocks:
    """The blocks of a LAGOON store's AMR set of 3D grids (``dims`` 3)."""

    def __init__(self, amr):
        self.amr = amr
        self._steps = [int(s) for s in amr.steps]
        try:
            times = [float(t) for t in amr.times]
        except (KeyError, ValueError, TypeError):
            times = []
        #: output number -> simulation time, as stored
        self.times = dict(zip(self._steps, times, strict=False)) if times else {}
        self._index = {step: t for t, step in enumerate(self._steps)}
        self._grids: dict[int, list] = {}
        #: each output's block levels (its grid table) and boxes (its coordinates):
        #: what opening a run and the volume's cells need, without the blocks' fields
        self._levels: dict[int, np.ndarray] = {}
        self._boxes: dict[int, list[tuple]] = {}
        self._lock = threading.Lock()

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

    def _levels_of(self, t: int) -> np.ndarray:
        with self._lock:
            found = self._levels.get(t)
        if found is None:
            found = np.asarray(self.amr.table(t)["level"], np.int64)
            with self._lock:
                self._levels[t] = found
        return found

    def _boxes_of(self, t: int) -> list[tuple]:
        with self._lock:
            found = self._boxes.get(t)
        if found is None:
            found = [
                (float(g.x[0]), float(g.x[-1]), float(g.y[0]), float(g.y[-1]),
                 float(g.z[0]), float(g.z[-1]))
                for g in self.amr.grids(t, [])
            ]  # fmt: skip
            with self._lock:
                if len(self._boxes) > 4096:
                    self._boxes.clear()
                self._boxes[t] = found
        return found

    def _of(self, t: int) -> list[Block]:
        with self._lock:
            found = self._grids.get(t)
        if found is not None:
            return found
        found = []
        for g, grid in enumerate(self.amr.grids(t)):
            cells = dict(grid.fields)
            if "velocity" not in cells and all(c in cells for c in ("u", "v", "w")):
                cells["velocity"] = np.stack([cells.pop(c) for c in ("u", "v", "w")], axis=-1)
            if "level" not in cells:
                cells["level"] = np.full((grid.nz, grid.ny, grid.nx), grid.level, np.float32)
            key = self.amr.group.path / f"t{t}" / f"g{g}.block"
            found.append(Block(key, grid.x, grid.y, grid.z, cells))
        with self._lock:
            if len(self._grids) > 4:
                self._grids.clear()
            self._grids[t] = found
        return found

    def keys(self, step: int) -> list[Path]:
        """The blocks of an output, from its grid table alone (opening a run read
        every block of every output, its fields too: 40 s for a CFD run of 140
        outputs)."""
        t = self._index.get(step)
        if t is None:
            return []
        path = self.amr.group.path / f"t{t}"
        return [path / f"g{g}.block" for g in range(len(self._levels_of(t)))]

    def _find(self, key: Path) -> Block:
        t = int(key.parent.name[1:])
        return self._of(t)[int(key.stem[1:])]

    def box(self, key: Path) -> tuple:
        return self._boxes_of(int(key.parent.name[1:]))[int(key.stem[1:])]

    def block(self, key: Path) -> Block:
        return self._find(key)

    def level(self, key: Path) -> int:
        return int(self._levels_of(int(key.parent.name[1:]))[int(key.stem[1:])])

    def stamp(self, step: int) -> Path:
        return self.amr.group.path / "time" / "zarr.json"


# ======================================================================== datasets
class RefinedVolume(Dataset):
    """A volume with the refined blocks of each output (module doc). It stands in
    for the volume (the same key): its pieces are the volume's, then the blocks."""

    def __init__(self, base: Dataset, blocks):
        self.base = base
        self.blocks = blocks
        self._frame_of: dict[Path, int] = {}
        self._block_keys: set[Path] = set()
        self._block_frame: dict[Path, int] = {}
        self._frame_blocks: list[list[Path]] = []
        self._plans: dict[tuple, VolumePlan] = {}
        self._own_cells: dict[tuple, tuple] = {}
        self._seams: dict[tuple, dict] = {}
        self._seam_arrays: dict[tuple, dict] = {}  # (frame, array) -> block -> nodes
        self._layouts: dict[int, int] = {}  # frame -> its layout (see _layout)
        self._layout_of: dict[tuple, int] = {}
        self._amr_lock = threading.Lock()
        frames = []
        for index, frame in enumerate(base.frames):
            for path in frame.pieces:
                self._frame_of[path] = index
            # REEF3D also writes each rank's level 0 round its blocks: the volume's
            # own cells (overlapping one another), drawn by the volume already
            keys = [k for k in blocks.keys(frame.step) if blocks.level(k) > 0]
            self._block_keys.update(keys)
            self._block_frame.update(dict.fromkeys(keys, index))
            self._frame_blocks.append(keys)
            frames.append(Frame(frame.step, frame.path, [*frame.pieces, *keys]))
        super().__init__(base.key, base.solver, base.role, frames, base.folder)
        self.run_id = base.run_id
        self.run_started = base.run_started
        self.times = base.times
        self.earlier_run = base.earlier_run
        self.source = getattr(base, "source", None)

    # ------------------------------------------------------------- description
    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        return self.base.kind, {**self.base.components, "level": 1}

    @property
    def label(self) -> str:
        return self.base.label + " with AMR"

    def written_span(self) -> tuple[float, float]:
        return self.base.written_span()

    def amr_blocks(self, index: int) -> list[Path]:
        """The refined blocks of a frame."""
        return self._frame_blocks[index]

    def _layout(self, index: int) -> int:
        """Which blocks a frame has, as a number: the same for frames whose blocks
        have the same levels and lie in the same places, one after the other (a
        refinement that stays, CFD's G 15 box: every output). What is planned for
        the cells of one such frame (the outer surface, the cuts of a slice) holds
        for the others."""
        with self._amr_lock:
            known = self._layouts.get(index)
        if known is not None:
            return known
        where = tuple(
            (self.blocks.level(key), *(round(v, 9) for v in self.blocks.box(key)))
            for key in self.amr_blocks(index)
        )
        with self._amr_lock:
            return self._layouts.setdefault(
                index, self._layout_of.setdefault(where, len(self._layout_of))
            )

    def _seamless(self, index: int, names: list[str]) -> dict[Path, dict]:
        """The node values of a frame's blocks, alike on both sides of their faces
        (:func:`seamless_nodes`), kept for the frames being looked at. Each array is
        worked out once per frame, whichever items ask for it (the free surface, a
        slice and the volume ask for different ones: each made all of its own)."""
        names = sorted(n for n in names if n not in ("level", "elevation"))
        key = (index, tuple(names))
        with self._amr_lock:
            found = self._seams.get(key)
            missing = [n for n in names if (index, n) not in self._seam_arrays]
        if found is not None:
            return found
        made = {}
        if missing:
            keys = self.amr_blocks(index)
            blocks = [self.blocks.block(k) for k in keys]
            wanted = [n for n in missing if n in self.base.components]
            pieces = [self._near_piece(p, blocks, wanted) for p in self.base.frames[index].pieces]
            nodes = seamless_nodes(blocks, missing, volume_lattice(pieces, wanted))
            made = {
                name: {path: of[name] for path, of in nodes.items() if name in of}
                for name in missing
            }
        with self._amr_lock:
            for old in [k for k in self._seams if k[0] < index - 2 or k[0] > index + 2]:
                del self._seams[old]
            for old in [k for k in self._seam_arrays if k[0] < index - 2 or k[0] > index + 2]:
                del self._seam_arrays[old]
            if len(self._seams) > 16:
                self._seams.clear()
            for name, of in made.items():
                self._seam_arrays.setdefault((index, name), of)
            arrays = {name: self._seam_arrays.get((index, name), {}) for name in names}
            found = {}
            if names:
                for path in self.amr_blocks(index):
                    found[path] = {n: of[path] for n, of in arrays.items() if path in of}
            self._seams[key] = found
        return found

    def _near_piece(self, path: Path, blocks: list[Block], names: list[str]) -> Piece:
        """A piece of the volume for the lattice the blocks' faces take their values
        from: its points, and ``names`` only if it reaches within two of its cells of
        a block (the faces look into the coarse cells round them; the rest of the
        volume, most of it, was read in vain)."""
        light = self.base._piece(path, point_arrays=[], with_cells=False)
        if not len(light.points) or not names:
            return light
        key = ("near", hash(np.ascontiguousarray(light.points).tobytes()))
        with self._amr_lock:
            box = self._own_cells.get(key)
        if box is None:
            points = light.points.astype(np.float64)
            low, high = points.min(axis=0), points.max(axis=0)
            reach = np.zeros(3)
            for d in range(3):
                axis = np.unique(np.round(points[:, d], 5))
                reach[d] = 2.0 * float(np.max(np.diff(axis))) if len(axis) > 1 else 0.0
            box = (low - reach, high + reach)
            with self._amr_lock:
                if len(self._own_cells) > 256:
                    self._own_cells.clear()
                self._own_cells[key] = box
        low, high = box
        for block in blocks:
            b = block.box
            if all(b[2 * d + 1] >= low[d] and b[2 * d] <= high[d] for d in range(3)):
                return self.base._piece(path, point_arrays=names, with_cells=False)
        return light

    # ------------------------------------------------------------- pieces
    def _piece(self, path: Path, point_arrays: list[str] | None = None, with_cells: bool = True):
        names = list(self.components) if point_arrays is None else list(point_arrays)
        if path in self._block_keys:
            seams = self._seamless(self._block_frame[path], names)
            return block_piece(self.blocks.block(path), names, with_cells, seams.get(path))
        level = "level" in names
        own = [n for n in names if n != "level"]
        piece = self.base._piece(path, point_arrays=own, with_cells=with_cells)
        if level:
            piece.point_data["level"] = np.zeros(len(piece.points), np.float32)
        if with_cells and piece.offsets.size:
            index = self._frame_of.get(path)
            files = self.amr_blocks(index) if index is not None else []
            if files:
                piece = _without_covered(piece, [self.blocks.box(f) for f in files])
        return piece

    def _piece_cells(self, number: int, path: Path, piece) -> tuple:
        """The cells of one piece of a frame: the volume's without those under the
        frame's blocks. Kept by where the blocks are, not by frame: a refinement that
        stays (CFD's G 15 box around a structure) has the same cells in every output,
        and finding the covered ones took a third of each iso-surface (the free
        surface, as the run plays)."""
        if path in self._block_keys:  # a refined block itself: its own cells
            block = self.blocks.block(path)
            nz, ny, nx = len(block.z) - 1, len(block.y) - 1, len(block.x) - 1
            return (piece.points.shape[0], *block_cells(nz, ny, nx, block.cells.get("covered")))
        index = self._frame_of.get(path, -1)
        files = self.amr_blocks(index) if index >= 0 else []
        boxes = tuple(tuple(round(v, 9) for v in self.blocks.box(f)) for f in files)
        key = (number, piece.points.shape[0], boxes)
        with self._amr_lock:
            cells = self._own_cells.get(key)
        if cells is None or cells[0] != piece.points.shape[0]:
            full = self._piece(path, point_arrays=[])
            cells = (
                full.points.shape[0],
                full.connectivity.astype(np.int32),
                full.offsets.astype(np.int64),
                full.types,
            )
            with self._amr_lock:
                if len(self._own_cells) > 256:
                    self._own_cells.clear()
                self._own_cells[key] = cells
        return cells

    def _iso_piece(
        self,
        number: int,
        path: Path,
        variable: str,
        value: float,
        arrays,
        extra: float | None = 0.0,
        near: bool = False,
        stride: int = 1,
    ):
        """A piece to cut iso-surfaces from: a block whole; the volume's own pieces
        as their dataset gives them (a store's: only the levels the iso-surfaces pass
        through), without the cells under the frame's blocks."""
        if path in self._block_keys:
            # a block none of the iso-surfaces passes through (most blocks of a
            # refinement many cells high): not made
            seams = self._seamless(self._block_frame[path], list(arrays))
            nodes = seams.get(path, {}).get(variable.partition(".")[0])
            if extra is not None and nodes is not None and nodes.ndim == 3 and nodes.size:
                low, high = float(np.nanmin(nodes)), float(np.nanmax(nodes))
                if high <= value or low > value + extra:  # (nothing known: NaN, kept)
                    return None
            piece = self._piece(path, point_arrays=arrays)  # once: points and cells
            piece.block = True
            return piece, piece.connectivity, piece.offsets, piece.types
        own = [n for n in arrays if n != "level"]
        cut = self.base._iso_piece(number, path, variable, value, own, extra, near)
        if cut is None:
            return None
        piece, connectivity, offsets, types = cut
        if "level" in arrays:
            piece.point_data["level"] = np.zeros(len(piece.points), np.float32)
        index = self._frame_of.get(path, -1)
        files = self.amr_blocks(index) if index >= 0 else []
        if not files or not len(offsets):
            return piece, connectivity, offsets, types
        boxes = tuple(tuple(round(v, 9) for v in self.blocks.box(f)) for f in files)
        key = ("iso", number, getattr(piece, "band", None), len(piece.points), boxes)
        with self._amr_lock:
            cells = self._own_cells.get(key)
        if cells is None:
            whole = Piece(piece.kind, piece.points, connectivity, offsets, types, {})
            kept = _without_covered(whole, list(boxes))
            cells = (kept.connectivity.astype(np.int32), kept.offsets.astype(np.int64), kept.types)
            with self._amr_lock:
                if len(self._own_cells) > 256:
                    self._own_cells.clear()
                self._own_cells[key] = cells
        return (piece, *cells)

    def _cuts_together(self, cuts: list[tuple]) -> list[tuple]:
        """The blocks among the pieces as one piece (their points and cells one
        after the other, where the first block stood): an iso-surface is cut from
        them at once. A refinement of a thousand blocks was a thousand small cuts
        per iso-surface, three with the water-side colours, most of their time
        spent getting each started. The points and their values are those of the
        blocks cut one by one, in the same order; the triangles come in another."""
        blocks = [n for n, cut in enumerate(cuts) if getattr(cut[0], "block", False)]
        if len(blocks) < 2:
            return cuts
        pieces = [cuts[n][0] for n in blocks]
        starts = np.cumsum([0] + [len(piece.points) for piece in pieces])
        names = set.intersection(*(set(piece.point_data) for piece in pieces))
        connectivity = np.concatenate(
            [
                cuts[n][1].astype(np.int64, copy=False) + start
                for n, start in zip(blocks, starts[:-1], strict=True)
            ]
        )
        ends = np.cumsum([0] + [len(cuts[n][1]) for n in blocks])
        offsets = np.concatenate(
            [
                cuts[n][2].astype(np.int64, copy=False) + end
                for n, end in zip(blocks, ends[:-1], strict=True)
            ]
        )
        types = np.concatenate([cuts[n][3] for n in blocks])
        one = Piece(
            pieces[0].kind,
            np.concatenate([piece.points for piece in pieces]),
            connectivity,
            offsets,
            types,
            {name: np.concatenate([piece.point_data[name] for piece in pieces]) for name in names},
        )
        first = blocks[0]
        rest = set(blocks[1:])
        return [
            (one, connectivity, offsets, types) if n == first else cut
            for n, cut in enumerate(cuts)
            if n not in rest
        ]

    def _water_margin(self) -> float | None:
        return self.base._water_margin()  # the volume's cells are the coarsest

    # ------------------------------------------------------------- reading
    def read(self, index: int, variables: list[str] | None = None) -> SurfaceMesh:
        """The outer surface of a frame (the volume's and its blocks')."""
        frame = self.frames[index]
        arrays = self._arrays(variables)
        light = [self._piece(p, point_arrays=arrays, with_cells=False) for p in frame.pieces]
        # by where the blocks lie, not by their files (each output's own): the outer
        # faces were found again for every frame, most of its time
        key = (self._layout(index),)
        with self._lock:
            plan = self._plans.get(key)
            if plan is None or not plan.accepts(light):
                full = [self._piece(p, point_arrays=[]) for p in frame.pieces]
                plan = VolumePlan.build(full)
                if len(self._plans) > 8:
                    self._plans.clear()
                self._plans[key] = plan
        points, data = plan.apply(light)
        mesh = SurfaceMesh(
            points=np.ascontiguousarray(points, dtype=np.float32),
            triangles=plan.triangles,
            point_data=data,
        )
        return mesh.with_normals()

    def _cut_layout(self, index: int):
        return self._layout(index)  # the cells change with the blocks: their cuts too

    def read_values(self, index: int, variable: str):
        return self.base.read_values(index, variable)

    def sample_values(self, index: int, variable: str):
        if variable.partition(".")[0] == "level":
            levels = [float(self.blocks.level(p)) for p in self.amr_blocks(index)]
            return np.array([0.0, *levels])
        return self.base.sample_values(index, variable)

    def point_series(self, variable: str, x: float, y: float):
        return self.base.point_series(variable, x, y)

    def read_streamlines(self, index, vector, axis, position, count, variables=None,
                         lines_in="volume"):  # fmt: skip
        return self.base.read_streamlines(index, vector, axis, position, count,
                                          [v for v in (variables or []) if v != "level"],
                                          lines_in)  # fmt: skip

    def points_per_frame(self) -> int | None:
        return self.base.points_per_frame()


def _without_covered(piece: Piece, boxes: list[tuple]) -> Piece:
    """A piece of the volume without its cells whose centre lies in a block."""
    count = piece.offsets.size
    if not count or piece.connectivity.size != 8 * count:  # hexahedra only
        return piece
    cells = piece.connectivity.reshape(-1, 8)
    if not boxes:
        return piece
    # the centres, corner by corner (np.take: a fraction of the time of
    # points[cells].mean(axis=1)), and only the cells in the box round all blocks
    # asked block by block: most of a piece lies far from any (with some dozen
    # blocks this was the largest part of a frame)
    points = np.asarray(piece.points, np.float32)
    centres = np.take(points, cells[:, 0], axis=0)
    for corner in range(1, 8):
        centres += np.take(points, cells[:, corner], axis=0)
    centres /= np.float32(8.0)
    low = [min(b[2 * d] for b in boxes) for d in range(3)]
    high = [max(b[2 * d + 1] for b in boxes) for d in range(3)]
    near = np.ones(len(cells), bool)
    for d in range(3):
        near &= (centres[:, d] > low[d]) & (centres[:, d] < high[d])
    which = np.flatnonzero(near)
    inside = np.zeros(len(cells), bool)
    if which.size:
        cx, cy, cz = (centres[which, d] for d in range(3))
        within = np.zeros(which.size, bool)
        for x0, x1, y0, y1, z0, z1 in boxes:
            within |= (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1) & (cz > z0) & (cz < z1)
        inside[which[within]] = True
    if not inside.any():
        return piece
    kept = cells[~inside]
    return Piece(
        piece.kind,
        piece.points,
        kept.ravel().astype(np.int64),
        (np.arange(1, len(kept) + 1) * 8).astype(np.int64),
        piece.types[~inside] if piece.types is not None else None,
        piece.point_data,
    )


class AmrBlocks(Dataset):
    """The outlines of the refined blocks of each output, coloured by level."""

    def __init__(self, key: str, solver: str, blocks, folder: Path):
        self.blocks = blocks
        frames = []
        for step in blocks.steps:
            keys = blocks.keys(step)
            if keys:
                frames.append(Frame(step, blocks.stamp(step), keys))
        super().__init__(key, solver, "amr_blocks", frames, folder)

    def _describe(self, first: Frame) -> tuple[str, dict[str, int]]:
        return "Lines", {"level": 1}

    @property
    def label(self) -> str:
        return f"{self.solver} AMR blocks"

    def default_variable(self) -> str:
        return "level"

    def read(self, index: int, variables: list[str] | None = None) -> SurfaceMesh:
        points, segments, levels = [], [], []
        edges = np.array(
            [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
        )  # fmt: skip
        for number, key in enumerate(self.frames[index].pieces):
            x0, x1, y0, y1, z0, z1 = self.blocks.box(key)
            corners = np.array(
                [[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], np.float32
            )
            segments.append(edges + 8 * number)
            points.append(corners)
            levels.append(np.full(8, float(self.blocks.level(key)), np.float32))
        if not points:
            empty = np.zeros((0, 3), np.float32)
            return SurfaceMesh(empty, np.zeros((0, 3), np.uint32), {"level": np.zeros(0)})
        return SurfaceMesh(
            points=np.concatenate(points),
            triangles=np.zeros((0, 3), np.uint32),
            point_data={"level": np.concatenate(levels)},
            segments=np.concatenate(segments).astype(np.uint32),
        )

    def written_span(self) -> tuple[float, float]:
        if self._span is None:
            try:
                first = self.frames[0].path.stat().st_mtime
                last = self.frames[-1].path.stat().st_mtime
            except OSError:
                first = last = 0.0
            self._span = (min(first, last), max(first, last))
        return self._span
