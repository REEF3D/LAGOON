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

    @property
    def level(self) -> int:
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
    cells = _hexahedra(nz, ny, nx)
    covered = block.cells.get("covered")
    if covered is not None:
        cells = cells[np.asarray(covered).ravel() < 0.5]
    connectivity = cells.ravel().astype(np.int64)
    offsets = (np.arange(1, len(cells) + 1) * 8).astype(np.int64)
    types = np.full(len(cells), 12, np.uint8)  # VTK_HEXAHEDRON
    return Piece("UnstructuredGrid", points, connectivity, offsets, types, data)


# ======================================================================== seams
#: the eight cells round a node, as offsets (dz, dy, dx)
_EIGHT = [(dz, dy, dx) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]


def _simplex(x, y, z, values: np.ndarray, px, py, pz) -> np.ndarray:
    """Values on a node lattice (len(z), len(y), len(x)[, c]) at points, linear in
    each of the six tetrahedra a cell is cut into (as :func:`plan_level` cuts a
    hexahedron: round its diagonal from the lowest to the highest corner). A finer
    block's face takes so exactly the values the coarser cells' cut has there, and
    the two meet without a crack."""
    index, fraction = [], []
    for axis, p in zip((x, y, z), (px, py, pz), strict=True):
        axis = np.asarray(axis, np.float64)
        i = np.clip(np.searchsorted(axis, p, "right") - 1, 0, len(axis) - 2)
        f = np.clip((p - axis[i]) / (axis[i + 1] - axis[i]), 0.0, 1.0)
        index.append(i)
        fraction.append(f)
    f = np.stack(fraction, axis=1)
    order = np.argsort(-f, axis=1, kind="stable")  # the axes from the largest step
    fs = np.take_along_axis(f, order, axis=1)
    weights = (1.0 - fs[:, 0], fs[:, 0] - fs[:, 1], fs[:, 1] - fs[:, 2], fs[:, 2])
    corner = np.zeros((len(f), 3), np.int64)
    rows = np.arange(len(f))
    out = None
    for step, w in enumerate(weights):
        if step:
            corner[rows, order[:, step - 1]] = 1
        v = values[index[2] + corner[:, 2], index[1] + corner[:, 1], index[0] + corner[:, 0]]
        part = (w[:, None] if v.ndim == 2 else w) * v.astype(np.float64)
        out = part if out is None else out + part
    return out.astype(np.float32)


def volume_lattice(pieces: list[Piece], names: list[str]):
    """The volume's nodes as a lattice: (x, y, z, {name: (nz, ny, nx[, c])}), NaN
    where no piece has a node; None if its points are no lattice."""
    pieces = [p for p in pieces if len(p.points)]
    if not pieces:
        return None
    points = np.concatenate([p.points for p in pieces]).astype(np.float64)
    axes = [np.unique(np.round(points[:, d], 5)) for d in range(3)]
    if (
        np.prod([a.size for a in axes], dtype=np.float64) > 8.0 * len(points)
        or min(a.size for a in axes) < 2
    ):
        return None
    i, j, k = (np.searchsorted(axes[d], np.round(points[:, d], 5)) for d in range(3))
    arrays = {}
    for name in names:
        parts = [p.point_data.get(name) for p in pieces]
        if any(v is None for v in parts):
            continue
        values = np.concatenate([np.asarray(v, np.float32) for v in parts])
        shape = (axes[2].size, axes[1].size, axes[0].size, *values.shape[1:])
        lattice = np.full(shape, np.nan, np.float32)
        lattice[k, j, i] = values
        arrays[name] = lattice
    return axes[0], axes[1], axes[2], arrays


def _touch(a: tuple, b: tuple, reach: tuple) -> bool:
    """Whether box ``b`` reaches into box ``a`` widened by ``reach`` (dx, dy, dz)."""
    return all(
        b[2 * d + 1] >= a[2 * d] - reach[d] and b[2 * d] <= a[2 * d + 1] + reach[d]
        for d in range(3)
    )


def _centres(edges: np.ndarray) -> np.ndarray:
    """Cell centres of a lattice and of one cell beyond each end."""
    e = np.asarray(edges, np.float64)
    c = 0.5 * (e[:-1] + e[1:])
    return np.concatenate([[c[0] - (e[1] - e[0])], c, [c[-1] + (e[-1] - e[-2])]])


def _padded(cells: np.ndarray) -> np.ndarray:
    """Cells (nz, ny, nx[, c]) with a layer beyond each face, carried on linearly."""
    padded = np.asarray(cells, np.float32)
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
    return padded


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
    done: dict[int, dict[str, np.ndarray]] = {}
    for n in order:
        block = blocks[n]
        level = block.level
        reach = tuple(float(max(a[1] - a[0], a[-1] - a[-2])) for a in (block.x, block.y, block.z))
        near = [m for m in range(len(blocks)) if m != n and _touch(boxes[n], boxes[m], reach)]
        same = [m for m in near if blocks[m].level == level]
        coarser = sorted(
            (m for m in near if blocks[m].level < level), key=lambda m: -blocks[m].level
        )
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
            oi = np.clip(np.searchsorted(ox, xc[ii], "right") - 1, 0, len(ox) - 2)
            oj = np.clip(np.searchsorted(oy, yc[jj], "right") - 1, 0, len(oy) - 2)
            ok = np.clip(np.searchsorted(oz, zc[kk], "right") - 1, 0, len(oz) - 2)
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
                for name in own:
                    if name in done[m]:
                        nodes[name][kk[at], jj[at], ii[at]] = _simplex(
                            other.x, other.y, other.z, done[m][name], px[at], py[at], pz[at]
                        )
                open_ &= ~at
            if lattice is not None and open_.any():
                lx, ly, lz, arrays = lattice
                for name in own:
                    if name not in arrays:
                        continue
                    found = _simplex(lx, ly, lz, arrays[name], px[open_], py[open_], pz[open_])
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
        self._lock = threading.Lock()

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

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
        t = self._index.get(step)
        return [] if t is None else [b.path for b in self._of(t)]

    def _find(self, key: Path) -> Block:
        t = int(key.parent.name[1:])
        return self._of(t)[int(key.stem[1:])]

    def box(self, key: Path) -> tuple:
        return self._find(key).box

    def block(self, key: Path) -> Block:
        return self._find(key)

    def level(self, key: Path) -> int:
        return self._find(key).level

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

    def _seamless(self, index: int, names: list[str]) -> dict[Path, dict]:
        """The node values of a frame's blocks, alike on both sides of their faces
        (:func:`seamless_nodes`), kept for the frames being looked at."""
        names = sorted(n for n in names if n not in ("level", "elevation"))
        key = (index, tuple(names))
        with self._amr_lock:
            found = self._seams.get(key)
        if found is not None:
            return found
        found = {}
        if names:
            keys = self.amr_blocks(index)
            blocks = [self.blocks.block(k) for k in keys]
            wanted = [n for n in names if n in self.base.components]
            pieces = [
                self.base._piece(p, point_arrays=wanted, with_cells=False)
                for p in self.base.frames[index].pieces
            ]
            found = seamless_nodes(blocks, names, volume_lattice(pieces, wanted))
        with self._amr_lock:
            for old in [k for k in self._seams if k[0] < index - 2 or k[0] > index + 2]:
                del self._seams[old]
            if len(self._seams) > 16:
                self._seams.clear()
            self._seams[key] = found
        return found

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
        frame's blocks change from output to output, so they are kept by frame."""
        index = self._frame_of.get(path, -1)
        key = (number, path)
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
                # the cells of the frames being looked at (the latest few)
                for old in [k for k in self._own_cells if self._frame_of.get(k[1], -2) < index - 2]:
                    del self._own_cells[old]
                if len(self._own_cells) > 512:
                    self._own_cells.clear()
                self._own_cells[key] = cells
        return cells

    # ------------------------------------------------------------- reading
    def read(self, index: int, variables: list[str] | None = None) -> SurfaceMesh:
        """The outer surface of a frame (the volume's and its blocks')."""
        frame = self.frames[index]
        arrays = self._arrays(variables)
        light = [self._piece(p, point_arrays=arrays, with_cells=False) for p in frame.pieces]
        key = tuple(str(p) for p in self.amr_blocks(index)) or ("",)
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

    def read_slice(self, index, axis, position, variables=None) -> SurfaceMesh:
        self._cuts.clear()  # the cells change with the blocks: cut anew
        return super().read_slice(index, axis, position, variables)

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
    centres = piece.points[cells].mean(axis=1)
    inside = np.zeros(len(cells), bool)
    for x0, x1, y0, y1, z0, z1 in boxes:
        inside |= (
            (centres[:, 0] > x0)
            & (centres[:, 0] < x1)
            & (centres[:, 1] > y0)
            & (centres[:, 1] < y1)
            & (centres[:, 2] > z0)
            & (centres[:, 2] < z1)
        )
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
