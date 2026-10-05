# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""LAGOON stores (SPEC.md): the reference reader and writer.

Reading::

    store = Store("case/REEF3D_FNPF.lagoon")
    volume = store["volume"]
    volume.times                       # simulation time of every output
    u = volume.read("velocity", 10)    # (nz, ny, nx, 3) at output 10, all blocks
    z = volume.heights(10)             # (nz, ny, nx) level heights
    eta = store["free_surface"].series("eta", i=40, j=12)   # over all outputs

Writing (what the converter and the tests do)::

    store = StoreWriter.create("run.lagoon", solver="NHFLOW")
    out = store.add_output("volume", grid="sigma", x=x, y=y, sigma=sigma,
                           variables={"velocity": 3, "pressure": 1},
                           blocks=[((0, 0), (nx, ny))])
    out.append(time, step, [{"z_bed": ..., "z_surface": ..., "velocity": ...}])
    store.close()
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lagoon_format.amr import AmrSet, AmrWriter
from lagoon_format.bodies import BodySet, BodyWriter
from lagoon_format.particles import ParticleSet, ParticleWriter
from lagoon_format.series import ProfileSet, ProfileWriter, Table, write_table
from lagoon_format.zarr3 import Array, Group, ZarrError, codecs_for, node_type

FORMAT = "lagoon"
VERSION = "0.3"
GRIDS = ("sigma", "cartesian", "surface")
#: per-output arrays of the grid itself, not variables
GRID_ARRAYS = {"sigma": ("z_bed", "z_surface", "z_offset"), "cartesian": (), "surface": ("z",)}
UNITS = {
    "velocity": "m/s",
    "pressure": "Pa",
    "elevation": "m",
    "eta": "m",
    "depth": "m",
    "WL": "m",
    "Fi": "m2/s",
    "Fifsf": "m2/s",
}


class StoreError(ValueError):
    """Not a LAGOON store, or one this reader cannot read."""


def is_store(path: str | Path) -> bool:
    """True for a directory holding a LAGOON store."""
    path = Path(path)
    if node_type(path) != "group":
        return False
    try:
        return Group(path).attrs.get("lagoon", {}).get("format") == FORMAT
    except (ZarrError, OSError, ValueError):
        return False


def _block_name(number: int) -> str:
    return f"b{number:04d}"


def tile_size(n: int, most: int = 64) -> int:
    """The tile length that splits ``n`` points into equal tiles of at most ``most``."""
    if n <= most:
        return max(n, 1)
    return math.ceil(n / math.ceil(n / most))


#: values a chunk should hold at least, so it compresses well
CHUNK_VALUES = 2048


def chunk_tiles(ny: int, nx: int, nz: int = 1, most: int = 64) -> tuple[int, int, int]:
    """Inner chunk (cy, cx, cz) of a block: tiles of at most ``most`` points in y
    and x (longer in x when the block is narrow in y, as in a 2D flume), one level
    when a tile holds enough values, else as many levels as make it hold enough."""
    cy = tile_size(ny, most)
    cx = tile_size(nx, max(most, 2 * CHUNK_VALUES // cy))
    cz = 1 if cy * cx >= CHUNK_VALUES else min(nz, math.ceil(CHUNK_VALUES / (cy * cx)))
    return cy, cx, max(cz, 1)


# ======================================================================== reading
@dataclass
class Block:
    number: int
    start: tuple[int, int]  # (i0, j0) into x and y
    size: tuple[int, int]  # (nx_b, ny_b)
    group: Group
    #: Cartesian grids split in z too: the block's first level and its number of
    #: levels (None: all levels)
    k0: int = 0
    nz: int | None = None

    @property
    def stop(self) -> tuple[int, int]:
        return (self.start[0] + self.size[0], self.start[1] + self.size[1])

    def contains(self, i: int, j: int) -> bool:
        return self.start[0] <= i < self.stop[0] and self.start[1] <= j < self.stop[1]


class Output:
    """One output stream of a run (``volume``, ``free_surface``, ``bed``)."""

    def __init__(self, group: Group):
        self.group = group
        self.name = group.path.name
        info = group.attrs.get("lagoon")
        if not isinstance(info, dict):
            raise StoreError(f"{group.path}: not a LAGOON output group")
        self.info = info
        self.role = str(info.get("role", self.name))
        self.grid = str(info.get("grid", ""))
        if self.grid not in GRIDS:
            raise StoreError(f"{group.path}: grid {self.grid!r} is not supported")
        self.variables: dict[str, int] = {
            name: int(spec.get("components", 1)) for name, spec in info["variables"].items()
        }
        self.units = {
            name: spec["units"] for name, spec in info["variables"].items() if "units" in spec
        }
        self.x = group.array("x")[:]
        self.y = group.array("y")[:]
        self.sigma = group.array("sigma")[:] if self.grid == "sigma" else None
        #: heights of the levels of a Cartesian grid
        self.z = group.array("z")[:] if self.grid == "cartesian" else None
        levels = self.sigma if self.sigma is not None else self.z
        #: number of levels (0 for surfaces)
        self.nlevels = 0 if levels is None else len(levels)
        blocks = group.group("blocks")
        self.blocks: list[Block] = []
        for number in range(int(info.get("blocks", 0))):
            block = blocks.group(_block_name(number))
            meta = block.attrs["lagoon"]
            start, size = list(meta["start"]), list(meta["size"])
            k0 = int(start[2]) if len(start) > 2 else 0
            nz = int(size[2]) if len(size) > 2 else None
            self.blocks.append(Block(number, tuple(start[:2]), tuple(size[:2]), block, k0, nz))
        for b in self.blocks:
            if b.nz is None and self.nlevels:
                b.nz = self.nlevels
        self._arrays: dict[tuple[int, str], Array] = {}
        self._owner: np.ndarray | None = None
        self.refresh()

    # ------------------------------------------------------------- description
    @property
    def shape(self) -> tuple[int, ...]:
        """(nz, ny, nx) of the grid, (ny, nx) for surfaces."""
        if self.nlevels:
            return (self.nlevels, len(self.y), len(self.x))
        return (len(self.y), len(self.x))

    def __len__(self) -> int:
        return len(self.times)

    def refresh(self) -> int:
        """Read the output times again (a store still being written); the count."""
        time = self.group.array("time")
        step = self.group.array("step")
        n = min(time.shape[0], step.shape[0])
        self.times = time[:n].astype(np.float64)
        self.steps = step[:n].astype(np.int64)
        for array in self._arrays.values():
            array.reload()
        return n

    def array(self, block: int | Block, name: str) -> Array:
        number = block.number if isinstance(block, Block) else int(block)
        key = (number, name)
        array = self._arrays.get(key)
        if array is None:
            array = self._arrays[key] = self.blocks[number].group.array(name)
        return array

    def has(self, name: str, block: int | None = None) -> bool:
        """True if the block (any block when None) has the array."""
        if block is None:
            return any(name in b.group for b in self.blocks)
        return name in self.blocks[block].group

    @property
    def owner(self) -> np.ndarray:
        """(ny, nx): the block that provides each grid point (-1: none), the lowest
        numbered where blocks overlap."""
        if self._owner is None:
            owner = np.full((len(self.y), len(self.x)), -1, np.int32)
            for block in reversed(self.blocks):
                (i0, j0), (i1, j1) = block.start, block.stop
                owner[j0:j1, i0:i1] = block.number
            self._owner = owner
        return self._owner

    def block_at(self, i: int, j: int, k: int | None = None) -> Block | None:
        """The block that provides grid point (i, j[, level k]): the lowest numbered
        that holds it (the top level when k is None)."""
        if k is None and self.nlevels:
            k = self.nlevels - 1
        for block in self.blocks:
            inside = block.contains(i, j)
            if inside and (k is None or block.k0 <= k < block.k0 + (block.nz or 0)):
                return block
        return None

    def nearest(self, x: float, y: float) -> tuple[int, int]:
        """Grid indices (i, j) of the point nearest to (x, y)."""
        return int(np.argmin(np.abs(self.x - x))), int(np.argmin(np.abs(self.y - y)))

    # ------------------------------------------------------------- reading
    def read_block(
        self, name: str, t: int, block: int | Block, level: int | None = None
    ) -> np.ndarray:
        """One block's array at output ``t``: (nz, ny_b, nx_b[, 3]) or (ny_b, nx_b[, 3]);
        only one level of a volume field with ``level``."""
        if name == "z_offset" and not self.has("z_offset", _number(block)):
            shape = self._block_shape(block)
            return np.zeros(shape[1:] if level is not None else shape, np.float32)
        array = self.array(block, name)
        if level is not None and self._has_levels(name, array):
            return array[t, level - self.blocks[_number(block)].k0]
        return array[t]

    def _has_levels(self, name: str, array: Array) -> bool:
        return bool(self.nlevels) and name not in ("z_bed", "z_surface") and array.ndim >= 4

    def _block_shape(self, block) -> tuple[int, ...]:
        b = self.blocks[_number(block)]
        if self.nlevels:
            return (b.nz, b.size[1], b.size[0])
        return (b.size[1], b.size[0])

    def read(self, name: str, t: int, threads: int = 4, level: int | None = None) -> np.ndarray:
        """An array of the whole grid at output ``t``, joined from the blocks:
        (nz, ny, nx[, 3]) for volume fields (one level: (ny, nx[, 3])), (ny, nx[, 3])
        for surfaces and the column heights ``z_bed``, ``z_surface``."""
        holders = [b.number for b in self.blocks if self.has(name, b.number)]
        if name == "z_offset" and not holders:
            return np.zeros(self.shape if level is None else self.shape[1:], np.float32)
        sample = self.array(holders[0] if holders else 0, name)
        levels = self._has_levels(name, sample)
        head = (self.nlevels,) if levels and level is None else ()
        tail = tuple(sample.shape[(4 if levels else 3) :])
        out = np.full((*head, len(self.y), len(self.x), *tail), np.nan, sample.dtype)
        rest = (slice(None),) * len(tail)
        wanted = [
            b
            for b in self.blocks
            if not (levels and level is not None) or b.k0 <= level < b.k0 + (b.nz or 0)
        ]
        if threads > 1 and len(wanted) > 1:
            with ThreadPoolExecutor(min(threads, len(wanted))) as pool:
                parts = list(pool.map(lambda b: self.read_block(name, t, b, level), wanted))
        else:
            parts = [self.read_block(name, t, b, level) for b in wanted]
        # the highest numbered first: where blocks overlap, the lowest numbered wins
        for block, values in sorted(zip(wanted, parts, strict=True), key=lambda bp: -bp[0].number):
            (i0, j0), (i1, j1) = block.start, block.stop
            lead = (slice(block.k0, block.k0 + values.shape[0]),) if head else ()
            out[lead + (slice(j0, j1), slice(i0, i1)) + rest] = values
        return out

    def heights(
        self, t: int, block: int | Block | None = None, level: int | None = None
    ) -> np.ndarray:
        """z of every point at output ``t``: (nz, ny, nx) for σ-grids, (ny, nx) for
        surfaces; one block's part when ``block`` is given, one level with ``level``."""
        if block is None:
            parts = {n: self.read(n, t, level=level) for n in GRID_ARRAYS[self.grid]}
        else:
            parts = {n: self.read_block(n, t, block, level) for n in GRID_ARRAYS[self.grid]}
        if self.grid == "surface":
            return parts["z"]
        if self.grid == "cartesian":
            b = self.blocks[_number(block)] if block is not None else None
            ny, nx = (b.size[1], b.size[0]) if b is not None else (len(self.y), len(self.x))
            if level is not None:
                return np.full((ny, nx), self.z[level], np.float32)
            z = self.z if b is None else self.z[b.k0 : b.k0 + b.nz]
            return np.broadcast_to(z.astype(np.float32)[:, None, None], (len(z), ny, nx))
        if level is None:
            return sigma_heights(parts["z_bed"], parts["z_surface"], self.sigma, parts["z_offset"])
        bed = parts["z_bed"].astype(np.float64)
        z = bed + self.sigma[level] * (parts["z_surface"] - bed) + parts["z_offset"]
        return z.astype(np.float32)

    def series(self, name: str, i: int, j: int, k: int | None = None) -> np.ndarray:
        """Values at grid point (i, j[, level k]) over all outputs: (nt,) or (nt, 3).
        Reads one small inner chunk per output."""
        flat = name in ("z_bed", "z_surface", "z") or not self.nlevels
        block = self.block_at(i, j, None if flat else k)
        if block is None:
            raise IndexError(f"no block holds point ({i}, {j})")
        bi, bj = i - block.start[0], j - block.start[1]
        array = self.array(block, name)
        n = len(self.times)
        if not flat:
            level = self.nlevels - 1 if k is None else k
            return array[:n, level - block.k0, bj, bi]
        return array[:n, bj, bi]

    def column_heights(self, i: int, j: int) -> np.ndarray:
        """(nt, nz) heights of the levels of column (i, j) over all outputs."""
        block = self.block_at(i, j)
        if block is None:
            raise IndexError(f"no block holds point ({i}, {j})")
        bi, bj = i - block.start[0], j - block.start[1]
        n = len(self.times)
        if self.grid == "cartesian":
            return np.tile(self.z.astype(np.float64), (n, 1))
        bed = self.array(block, "z_bed")[:n, bj, bi].astype(np.float64)
        top = self.array(block, "z_surface")[:n, bj, bi].astype(np.float64)
        z = bed[:, None] + self.sigma[None, :] * (top - bed)[:, None]
        if self.has("z_offset", block.number):
            z += self.array(block, "z_offset")[:n, :, bj, bi]
        return z


def _remove_tree(path: Path) -> None:
    """Remove a group of the store (a body set written again)."""
    if node_type(path) == "group":
        shutil.rmtree(path)


def _number(block) -> int:
    return block.number if isinstance(block, Block) else int(block)


def sigma_heights(z_bed, z_surface, sigma, z_offset=None) -> np.ndarray:
    """z = z_bed + σ (z_surface − z_bed) + z_offset, levels first."""
    bed = np.asarray(z_bed, np.float64)
    depth = np.asarray(z_surface, np.float64) - bed
    shape = (-1,) + (1,) * bed.ndim
    z = bed[None] + np.asarray(sigma, np.float64).reshape(shape) * depth[None]
    if z_offset is not None:
        z += z_offset
    return z.astype(np.float32)


class Store:
    """A LAGOON store opened for reading."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not is_store(self.path):
            raise StoreError(f"{self.path}: not a LAGOON store")
        self.root = Group(self.path)
        info = self.root.attrs["lagoon"]
        major = str(info.get("version", "0")).split(".")[0]
        if major != VERSION.split(".")[0]:
            raise StoreError(f"{self.path}: format version {info.get('version')} not supported")
        self.info = info
        self.solver = str(info.get("solver", ""))
        self.run: dict[str, Any] | None = self.root.attrs.get("reef3d_run")
        self.outputs: dict[str, Output] = {}
        for name in info.get("outputs", []):
            if name in self.root:
                self.outputs[name] = Output(self.root.group(name))
        #: time series tables (SPEC 7) and wave profile sets (SPEC 8), by key
        self.tables: dict[str, Table] = {}
        self.profiles: dict[str, ProfileSet] = {}
        if "series" in self.root:
            series = self.root.group("series")
            for key in info.get("series", series.members()):
                if key in series:
                    self.tables[key] = Table.open(series.group(key))
        if "profiles" in self.root:
            profiles = self.root.group("profiles")
            for key in info.get("profiles", profiles.members()):
                if key in profiles:
                    self.profiles[key] = ProfileSet.open(profiles.group(key))
        #: floating bodies (SPEC 9), by key
        self.bodies: dict[str, BodySet] = {}
        if "bodies" in self.root:
            bodies = self.root.group("bodies")
            for key in info.get("bodies", bodies.members()):
                if key in bodies:
                    self.bodies[key] = BodySet.open(bodies.group(key))
        #: particles, DEM elements, ice floes, rods, mooring lines (SPEC 10), by key
        self.particles: dict[str, ParticleSet] = {}
        if "particles" in self.root:
            particles = self.root.group("particles")
            for key in info.get("particles", particles.members()):
                if key in particles:
                    self.particles[key] = ParticleSet.open(particles.group(key))
        #: adaptive mesh refinement surfaces (SPEC 11), by key
        self.amr: dict[str, AmrSet] = {}
        if "amr" in self.root:
            amr = self.root.group("amr")
            for key in info.get("amr", amr.members()):
                if key in amr:
                    self.amr[key] = AmrSet.open(amr.group(key))

    @property
    def run_id(self) -> str | None:
        return self.run.get("run") if isinstance(self.run, dict) else None

    def __getitem__(self, name: str) -> Output:
        return self.outputs[name]

    def __contains__(self, name: str) -> bool:
        return name in self.outputs

    def refresh(self) -> None:
        for output in self.outputs.values():
            output.refresh()


# ======================================================================== writing
class OutputWriter:
    """Appends outputs to one output group; whole shards are written at once."""

    def __init__(self, group: Group, shard_time: int = 16, tile: int = 64, level: int = 1):
        self.group = group
        self.output = Output(group)
        self.shard_time = int(shard_time)
        self.tile = int(tile)
        self.level = int(level)
        self.pending: list[tuple[float, int, list[dict[str, np.ndarray]]]] = []
        self._pool = ThreadPoolExecutor(min(8, os.cpu_count() or 2))

    @property
    def count(self) -> int:
        """Outputs written (on disk) plus waiting."""
        return len(self.output.times) + len(self.pending)

    def append(self, time: float, step: int, blocks: list[dict[str, np.ndarray]]) -> None:
        """One output: per block a dict of its arrays (grid arrays and variables).

        Volumes need ``z_bed``, ``z_surface`` (ny_b, nx_b), optionally ``z_offset`` and
        the variables (nz, ny_b, nx_b[, 3]); surfaces need ``z`` and the variables.
        """
        if len(blocks) != len(self.output.blocks):
            have = len(self.output.blocks)
            raise StoreError(f"{len(blocks)} blocks given, the output has {have}")
        self.pending.append((float(time), int(step), blocks))
        start = len(self.output.times)
        if (start + len(self.pending)) % self.shard_time == 0:
            self.flush()

    def flush(self) -> None:
        """Write what is waiting (a partial shard is written now and completed by
        read-modify-write later)."""
        if not self.pending:
            return
        out = self.output
        start = len(out.times)
        stop = start + len(self.pending)
        jobs = []
        for number, block in enumerate(out.blocks):
            names = {name for _, _, blocks in self.pending for name in blocks[number]}
            for name in sorted(names):
                stack = np.stack(
                    [self._value(blocks[number], name, number) for _, _, blocks in self.pending]
                )
                jobs.append((block, name, stack))
        list(self._pool.map(lambda job: self._write(start, stop, *job), jobs))
        # time and step last: readers count outputs by them (SPEC 2.1)
        times = np.concatenate([out.times, [t for t, _, _ in self.pending]])
        steps = np.concatenate([out.steps, [s for _, s, _ in self.pending]])
        for name, values in (("step", steps), ("time", times)):
            array = self.group.array(name)
            array.resize([stop])
            array[start:stop] = values[start:stop]
        self.pending = []
        out.refresh()

    def _value(self, arrays: dict[str, np.ndarray], name: str, number: int) -> np.ndarray:
        if name in arrays:
            return np.asarray(arrays[name], np.float32)
        shape = self._shape(name, number)
        return np.full(shape, 0.0 if name == "z_offset" else np.nan, np.float32)

    def _shape(self, name: str, number: int) -> tuple[int, ...]:
        out = self.output
        nx_b, ny_b = out.blocks[number].size
        flat = name in ("z_bed", "z_surface", "z") or not out.nlevels
        shape = (ny_b, nx_b) if flat else (out.blocks[number].nz, ny_b, nx_b)
        if out.variables.get(name, 1) > 1:
            shape += (out.variables[name],)
        return shape

    def _write(self, start: int, stop: int, block: Block, name: str, stack: np.ndarray) -> None:
        group = block.group
        if name in group:
            array = self.output.array(block, name)
            array.reload()
        else:
            array = self._create(block, name, stack.shape[1:])
        if array.shape[0] < stop:
            array.resize([stop, *array.shape[1:]])
        array[start:stop] = stack

    def _create(self, block: Block, name: str, shape: tuple[int, ...]) -> Array:
        out = self.output
        volume = bool(out.nlevels) and name not in ("z_bed", "z_surface", "z")
        dims = ["time"] + (["level"] if volume else []) + ["y", "x"]
        if len(shape) > len(dims) - 1:
            dims.append("component")
        ny_b, nx_b = block.size[1], block.size[0]
        ty, tx, tz = chunk_tiles(ny_b, nx_b, shape[0] if volume else 1, self.tile)
        inner = [1] + ([tz] if volume else []) + [ty, tx]
        shard = [self.shard_time] + ([math.ceil(shape[0] / tz) * tz] if volume else [])
        shard += [math.ceil(ny_b / ty) * ty, math.ceil(nx_b / tx) * tx]
        if dims[-1] == "component":
            inner.append(shape[-1])
            shard.append(shape[-1])
        attributes = {}
        if name in out.units:
            attributes["units"] = out.units[name]
        elif name.startswith("z"):
            attributes["units"] = "m"
        array = Array.create(
            block.group.path / name,
            [0, *shape],
            "float32",
            shard,
            inner=inner,
            codecs=codecs_for("gzip", self.level, 4),
            fill_value=0.0 if name == "z_offset" else np.nan,  # SPEC 3: mostly zeros
            dimension_names=dims,
            attributes=attributes,
        )
        out._arrays[(block.number, name)] = array
        return array

    def close(self) -> None:
        self.flush()
        self._pool.shutdown()


class StoreWriter:
    """A LAGOON store opened for writing (new, or to append to)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.root = Group(self.path)
        self.writers: dict[str, OutputWriter] = {}
        self.body_writers: list[BodyWriter] = []
        self.particle_writers: list[ParticleWriter] = []
        self.amr_writers: list[AmrWriter] = []

    @classmethod
    def create(
        cls,
        path: str | Path,
        solver: str,
        run: dict[str, Any] | None = None,
        created_by: str = "lagoon-format",
        source: str = "",
    ) -> StoreWriter:
        path = Path(path)
        if path.exists() and any(path.iterdir()):
            raise StoreError(f"{path}: exists and is not empty")
        from lagoon_format import __version__

        info = {
            "format": FORMAT,
            "version": VERSION,
            "solver": solver,
            "outputs": [],
            "created_by": f"{created_by} (lagoon-format {__version__})",
            "created": _dt.datetime.now(_dt.timezone.utc).strftime(  # noqa: UP017 (3.10)
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        if source:
            info["source"] = source
        attributes: dict[str, Any] = {"lagoon": info}
        if run:
            attributes["reef3d_run"] = run
        Group.create(path, attributes)
        return cls(path)

    @classmethod
    def open(cls, path: str | Path) -> StoreWriter:
        if not is_store(path):
            raise StoreError(f"{path}: not a LAGOON store")
        return cls(path)

    def add_output(
        self,
        name: str,
        grid: str,
        x: Sequence[float],
        y: Sequence[float],
        variables: dict[str, int],
        blocks: list[tuple[tuple[int, int], tuple[int, int]]],
        sigma: Sequence[float] | None = None,
        z: Sequence[float] | None = None,
        role: str | None = None,
        source: str = "",
        attributes: dict[str, Any] | None = None,
        shard_time: int = 16,
        tile: int = 64,
        level: int = 1,
    ) -> OutputWriter:
        """A new output group; ``blocks`` are ((i0, j0), (nx_b, ny_b)) per block, or
        ((i0, j0, k0), (nx_b, ny_b, nz_b)) for a Cartesian grid split in z too."""
        if grid not in GRIDS:
            raise StoreError(f"grid {grid!r} is not supported")
        if grid == "sigma" and sigma is None:
            raise StoreError("a σ-grid needs its levels (sigma)")
        if grid == "cartesian" and z is None:
            raise StoreError("a Cartesian grid needs the heights of its levels (z)")
        shape = {"x": len(x), "y": len(y)}
        if sigma is not None or z is not None:
            shape["level"] = len(sigma if sigma is not None else z)
        info = {
            "role": role or name,
            "grid": grid,
            "shape": shape,
            "variables": {
                v: {"components": int(c), **({"units": UNITS[v]} if v in UNITS else {})}
                for v, c in variables.items()
            },
            "blocks": len(blocks),
        }
        if source:
            info["source"] = source
        info.update(attributes or {})  # e.g. "dataset": what LAGOON called the source
        group = Group.create(self.path / name, {"lagoon": info})
        for array_name, values, dims in (
            ("x", x, ["x"]),
            ("y", y, ["y"]),
            *([("sigma", sigma, ["level"])] if sigma is not None else []),
            *([("z", z, ["level"])] if grid == "cartesian" else []),
        ):
            values = np.asarray(values, np.float64)
            Array.create(group.path / array_name, values.shape, "float64", dimension_names=dims)[
                :
            ] = values
        for array_name, dtype in (("time", "float64"), ("step", "int64")):
            Array.create(
                group.path / array_name,
                [0],
                dtype,
                [4096],
                dimension_names=["time"],
                attributes={"units": "s"} if array_name == "time" else {},
            )
        Group.create(group.path / "blocks")
        for number, (start, size) in enumerate(blocks):
            Group.create(
                group.path / "blocks" / _block_name(number),
                {"lagoon": {"start": [int(s) for s in start], "size": [int(s) for s in size]}},
            )
        attrs = dict(self.root.attrs)
        attrs["lagoon"] = dict(attrs["lagoon"])
        attrs["lagoon"]["outputs"] = sorted({*attrs["lagoon"].get("outputs", []), name})
        self.root.set_attrs(attrs)
        writer = OutputWriter(group, shard_time, tile, level)
        self.writers[name] = writer
        return writer

    def _list(self, entry: str, key: str) -> None:
        """Name a table or profile set in the root's list of them."""
        attrs = dict(self.root.attrs)
        info = dict(attrs["lagoon"])
        info[entry] = sorted({*info.get(entry, []), key})
        info["version"] = VERSION
        attrs["lagoon"] = info
        self.root.set_attrs(attrs)

    def write_table(self, key: str, time, columns, **kwargs) -> Table:
        """Write (or replace) a time series table (see :func:`write_table`)."""
        if "series" not in self.root:
            Group.create(self.path / "series")
        table = write_table(self.root.group("series"), key, time, columns, **kwargs)
        self._list("series", key)
        return table

    def profile_writer(
        self,
        key: str,
        lines: dict[int, tuple[float, np.ndarray, bool]] | None = None,
        source: str = "",
    ) -> ProfileWriter:
        """The writer of a wave profile set; ``lines`` creates it (y, x, has theory)."""
        if "profiles" not in self.root:
            Group.create(self.path / "profiles")
        parent = self.root.group("profiles")
        if key in parent and lines is None:
            return ProfileWriter(parent.group(key))
        writer = ProfileWriter.create(parent, key, lines or {}, source)
        self._list("profiles", key)
        return writer

    def body_writer(
        self,
        key: str,
        bodies: dict | None = None,
        source: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> BodyWriter:
        """The writer of a body set; ``bodies`` creates it (or replaces it), see
        :meth:`BodyWriter.create`."""
        if "bodies" not in self.root:
            Group.create(self.path / "bodies")
        parent = self.root.group("bodies")
        if key in parent and bodies is None:
            writer = BodyWriter(parent.group(key))
        else:
            if key in parent:
                _remove_tree(parent.path / key)
            writer = BodyWriter.create(parent, key, bodies or {}, source, attributes)
        self.body_writers.append(writer)
        self._list("bodies", key)
        return writer

    def particle_writer(self, key: str, create: dict[str, Any] | None = None) -> ParticleWriter:
        """The writer of a particle set; ``create`` (the arguments of
        :meth:`ParticleWriter.create` after the key) makes it new (or replaces it)."""
        if "particles" not in self.root:
            Group.create(self.path / "particles")
        parent = self.root.group("particles")
        if key in parent and create is None:
            writer = ParticleWriter(parent.group(key))
        else:
            if key in parent:
                _remove_tree(parent.path / key)
            writer = ParticleWriter.create(parent, key, **(create or {"point_fields": {}}))
        self.particle_writers.append(writer)
        self._list("particles", key)
        return writer

    def amr_writer(self, key: str, create: dict[str, Any] | None = None) -> AmrWriter:
        """The writer of an AMR set; ``create`` (the arguments of
        :meth:`AmrWriter.create` after the key) makes it new (or replaces it)."""
        if "amr" not in self.root:
            Group.create(self.path / "amr")
        parent = self.root.group("amr")
        if key in parent and create is None:
            writer = AmrWriter(parent.group(key))
        else:
            if key in parent:
                _remove_tree(parent.path / key)
            writer = AmrWriter.create(parent, key, **(create or {"fields": {}}))
        self.amr_writers.append(writer)
        self._list("amr", key)
        return writer

    def output(self, name: str, shard_time: int = 16, tile: int = 64) -> OutputWriter:
        """The writer of an existing output group (to append to it)."""
        writer = self.writers.get(name)
        if writer is None:
            writer = self.writers[name] = OutputWriter(self.root.group(name), shard_time, tile)
        return writer

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        for body_writer in self.body_writers:
            body_writer.close()
        for particle_writer in self.particle_writers:
            particle_writer.close()
        for amr_writer in self.amr_writers:
            amr_writer.close()
