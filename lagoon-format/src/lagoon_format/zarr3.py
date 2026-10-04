# SPDX-License-Identifier: MIT
# Architect: Hans Bihs
"""A small Zarr v3 reader and writer (NumPy and the standard library only).

It covers what LAGOON stores use (SPEC.md, section 5): groups with attributes,
arrays with a regular chunk grid, the ``default`` chunk key encoding, the codecs
``bytes``, ``numcodecs.shuffle``, ``gzip``, ``zlib``, ``crc32c`` (read) and
``sharding_indexed``, and resizing along any dimension. Stores written here open in
zarr-python; stores written by zarr-python with these codecs open here.

Reads touch only the chunks (and, in a shard, only the inner chunks) a selection
needs. Writes are read-modify-write per chunk or shard, so writing whole shards at
a time (as the converter does) never reads anything.
"""

from __future__ import annotations

import json
import math
import os
import struct
import threading
import zlib
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ZARR_JSON = "zarr.json"
#: index entry of an inner chunk that is not written
EMPTY = 2**64 - 1

DTYPES = {
    "bool": np.dtype(bool),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "uint64": np.dtype("uint64"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
}


class ZarrError(ValueError):
    """Not a Zarr v3 node, or a feature this reader does not support."""


# ----------------------------------------------------------------- metadata files
def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads((path / ZARR_JSON).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ZarrError(f"{path}: no {ZARR_JSON}") from None


def write_json(path: Path, meta: dict[str, Any]) -> None:
    """Write ``zarr.json`` atomically (a reader never sees half a file)."""
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write(path / ZARR_JSON, json.dumps(meta, indent=2, allow_nan=False).encode())


def _atomic_write(target: Path, data: bytes) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, target)


def _fill_to_json(value, dtype: np.dtype):
    if dtype.kind == "f":
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(value)
    if dtype.kind == "b":
        return bool(value)
    return int(value)


def _fill_from_json(value, dtype: np.dtype):
    if isinstance(value, str):
        value = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}.get(value, value)
        if isinstance(value, str) and value.startswith("0x"):  # raw bit pattern
            raw = int(value, 16).to_bytes(dtype.itemsize, "big")
            return np.frombuffer(raw, dtype.newbyteorder(">"))[0]
    return dtype.type(value)


# ----------------------------------------------------------------- groups
class Group:
    """A Zarr v3 group: a directory with ``zarr.json`` and attributes."""

    def __init__(self, path: str | Path, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.meta = meta if meta is not None else read_json(self.path)
        if self.meta.get("zarr_format") != 3 or self.meta.get("node_type") != "group":
            raise ZarrError(f"{self.path}: not a Zarr v3 group")

    @classmethod
    def create(cls, path: str | Path, attributes: dict[str, Any] | None = None) -> Group:
        meta = {"zarr_format": 3, "node_type": "group", "attributes": attributes or {}}
        write_json(Path(path), meta)
        return cls(path, meta)

    @property
    def attrs(self) -> dict[str, Any]:
        return self.meta.setdefault("attributes", {})

    def set_attrs(self, attributes: dict[str, Any]) -> None:
        self.meta["attributes"] = attributes
        write_json(self.path, self.meta)

    def group(self, name: str) -> Group:
        return Group(self.path / name)

    def array(self, name: str) -> Array:
        return Array(self.path / name)

    def __contains__(self, name: str) -> bool:
        return (self.path / name / ZARR_JSON).is_file()

    def members(self) -> list[str]:
        if not self.path.is_dir():
            return []
        return sorted(p.name for p in self.path.iterdir() if (p / ZARR_JSON).is_file())


def node_type(path: str | Path) -> str | None:
    """``"group"``, ``"array"`` or None when ``path`` is no Zarr v3 node."""
    try:
        meta = read_json(Path(path))
    except (ZarrError, ValueError, OSError):
        return None
    return meta.get("node_type") if meta.get("zarr_format") == 3 else None


# ----------------------------------------------------------------- codecs
def _shuffle(data: bytes, size: int) -> bytes:
    if size <= 1 or len(data) % size:
        return data
    return np.frombuffer(data, np.uint8).reshape(-1, size).T.tobytes()


def _unshuffle(data: bytes, size: int):
    """The bytes of each value together again (a NumPy byte array, no extra copy)."""
    if size <= 1 or len(data) % size:
        return data
    planes = np.frombuffer(data, np.uint8).reshape(size, -1)
    out = np.empty((planes.shape[1], size), np.uint8)
    out[:] = planes.T
    return out


def _gzip(data: bytes, level: int) -> bytes:
    packer = zlib.compressobj(level, zlib.DEFLATED, 31)  # 31: gzip container
    return packer.compress(data) + packer.flush()


_CRC32C_TABLE: list[int] | None = None


def crc32c(data: bytes) -> int:
    """CRC-32C (Castagnoli), as Zarr's ``crc32c`` codec appends it."""
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        table = []
        for n in range(256):
            c = n
            for _ in range(8):
                c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
            table.append(c)
        _CRC32C_TABLE = table
    crc = 0xFFFFFFFF
    table = _CRC32C_TABLE
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


class Pipeline:
    """The ``bytes`` codec plus bytes-to-bytes codecs, for one chunk."""

    def __init__(self, codecs: list[dict[str, Any]], dtype: np.dtype):
        self.dtype = dtype
        self.endian = "little"
        self.steps: list[tuple[str, dict[str, Any]]] = []
        seen_bytes = False
        for codec in codecs:
            name = codec["name"] if isinstance(codec, dict) else str(codec)
            config = codec.get("configuration", {}) if isinstance(codec, dict) else {}
            if name == "bytes":
                self.endian = config.get("endian", "little")
                seen_bytes = True
            elif name in ("numcodecs.shuffle", "shuffle"):
                self.steps.append(("shuffle", {"size": int(config.get("elementsize", 4))}))
            elif name == "gzip":
                self.steps.append(("gzip", {"level": int(config.get("level", 1))}))
            elif name in ("zlib", "numcodecs.zlib"):
                self.steps.append(("zlib", {"level": int(config.get("level", 1))}))
            elif name == "crc32c":
                self.steps.append(("crc32c", {}))
            elif name == "sharding_indexed":
                raise ZarrError("sharding is only supported as the array's single codec")
            else:
                raise ZarrError(f"codec {name!r} is not supported")
        if not seen_bytes and dtype.itemsize > 1:
            raise ZarrError("an array-to-bytes codec ('bytes') is required")
        order = "<" if self.endian == "little" else ">"
        self.stored = dtype.newbyteorder(order) if dtype.itemsize > 1 else dtype

    def encode(self, chunk: np.ndarray) -> bytes:
        data = np.ascontiguousarray(chunk, dtype=self.stored).tobytes()
        for name, config in self.steps:
            if name == "shuffle":
                data = _shuffle(data, config["size"])
            elif name == "gzip":
                data = _gzip(data, config["level"])
            elif name == "zlib":
                data = zlib.compress(data, config["level"])
            elif name == "crc32c":
                data = data + struct.pack("<I", crc32c(data))
        return data

    def decode(self, data: bytes, shape: tuple[int, ...]) -> np.ndarray:
        for name, config in reversed(self.steps):
            if name == "shuffle":
                data = _unshuffle(data, config["size"])
            elif name == "gzip":
                data = zlib.decompress(data, 47)  # 47: gzip or zlib header, detected
            elif name == "zlib":
                data = zlib.decompress(data)
            elif name == "crc32c":
                data = data[:-4]
        values = np.frombuffer(data, self.stored)
        if values.size != math.prod(shape):
            raise ZarrError(f"chunk has {values.size} values, expected {math.prod(shape)}")
        return values.reshape(shape).astype(self.dtype, copy=False)


def codecs_for(
    compression: str | None = "gzip", level: int = 1, shuffle: int | None = 4
) -> list[dict[str, Any]]:
    """The codec list of a chunk: bytes (little endian), shuffle, gzip/zlib."""
    codecs: list[dict[str, Any]] = [{"name": "bytes", "configuration": {"endian": "little"}}]
    if compression and shuffle:
        codecs.append({"name": "numcodecs.shuffle", "configuration": {"elementsize": shuffle}})
    if compression:
        codecs.append({"name": compression, "configuration": {"level": level}})
    return codecs


# ----------------------------------------------------------------- arrays
class _IndexCache:
    """Shard indexes read lately (a probe reads one inner chunk of many shards)."""

    def __init__(self, size: int = 4096):
        self.size = size
        self.items: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            value = self.items.get(key)
            if value is not None:
                self.items.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        with self.lock:
            self.items[key] = value
            if len(self.items) > self.size:
                self.items.popitem(last=False)


_INDEXES = _IndexCache()
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _decoder_pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(min(8, os.cpu_count() or 2), thread_name_prefix="zarr")
        return _POOL


class Array:
    """A Zarr v3 array on disk; ``a[...]`` reads, ``a[...] = v`` writes."""

    def __init__(self, path: str | Path, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.meta = meta if meta is not None else read_json(self.path)
        if self.meta.get("zarr_format") != 3 or self.meta.get("node_type") != "array":
            raise ZarrError(f"{self.path}: not a Zarr v3 array")
        self._parse()

    def _parse(self) -> None:
        meta = self.meta
        name = meta["data_type"]
        if name not in DTYPES:
            raise ZarrError(f"{self.path}: data type {name!r} is not supported")
        self.dtype = DTYPES[name]
        self.shape = tuple(int(n) for n in meta["shape"])
        grid = meta["chunk_grid"]
        if grid.get("name") != "regular":
            raise ZarrError(f"{self.path}: only regular chunk grids are supported")
        self.chunks = tuple(int(n) for n in grid["configuration"]["chunk_shape"])
        encoding = meta.get("chunk_key_encoding", {"name": "default"})
        self._separator = encoding.get("configuration", {}).get(
            "separator", "/" if encoding.get("name", "default") == "default" else "."
        )
        self._v2_keys = encoding.get("name") == "v2"
        self.fill_value = _fill_from_json(meta.get("fill_value", 0), self.dtype)
        codecs = meta.get("codecs", [])
        self.inner: tuple[int, ...] | None = None
        if len(codecs) == 1 and codecs[0]["name"] == "sharding_indexed":
            config = codecs[0]["configuration"]
            self.inner = tuple(int(n) for n in config["chunk_shape"])
            if any(c % i for c, i in zip(self.chunks, self.inner, strict=True)):
                raise ZarrError(f"{self.path}: shard shape is not a multiple of its chunks")
            self.pipeline = Pipeline(config.get("codecs", []), self.dtype)
            index_codecs = [c["name"] for c in config.get("index_codecs", [{"name": "bytes"}])]
            self._index_crc = "crc32c" in index_codecs
            self._index_at_end = config.get("index_location", "end") == "end"
            self._per_shard = tuple(c // i for c, i in zip(self.chunks, self.inner, strict=True))
        else:
            self.pipeline = Pipeline(codecs, self.dtype)

    # ------------------------------------------------------------- creating
    @classmethod
    def create(
        cls,
        path: str | Path,
        shape: Sequence[int],
        dtype: str | np.dtype,
        chunks: Sequence[int] | None = None,
        *,
        inner: Sequence[int] | None = None,
        codecs: list[dict[str, Any]] | None = None,
        fill_value=None,
        dimension_names: Sequence[str] | None = None,
        attributes: dict[str, Any] | None = None,
        index_crc: bool = False,
    ) -> Array:
        """Create (or replace) an array. With ``inner``, ``chunks`` is the shard shape
        and ``inner`` the chunk shape inside it."""
        dtype = np.dtype(dtype)
        name = next((k for k, v in DTYPES.items() if v == dtype), None)
        if name is None:
            raise ZarrError(f"data type {dtype} is not supported")
        shape = [int(n) for n in shape]
        chunks = [int(n) for n in (chunks or [max(n, 1) for n in shape])]
        if fill_value is None:
            fill_value = np.nan if dtype.kind == "f" else 0
        chunk_codecs = codecs if codecs is not None else codecs_for(None)
        if inner is not None:
            inner = [int(n) for n in inner]
            array_codecs = [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": inner,
                        "codecs": chunk_codecs,
                        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
                        + ([{"name": "crc32c"}] if index_crc else []),
                        "index_location": "end",
                    },
                }
            ]
        else:
            array_codecs = chunk_codecs
        meta: dict[str, Any] = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": name,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunks}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": _fill_to_json(fill_value, dtype),
            "codecs": array_codecs,
            "attributes": attributes or {},
        }
        if dimension_names is not None:
            meta["dimension_names"] = list(dimension_names)
        write_json(Path(path), meta)
        return cls(path, meta)

    def resize(self, shape: Sequence[int]) -> None:
        """New shape (chunks beyond it stay on disk and read again if it grows)."""
        self.meta["shape"] = [int(n) for n in shape]
        write_json(self.path, self.meta)
        self.shape = tuple(self.meta["shape"])

    def reload(self) -> None:
        """Read ``zarr.json`` again (another process may have grown the array)."""
        self.meta = read_json(self.path)
        self._parse()

    @property
    def attrs(self) -> dict[str, Any]:
        return self.meta.setdefault("attributes", {})

    @property
    def dimension_names(self) -> list[str] | None:
        return self.meta.get("dimension_names")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # ------------------------------------------------------------- chunk files
    def _key(self, coords: Sequence[int]) -> Path:
        if self._v2_keys:
            name = self._separator.join(str(c) for c in coords) or "0"
            return self.path / name
        if self._separator == "/":
            return self.path.joinpath("c", *[str(c) for c in coords])
        return self.path / self._separator.join(["c", *[str(c) for c in coords]])

    def _read_index(self, file: Path, data: bytes | None = None) -> np.ndarray | None:
        """The shard's index: (*inner chunks per shard, 2) offsets and sizes."""
        count = math.prod(self._per_shard)
        size = 16 * count + (4 if self._index_crc else 0)
        try:
            stat = file.stat()
        except FileNotFoundError:
            return None
        key = (str(file), stat.st_mtime_ns, stat.st_size)
        if data is None:
            cached = _INDEXES.get(key)
            if cached is not None:
                return cached
            with open(file, "rb") as handle:
                if self._index_at_end:
                    handle.seek(stat.st_size - size)
                raw = handle.read(size)
        else:
            raw = data[-size:] if self._index_at_end else data[:size]
        if len(raw) != size:
            raise ZarrError(f"{file}: shard index is cut short")
        if self._index_crc and struct.unpack("<I", raw[-4:])[0] != crc32c(raw[:-4]):
            return None  # being written (the writer appends a new index): not there yet
        index = np.frombuffer(raw[: 16 * count], "<u8").reshape(*self._per_shard, 2)
        _INDEXES.put(key, index)
        return index

    def _read_inner(self, file: Path, index: np.ndarray, local: tuple[int, ...]):
        offset, nbytes = (int(v) for v in index[local])
        if offset == EMPTY and nbytes == EMPTY:
            return None
        with open(file, "rb") as handle:
            handle.seek(offset)
            data = handle.read(nbytes)
        return self.pipeline.decode(data, self.inner)

    def read_chunk(self, coords: Sequence[int]) -> np.ndarray | None:
        """One whole (outer) chunk or shard, decoded; None when it is not written."""
        file = self._key(coords)
        if self.inner is None:
            try:
                data = file.read_bytes()
            except FileNotFoundError:
                return None
            return self.pipeline.decode(data, self.chunks)
        if not file.is_file():
            return None
        data = file.read_bytes()
        index = self._read_index(file, data)
        if index is None:
            return None
        out = np.full(self.chunks, self.fill_value, self.dtype)
        for local in np.ndindex(*self._per_shard):
            offset, nbytes = (int(v) for v in index[local])
            if offset == EMPTY and nbytes == EMPTY:
                continue
            region = self._inner_region(local)
            out[region] = self.pipeline.decode(data[offset : offset + nbytes], self.inner)
        return out

    def _inner_region(self, local: tuple[int, ...]) -> tuple[slice, ...]:
        return tuple(slice(n * i, (n + 1) * i) for n, i in zip(local, self.inner, strict=True))

    def write_chunk(self, coords: Sequence[int], chunk: np.ndarray) -> None:
        """Write one whole (outer) chunk or shard; in a shard, inner chunks that are
        all fill values (or lie outside the array) are left out."""
        chunk = np.asarray(chunk, dtype=self.dtype)
        if chunk.shape != self.chunks:
            raise ZarrError(f"chunk shape {chunk.shape} is not {self.chunks}")
        file = self._key(coords)
        file.parent.mkdir(parents=True, exist_ok=True)
        if self.inner is None:
            _atomic_write(file, self.pipeline.encode(chunk))
            return
        start = [c * s for c, s in zip(coords, self.chunks, strict=True)]
        index = np.full((*self._per_shard, 2), EMPTY, "<u8")
        parts: list[bytes] = []
        offset = 0
        for local in np.ndindex(*self._per_shard):
            low = [s + n * i for s, n, i in zip(start, local, self.inner, strict=True)]
            if any(lo >= n for lo, n in zip(low, self.shape, strict=True)):
                continue  # outside the array
            region = self._inner_region(local)
            part = chunk[region]
            if _all_fill(part, self.fill_value):
                continue
            data = self.pipeline.encode(part)
            index[local] = (offset, len(data))
            parts.append(data)
            offset += len(data)
        if not parts:
            file.unlink(missing_ok=True)
            return
        raw = index.tobytes()
        parts.append(raw)
        if self._index_crc:
            parts.append(struct.pack("<I", crc32c(raw)))
        _atomic_write(file, b"".join(parts))

    # ------------------------------------------------------------- selections
    def _region(self, key) -> tuple[list[slice], list[int]]:
        """Slices (step 1) per dimension and the dimensions dropped by an integer."""
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            at = key.index(Ellipsis)
            key = key[:at] + (slice(None),) * (self.ndim - len(key) + 1) + key[at + 1 :]
        key = key + (slice(None),) * (self.ndim - len(key))
        if len(key) != self.ndim:
            raise IndexError(f"{len(key)} indices for an array with {self.ndim} dimensions")
        slices, dropped = [], []
        for dim, (k, n) in enumerate(zip(key, self.shape, strict=True)):
            if isinstance(k, (int, np.integer)):
                k = int(k) + (n if k < 0 else 0)
                if not 0 <= k < n:
                    raise IndexError(f"index {k} out of range 0..{n - 1}")
                slices.append(slice(k, k + 1))
                dropped.append(dim)
            elif isinstance(k, slice):
                lo, hi, step = k.indices(n)
                if step != 1:
                    raise IndexError("only slices with step 1 are supported")
                slices.append(slice(lo, max(lo, hi)))
            else:
                raise IndexError(f"unsupported index {k!r}")
        return slices, dropped

    def __getitem__(self, key) -> np.ndarray:
        slices, dropped = self._region(key)
        out = np.full([s.stop - s.start for s in slices], self.fill_value, self.dtype)
        if out.size == 0:
            return out.reshape([n for d, n in enumerate(out.shape) if d not in dropped])
        if self.inner is None:
            self._gather(out, slices, self.chunks, self._read_plain_chunk)
        else:
            self._gather_sharded(out, slices)
        return out.reshape([n for d, n in enumerate(out.shape) if d not in dropped])

    def _gather_sharded(self, out: np.ndarray, slices: list[slice]) -> None:
        """Read the inner chunks a selection needs: per shard file one read of the
        span that holds them (a writer puts the chunks of an output together), then
        decode them, in parallel for big selections."""
        size = self.inner
        ranges = [
            range(s.start // c, (s.stop - 1) // c + 1) for s, c in zip(slices, size, strict=True)
        ]
        by_shard: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        for at in np.ndindex(*[len(r) for r in ranges]):
            coords = tuple(r[i] for r, i in zip(ranges, at, strict=True))
            shard = tuple(c // p for c, p in zip(coords, self._per_shard, strict=True))
            by_shard.setdefault(shard, []).append(coords)
        jobs = []  # (coords, raw bytes)
        for shard, wanted in by_shard.items():
            file = self._key(shard)
            index = self._read_index(file)
            if index is None:
                continue
            entries = []
            for coords in wanted:
                local = tuple(c % p for c, p in zip(coords, self._per_shard, strict=True))
                offset, nbytes = (int(v) for v in index[local])
                if not (offset == EMPTY and nbytes == EMPTY):
                    entries.append((coords, offset, nbytes))
            if not entries:
                continue
            low = min(e[1] for e in entries)
            high = max(e[1] + e[2] for e in entries)
            needed = sum(e[2] for e in entries)
            with open(file, "rb") as handle:
                if high - low <= 2 * needed + (1 << 16):  # one read for all of them
                    handle.seek(low)
                    span = handle.read(high - low)
                    jobs += [(c, span[o - low : o - low + n]) for c, o, n in entries]
                else:
                    for c, o, n in entries:
                        handle.seek(o)
                        jobs.append((c, handle.read(n)))

        def place(job) -> None:
            coords, raw = job
            chunk = self.pipeline.decode(raw, size)
            src, dst = [], []
            for c, s, n in zip(coords, slices, size, strict=True):
                lo, hi = max(s.start, c * n), min(s.stop, (c + 1) * n)
                src.append(slice(lo - c * n, hi - c * n))
                dst.append(slice(lo - s.start, hi - s.start))
            out[tuple(dst)] = chunk[tuple(src)]

        if len(jobs) > 1 and out.nbytes > (1 << 18):
            list(_decoder_pool().map(place, jobs))
        else:
            for job in jobs:
                place(job)

    def _read_plain_chunk(self, coords: tuple[int, ...]) -> np.ndarray | None:
        return self.read_chunk(coords)

    def _read_sharded_chunk(self, coords: tuple[int, ...]) -> np.ndarray | None:
        shard = tuple(c // p for c, p in zip(coords, self._per_shard, strict=True))
        local = tuple(c % p for c, p in zip(coords, self._per_shard, strict=True))
        file = self._key(shard)
        index = self._read_index(file)
        if index is None:
            return None
        return self._read_inner(file, index, local)

    def _gather(self, out: np.ndarray, slices: list[slice], size: tuple[int, ...], read) -> None:
        ranges = [
            range(s.start // c, (s.stop - 1) // c + 1) for s, c in zip(slices, size, strict=True)
        ]
        wanted = [
            tuple(r[i] for r, i in zip(ranges, coords, strict=True))
            for coords in np.ndindex(*[len(r) for r in ranges])
        ]

        def place(chunk_coords) -> None:
            chunk = read(chunk_coords)
            if chunk is None:
                return
            src, dst = [], []
            for c, s, n in zip(chunk_coords, slices, size, strict=True):
                lo, hi = max(s.start, c * n), min(s.stop, (c + 1) * n)
                src.append(slice(lo - c * n, hi - c * n))
                dst.append(slice(lo - s.start, hi - s.start))
            out[tuple(dst)] = chunk[tuple(src)]

        # zlib and NumPy copies release the GIL: big reads decode their chunks in parallel
        if len(wanted) > 1 and out.nbytes > (1 << 18):
            list(_decoder_pool().map(place, wanted))
        else:
            for chunk_coords in wanted:
                place(chunk_coords)

    def __setitem__(self, key, value) -> None:
        slices, _ = self._region(key)
        shape = [s.stop - s.start for s in slices]
        value = np.broadcast_to(np.asarray(value, dtype=self.dtype), shape) if shape else value
        if not shape or math.prod(shape) == 0:
            return
        ranges = [
            range(s.start // c, (s.stop - 1) // c + 1)
            for s, c in zip(slices, self.chunks, strict=True)
        ]
        for coords in np.ndindex(*[len(r) for r in ranges]):
            chunk_coords = tuple(r[i] for r, i in zip(ranges, coords, strict=True))
            src, dst, whole = [], [], True
            for c, s, n, total in zip(chunk_coords, slices, self.chunks, self.shape, strict=True):
                lo, hi = max(s.start, c * n), min(s.stop, (c + 1) * n)
                src.append(slice(lo - s.start, hi - s.start))
                dst.append(slice(lo - c * n, hi - c * n))
                # the whole chunk, as far as it lies inside the array
                whole &= lo == c * n and hi >= min((c + 1) * n, total)
            chunk = None if whole else self.read_chunk(chunk_coords)
            if chunk is None:
                chunk = np.full(self.chunks, self.fill_value, self.dtype)
            else:
                chunk = np.array(chunk)  # writable
            chunk[tuple(dst)] = value[tuple(src)]
            self.write_chunk(chunk_coords, chunk)


def _all_fill(part: np.ndarray, fill) -> bool:
    if part.dtype.kind == "f" and np.isnan(fill):
        return bool(np.isnan(part).all())
    return bool((part == fill).all())
