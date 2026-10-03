# SPDX-License-Identifier: GPL-3.0-or-later
"""Fast reader for VTK XML files (.vtu, .vtp) and their parallel forms (.pvtu, .pvtp).

Written for REEF3D output, which uses raw appended binary data, but it also reads
inline base64 and ASCII arrays and zlib-compressed data as written by ParaView.
Only NumPy is needed. Arrays are read on demand, so asking for one variable of a
large file reads only that variable from disk.
"""

from __future__ import annotations

import base64
import re
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VTK_TYPES = {
    "Int8": np.int8,
    "UInt8": np.uint8,
    "Int16": np.int16,
    "UInt16": np.uint16,
    "Int32": np.int32,
    "UInt32": np.uint32,
    "Int64": np.int64,
    "UInt64": np.uint64,
    "Float32": np.float32,
    "Float64": np.float64,
}

#: VTK cell type ids used by REEF3D and their number of nodes
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_TETRA = 10
VTK_QUAD = 9
VTK_TRIANGLE = 5

_HEADER_CHUNK = 1 << 16


class VTKFormatError(ValueError):
    """The file is not a VTK XML file this reader understands."""


@dataclass(frozen=True)
class ArrayInfo:
    """Where one data array lives in a file."""

    name: str
    dtype: np.dtype
    components: int
    format: str  # "appended", "binary" or "ascii"
    offset: int | None = None  # appended: byte offset after the "_" marker
    text: str | None = None  # binary or ascii: the element text


@dataclass
class PieceHeader:
    """The XML part of a serial VTK XML file: sizes and array locations."""

    path: Path
    kind: str  # "UnstructuredGrid" or "PolyData"
    n_points: int
    n_cells: int  # cells (UnstructuredGrid) or polygons (PolyData)
    points: ArrayInfo | None
    point_data: dict[str, ArrayInfo] = field(default_factory=dict)
    cell_data: dict[str, ArrayInfo] = field(default_factory=dict)
    cells: dict[str, ArrayInfo] = field(default_factory=dict)  # connectivity, offsets, types
    header_dtype: np.dtype = field(default_factory=lambda: np.dtype("<u4"))
    byte_order: str = "<"
    compressed: bool = False
    data_start: int | None = None  # file offset of the first appended byte
    appended_encoding: str = "raw"


@dataclass
class ParallelHeader:
    """A .pvtu or .pvtp file: the pieces and the arrays they hold."""

    path: Path
    kind: str  # "UnstructuredGrid" or "PolyData"
    pieces: list[Path]
    point_data: dict[str, int]  # name -> components
    cell_data: dict[str, int]


# ----------------------------------------------------------------- headers
def _read_xml_head(path: Path) -> tuple[str, int | None]:
    """Return the XML text before the appended data and the data start offset."""
    with open(path, "rb") as handle:
        buffer = b""
        while True:
            chunk = handle.read(_HEADER_CHUNK)
            buffer += chunk
            marker = buffer.find(b"<AppendedData")
            if marker >= 0:
                # data starts after the "_" that follows the AppendedData tag
                tag_end = buffer.find(b">", marker)
                while tag_end < 0 or buffer.find(b"_", tag_end) < 0:
                    more = handle.read(_HEADER_CHUNK)
                    if not more:
                        raise VTKFormatError(f"{path}: truncated AppendedData section")
                    buffer += more
                    tag_end = buffer.find(b">", marker)
                underscore = buffer.find(b"_", tag_end)
                tag = buffer[marker : tag_end + 1].decode("ascii", "replace")
                head = buffer[:marker].decode("utf-8", "replace") + tag + "</AppendedData>"
                return head + "</VTKFile>", underscore + 1
            if not chunk:
                return buffer.decode("utf-8", "replace"), None


def _array_info(element: ET.Element, byte_order: str) -> ArrayInfo:
    type_name = element.get("type", "Float32")
    if type_name not in VTK_TYPES:
        raise VTKFormatError(f"unsupported data type {type_name!r}")
    dtype = np.dtype(VTK_TYPES[type_name]).newbyteorder(byte_order)
    fmt = element.get("format", "ascii")
    offset = element.get("offset")
    return ArrayInfo(
        name=element.get("Name", ""),
        dtype=dtype,
        components=int(element.get("NumberOfComponents", "1")),
        format=fmt,
        offset=int(offset) if offset is not None else None,
        text=None if fmt == "appended" else (element.text or ""),
    )


def read_header(path: str | Path) -> PieceHeader:
    """Parse the XML header of a .vtu or .vtp file (does not read the data)."""
    path = Path(path)
    text, data_start = _read_xml_head(path)
    # an unclosed head (no appended data) is a complete document already
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise VTKFormatError(f"{path}: not a VTK XML file ({error})") from None
    if root.tag != "VTKFile":
        raise VTKFormatError(f"{path}: not a VTK XML file")
    kind = root.get("type", "")
    if kind not in ("UnstructuredGrid", "PolyData"):
        raise VTKFormatError(f"{path}: unsupported VTK file type {kind!r}")
    byte_order = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    header_type = root.get("header_type", "UInt32")
    header_dtype = np.dtype(VTK_TYPES[header_type]).newbyteorder(byte_order)
    compressed = bool(root.get("compressor"))

    pieces = root.findall(f"{kind}/Piece")
    if len(pieces) != 1:
        raise VTKFormatError(f"{path}: expected one Piece, found {len(pieces)}")
    piece = pieces[0]
    n_points = int(piece.get("NumberOfPoints", "0"))
    if kind == "UnstructuredGrid":
        n_cells = int(piece.get("NumberOfCells", "0"))
        cell_tag = "Cells"
    else:
        n_cells = int(piece.get("NumberOfPolys", "0"))
        cell_tag = "Polys"

    points_el = piece.find("Points/DataArray")
    header = PieceHeader(
        path=path,
        kind=kind,
        n_points=n_points,
        n_cells=n_cells,
        points=_array_info(points_el, byte_order) if points_el is not None else None,
        header_dtype=header_dtype,
        byte_order=byte_order,
        compressed=compressed,
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
    for element in piece.findall(f"{cell_tag}/DataArray"):
        info = _array_info(element, byte_order)
        header.cells[info.name] = info
    return header


def read_parallel_header(path: str | Path) -> ParallelHeader:
    """Parse a .pvtu or .pvtp file."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise VTKFormatError(f"{path}: not a VTK XML file ({error})") from None
    kind = root.get("type", "")
    if kind not in ("PUnstructuredGrid", "PPolyData"):
        raise VTKFormatError(f"{path}: unsupported parallel file type {kind!r}")
    grid = root.find(kind)
    if grid is None:
        raise VTKFormatError(f"{path}: missing {kind} element")
    pieces = [path.parent / p.get("Source", "") for p in grid.findall("Piece")]
    point_data = {
        a.get("Name", ""): int(a.get("NumberOfComponents", "1"))
        for a in grid.findall("PPointData/PDataArray")
    }
    cell_data = {
        a.get("Name", ""): int(a.get("NumberOfComponents", "1"))
        for a in grid.findall("PCellData/PDataArray")
    }
    return ParallelHeader(path, kind[1:], pieces, point_data, cell_data)


# ----------------------------------------------------------------- arrays
def _decompress(blob: bytes, header_dtype: np.dtype) -> bytes:
    """Undo vtkZLibDataCompressor block compression."""
    size = header_dtype.itemsize
    n_blocks = int(np.frombuffer(blob, header_dtype, 1, 0)[0])
    sizes = np.frombuffer(blob, header_dtype, 3 + n_blocks, 0)
    compressed_sizes = sizes[3:].astype(np.int64)
    position = size * (3 + n_blocks)
    parts = []
    for length in compressed_sizes:
        parts.append(zlib.decompress(blob[position : position + length]))
        position += int(length)
    return b"".join(parts)


def _read_appended(header: PieceHeader, info: ArrayInfo, count: int) -> np.ndarray:
    assert header.data_start is not None and info.offset is not None
    if header.appended_encoding != "raw":
        raise VTKFormatError(f"{header.path}: base64 appended data is not supported")
    hsize = header.header_dtype.itemsize
    with open(header.path, "rb") as handle:
        handle.seek(header.data_start + info.offset)
        if header.compressed:
            first = np.frombuffer(handle.read(hsize), header.header_dtype)[0]
            rest = np.frombuffer(handle.read(hsize * (2 + int(first))), header.header_dtype)
            total = int(rest[2:].astype(np.int64).sum())
            blob = first.tobytes() + rest.tobytes() + handle.read(total)
            return np.frombuffer(_decompress(blob, header.header_dtype), info.dtype)
        nbytes = int(np.frombuffer(handle.read(hsize), header.header_dtype)[0])
        expected = count * info.dtype.itemsize
        # A larger byte count than the XML implies is tolerated, like VTK does:
        # REEF3D::FNPF (hans_dev, Sep 2026) writes the vector size in the block
        # header of its scalar "elevation" array, while the data itself is right.
        if nbytes < expected:
            raise VTKFormatError(
                f"{header.path}: array {info.name or 'points'!r} has {nbytes} bytes, "
                f"expected {expected}"
            )
        data = np.fromfile(handle, dtype=info.dtype, count=count)
    if data.size != count:
        raise VTKFormatError(f"{header.path}: file is truncated")
    return data


def _read_inline(header: PieceHeader, info: ArrayInfo) -> np.ndarray:
    text = (info.text or "").strip()
    if info.format == "ascii":
        return np.array(text.split(), dtype=info.dtype)
    if info.format != "binary":
        raise VTKFormatError(f"{header.path}: unknown array format {info.format!r}")
    hsize = header.header_dtype.itemsize
    if header.compressed:
        # header and data are encoded separately
        n_blocks_b64 = base64.b64decode(text[: (hsize * 4 + 2) // 3 * 4])
        n_blocks = int(np.frombuffer(n_blocks_b64[:hsize], header.header_dtype)[0])
        head_len = (hsize * (3 + n_blocks) + 2) // 3 * 4
        head = base64.b64decode(text[:head_len])[: hsize * (3 + n_blocks)]
        body = base64.b64decode(text[head_len:])
        return np.frombuffer(_decompress(head + body, header.header_dtype), info.dtype)
    raw = base64.b64decode(text)
    nbytes = int(np.frombuffer(raw[:hsize], header.header_dtype)[0])
    return np.frombuffer(raw[hsize : hsize + nbytes], info.dtype)


def read_array(header: PieceHeader, info: ArrayInfo, count: int) -> np.ndarray:
    """Read one array. ``count`` is the number of tuples (points, cells, ...)."""
    total = count * info.components
    if info.format == "appended":
        data = _read_appended(header, info, total)
    else:
        data = _read_inline(header, info)
    if data.size != total:
        raise VTKFormatError(
            f"{header.path}: array {info.name!r} has {data.size} values, expected {total}"
        )
    data = data.astype(data.dtype.newbyteorder("="), copy=False)
    return data.reshape(count, info.components) if info.components > 1 else data


def _cell_array(header: PieceHeader, name: str) -> np.ndarray:
    """Connectivity (length unknown up front), offsets or types."""
    info = header.cells.get(name)
    if info is None:
        raise VTKFormatError(f"{header.path}: missing {name} array")
    if name in ("offsets", "types"):
        return read_array(header, info, header.n_cells)
    # connectivity: its length is the last offset
    offsets = _cell_array(header, "offsets")
    count = int(offsets[-1]) if offsets.size else 0
    return read_array(header, info, count)


# ----------------------------------------------------------------- pieces
@dataclass
class Piece:
    """The data of one serial file, as plain NumPy arrays."""

    kind: str
    points: np.ndarray  # (n, 3) float32
    connectivity: np.ndarray
    offsets: np.ndarray
    types: np.ndarray | None  # UnstructuredGrid only
    point_data: dict[str, np.ndarray]


def read_piece(
    path: str | Path,
    point_arrays: list[str] | None = None,
    header: PieceHeader | None = None,
    with_cells: bool = True,
) -> Piece:
    """Read a .vtu or .vtp file. ``point_arrays=None`` reads all point data arrays."""
    header = header or read_header(path)
    if header.points is None:
        points = np.zeros((0, 3), np.float32)
    else:
        points = read_array(header, header.points, header.n_points)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
    names = list(header.point_data) if point_arrays is None else point_arrays
    data = {}
    for name in names:
        info = header.point_data.get(name)
        if info is None:
            raise KeyError(f"{header.path.name}: no point array {name!r}")
        data[name] = read_array(header, info, header.n_points)
    if with_cells and header.n_cells:
        offsets = _cell_array(header, "offsets").astype(np.int64)
        connectivity = read_array(header, header.cells["connectivity"], int(offsets[-1])).astype(
            np.int64
        )
        types = _cell_array(header, "types") if header.kind == "UnstructuredGrid" else None
    else:
        offsets = np.zeros(0, np.int64)
        connectivity = np.zeros(0, np.int64)
        types = np.zeros(0, np.uint8) if header.kind == "UnstructuredGrid" else None
    return Piece(header.kind, points, connectivity, offsets, types, data)


_PIECE_NAME = re.compile(r"^(?P<stem>.+?)-(?P<step>\d+)(?:-(?P<rank>\d+))?$")


_TIME_VALUE = re.compile(rb'Name="TimeValue"[^>]*>\s*([-+0-9.eE]+)\s*<')


def read_time_value(path: str | Path, head: int = 4096) -> float | None:
    """The simulation time in a VTK XML file's ``TimeValue`` field data, or None.

    REEF3D writes it before the data, so only the start of the file is read.
    """
    try:
        with open(path, "rb") as file:
            match = _TIME_VALUE.search(file.read(head))
    except OSError:
        return None
    return float(match.group(1)) if match else None


def split_name(path: str | Path) -> tuple[str, int | None, int | None]:
    """Split ``REEF3D-NHFLOW-FSF-00000696-000001.vtp`` into (stem, step, rank)."""
    match = _PIECE_NAME.match(Path(path).stem)
    if not match:
        return Path(path).stem, None, None
    rank = match.group("rank")
    return match.group("stem"), int(match.group("step")), int(rank) if rank else None


# ----------------------------------------------------------------- legacy .vtk
def read_legacy(path: str | Path, point_arrays: list[str] | None = None) -> Piece:
    """Read an ASCII legacy VTK POLYDATA file (points, lines, polygons, point data).

    REEF3D writes mooring lines this way. Lines are returned as two-node segments
    in ``connectivity``/``offsets`` with ``kind="Lines"``; polygons as ``"PolyData"``.
    """
    path = Path(path)
    tokens = path.read_text(errors="replace").split()
    position = 0

    def take(n: int) -> list[str]:
        nonlocal position
        chunk = tokens[position : position + n]
        position += n
        return chunk

    points = np.zeros((0, 3), np.float32)
    segments: list[np.ndarray] = []
    polys_conn: list[int] = []
    polys_off: list[int] = []
    data: dict[str, np.ndarray] = {}
    n_points = 0
    try:
        start = tokens.index("POINTS")
    except ValueError:
        raise VTKFormatError(f"{path}: no POINTS in legacy VTK file") from None
    if "POLYDATA" not in tokens[:start]:
        raise VTKFormatError(f"{path}: only legacy POLYDATA files are supported")
    position = start
    while position < len(tokens):
        word = tokens[position].upper()
        position += 1
        if word == "POINTS":
            n_points = int(take(2)[0])
            points = np.array(take(3 * n_points), dtype=np.float32).reshape(-1, 3)
        elif word in ("LINES", "POLYGONS", "VERTICES", "TRIANGLE_STRIPS"):
            n_cells, size = (int(v) for v in take(2))
            values = np.array(take(size), dtype=np.int64)
            index = 0
            for _ in range(n_cells):
                count = int(values[index])
                cell = values[index + 1 : index + 1 + count]
                index += 1 + count
                if word == "LINES":
                    segments.append(np.stack([cell[:-1], cell[1:]], axis=1))
                elif word == "POLYGONS":
                    polys_conn.extend(cell.tolist())
                    polys_off.append(len(polys_conn))
        elif word == "POINT_DATA":
            take(1)
        elif word == "SCALARS":
            name, _dtype = take(2)
            components = 1
            if position < len(tokens) and tokens[position].isdigit():
                components = int(take(1)[0])
            if position < len(tokens) and tokens[position].upper() == "LOOKUP_TABLE":
                take(2)
            values = np.array(take(n_points * components), dtype=np.float32)
            data[name] = values.reshape(n_points, components) if components > 1 else values
        elif word in ("VECTORS", "NORMALS"):
            name, _dtype = take(2)
            data[name] = np.array(take(3 * n_points), dtype=np.float32).reshape(n_points, 3)
        elif word in ("CELL_DATA", "FIELD"):
            break  # cell data is not used
    if point_arrays is not None:
        data = {name: values for name, values in data.items() if name in point_arrays}
    if segments:
        lines = np.concatenate(segments)
        return Piece(
            "Lines", points, lines.ravel(), np.arange(2, 2 * len(lines) + 1, 2), None, data
        )
    return Piece(
        "PolyData",
        points,
        np.array(polys_conn, np.int64),
        np.array(polys_off, np.int64),
        None,
        data,
    )


def legacy_point_arrays(path: str | Path) -> dict[str, int]:
    """Names and components of the point arrays of a legacy file (reads the file)."""
    return {
        name: (values.shape[1] if values.ndim == 2 else 1)
        for name, values in read_legacy(path).point_data.items()
    }
