"""Small synthetic REEF3D runs for tests, written like REEF3D writes them.

Each MPI rank writes its own piece; neighbouring ranks both write the points on
their shared boundary. Files use raw appended data by default, and optionally the
inline formats ParaView writes (base64 binary, zlib-compressed, ASCII).
"""

from __future__ import annotations

import base64
import zlib
from pathlib import Path

import numpy as np

TYPE_NAMES = {
    np.dtype(np.float32): "Float32",
    np.dtype(np.float64): "Float64",
    np.dtype(np.int32): "Int32",
    np.dtype(np.int64): "Int64",
    np.dtype(np.uint8): "UInt8",
}


def _encode(array: np.ndarray, fmt: str, compressed: bool) -> str:
    raw = np.ascontiguousarray(array).tobytes()
    if fmt == "ascii":
        return " ".join(str(v) for v in np.asarray(array).ravel())
    if compressed:
        body = zlib.compress(raw)
        head = np.array([1, len(raw), len(raw), len(body)], np.uint32).tobytes()
        return base64.b64encode(head).decode() + base64.b64encode(body).decode()
    return base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode()


def write_piece(
    path: Path,
    kind: str,
    points: np.ndarray,
    point_data: dict[str, np.ndarray],
    connectivity: np.ndarray,
    offsets: np.ndarray,
    types: np.ndarray | None = None,
    fmt: str = "appended",
    compressed: bool = False,
    bad_count: str | None = None,
) -> None:
    """Write a .vtu (kind="UnstructuredGrid") or .vtp (kind="PolyData") file.

    ``bad_count`` names an array whose block header claims three times its size,
    like REEF3D::FNPF's ``elevation`` array.
    """
    arrays: list[tuple[str, str, np.ndarray]] = [("Points", "", points.astype(np.float32))]
    arrays += [("PointData", name, values) for name, values in point_data.items()]
    cell_section = "Cells" if kind == "UnstructuredGrid" else "Polys"
    arrays += [
        (cell_section, "connectivity", connectivity.astype(np.int32)),
        (cell_section, "offsets", offsets.astype(np.int32)),
    ]
    if types is not None:
        arrays.append((cell_section, "types", types.astype(np.uint8)))

    blobs = []
    offset = 0
    elements: dict[str, list[str]] = {"Points": [], "PointData": [], cell_section: []}
    for section, name, values in arrays:
        values = np.ascontiguousarray(values)
        components = values.shape[1] if values.ndim == 2 else 1
        attrs = f'type="{TYPE_NAMES[values.dtype]}"'
        if name:
            attrs += f' Name="{name}"'
        if components > 1:
            attrs += f' NumberOfComponents="{components}"'
        if fmt == "appended":
            elements[section].append(f'<DataArray {attrs} format="appended" offset="{offset}"/>')
            count = values.nbytes * (3 if name == bad_count else 1)
            blob = np.uint32(count).tobytes() + values.tobytes()
            blobs.append(blob)
            offset += len(blob)
        else:
            text = _encode(values, fmt, compressed)
            elements[section].append(f'<DataArray {attrs} format="{fmt}">{text}</DataArray>')

    n_cells = len(offsets)
    if kind == "UnstructuredGrid":
        sizes = f'NumberOfPoints="{len(points)}" NumberOfCells="{n_cells}"'
    else:
        sizes = (
            f'NumberOfPoints="{len(points)}" NumberOfVerts="0" NumberOfLines="0" '
            f'NumberOfStrips="0" NumberOfPolys="{n_cells}"'
        )
    compressor = ' compressor="vtkZLibDataCompressor"' if compressed else ""
    head = (
        '<?xml version="1.0"?>\n'
        f'<VTKFile type="{kind}" version="1.0" byte_order="LittleEndian"{compressor}>\n'
        f"<{kind}>\n<Piece {sizes}>\n"
        f"<PointData>\n{chr(10).join(elements['PointData'])}\n</PointData>\n"
        f"<Points>\n{elements['Points'][0]}\n</Points>\n"
        f"<{cell_section}>\n{chr(10).join(elements[cell_section])}\n</{cell_section}>\n"
        f"</Piece>\n</{kind}>\n"
    )
    with open(path, "wb") as handle:
        handle.write(head.encode())
        if fmt == "appended":
            handle.write(b'<AppendedData encoding="raw">\n_')
            for blob in blobs:
                handle.write(blob)
            handle.write(b"\n</AppendedData>\n")
        handle.write(b"</VTKFile>\n")


def write_parallel(path: Path, kind: str, pieces: list[str], point_data: dict[str, int]) -> None:
    pkind = f"P{kind}"
    arrays = "\n".join(
        f'<PDataArray type="Float32" Name="{n}"'
        + (f' NumberOfComponents="{c}"' if c > 1 else "")
        + "/>"
        for n, c in point_data.items()
    )
    sources = "\n".join(f'<Piece Source="{p}"/>' for p in pieces)
    path.write_text(
        '<?xml version="1.0"?>\n'
        f'<VTKFile type="{pkind}" version="1.0" byte_order="LittleEndian">\n'
        f'<{pkind} GhostLevel="0">\n<PPoints>\n<PDataArray type="Float32" '
        'NumberOfComponents="3"/>\n</PPoints>\n'
        f"<PPointData>\n{arrays}\n</PPointData>\n{sources}\n</{pkind}>\n</VTKFile>\n"
    )


# ----------------------------------------------------------------- a small run
NX, NY = 9, 5  # points per direction of the whole grid
DX = 1.0
DEPTH = 2.0


def eta(x: np.ndarray, y: np.ndarray, frame: int) -> np.ndarray:
    return (0.1 * np.sin(x - 0.5 * frame) + 0.02 * y).astype(np.float32)


def _surface_piece(i0: int, i1: int, frame: int, role: str):
    """Grid points i0..i1 (inclusive) in x, all of y; triangles like REEF3D."""
    xs = np.arange(i0, i1 + 1) * DX
    ys = np.arange(NY) * DX
    x, y = np.meshgrid(xs, ys, indexing="ij")
    x, y = x.ravel(), y.ravel()
    if role == "fsf":
        e = eta(x, y, frame)
        z = DEPTH + e
        velocity = np.stack([np.cos(x), 0 * x, 0.1 * e], axis=1).astype(np.float32)
        data = {"velocity": velocity, "eta": e}
    else:
        z = (0.1 * x).astype(np.float32)
        data = {"elevation": z.astype(np.float32), "depth": (DEPTH - z).astype(np.float32)}
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    nx = i1 - i0 + 1
    quads = []
    for i in range(nx - 1):
        for j in range(NY - 1):
            a = i * NY + j
            quads.append((a, a + NY, a + NY + 1, a + 1))
    quads = np.array(quads)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    offsets = np.arange(1, len(tris) + 1) * 3
    return points, data, tris.ravel(), offsets


def _volume_piece(i0: int, i1: int, frame: int, nz: int = 3):
    xs = np.arange(i0, i1 + 1) * DX
    ys = np.arange(NY) * DX
    sig = np.linspace(0.0, 1.0, nz)
    x, y, s = np.meshgrid(xs, ys, sig, indexing="ij")
    e = eta(x, y, frame)
    z = s * (DEPTH + e)
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
    nx = i1 - i0 + 1

    def node(i, j, k):
        return (i * NY + j) * nz + k

    cells = []
    for i in range(nx - 1):
        for j in range(NY - 1):
            for k in range(nz - 1):
                cells.append(
                    [
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    ]  # fmt: skip
                )
    cells = np.array(cells)
    velocity = np.stack([np.cos(points[:, 0]), 0 * points[:, 0], points[:, 2]], axis=1)
    data = {
        "velocity": velocity.astype(np.float32),
        "pressure": (1000 * 9.81 * (DEPTH - points[:, 2])).astype(np.float32),
    }
    offsets = np.arange(1, len(cells) + 1) * 8
    types = np.full(len(cells), 12, np.uint8)
    return points, data, cells.ravel(), offsets, types


def make_run(
    root: Path,
    solver: str = "NHFLOW",
    frames: int = 3,
    ranks: int = 2,
    volume: bool = True,
    print_interval: float | None = None,
) -> Path:
    """Write a REEF3D-like case folder with free surface, bed and volume output."""
    root.mkdir(parents=True, exist_ok=True)
    if print_interval is not None:
        (root / "ctrl.txt").write_text(f"N 40 3\nP 30 {print_interval}\nP 10 1\n")
    # split x points among ranks; neighbours share one column of points
    cuts = np.linspace(0, NX - 1, ranks + 1).round().astype(int)
    outputs = [("VTP_FSF", "FSF", "fsf"), ("VTP_BED", "BED", "bed")]
    if volume:
        outputs.append(("VTU", "", "volume"))
    for folder_suffix, tag, role in outputs:
        folder = root / f"REEF3D_{solver}_{folder_suffix}"
        folder.mkdir(exist_ok=True)
        stem = f"REEF3D-{solver}" + (f"-{tag}" if tag else "")
        ext = "vtu" if role == "volume" else "vtp"
        kind = "UnstructuredGrid" if role == "volume" else "PolyData"
        for frame in range(frames):
            names = []
            for rank in range(ranks):
                i0, i1 = cuts[rank], cuts[rank + 1]
                name = f"{stem}-{frame:08d}-{rank + 1:06d}.{ext}"
                if role == "volume":
                    points, data, conn, offs, types = _volume_piece(i0, i1, frame)
                else:
                    points, data, conn, offs = _surface_piece(i0, i1, frame, role)
                    types = None
                write_piece(folder / name, kind, points, data, conn, offs, types)
                names.append(name)
            components = {n: (v.shape[1] if v.ndim == 2 else 1) for n, v in data.items()}
            write_parallel(folder / f"{stem}-{frame:08d}.p{ext}", kind, names, components)
    return root
