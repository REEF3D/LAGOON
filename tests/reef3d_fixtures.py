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


def _volume_piece(i0: int, i1: int, frame: int, nz: int = 3, cartesian: bool = False):
    xs = np.arange(i0, i1 + 1) * DX
    ys = np.arange(NY) * DX
    sig = np.linspace(0.0, 1.0, nz)
    x, y, s = np.meshgrid(xs, ys, sig, indexing="ij")
    e = eta(x, y, frame)
    z = s * (DEPTH + 1.0) if cartesian else s * (DEPTH + e)  # CFD: fixed levels
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
    if cartesian:  # CFD: the level set, the distance below the free surface
        data["phi"] = (DEPTH + e.ravel() - points[:, 2]).astype(np.float32)
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
    series: bool = False,
    objects: bool = False,
    profiles: bool = False,
    dem: float | None = None,
    ice: float | None = None,
    rods: float | None = None,
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
                    points, data, conn, offs, types = _volume_piece(
                        i0, i1, frame, cartesian=solver == "CFD"
                    )
                else:
                    points, data, conn, offs = _surface_piece(i0, i1, frame, role)
                    types = None
                write_piece(folder / name, kind, points, data, conn, offs, types)
                names.append(name)
            components = {n: (v.shape[1] if v.ndim == 2 else 1) for n, v in data.items()}
            write_parallel(folder / f"{stem}-{frame:08d}.p{ext}", kind, names, components)
    if series:
        write_series(root, solver, (frames - 1) * (print_interval or 1.0))
    if objects:
        write_objects(root, solver, frames)
    if profiles:
        write_profiles(root, solver, (frames - 1) * (print_interval or 1.0))
    if dem is not None:  # DEM output every ``dem`` seconds over the same span
        write_dem(root, (frames - 1) * (print_interval or 1.0), dem)
    if ice is not None:
        write_ice(root, (frames - 1) * (print_interval or 1.0), ice, solver=solver)
    if rods is not None:
        write_rodtree(root, (frames - 1) * (print_interval or 1.0), rods)
    return root


#: DEM elements of the test case: two cubes of 0.2 m that fall towards the bed
DEM_START = ((2.0, 1.0, 3.0), (5.0, 3.0, 3.5))


def dem_centres(t: float) -> np.ndarray:
    return np.array(DEM_START) - np.array([0.0, 0.0, 0.5 * t])


def write_dem(root: Path, end: float, interval: float) -> None:
    """DEM output as REEF3D's DEM writes it (dem_f_print.cpp): one serial ASCII
    PolyData file per output in REEF3D_DEM_VTP, numbered from 0 with six digits,
    the simulation time in its TimeValue field data, triangles of every element."""
    folder = root / "REEF3D_DEM_VTP"
    folder.mkdir(exist_ok=True)
    tris = np.concatenate([BOX_FACES[:, [0, 1, 2]], BOX_FACES[:, [0, 2, 3]]])
    for count, t in enumerate(np.arange(0.0, end + 1e-9, interval)):
        points, velocity, ids, cells = [], [], [], []
        for body, centre in enumerate(dem_centres(t)):
            vertices = (BOX - 0.5) * 0.2 + centre
            cells.append(tris + 8 * body)
            points.append(vertices)
            velocity.append(np.tile([0.0, 0.0, -0.5], (8, 1)))
            ids += [body + 1] * 8
        points, velocity, cells = (
            np.concatenate(points),
            np.concatenate(velocity),
            np.concatenate(cells),
        )
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
            "<PolyData>",
            "<FieldData>",
            f'<DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{t:.12g}</DataArray>',
            "</FieldData>",
            f'<Piece NumberOfPoints="{len(points)}" NumberOfPolys="{len(cells)}">',
            '<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="ascii">',
            *(f"{x:.8g} {y:.8g} {z:.8g}" for x, y, z in points),
            "</DataArray>\n</Points>",
            "<PointData>",
            '<DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">',
            *(f"{u:g} {v:g} {w:g}" for u, v, w in velocity),
            "</DataArray>",
            '<DataArray type="Int32" Name="id" format="ascii">',
            *(str(i) for i in ids),
            "</DataArray>",
            '<DataArray type="Int32" Name="resolved" format="ascii">',
            *("0" for _ in ids),
            "</DataArray>",
            "</PointData>",
            '<Polys>\n<DataArray type="Int32" Name="connectivity" format="ascii">',
            *(f"{a} {b} {c}" for a, b, c in cells),
            '</DataArray>\n<DataArray type="Int32" Name="offsets" format="ascii">',
            *(str(3 * (i + 1)) for i in range(len(cells))),
            "</DataArray>\n</Polys>",
            "</Piece>\n</PolyData>\n</VTKFile>",
        ]
        (folder / f"REEF3D-DEM-{count:06d}.vtp").write_text("\n".join(lines) + "\n")


#: y of the wave profile lines, and their x points
PROFILE_LINES = {1: 1.0, 2: 3.0}
PROFILE_X = np.linspace(0.0, 8.0, 17)


def profile_eta(x: np.ndarray, y: float, t: float) -> np.ndarray:
    return 0.1 * np.sin(x - 0.5 * t) + 0.01 * y


def write_profiles(root: Path, solver: str, end: float, files: int = 6) -> None:
    """WSFLINE files as REEF3D writes them: numbered by iteration, timed by simtime;
    FNPF adds a wave theory column."""
    folder = root / f"REEF3D_{solver}_WSFLINE"
    folder.mkdir(exist_ok=True)
    theory = solver == "FNPF"
    for number, t in enumerate(np.linspace(0.0, end, files)):
        lines = [f"simtime:  {t:.12g}", f"number of wsf-lines:  {len(PROFILE_LINES)}", ""]
        lines += ["line_No     y_coord"]
        lines += [f"{n}\t {y:g}" for n, y in PROFILE_LINES.items()]
        lines += ["", ""]
        header = []
        for n in PROFILE_LINES:
            header += [f"X {n}", f"P {n}"] + ([f"W {n}"] if theory else [])
        lines.append("\t ".join(header))
        for x in PROFILE_X:
            row = []
            for y in PROFILE_LINES.values():
                eta = profile_eta(np.array([x]), y, t)[0]
                row += [f"{x:g}", f"{eta:.9g}"] + ([f"{eta * 0.9:.9g}"] if theory else [])
            lines.append(" \t ".join(row) + " \t ")
        name = f"REEF3D-{solver}-wsfline-{number * 37:06d}.dat"
        (folder / name).write_text("\n".join(lines) + "\n")


BOX = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    np.float32,
)
BOX_FACES = np.array(
    [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
)


def body_offset(frame: int) -> np.ndarray:
    """Where the floating box is at a frame (it heaves)."""
    return np.array([3.0, 1.5, 1.5 + 0.2 * np.sin(frame)], np.float32)


def write_objects(root: Path, solver: str, frames: int) -> None:
    """A floating box (6DOF VTP, triangles) and two mooring lines (legacy VTK)."""
    bodies = root / f"REEF3D_{solver}_6DOF_VTP"
    bodies.mkdir(exist_ok=True)
    tris = np.concatenate([BOX_FACES[:, [0, 1, 2]], BOX_FACES[:, [0, 2, 3]]])
    for frame in range(frames):
        points = (BOX * 0.5 + body_offset(frame)).astype(np.float32)
        # REEF3D writes every triangle with its own three points
        flat = points[tris.ravel()]
        write_piece(
            bodies / f"REEF3D-6DOF-0-{frame:06d}.vtp",
            "PolyData",
            flat,
            {},
            np.arange(flat.shape[0]),
            np.arange(3, flat.shape[0] + 1, 3),
        )
    mooring = root / f"REEF3D_{solver}_6DOF_Mooring"
    mooring.mkdir(exist_ok=True)
    for line in range(2):
        for frame in range(1, frames):  # REEF3D starts mooring output at 1
            anchor = np.array([0.0, 1.0 + line, 0.0])
            top = body_offset(frame) + np.array([0.0, 0.25 * line, 0.0])
            points = np.linspace(anchor, top, 6)
            tension = 1000.0 + 100 * frame + np.arange(6)
            lines = [
                "# vtk DataFile Version 2.0",
                f"Mooring line {line}",
                "ASCII ",
                "DATASET POLYDATA",
                f"POINTS {len(points)} float",
                *(f"{x:g} {y:g} {z:g}" for x, y, z in points),
                f"LINES {len(points) - 1} {3 * (len(points) - 1)}",
                *(f"2 {i} {i + 1}" for i in range(len(points) - 1)),
                f"POINT_DATA {len(points)}",
                "SCALARS Tension float 1 ",
                "LOOKUP_TABLE default",
                *(f"{t:g}" for t in tension),
            ]
            (mooring / f"REEF3D-Mooring-{line:08d}-{frame:06d}.vtk").write_text(
                "\n".join(lines) + "\n"
            )


GAUGES = [(1.0, 2.0), (4.0, 2.0), (7.0, 2.0)]


def gauge_eta(x: float, t: np.ndarray) -> np.ndarray:
    return 0.1 * np.sin(x - 0.5 * t)


def write_series(root: Path, solver: str, end: float, steps: int = 41) -> None:
    """Wave gauges and 6DOF files written like REEF3D writes them."""
    time = np.linspace(0.0, end, steps)
    wsf = root / f"REEF3D_{solver}_WSF"
    wsf.mkdir(exist_ok=True)
    lines = [f"number of gauges:  {len(GAUGES)}", "", "x_coord     y_coord"]
    lines += [f"{i + 1}\t {x:g}\t {y:g}" for i, (x, y) in enumerate(GAUGES)]
    lines += ["", "", "time\t " + "\t ".join(f"P{i + 1}" for i in range(len(GAUGES)))]
    for t in time:
        values = [gauge_eta(x, np.array([t]))[0] for x, _ in GAUGES]
        lines.append(f"{t:g}\t " + "\t ".join(f"{v:g}" for v in values))
    lines.append(f"{end + 1:g}\t -1e+20\t nan\t 0")  # REEF3D's "no value" markers
    (wsf / f"REEF3D-{solver}-WSF-HG.dat").write_text("\n".join(lines) + "\n")

    dof = root / f"REEF3D_{solver}_6DOF"
    dof.mkdir(exist_ok=True)
    forces = ["time \t Fx \t Fy \t Fz"]
    forces += [f"{t:g} \t {1000 * np.sin(t):g} \t 0 \t {2000 + 10 * t:g}" for t in time]
    (dof / "REEF3D_6DOF_forces_0.dat").write_text("\n".join(forces) + "\n")
    position = ["time \t XG \t YG \t ZG \t Phi \t Theta \t Psi"]
    position += [f"{t:g} \t 5 \t 2 \t {1.5 + 0.1 * np.sin(t):g} \t 0 \t 0 \t 0" for t in time]
    (dof / "REEF3D_6DOF_position_0.dat").write_text("\n".join(position) + "\n")
    velocity = ["time \t Ue [m/s] \t We [m/s] \t Qe [rad/s]"]
    velocity += [f"{t:g} \t 0 \t {0.1 * np.cos(t):g} \t 0" for t in time]
    (dof / "REEF3D_6DOF_velocity_0.dat").write_text("\n".join(velocity) + "\n")


def write_run_log(root: Path, solver: str, runs: list[dict]) -> Path:
    """A REEF3D run log (lagoon-format RUN.md) for files already in ``root``.

    ``runs``: dicts with ``id``, ``started`` and ``outputs`` = {stream: [(step, time)]},
    optionally ``domain`` = {"min": [x, y, z], "max": [x, y, z]}; the streams are the
    free surface and bed folders ``make_run`` writes.
    """
    import json

    streams = {
        "fsf": {
            "role": "free_surface",
            "folder": f"REEF3D_{solver}_VTP_FSF",
            "files": f"REEF3D-{solver}-FSF-{{step:08d}}.pvtp",
            "format": "vtp",
        },
        "bed": {
            "role": "bed",
            "folder": f"REEF3D_{solver}_VTP_BED",
            "files": f"REEF3D-{solver}-BED-{{step:08d}}.pvtp",
            "format": "vtp",
        },
    }
    lines = []
    for run in runs:
        lines.append(
            {
                "type": "run",
                "format": "reef3d-run",
                "version": 1,
                "run": run["id"],
                "solver": solver,
                "started": run["started"],
                "streams": streams,
                **({"domain": run["domain"]} if "domain" in run else {}),
            }
        )
        for stream, outputs in run["outputs"].items():
            for step, time in outputs:
                lines.append(
                    {
                        "type": "output",
                        "run": run["id"],
                        "step": step,
                        "time": time,
                        "streams": [stream],
                    }
                )
        lines.append({"type": "end", "run": run["id"], "status": "finished"})
    folder = root / "REEF3D_Case"
    folder.mkdir(exist_ok=True)
    path = folder / f"REEF3D_{solver}_run.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _xml_array(name: str, values, kind: str = "Float32", components: int = 1) -> list[str]:
    head = f'<DataArray type="{kind}" Name="{name}"'
    head += f' NumberOfComponents="{components}"' if components > 1 else ""
    rows = [" ".join(f"{v:.8g}" for v in np.atleast_1d(row)) for row in values]
    return [head + ' format="ascii">', *rows, "</DataArray>"]


def _time_value(t: float) -> list[str]:
    return [
        "<FieldData>",
        f'<DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{t:.12g}</DataArray>',
        "</FieldData>",
    ]


#: ice floes: two square floes of 1 x 1 m, 0.2 m thick, drifting in x
ICE_START = ((2.0, 1.0), (5.0, 3.0))


def write_ice(
    root: Path, end: float, interval: float, times: bool = True, solver: str = "FNPF"
) -> None:
    """FNPF ice floes as REEF3D writes them (fnpf_ice_print.cpp): prisms of a bottom
    and a top polygon and side quads, values per floe as cell data."""
    folder = root / f"REEF3D_{solver}_ICE"
    folder.mkdir(exist_ok=True)
    (folder / f"REEF3D-{solver}-ICE-floes.dat").write_text("# t id x y z ...\n")  # not a frame
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    for count, t in enumerate(np.arange(0.0, end + 1e-9, interval)):
        points, polys, ids = [], [], []
        for floe, (x0, y0) in enumerate(ICE_START):
            base = len(points)
            for z in (DEPTH - 0.1, DEPTH + 0.1):
                points += [(x0 + 0.3 * t + cx, y0 + cy, z) for cx, cy in corners]
            polys.append([base + 3, base + 2, base + 1, base])  # bottom, facing down
            polys.append([base + 4, base + 5, base + 6, base + 7])
            for q in range(4):
                q2 = (q + 1) % 4
                polys.append([base + q, base + q2, base + 4 + q2, base + 4 + q])
            ids += [floe + 1] * 6
        offsets = np.cumsum([len(p) for p in polys])
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
            "<PolyData>",
        ]
        lines += _time_value(t) if times else []
        lines += [
            f'<Piece NumberOfPoints="{len(points)}" NumberOfPolys="{len(polys)}">',
            "<Points>",
        ]
        lines += _xml_array("", points, components=3)[0:0]
        lines += ['<DataArray type="Float32" NumberOfComponents="3" format="ascii">']
        lines += [f"{x:.8g} {y:.8g} {z:.8g}" for x, y, z in points] + ["</DataArray>", "</Points>"]
        lines += ['<CellData Scalars="id">']
        lines += _xml_array("id", ids, "Int32") + _xml_array("type", [0] * len(ids), "Int32")
        lines += _xml_array("speed", [0.3] * len(ids)) + _xml_array(
            "contact_force", [10.0 * f for f in ids]
        )
        lines += [
            "</CellData>",
            "<Polys>",
            '<DataArray type="Int32" Name="connectivity" format="ascii">',
        ]
        lines += [" ".join(map(str, p)) for p in polys] + ["</DataArray>"]
        lines += _xml_array("offsets", offsets, "Int32") + [
            "</Polys>",
            "</Piece>",
            "</PolyData>",
            "</VTKFile>",
        ]
        (folder / f"REEF3D-{solver}-ICE-{count:08d}.vtp").write_text("\n".join(lines) + "\n")


#: rod trees: one colony of three rods (a stem and two branches) that sway in x
def rod_ends(t: float) -> np.ndarray:
    sway = 0.1 * np.sin(t)
    stem = [(4.0, 2.0, 0.4), (4.0 + sway, 2.0, 1.0)]
    left = [(4.0 + sway, 2.0, 1.0), (3.7 + 2 * sway, 2.0, 1.3)]
    right = [(4.0 + sway, 2.0, 1.0), (4.3 + 2 * sway, 2.0, 1.3)]
    return np.array([stem, left, right])  # (rods, 2 ends, 3)


def write_rodtree(root: Path, end: float, interval: float, times: bool = True) -> None:
    """Rod trees as REEF3D writes them (rodtree.cpp write_vtp): every rod a line of
    two points with a radius at each end, colony and forces as cell data."""
    folder = root / "REEF3D_RodTree"
    folder.mkdir(exist_ok=True)
    for count, t in enumerate(np.arange(0.0, end + 1e-9, interval)):
        ends = rod_ends(t)
        n = len(ends)
        points = ends.reshape(-1, 3)
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
            "<PolyData>",
        ]
        lines += _time_value(t) if times else []
        lines += [f'<Piece NumberOfPoints="{2 * n}" NumberOfLines="{n}">', "<Points>"]
        lines += ['<DataArray type="Float64" NumberOfComponents="3" format="ascii">']
        lines += [f"{x:.8g} {y:.8g} {z:.8g}" for x, y, z in points] + ["</DataArray>", "</Points>"]
        lines += ['<PointData Scalars="radius">']
        lines += _xml_array("radius", [0.05, 0.04, 0.03, 0.02, 0.03, 0.02], "Float64")
        lines += _xml_array("velocity", [(0.1 * np.cos(t), 0.0, 0.0)] * (2 * n), "Float64", 3)
        lines += ["</PointData>", "<CellData>"]
        lines += _xml_array("colony", [0] * n, "Int32")
        lines += _xml_array("hydro_force", [(1.0 + i, 0.0, 0.0) for i in range(n)], "Float64", 3)
        lines += _xml_array("submerged", [1.0] * n, "Float64")
        lines += [
            "</CellData>",
            "<Lines>",
            '<DataArray type="Int32" Name="connectivity" format="ascii">',
        ]
        lines += [f"{2 * e} {2 * e + 1}" for e in range(n)] + ["</DataArray>"]
        lines += _xml_array("offsets", [2 * e + 2 for e in range(n)], "Int32")
        lines += ["</Lines>", "</Piece>", "</PolyData>", "</VTKFile>"]
        (folder / f"REEF3D-RodTree-{count:08d}.vtp").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- particles
def particle_cloud(t: float, rank: int = 0, n0: int = 30) -> np.ndarray:
    """Positions of a sediment cloud that grows and drifts: (n, 3) float32."""
    rng = np.random.default_rng(1000 * rank + int(round(100 * t)))
    n = n0 + 5 * int(round(4 * t)) + 3 * rank
    base = rng.uniform([0.5 + 4 * rank, 0.5, 0.2], [4.0 + 4 * rank, 3.5, 1.5], (n, 3))
    return (base + [0.3 * t, 0.0, 0.05 * t]).astype(np.float32)


def _particle_vtp(path: Path, t: float, arrays: list[tuple[str, np.ndarray]], points, time_value):
    """A particle VTP as REEF3D writes it: Verts, raw appended data, int32 sizes,
    the arrays in the order of the header, then points, connectivity and offsets."""
    n = len(points)
    blobs, offset, elements = [], 0, []
    for name, values in arrays:
        values = np.ascontiguousarray(values)
        components = values.shape[1] if values.ndim == 2 else 1
        attrs = f'type="{TYPE_NAMES[values.dtype]}" Name="{name}"'
        if components > 1:
            attrs += f' NumberOfComponents="{components}"'
        elements.append(f'<DataArray {attrs} format="appended" offset="{offset}"/>')
        blob = np.int32(values.nbytes).tobytes() + values.tobytes()
        blobs.append(blob)
        offset += len(blob)
    tail = []
    for values in (
        points.astype(np.float32),
        np.arange(n, dtype=np.int32),
        np.arange(1, n + 1, dtype=np.int32),
    ):
        tail.append(offset)
        blob = np.int32(values.nbytes).tobytes() + values.tobytes()
        blobs.append(blob)
        offset += len(blob)
    field = (
        f'<FieldData>\n<DataArray type="Float64" Name="TimeValue" NumberOfTuples="1"> {t:g}\n'
        "</DataArray>\n</FieldData>\n"
        if time_value
        else "<FieldData>\n</FieldData>\n"
    )
    head = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n<PolyData>\n'
        f'<Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" '
        'NumberOfStrips="0" NumberOfPolys="0">\n'
        + field
        + "<PointData>\n"
        + "\n".join(elements)
        + "\n</PointData>\n<Points>\n"
        f'<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="{tail[0]}"/>\n'
        "</Points>\n<Verts>\n"
        f'<DataArray type="Int32" Name="connectivity" format="appended" offset="{tail[1]}"/>\n'
        f'<DataArray type="Int32" Name="offsets" format="appended" offset="{tail[2]}"/>\n'
        "</Verts>\n</Piece>\n</PolyData>\n"
    )
    with open(path, "wb") as handle:
        handle.write(head.encode())
        handle.write(b'<AppendedData encoding="raw">\n_')
        for blob in blobs:
            handle.write(blob)
        handle.write(b"\n</AppendedData>\n</VTKFile>\n")


def cpm_arrays(points: np.ndarray, t: float) -> list[tuple[str, np.ndarray]]:
    n = len(points)
    velocity = np.stack([np.full(n, 0.3), np.zeros(n), np.full(n, 0.05)], axis=1)
    return [
        ("Flag", np.ones(n, np.float32)),
        ("velocity", velocity.astype(np.float32)),
        ("radius", np.full(n, 0.0005, np.float32)),
        ("fluid velocity", (velocity * 1.5).astype(np.float32)),
        ("bedChange", (points[:, 0] * 0.01 * t).astype(np.float32)),
    ]


def write_cpm(root: Path, end: float, interval: float, ranks: int = 2) -> None:
    """CFD's CPM particles (REEF3D_CFD_CPM_Particle): per output a .pvtp and a VTP per
    rank, at their own interval (Q 182), with TimeValue. The .pvtp names the vector
    arrays with <DataArray> as REEF3D does."""
    folder = root / "REEF3D_CFD_CPM_Particle"
    folder.mkdir(exist_ok=True)
    (folder / "REEF3D-CFD-CPM-Log.dat").write_text("log\n")
    for count, t in enumerate(np.arange(0.0, end + 1e-9, interval)):
        names = []
        for rank in range(ranks):
            points = particle_cloud(float(t), rank)
            name = f"REEF3D-Particle-{count:08d}-{rank + 1:06d}.vtp"
            _particle_vtp(folder / name, float(t), cpm_arrays(points, float(t)), points, True)
            names.append(name)
        (folder / f"REEF3D-Particle-{count:08d}.pvtp").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="PPolyData" version="1.0" '
            'byte_order="LittleEndian">\n<PPolyData GhostLevel="0">\n<PPointData>\n'
            '<PDataArray type="Float32" Name="Flag"/>\n'
            '<DataArray type="Float32" Name="velocity" NumberOfComponents="3"/>\n'
            '<PDataArray type="Float32" Name="radius"/>\n'
            '<DataArray type="Float32" Name="fluid velocity" NumberOfComponents="3"/>\n'
            '<PDataArray type="Float32" Name="bedChange"/>\n</PPointData>\n<PPoints>\n'
            '<PDataArray type="Float32" NumberOfComponents="3"/>\n</PPoints>\n'
            + "".join(f'<Piece Source="{n}"/>\n' for n in names)
            + "</PPolyData>\n</VTKFile>\n"
        )


def write_nhflow_particles(root: Path, end: float, interval: float) -> None:
    """NHFLOW's particles (REEF3D_NHFLOW_Particles): one serial VTP per output."""
    folder = root / "REEF3D_NHFLOW_Particles"
    folder.mkdir(exist_ok=True)
    (folder / "REEF3D-NHFLOW-Particles-Tracks.csv").write_text("id,t,x,y,z\n")
    for count, t in enumerate(np.arange(0.0, end + 1e-9, interval)):
        points = particle_cloud(float(t), 0, n0=20)
        n = len(points)
        arrays = [
            ("velocity", np.tile(np.float32([0.3, 0, 0.05]), (n, 1))),
            ("id", np.arange(n, dtype=np.int32)),
            ("release", np.zeros(n, np.int32)),
            ("state", np.ones(n, np.int32)),
            ("age", np.full(n, float(t), np.float32)),
        ]
        path = folder / f"REEF3D-NHFLOW-Particles-{count:08d}.vtp"
        _particle_vtp(path, float(t), arrays, points, True)


# ======================================================================== AMR
#: AMR fields per solver, in REEF3D's order (fnpf_amr_print.cpp, nhflow_amr_print.cpp,
#: sflow_amr.cpp)
AMR_FIELDS = {
    "FNPF": ("eta", "elevation", "Fifsf", "Fz", "bed"),
    "NHFLOW": ("eta", "elevation", "bed", "u_top", "v_top", "wet", "WL"),
    "SFLOW": ("eta", "elevation", "WL", "u", "v", "bed", "press", "w", "wetdry"),
}
#: fields REEF3D writes as Int32 (the others as Float64)
AMR_INTEGER = ("wetdry",)
AMR_DEPTH = 0.5  # still water depth (wd)


def amr_eta(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return 0.05 * np.sin(2.0 * x - 1.5 * t) * np.cos(0.5 * y)


def amr_patches(frame: int) -> list[tuple[int, int, float, float, float, float]]:
    """(rank, level, x0, x1, y0, y1) of the refined patches of an output: a level-1
    patch that moves along x (regridding) and a level-2 patch inside it; on the
    second rank a second level-1 patch from the second output on."""
    shift = 0.5 * frame
    patches = [
        (1, 1, 1.0 + shift, 3.0 + shift, 0.5, 2.5),
        (1, 2, 1.5 + shift, 2.5 + shift, 1.0, 2.0),
    ]
    if frame >= 1:
        patches.append((2, 1, 6.0, 7.5, 0.0, 1.5))
    return patches


def _amr_fields(solver: str, x: np.ndarray, y: np.ndarray, t: float) -> dict[str, np.ndarray]:
    """Cell values (ny, nx) of the solver's AMR fields at cell centres x, y."""
    xx, yy = np.meshgrid(x, y)
    eta = amr_eta(xx, yy, t)
    values = {
        "eta": eta,
        "elevation": eta + AMR_DEPTH,
        "bed": np.zeros_like(eta),
        "Fifsf": 0.1 * eta,
        "Fz": 0.2 * eta,
        "u_top": np.cos(2.0 * xx - 1.5 * t),
        "v_top": 0.1 * np.sin(yy),
        "wet": np.ones_like(eta),
        "WL": eta + AMR_DEPTH,
        "u": np.cos(2.0 * xx - 1.5 * t),
        "v": 0.1 * np.sin(yy),
        "press": np.zeros_like(eta),
        "w": 0.01 * eta,
        "wetdry": np.ones_like(eta, dtype=np.int32),
    }
    return {name: values[name] for name in AMR_FIELDS[solver]}


def write_amr_grid(
    path: Path, x: np.ndarray, y: np.ndarray, fields: dict[str, np.ndarray], level: int, t: float
) -> None:
    """A grid as REEF3D's AMR output writes it: ASCII rectilinear grid, cell data, rows
    of x values per y, TimeValue and level in the field data."""
    nx, ny = x.size - 1, y.size - 1
    out = ['<?xml version="1.0"?>\n']
    out.append('<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n')
    out.append(f'<RectilinearGrid WholeExtent="0 {nx} 0 {ny} 0 0">\n')
    out.append(
        f'<FieldData><DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{float(t)!r}'
        "</DataArray>"
    )
    out.append(
        f'<DataArray type="Int32" Name="level" NumberOfTuples="1">{level}</DataArray></FieldData>\n'
    )
    out.append(f'<Piece Extent="0 {nx} 0 {ny} 0 0">\n<CellData Scalars="eta">\n')
    for name, values in fields.items():
        integer = name in AMR_INTEGER
        kind = "Int32" if integer else "Float64"
        out.append(f'<DataArray type="{kind}" Name="{name}" format="ascii">\n')
        for row in values:
            if integer:
                out.append("".join(f"{int(v)} " for v in row) + "\n")
            else:
                out.append("".join(f"{float(v)!r} " for v in row) + "\n")
        out.append("</DataArray>\n")
    out.append("</CellData>\n<Coordinates>\n")
    out.append('<DataArray type="Float64" Name="x" format="ascii">')
    out.append("".join(f"{float(v)!r} " for v in x))
    out.append('</DataArray>\n<DataArray type="Float64" Name="y" format="ascii">')
    out.append("".join(f"{float(v)!r} " for v in y))
    out.append('</DataArray>\n<DataArray type="Float64" Name="z" format="ascii">0</DataArray>\n')
    out.append("</Coordinates>\n</Piece>\n</RectilinearGrid>\n</VTKFile>\n")
    path.write_text("".join(out))


def make_amr(
    root: Path,
    solver: str = "NHFLOW",
    frames: int = 3,
    ranks: int = 2,
    interval: float = 0.5,
    nx: int = 16,
    ny: int = 6,
    length: float = 8.0,
    width: float = 3.0,
) -> Path:
    """REEF3D_<SOLVER>_AMR in ``root``: level 0 split along x over the ranks, the
    patches of :func:`amr_patches` refined by 2 per level. Returns the folder."""
    folder = root / f"REEF3D_{solver}_AMR"
    folder.mkdir(parents=True, exist_ok=True)
    dx0 = length / nx
    x0 = np.linspace(0.0, length, nx + 1)
    y0 = np.linspace(0.0, width, ny + 1)
    bounds = np.linspace(0, nx, ranks + 1).round().astype(int)
    for frame in range(frames):
        t = frame * interval
        stamp = f"{frame:08d}"
        for rank in range(ranks):
            x = x0[bounds[rank] : bounds[rank + 1] + 1]
            centres = 0.5 * (x[:-1] + x[1:]), 0.5 * (y0[:-1] + y0[1:])
            write_amr_grid(
                folder / f"REEF3D-{solver}-AMR-L0-{stamp}-{rank + 1:04d}.vtr",
                x,
                y0,
                _amr_fields(solver, *centres, t),
                0,
                t,
            )
        per_rank: dict[int, int] = {}
        for rank, level, px0, px1, py0, py1 in amr_patches(frame):
            if rank > ranks:
                continue
            per_rank[rank] = per_rank.get(rank, 0) + 1
            step = dx0 / 2**level
            x = np.arange(px0, px1 + 0.5 * step, step)
            y = np.arange(py0, py1 + 0.5 * step, step)
            centres = 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:])
            write_amr_grid(
                folder / f"REEF3D-{solver}-AMR-{stamp}-{rank:04d}-{per_rank[rank]:04d}.vtr",
                x,
                y,
                _amr_fields(solver, *centres, t),
                level,
                t,
            )
        out = [
            '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n'
            "<vtkMultiBlockDataSet>\n"
        ]
        out.append('<Block index="0" name="level 0">\n')
        for rank in range(ranks):
            out.append(
                f'<DataSet index="{rank}" '
                f'file="REEF3D-{solver}-AMR-L0-{stamp}-{rank + 1:04d}.vtr"/>\n'
            )
        out.append('</Block>\n<Block index="1" name="patches">\n')
        index = 0
        for rank in range(1, ranks + 1):
            for patch in range(1, per_rank.get(rank, 0) + 1):
                out.append(
                    f'<DataSet index="{index}" '
                    f'file="REEF3D-{solver}-AMR-{stamp}-{rank:04d}-{patch:04d}.vtr"/>\n'
                )
                index += 1
        out.append("</Block>\n</vtkMultiBlockDataSet>\n</VTKFile>\n")
        (folder / f"REEF3D-{solver}-AMR-{stamp}.vtm").write_text("".join(out))
    return folder


def write_grid_preview(case: Path, x, y, z) -> Path:
    """DIVEMesh's grid preview (DIVEMesh_Paraview/DIVEMesh_grid-preview.vts) of the
    grid with node coordinates x, y, z, as DIVEMesh writes it: appended raw arrays
    topo, solid (float32 per node) and the points, each after its size (UInt32)."""
    x, y, z = (np.asarray(a, np.float32) for a in (x, y, z))
    nx, ny, nz = len(x), len(y), len(z)
    n = nx * ny * nz
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")  # x fastest, then y, then z
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    topo = np.ones(n, np.float32)
    solid = np.ones(n, np.float32)
    offsets = [0, 4 + topo.nbytes, 8 + topo.nbytes + solid.nbytes]
    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        f'<StructuredGrid WholeExtent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}">\n'
        f'<Piece Extent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}">\n<PointData>\n'
        f'<DataArray type="Float32" Name="topo" format="appended" offset="{offsets[0]}"/>\n'
        f'<DataArray type="Float32" Name="solid" format="appended" offset="{offsets[1]}"/>\n'
        "</PointData>\n<Points>\n"
        f'<DataArray type="Float32" NumberOfComponents="3" format="appended" '
        f'offset="{offsets[2]}"/>\n'
        "</Points>\n</Piece>\n</StructuredGrid>\n"
        '<AppendedData encoding="raw">\n_'
    )
    path = case / "DIVEMesh_Paraview" / "DIVEMesh_grid-preview.vts"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(header.encode())
        for array in (topo, solid, points):
            out.write(np.uint32(array.nbytes).tobytes())
            out.write(array.tobytes())
        out.write(b"\n</AppendedData>\n</VTKFile>\n")
    return path


# ----------------------------------------------------------------- SEASTATE
SEASTATE_WD = 3.0  # the still water level SEASTATE writes its map at
SEASTATE_DIR = 30.0  # degrees, counter-clockwise from x: where the waves go


def seastate_hs(x: np.ndarray, frame: int) -> np.ndarray:
    """Waves growing along x, higher at later outputs."""
    return (0.5 + 0.1 * x + 0.05 * frame).astype(np.float32)


def write_seastate(root: Path, frames: int = 3, ranks: int = 2, wind: bool = True) -> Path:
    """REEF3D::SEASTATE's maps, as seastate_vtp writes them: per output a .pvtp and a
    VTP per rank (REEF3D_SEASTATE_VTP/REEF3D-SEASTATE-<output>-<rank>.vtp), a plane at
    the still water level, triangles, the nodal sea state parameters. The last x
    column is dry (wet 0)."""
    folder = root / "REEF3D_SEASTATE_VTP"
    folder.mkdir(parents=True, exist_ok=True)
    cuts = np.linspace(0, NX - 1, ranks + 1).round().astype(int)
    for frame in range(frames):
        names = []
        for rank in range(ranks):
            points, data, conn, offs = _surface_piece(cuts[rank], cuts[rank + 1], frame, "bed")
            x = points[:, 0]
            points[:, 2] = SEASTATE_WD
            hs = seastate_hs(x, frame)
            wet = (x < (NX - 1) * DX - 1e-6).astype(np.float32)
            data = {
                "Hs": hs * wet,
                "Tm01": np.full_like(x, 6.0),
                "Tm-10": np.full_like(x, 7.0),
                "Tp": np.full_like(x, 8.0),
                "dir": np.full_like(x, SEASTATE_DIR),
                "spread": np.full_like(x, 25.0),
                "depth": (SEASTATE_WD - 0.1 * x).astype(np.float32),
                "wet": wet,
            }
            if wind:
                data["U10x"] = np.full_like(x, 10.0)
                data["U10y"] = np.full_like(x, 0.0)
            name = f"REEF3D-SEASTATE-{frame:08d}-{rank + 1:06d}.vtp"
            write_piece(folder / name, "PolyData", points, data, conn, offs)
            names.append(name)
        write_parallel(
            folder / f"REEF3D-SEASTATE-{frame:08d}.pvtp",
            "PolyData",
            names,
            {n: 1 for n in data},
        )
    return root


def write_seastate_amr(root: Path, frames: int = 2) -> Path:
    """SEASTATE's AMR output, as seastate_amr writes it: ASCII rectilinear grids (a
    coarse one and a refined patch at level 1) with cell data, and a .vtm per output
    listing them."""
    folder = root / "REEF3D_SEASTATE_AMR"
    folder.mkdir(parents=True, exist_ok=True)

    def grid(path: Path, xs, ys, level: int, t: float, frame: int) -> None:
        nx, ny = len(xs) - 1, len(ys) - 1
        cx = 0.5 * (np.asarray(xs[:-1]) + np.asarray(xs[1:]))
        cells = np.tile(cx, ny)
        arrays = {
            "Hs": seastate_hs(cells, frame),
            "Tm01": np.full(cells.size, 6.0),
            "Tp": np.full(cells.size, 8.0),
            "dir": np.full(cells.size, SEASTATE_DIR),
            "spread": np.full(cells.size, 25.0),
            "depth": np.full(cells.size, 2.0),
        }
        body = "".join(
            f'<DataArray type="Float32" Name="{name}" format="ascii">'
            + " ".join(f"{v:g}" for v in values)
            + "</DataArray>\n"
            for name, values in arrays.items()
        )
        ints = " ".join("1" for _ in range(cells.size))
        body += f'<DataArray type="Int32" Name="wet" format="ascii"> {ints}</DataArray>\n'
        zeros = " ".join("0" for _ in range(cells.size))
        body += f'<DataArray type="Int32" Name="covered" format="ascii"> {zeros}</DataArray>\n'
        path.write_text(
            '<?xml version="1.0"?>\n'
            '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n'
            f'<RectilinearGrid WholeExtent="0 {nx} 0 {ny} 0 0">\n'
            f'<Piece Extent="0 {nx} 0 {ny} 0 0">\n'
            f'<FieldData><DataArray type="Float64" Name="TIME" NumberOfTuples="1" '
            f'format="ascii">{t}</DataArray><DataArray type="Int32" Name="level" '
            f'NumberOfTuples="1" format="ascii">{level}</DataArray></FieldData>\n'
            "<Coordinates>\n"
            '<DataArray type="Float64" Name="x" format="ascii">'
            + " ".join(f"{v:g}" for v in xs)
            + "</DataArray>\n"
            '<DataArray type="Float64" Name="y" format="ascii">'
            + " ".join(f"{v:g}" for v in ys)
            + "</DataArray>\n"
            '<DataArray type="Float64" Name="z" format="ascii"> 0</DataArray>\n'
            "</Coordinates>\n<CellData>\n" + body + "</CellData>\n</Piece>\n"
            "</RectilinearGrid>\n</VTKFile>\n"
        )

    for frame in range(frames):
        t = 10.0 * frame
        files = []
        for patch, (xs, ys, level) in enumerate(
            (
                (np.arange(0.0, 9.0, 1.0), np.arange(0.0, 5.0, 1.0), 0),
                (np.arange(2.0, 4.01, 0.5), np.arange(1.0, 3.01, 0.5), 1),
            )
        ):
            name = f"REEF3D-SEASTATE-AMR-{frame:08d}-00000-{patch:04d}.vtr"
            grid(folder / name, xs, ys, level, t, frame)
            files.append(name)
        sets = "".join(f'<DataSet index="{i}" file="{f}"/>\n' for i, f in enumerate(files))
        (folder / f"REEF3D-SEASTATE-AMR-{frame:08d}.vtm").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0" '
            'byte_order="LittleEndian">\n<vtkMultiBlockDataSet>\n'
            + sets
            + "</vtkMultiBlockDataSet>\n</VTKFile>\n"
        )
    return root
