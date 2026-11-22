# LAGOON — NHFLOW's membranes (X 330) and the collar ring of a flexible bag
# Architect: Hans Bihs
"""REEF3D writes each membrane at its own print interval as
REEF3D_NHFLOW_Membrane_VTP/REEF3D-NHFLOW-Membrane-<membrane>-<output>.vtp (net_membrane_
print.cpp: velocity, displacement, load on the nodes; dp, force, tension, tag on the
triangles; a .pvd with the times), or with P 18 into the store (particles/
nhflow_membrane). A flexible bag with ``collar D ...`` in membrane.dat floats on a pipe
ring along its top edge, which REEF3D does not write: LAGOON draws it."""

from __future__ import annotations

import numpy as np
from reef3d_fixtures import make_run

from lagoonviz.core import Session
from lagoonviz.core.data.membranes import SIDES, read_membrane_dat, ring_tube
from lagoonviz.core.data.reef3d import open_case

XC, YC, R, BOTTOM, TOP = 5.0, 3.0, 1.0, 0.4, 1.2
AROUND, LEVELS = 24, 5
COLLAR_D = 0.2

MEMBRANE_DAT = f"""# two bags: a flexible one on a collar, a fixed box
membrane cylinder {XC} {YC} {R} {BOTTOM} {TOP}
name cage
structure flexible   # a floating pipe ring
collar {COLLAR_D} 5.0 1.0e6 1.0e3
mooring 0 0 0 100 10
membrane box 8 9 1 2 0.5 1.0
collar 0.5 1 1 1
"""


def _bag(frame: int):
    """A cylinder bag: rings of nodes up its wall and a node in the floor's middle,
    the top ring moved by the waves (heave and surge) and its wall bulging."""
    angle = np.linspace(0.0, 2 * np.pi, AROUND, endpoint=False)
    z = np.linspace(BOTTOM, TOP, LEVELS)
    start = [[XC, YC, BOTTOM]]
    for level in z:
        start += [[XC + R * np.cos(a), YC + R * np.sin(a), level] for a in angle]
    start = np.asarray(start)
    move = np.zeros_like(start)
    move[:, 0] = 0.05 * frame * (start[:, 2] - BOTTOM)
    move[:, 2] = 0.02 * frame
    points = start + move
    triangles = [[0, 1 + k, 1 + (k + 1) % AROUND] for k in range(AROUND)]
    for level in range(LEVELS - 1):
        a = 1 + level * AROUND
        b = a + AROUND
        for k in range(AROUND):
            k1 = (k + 1) % AROUND
            triangles += [[a + k, b + k, b + k1], [a + k, b + k1, a + k1]]
    return points, move, np.asarray(triangles)


def _box(frame: int):
    """A small fixed box membrane: one wall of two triangles."""
    points = np.array([[8, 1, 0.5], [9, 1, 0.5], [9, 1, 1.0], [8, 1, 1.0]], float)
    return points, np.zeros_like(points), np.array([[0, 1, 2], [0, 2, 3]])


def _array(name, values, kind="Float32"):
    values = np.atleast_2d(np.asarray(values).T).T
    components = values.shape[1] if values.ndim == 2 else 1
    head = f'<DataArray type="{kind}" Name="{name}"'
    head += f' NumberOfComponents="{components}"' if components > 1 else ""
    body = " ".join(f"{v:.9g}" for v in np.ravel(values))
    return f'{head} format="ascii">{body}</DataArray>'


def _vtp(path, t, points, move, triangles, time_value=True):
    n, m = len(points), len(triangles)
    field = (
        f'<FieldData><DataArray type="Float64" Name="TimeValue" NumberOfTuples="1">{t:g}'
        "</DataArray></FieldData>\n"
        if time_value
        else ""
    )
    text = (
        '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n'
        f"<PolyData>\n{field}"
        f'<Piece NumberOfPoints="{n}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" '
        f'NumberOfPolys="{m}">\n'
        f'<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">'
        f"{' '.join(f'{v:.9g}' for v in points.ravel())}</DataArray></Points>\n"
        "<PointData>\n"
        + _array("velocity", np.zeros((n, 3)))
        + _array("displacement", move)
        + _array("load", np.ones((n, 3)))
        + "</PointData>\n<CellData>\n"
        + _array("dp", np.arange(m, dtype=float))
        + _array("force", np.zeros((m, 3)))
        + _array("tension", np.full(m, 10.0 + t))
        + _array("tag", np.zeros(m, np.int32), "Int32")
        + "</CellData>\n<Polys>\n"
        + _array("connectivity", triangles.ravel(), "Int32")
        + _array("offsets", 3 * np.arange(1, m + 1), "Int32")
        + "</Polys>\n</Piece>\n</PolyData>\n</VTKFile>\n"
    )
    path.write_text(text)


def write_membranes(root, frames=3, interval=0.05, time_value=True):
    """REEF3D_NHFLOW_Membrane_VTP as net_membrane_print.cpp writes it, and
    membrane.dat in the case folder."""
    (root / "membrane.dat").write_text(MEMBRANE_DAT)
    folder = root / "REEF3D_NHFLOW_Membrane_VTP"
    folder.mkdir(parents=True, exist_ok=True)
    for number, shape in enumerate((_bag, _box)):
        entries = []
        for frame in range(frames):
            name = f"REEF3D-NHFLOW-Membrane-{number}-{frame:06d}.vtp"
            _vtp(folder / name, interval * frame, *shape(frame), time_value=time_value)
            entries.append(
                f'<DataSet timestep="{interval * frame:.10g}" group="" part="0" file="{name}"/>'
            )
        (folder / f"REEF3D-NHFLOW-Membrane-{number}.pvd").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n'
            + "\n".join(entries)
            + "\n</Collection>\n</VTKFile>\n"
        )
    return folder


def test_membrane_dat_is_read_as_reef3d_reads_it(tmp_path):
    (tmp_path / "membrane.dat").write_text(
        MEMBRANE_DAT + "membrane cylcone 1 1 0.5 0.1 0.3 0.9\nattach 0.8\n"
    )
    cage, box, cone = read_membrane_dat(tmp_path / "membrane.dat")
    assert (cage.number, cage.shape, cage.name, cage.structure) == (
        0,
        "cylinder",
        "cage",
        "flexible",
    )
    assert cage.collar == COLLAR_D and cage.top == TOP and cage.has_collar
    assert cage.footprint == (XC - R, XC + R, YC - R, YC + R)
    assert box.collar == 0.5 and not box.has_collar  # a fixed membrane has no collar ring
    assert cone.top == 0.9 and cone.attach == 0.8 and not cone.has_collar


def test_the_collar_tube_is_closed_and_round():
    angle = np.linspace(0.0, 2 * np.pi, 30, endpoint=False)
    ring = np.stack([2 + np.cos(angle), 3 + np.sin(angle), np.full_like(angle, 1.0)], axis=1)
    vertices, triangles, normals = ring_tube(ring, 0.1)
    assert vertices.shape == (30 * SIDES, 3) and triangles.shape == (2 * 30 * SIDES, 3)
    assert triangles.max() == len(vertices) - 1
    # every vertex 0.1 from the ring: the section square to the ring
    nearest = np.repeat(ring, SIDES, axis=0)
    np.testing.assert_allclose(np.linalg.norm(vertices - nearest, axis=1), 0.1, rtol=1e-5)
    # closed: every edge in two triangles
    edges = np.sort(
        np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    # faces wound outwards, as the normals point
    a, b, c = (vertices[triangles[:, k]].astype(np.float64) for k in range(3))
    face = np.cross(b - a, c - a)
    middle = (a + b + c) / 3 - np.repeat(ring, SIDES, axis=0)[triangles[:, 0]]
    assert ((face * middle).sum(axis=1) > 0).all()


def test_the_membranes_and_the_collar_are_shown(tmp_path):
    run = make_run(tmp_path / "nhflow", solver="NHFLOW", frames=3, print_interval=0.5)
    write_membranes(run, frames=4)
    case = open_case(run)
    membranes = case.datasets["nhflow_membrane"]
    assert membranes.role == "membrane" and len(membranes) == 4
    assert [len(f.pieces) for f in membranes.frames] == [2] * 4  # the two membranes
    np.testing.assert_allclose(membranes.times, [0.0, 0.05, 0.1, 0.15])
    assert {"dp", "tension", "displacement"} <= set(membranes.components)
    [collar] = membranes.collars  # the box is fixed: no collar
    assert collar.diameter == COLLAR_D and collar.z == TOP
    fabric = AROUND * LEVELS + 1 + 4
    mesh = membranes.read(3, ["tension"])
    assert mesh.n_points == fabric + AROUND * SIDES
    # the tube round the top ring where it is now: moved by the waves
    points, move, _ = _bag(3)
    top = points[-AROUND:]
    tube = mesh.points[fabric:].reshape(AROUND, SIDES, 3).mean(axis=1)
    found = np.linalg.norm(tube[:, None, :] - top[None, :, :], axis=2).min(axis=1)
    assert found.max() < 1e-3
    assert np.ptp(mesh.points[fabric:, 2]) > 0.9 * COLLAR_D  # round, of its diameter
    assert np.allclose(mesh.point_data["tension"][fabric:], 10.15)  # its nodes' values
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "membrane")
    assert item.visible and item.properties["variable"] == "none"
    shown = s.data.load(item.id)
    assert shown is not None and shown.n_points == fabric + AROUND * SIDES


def test_membrane_times_from_the_pvd(tmp_path):
    """Without a TimeValue (P 16 0) the times come from the .pvd collections."""
    run = make_run(tmp_path / "nhflow", solver="NHFLOW", frames=2, print_interval=0.5)
    write_membranes(run, frames=3, interval=0.2, time_value=False)
    membranes = open_case(run).datasets["nhflow_membrane"]
    np.testing.assert_allclose(membranes.times, [0.0, 0.2, 0.4])


def test_no_collar_without_membrane_dat(tmp_path):
    run = make_run(tmp_path / "nhflow", solver="NHFLOW", frames=2, print_interval=0.5)
    write_membranes(run, frames=2)
    (run / "membrane.dat").unlink()
    membranes = open_case(run).datasets["nhflow_membrane"]
    assert membranes.collars == []
    assert membranes.read(1, None).n_points == AROUND * LEVELS + 1 + 4


def test_membranes_written_by_reef3d(tmp_path):
    """As REEF3D writes them with P 18 (lagoon_membranes.cpp): the nodes of all
    membranes side by side, the triangles with the number of their membrane."""
    from lagoon_format.store import StoreWriter

    root = tmp_path / "nhflow"
    root.mkdir()
    (root / "membrane.dat").write_text(MEMBRANE_DAT)
    writer = StoreWriter.create(root / "REEF3D_NHFLOW.lagoon", solver="NHFLOW", run={"run": "R1"})
    vectors = {n: (3, "float32") for n in ("velocity", "displacement", "load")}
    cell_fields = {
        "membrane": (1, "int32"),
        "dp": (1, "float32"),
        "force": (3, "float32"),
        "tension": (1, "float32"),
        "tag": (1, "int32"),
    }
    membranes = writer.particle_writer(
        "nhflow_membrane",
        {
            "point_fields": vectors,
            "cells": "polys",
            "cell_fields": cell_fields,
            "source": "REEF3D_NHFLOW_Membrane_VTP",
            "attributes": {"role": "membrane"},
        },
    )
    for frame in range(2):
        a, move_a, tri_a = _bag(frame)
        b, move_b, tri_b = _box(frame)
        points = np.concatenate([a, b]).astype(np.float32)
        data = {
            "velocity": np.zeros_like(points),
            "displacement": np.concatenate([move_a, move_b]).astype(np.float32),
            "load": np.zeros_like(points),
        }
        cells = np.concatenate([tri_a, tri_b + len(a)]).astype(np.int32)
        m = len(cells)
        offsets = np.arange(3, 3 * m + 1, 3, dtype=np.int32)
        values = {
            "membrane": np.repeat(np.int32([0, 1]), [len(tri_a), len(tri_b)]),
            "dp": np.zeros(m, np.float32),
            "force": np.zeros((m, 3), np.float32),
            "tension": np.full(m, 5.0, np.float32),
            "tag": np.zeros(m, np.int32),
        }
        membranes.append(0.1 * frame, frame, points, data, cells.ravel(), offsets, values)
    membranes.flush()
    writer.close()
    s = Session()
    s.open_case(root)
    item = next(i for i in s.scene if i.kind == "membrane")
    dataset = s.data.dataset(item.id)
    assert "LAGOON" in dataset.label and len(dataset.collars) == 1
    mesh = dataset.read(1, ["dp"])
    assert mesh.n_points == AROUND * LEVELS + 1 + 4 + AROUND * SIDES


def test_membranes_are_drawn_in_their_colour(tmp_path):
    from lagoonviz.core.colormaps import PLAIN_COLORS

    run = make_run(tmp_path / "nhflow", solver="NHFLOW", frames=2, print_interval=0.5)
    write_membranes(run, frames=2)
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "membrane")
    surface = s.data.display(item.id, s.data.load(item.id))
    assert surface.values is None and surface.color == PLAIN_COLORS["membrane"]
    s.scene.update(item.id, properties={"variable": "tension"})
    surface = s.data.display(item.id, s.data.load(item.id))
    assert surface.values is not None and len(surface.values) == len(surface.positions)


def test_the_membranes_in_the_store(tmp_path):
    from lagoon_format.store import Store

    from lagoonviz.core.convert import convert

    run = make_run(tmp_path / "nhflow", solver="NHFLOW", frames=2, print_interval=0.5)
    write_membranes(run, frames=3)
    report = convert(run)
    assert report.outputs["membrane"] == 3
    stored = Store(report.store).particles["nhflow_membrane"]
    assert stored.cells == "polys" and stored.info["role"] == "membrane"
    membranes = open_case(run).datasets["nhflow_membrane"]
    assert "LAGOON" in membranes.label and len(membranes.collars) == 1
    np.testing.assert_allclose(membranes.times, [0.0, 0.05, 0.1])
    mesh = membranes.read(2, ["dp"])
    assert mesh.n_points == AROUND * LEVELS + 1 + 4 + AROUND * SIDES
