# LAGOON — nets of bars (X 320): REEF3D's .vtk files and the LAGOON store
# Architect: Hans Bihs
"""REEF3D writes each bar net (X 320 types 1 - 3, 12 - 14) at every output as a
legacy .vtk unstructured grid of line cells, REEF3D_<SOLVER>_6DOF_Net/
REEF3D-Net-<net>-<output>.vtk: the knots with Tension (and Velocity, Acceleration
for a dynamic net), the bars as lines. With P 18 it writes them into the store as
the particle set <solver>_net (role "net"); lagoon-convert does the same from the
files. LAGOON shows them as nets, coloured by tension."""

from __future__ import annotations

import numpy as np
from lagoon_format.store import Store
from reef3d_fixtures import make_run
from test_convert import _same_meshes

from lagoonviz.core import Session
from lagoonviz.core.convert import convert
from lagoonviz.core.data.reef3d import open_case
from lagoonviz.core.data.vtkxml import read_legacy

ROWS, COLUMNS = 4, 5  # knots of a wall net


def _net(number: int, frame: int):
    """A wall net: knots on a grid in the y-z plane, bent by the flow, its bars the
    grid lines (the border bars written after the inner ones, as REEF3D does)."""
    y, z = np.meshgrid(np.linspace(1.0, 3.0, COLUMNS), np.linspace(0.2, 1.0, ROWS))
    x = 2.0 + 3.0 * number + 0.05 * frame * np.sin(np.pi * (z - 0.2) / 0.8)
    knots = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    k = np.arange(ROWS * COLUMNS).reshape(ROWS, COLUMNS)
    across = np.stack([k[:, :-1].ravel(), k[:, 1:].ravel()], axis=1)
    down = np.stack([k[:-1, :].ravel(), k[1:, :].ravel()], axis=1)
    bars = np.concatenate([across, down])
    tension = 100.0 * (number + 1) + 10.0 * frame + np.arange(len(knots))
    velocity = np.zeros_like(knots)
    velocity[:, 0] = 0.1 * frame
    return knots, bars, tension, velocity


def write_nets(root, solver: str = "CFD", frames: int = 3, nets: int = 2):
    """REEF3D_<SOLVER>_6DOF_Net as REEF3D writes it (net_barDyn_print.cpp), with a
    sheet net's .stl and Lagrange points beside (not read)."""
    folder = root / f"REEF3D_{solver}_6DOF_Net"
    folder.mkdir(parents=True, exist_ok=True)
    for frame in range(frames):
        for number in range(nets):
            knots, bars, tension, velocity = _net(number, frame)
            lines = [
                "# vtk DataFile Version 2.0",
                f"Net {number}",
                "ASCII ",
                "DATASET UNSTRUCTURED_GRID",
                f"POINTS {len(knots)} float",
                *(f"{a:g} {b:g} {c:g}" for a, b, c in knots),
                "",
                f"CELLS {len(bars)} {3 * len(bars)}",
                *(f"2 {a} {b}" for a, b in bars),
                "",
                f"CELL_TYPES {len(bars)}",
                *(["3"] * len(bars)),
                "",
                f"POINT_DATA {len(knots)}",
                "SCALARS Tension float 1 ",
                "LOOKUP_TABLE default",
                *(f"{t:g}" for t in tension),
                "VECTORS Velocity float",
                *(f"{a:g} {b:g} {c:g}" for a, b, c in velocity),
                "VECTORS Acceleration float",
                *(f"{a:g} {b:g} {c:g}" for a, b, c in 0 * velocity),
            ]
            (folder / f"REEF3D-Net-{number:08d}-{frame:06d}.vtk").write_text(
                "\n".join(lines) + "\n"
            )
        (folder / f"REEF3D-Net-00000009-{frame:06d}.stl").write_text("solid net\nendsolid net\n")
        (folder / f"REEF3D-Net-Lagrange-{frame:08d}.pvtu").write_text("<VTKFile/>\n")
    return folder


def test_a_net_file_is_read_as_its_bars(tmp_path):
    folder = write_nets(tmp_path, frames=1, nets=1)
    piece = read_legacy(folder / "REEF3D-Net-00000000-000000.vtk")
    knots, bars, tension, velocity = _net(0, 0)
    assert piece.kind == "Lines"
    np.testing.assert_allclose(piece.points, knots, atol=1e-6)
    np.testing.assert_array_equal(piece.connectivity.reshape(-1, 2), bars)
    np.testing.assert_allclose(piece.point_data["Tension"], tension)
    assert piece.point_data["Velocity"].shape == (len(knots), 3)
    assert set(piece.point_data) == {"Tension", "Velocity", "Acceleration"}


def test_the_nets_are_shown_with_their_run(tmp_path):
    run = make_run(tmp_path / "cfd", solver="CFD", frames=3, print_interval=0.5)
    write_nets(run)
    case = open_case(run)
    nets = case.datasets["cfd_net"]
    assert nets.role == "net" and nets.kind == "Lines" and len(nets) == 3
    assert [len(f.pieces) for f in nets.frames] == [2, 2, 2]  # the two nets, side by side
    mesh = nets.read(2, ["Tension"])
    bars = (ROWS * (COLUMNS - 1) + (ROWS - 1) * COLUMNS) * 2
    assert mesh.segments is not None and len(mesh.segments) == bars
    assert mesh.point_data["Tension"].max() == 200.0 + 20.0 + ROWS * COLUMNS - 1
    s = Session()
    s.open_case(run)
    item = next(i for i in s.scene if i.kind == "net")
    assert item.visible and item.properties["variable"] == "Tension"
    shown = s.data.load(item.id)
    assert shown is not None and len(shown.segments) == bars


def test_the_nets_in_the_store(tmp_path):
    run = make_run(tmp_path / "cfd", solver="CFD", frames=3, print_interval=0.5)
    write_nets(run)
    vtk = open_case(run, stores=False)
    report = convert(run)
    assert report.outputs["net"] == 3
    store = Store(report.store)
    nets = store.particles["cfd_net"]
    assert nets.cells == "lines" and nets.info["role"] == "net"
    assert nets.group.array("cell_sets/cell_end").shape[0] == 1  # the bars stay the bars
    case = open_case(run)
    _same_meshes(vtk.datasets["cfd_net"], case.datasets["cfd_net"], range(3))
    assert "LAGOON" in case.datasets["cfd_net"].label


def test_nets_written_by_reef3d(tmp_path):
    """As REEF3D writes them with P 18 (lagoon_nets.cpp): the knots of all nets of an
    output, the bars with the number of their net, Velocity NaN for a quasi-static
    net among dynamic ones; no .vtk files (P 18 1)."""
    from lagoon_format.store import StoreWriter

    root = tmp_path / "cfd"
    root.mkdir()
    writer = StoreWriter.create(root / "REEF3D_CFD.lagoon", solver="CFD", run={"run": "R1"})
    fields = {"Tension": (1, "float32"), "Velocity": (3, "float32"), "Acceleration": (3, "float32")}
    nets = writer.particle_writer(
        "cfd_net",
        {
            "point_fields": fields,
            "cells": "lines",
            "cell_fields": {"net": (1, "int32")},
            "source": "REEF3D_CFD_6DOF_Net",
            "attributes": {"dataset": "cfd_net", "role": "net"},
        },
    )
    for frame in range(2):
        a, bars, tension, velocity = _net(0, frame)
        b, _, other, _ = _net(1, frame)
        points = np.concatenate([a, b]).astype(np.float32)
        nan = np.full_like(velocity, np.nan)  # net 1: quasi-static
        data = {
            "Tension": np.concatenate([tension, other]).astype(np.float32),
            "Velocity": np.concatenate([velocity, nan]).astype(np.float32),
            "Acceleration": np.concatenate([0 * velocity, nan]).astype(np.float32),
        }
        cells = np.concatenate([bars, bars + len(a)]).astype(np.int32)
        offsets = np.arange(2, 2 * len(cells) + 1, 2, dtype=np.int32)
        number = np.repeat(np.int32([0, 1]), len(bars))
        nets.append(0.5 * frame, frame, points, data, cells.ravel(), offsets, {"net": number})
    nets.flush()
    writer.close()
    s = Session()
    s.open_case(root)
    item = next(i for i in s.scene if i.kind == "net")
    mesh = s.data.load(item.id)
    assert mesh is not None and len(mesh.segments) == 2 * len(bars)
    np.testing.assert_allclose(mesh.point_data["Tension"][: len(a)], _net(0, 0)[2])  # time 0
    every = s.data.dataset(item.id).read(1, None)
    np.testing.assert_allclose(every.point_data["Velocity"][: len(a)], _net(0, 1)[3])
    assert np.isnan(every.point_data["Velocity"][len(a) :]).all()


def test_mooring_lines_written_by_reef3d(tmp_path):
    """REEF3D's mooring lines of every type in the store (P 18, lagoon_moorings): the
    points of the lines side by side with Tension, the segments with their line."""
    from lagoon_format.store import StoreWriter

    root = tmp_path / "cfd"
    root.mkdir()
    writer = StoreWriter.create(root / "REEF3D_CFD.lagoon", solver="CFD", run={"run": "R1"})
    lines = writer.particle_writer(
        "cfd_mooring",
        {
            "point_fields": {"Tension": (1, "float32")},
            "cells": "lines",
            "cell_fields": {"line": (1, "int32")},
            "source": "REEF3D_CFD_6DOF_Mooring",
            "attributes": {"dataset": "cfd_mooring", "role": "mooring"},
        },
    )
    for frame in range(2):
        anchors = np.array([[0.5, 0.5, 0.0], [5.5, 5.5, 0.0]])
        points = np.concatenate([np.linspace(a, [3.0, 3.0, 2.2 - 0.1 * frame], 6) for a in anchors])
        segments = np.concatenate(
            [[[i, i + 1] for i in range(5)], [[i, i + 1] for i in range(6, 11)]]
        )
        lines.append(
            0.25 * frame,
            frame,
            points.astype(np.float32),
            {"Tension": np.linspace(50.0, 60.0 + frame, 12).astype(np.float32)},
            segments.ravel().astype(np.int32),
            np.arange(2, 21, 2, dtype=np.int32),
            {"line": np.repeat(np.int32([0, 1]), 5)},
        )
    lines.flush()
    writer.close()
    s = Session()
    s.open_case(root)
    item = next(i for i in s.scene if i.kind == "mooring")
    assert item.properties["variable"] == "Tension"
    mesh = s.data.load(item.id, 0.25)
    assert mesh is not None and len(mesh.segments) == 10 and mesh.n_points == 12
    assert mesh.point_data["Tension"].max() == np.float32(61.0)
