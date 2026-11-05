"""DIVEMesh's solids: built from control.txt or read from DIVEMesh_Paraview, shown as
a structure, their faces in the bed left out."""

import math
import os

import numpy as np
import pytest
from reef3d_fixtures import write_piece, write_run_log

from lagoonviz.core import Session
from lagoonviz.core.data.divemesh import (
    SOLID_VTP,
    bed_faces,
    find_solids,
    left_over,
    solids_from_control,
)

#: the fixture's domain (x 0..8, y 0..4), its bed z = 0.1 x under still water at 2
CONTROL = """\
B 10 0.0 8.0 0.0 4.0 -0.5 3.0   // domain
S 33 4.0 2.0 0.3                 // monopile
S 61 0.0 8.0 0.0 4.0 0.0 0.8     // the sloping bed
S 10 0.0 8.0 0.0 4.0 -0.5 0.0    // under it
S 12 1 2 3 4 5 6 7 8             // a beam: only from DIVEMesh's own file
"""


def _volume(points, triangles):
    a, b, c = (np.asarray(points, float)[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def test_solids_from_control(tmp_path):
    control = tmp_path / "control.txt"
    control.write_text(CONTROL)
    points, triangles, skipped = solids_from_control(control)
    assert skipped == ["S 12"]
    cylinder = math.pi * 0.3**2 * 3.5
    wedge = 0.5 * 8.0 * 0.8 * 4.0
    box = 8.0 * 4.0 * 0.5
    assert _volume(points, triangles) == pytest.approx(cylinder + wedge + box, rel=0.01)
    assert find_solids(tmp_path) == control
    # the top of the wedge is the bed: those faces go
    x, y = np.meshgrid(np.linspace(0, 8, 9), np.linspace(0, 4, 5))
    bed = np.c_[x.ravel(), y.ravel(), 0.1 * x.ravel()]
    in_bed = bed_faces(points, triangles, bed)
    assert in_bed.sum() == 2
    tops = points[triangles[in_bed]]
    assert np.allclose(tops[..., 2], 0.1 * tops[..., 0], atol=1e-5)


@pytest.mark.parametrize(
    "line, volume",
    [
        ("S 51 0 0 0 1", 4 / 3 * math.pi),
        ("S 43 0 0 0 2 1 0", math.pi * 2 / 3),  # a cone
        ("S 62 0 1 0 2 0 3", 0.5 * 2 * 3 * 1),
        ("S 63 0 1 0 2 0 3", 0.5 * 1 * 3 * 2),
        ("S 84 0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1", 1.0),
    ],
)
def test_shapes_are_closed_and_face_out(tmp_path, line, volume):
    control = tmp_path / "control.txt"
    control.write_text(line + "\n")
    points, triangles, skipped = solids_from_control(control)
    assert not skipped
    assert _volume(points, triangles) == pytest.approx(volume, rel=0.02)


def test_case_shows_its_solids(reef3d_run):
    (reef3d_run / "control.txt").write_text(CONTROL)
    session = Session()
    session.open_case(reef3d_run)
    solid = next(i for i in session.scene if i.kind == "structure")
    assert solid.properties["dataset"] == "divemesh_solids"
    mesh = session.data.load(solid.id)
    assert len(mesh.triangles)
    # the monopile stands from the domain's bottom to its top
    assert mesh.points[:, 2].max() == pytest.approx(3.0)
    # no faces left in the bed
    bed_item = next(i for i in session.scene if i.kind == "bathymetry")
    bed = session.data.load(bed_item.id).points
    assert not bed_faces(mesh.points, mesh.triangles, bed).any()
    # shown at any time
    session.set_time(2.0)
    assert len(session.data.load(solid.id).triangles) == len(mesh.triangles)


def test_divemesh_output_is_preferred(reef3d_run):
    (reef3d_run / "control.txt").write_text(CONTROL)
    folder = reef3d_run / "DIVEMesh_Paraview"
    folder.mkdir()
    corners = np.array(
        [(x, y, z) for x in (1.0, 2.0) for y in (1.0, 2.0) for z in (2.5, 3.0)], np.float32
    )
    faces = np.array(
        [
            (0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
            (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5),
        ]
    )  # fmt: skip
    write_piece(
        folder / "REEF3D_Solid.vtp",
        "PolyData",
        corners,
        {},
        faces.ravel(),
        np.arange(1, len(faces) + 1) * 3,
    )
    session = Session()
    session.open_case(reef3d_run)
    solid = next(i for i in session.scene if i.kind == "structure")
    mesh = session.data.load(solid.id)
    assert len(mesh.triangles) == 12  # DIVEMesh's box, not the control.txt solids
    assert mesh.points[:, 2].min() == pytest.approx(2.5)


def _box_vtp(path, low, high):
    """A box (corners ``low``, ``high``) as DIVEMesh's REEF3D_Solid.vtp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    corners = np.array(
        [
            (x, y, z)
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        np.float32,
    )
    faces = np.array(
        [
            (0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
            (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2), (1, 3, 7), (1, 7, 5),
        ]
    )  # fmt: skip
    write_piece(path, "PolyData", corners, {}, faces.ravel(), np.arange(1, len(faces) + 1) * 3)
    return path


def _age(path, seconds):
    """Make ``path`` ``seconds`` older."""
    info = path.stat()
    os.utime(path, (info.st_atime - seconds, info.st_mtime - seconds))


def _structures(case):
    session = Session()
    session.open_case(case)
    return [i for i in session.scene if i.kind == "structure"]


def test_solids_of_another_case_are_not_opened(reef3d_run):
    """In a folder where several cases run, DIVEMesh's solid file can be another
    case's: control.txt without solids means the case has none."""
    _box_vtp(reef3d_run / SOLID_VTP, (1.0, 1.0, 2.5), (2.0, 2.0, 3.0))
    (reef3d_run / "control.txt").write_text("B 10 0.0 8.0 0.0 4.0 -0.5 3.0\n")
    assert find_solids(reef3d_run) is None
    assert not _structures(reef3d_run)


def test_solids_left_over_by_an_earlier_divemesh_run(reef3d_run):
    """DIVEMesh ran since (its partition is newer) without writing the solid file:
    the file is an earlier case's, also without control.txt to tell."""
    vtp = _box_vtp(reef3d_run / SOLID_VTP, (1.0, 1.0, 2.5), (2.0, 2.0, 3.0))
    partition = reef3d_run / "DIVEMesh_Paraview" / "DIVEMesh_Partition.vtp"
    partition.write_text("<VTKFile/>")
    assert find_solids(reef3d_run) == vtp  # written by the same DIVEMesh run
    _age(vtp, 2 * 86400.0)  # two days before DIVEMesh's last run
    assert left_over(vtp, reef3d_run)
    assert find_solids(reef3d_run) is None
    assert not _structures(reef3d_run)
    # with the solids in control.txt, they are built from it
    (reef3d_run / "control.txt").write_text(CONTROL)
    assert find_solids(reef3d_run) == reef3d_run / "control.txt"


def test_solids_older_than_control_are_built_from_it(reef3d_run):
    """control.txt changed since DIVEMesh ran: its solids, not the old file's."""
    vtp = _box_vtp(reef3d_run / SOLID_VTP, (1.0, 1.0, 2.5), (2.0, 2.0, 3.0))
    (reef3d_run / "control.txt").write_text(CONTROL)
    _age(vtp, 3600.0)
    assert find_solids(reef3d_run) == reef3d_run / "control.txt"
    # only solids that cannot be built from control.txt: DIVEMesh's file after all
    (reef3d_run / "control.txt").write_text("S 12 1 2 3 4 5 6 7 8\n")
    assert find_solids(reef3d_run) == vtp


def test_solids_outside_the_runs_domain_are_not_shown(reef3d_run):
    """The run log gives the run's domain: solids elsewhere are another case's."""
    write_run_log(
        reef3d_run,
        "NHFLOW",
        [
            {
                "id": "run-1",
                "started": "2026-10-14T06:00:00Z",
                "outputs": {"fsf": [(i, float(i)) for i in range(3)]},
                "domain": {"min": [0.0, 0.0, -0.5], "max": [8.0, 4.0, 3.0]},
            }
        ],
    )
    vtp = _box_vtp(reef3d_run / SOLID_VTP, (1.0, 1.0, 2.5), (2.0, 2.0, 3.0))
    assert [i.properties["dataset"] for i in _structures(reef3d_run)] == ["divemesh_solids"]
    _box_vtp(vtp, (15000.0, 6000.0, 0.0), (15010.0, 6010.0, 1.0))  # a fjord far away
    assert not _structures(reef3d_run)


def _cube_stl(path, binary):
    """A unit cube 0..1 as an STL file, ASCII or binary (outward normals)."""
    faces = []
    for axis in range(3):
        for side in (0.0, 1.0):
            square = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                corner = [0.0, 0.0, 0.0]
                corner[axis] = side
                corner[(axis + 1) % 3], corner[(axis + 2) % 3] = float(a), float(b)
                square.append(corner)
            first, second = [square[0], square[1], square[2]], [square[0], square[2], square[3]]
            if side == 0.0:
                first, second = first[::-1], second[::-1]
            faces += [first, second]
    if binary:
        record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("a", "<u2")])
        table = np.zeros(len(faces), record)
        table["corners"] = faces
        path.write_bytes(b"\0" * 80 + np.uint32(len(faces)).tobytes() + table.tobytes())
    else:
        lines = ["solid cube"]
        for face in faces:
            lines += ["facet normal 0 0 0", "outer loop"]
            lines += [f"vertex {x} {y} {z}" for x, y, z in face]
            lines += ["endloop", "endfacet"]
        path.write_text("\n".join(lines + ["endsolid cube"]) + "\n")


@pytest.mark.parametrize("binary", [False, True])
def test_the_stl_geometry_from_control(tmp_path, binary):
    """S 1 1: DIVEMesh's STL geometry (solid.stl), placed as DIVEMesh places it
    (scaled by S 4, moved by S 7, turned by S 8 about z), with the other solids when
    they are built from control.txt (bug: the ogee weir of an STL was missing when
    DIVEMesh's own solid file was not there or older than control.txt)."""
    _cube_stl(tmp_path / "solid.stl", binary)
    control = tmp_path / "control.txt"
    control.write_text(
        "B 10 -5 5 -5 5 0 3\nS 1 1   // solid.stl\nS 4 2.0\nS 7 1.0 0.0 0.0\nS 8 90.0\n"
        "S 10 3 4 3 4 0 1   // a box\n"
    )
    points, triangles, skipped = solids_from_control(control)
    assert skipped == []
    assert len(triangles) == 12 + 12
    stl = points[: 12 * 3]
    # 2 m wide, moved 1 m in x (1..3), turned 90° about z: x -2..0, y 1..3
    np.testing.assert_allclose(stl.min(axis=0), [-2.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(stl.max(axis=0), [0.0, 3.0, 2.0], atol=1e-6)
    assert _volume(points, triangles) == pytest.approx(8.0 + 1.0)
    assert find_solids(tmp_path) == control  # buildable: the STL alone would do
    (tmp_path / "solid.stl").unlink()
    assert solids_from_control(control)[2] == ["S 1"]  # its file gone: left out
