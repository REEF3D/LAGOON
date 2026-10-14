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
