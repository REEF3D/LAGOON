"""DIVEMesh's solids: built from control.txt or read from DIVEMesh_Paraview, shown as
a structure, their faces in the bed left out."""

import math

import numpy as np
import pytest
from reef3d_fixtures import write_piece

from lagoonviz.core import Session
from lagoonviz.core.data.divemesh import bed_faces, find_solids, solids_from_control

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
