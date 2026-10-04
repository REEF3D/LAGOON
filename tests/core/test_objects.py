"""Floating bodies (6DOF VTP), mooring lines (legacy VTK) and structures."""

import numpy as np
import pytest
from reef3d_fixtures import body_offset, make_run

from lagoonviz.core import Session
from lagoonviz.core.data.reef3d import open_case
from lagoonviz.core.data.vtkxml import VTKFormatError, read_legacy


@pytest.fixture
def run(tmp_path):
    return make_run(tmp_path / "moored", frames=4, volume=False, objects=True)


def test_objects_are_found(run):
    case = open_case(run)
    body = case.datasets["nhflow_body"]
    mooring = case.datasets["nhflow_mooring"]
    assert body.steps == [0, 1, 2, 3] and body.label == "NHFLOW bodies"
    assert mooring.steps == [1, 2, 3] and mooring.variables() == ["Tension"]
    assert len(mooring.frames[0].pieces) == 2  # one piece per line


def test_body_moves_with_its_frames(run):
    body = open_case(run).datasets["nhflow_body"]
    for frame in (0, 2):
        mesh = body.read(frame, None)
        assert mesh.n_triangles == 12 and mesh.segments is None
        np.testing.assert_allclose(mesh.points.min(axis=0), body_offset(frame), atol=1e-6)


def test_mooring_lines(run):
    mooring = open_case(run).datasets["nhflow_mooring"]
    mesh = mooring.read(1, ["Tension"])  # output 2
    assert mesh.n_triangles == 0 and mesh.segments.shape == (10, 2)
    assert mesh.segments.max() < mesh.n_points == 12
    np.testing.assert_allclose(mesh.point_data["Tension"][:6], 1200 + np.arange(6))


def test_open_case_shows_body_and_mooring(run):
    s = Session()
    s.open_case(run)
    kinds = [i.kind for i in s.scene]
    assert kinds == ["case", "bathymetry", "body", "mooring", "free_surface"]
    shown = s.data.display("mooring-1", s.data.load("mooring-1"))
    assert shown.segments is not None and shown.variable == "Tension"
    body = s.data.display("body-1", s.data.load("body-1"))
    assert body.values is None and body.color == "#8f9aa6"  # steel
    s.set_property("body-1", "material", "concrete")
    assert s.data.display("body-1", s.data.load("body-1")).color == "#b4afa6"
    # the mooring starts at output 1: at time 0 its first frame is shown
    assert s.data.frame_index("mooring-1", 0) == 0


def test_static_structure(run):
    from reef3d_fixtures import BOX, write_piece

    folder = run / "REEF3D_NHFLOW_VRANS_VTP"
    folder.mkdir()
    write_piece(
        folder / "REEF3D-NHFLOW-VRANS.vtp", "PolyData", BOX, {}, np.arange(4), np.array([4])
    )
    case = open_case(run)
    structure = case.datasets["nhflow_structure"]
    assert structure.steps == [0] and structure.frame_index(500) == 0
    mesh = structure.read(0, None)
    assert mesh.n_triangles == 2


def test_legacy_reader_errors(tmp_path):
    bad = tmp_path / "grid.vtk"
    bad.write_text(
        "# vtk DataFile Version 2.0\nx\nASCII\nDATASET STRUCTURED_POINTS\nPOINTS 1 float\n0 0 0\n"
    )
    with pytest.raises(VTKFormatError):
        read_legacy(bad)


def test_a_converted_case_shows_its_bodies_from_the_store(run):
    from lagoonviz.core.convert import convert
    from lagoonviz.core.data.lagoon import StoreBodyDataset

    convert(run)
    s = Session()
    s.open_case(run)
    assert [i.kind for i in s.scene] == ["case", "bathymetry", "body", "mooring", "free_surface"]
    dataset = s.data.case("case-1").datasets["nhflow_body"]
    assert isinstance(dataset, StoreBodyDataset)
    body = s.data.display("body-1", s.data.load("body-1"))
    assert body.values is None and body.color == "#8f9aa6"
    mesh = dataset.read(2)
    assert mesh.n_triangles == 12
    np.testing.assert_allclose(mesh.points.min(axis=0), body_offset(2), atol=1e-5)
