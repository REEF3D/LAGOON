"""Slices through volume output: marching tetrahedra on REEF3D's hexahedra."""

import numpy as np
import pytest
from reef3d_fixtures import DEPTH, NX, NY, eta

from lagoonviz.core import Session
from lagoonviz.core.data.mesh import cut_cells, plan_cut

CUBE = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float
)


def area(mesh) -> float:
    p = mesh.points.astype(float)
    t = mesh.triangles.astype(np.int64)
    return 0.5 * float(
        np.linalg.norm(np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]), axis=1).sum()
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("position", [0.25, 0.5, 0.9])
def test_cube_sections(axis, position):
    data = {"f": CUBE[:, axis].copy(), "v": CUBE.copy()}
    mesh = cut_cells(CUBE, np.arange(8), np.array([8]), np.array([12]), data, axis, position)
    assert area(mesh) == pytest.approx(1.0)
    np.testing.assert_allclose(mesh.points[:, axis], position, atol=1e-6)
    np.testing.assert_allclose(mesh.point_data["f"], position, atol=1e-6)  # linear data is exact
    np.testing.assert_allclose(mesh.point_data["v"][:, axis], position, atol=1e-6)


def test_plane_outside_cuts_nothing():
    plan = plan_cut(CUBE, np.arange(8), np.array([8]), np.array([12]), 0, 3.0)
    assert plan.n_points == 0 and plan.triangles.shape == (0, 3)


def test_slice_item_in_a_run(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    slice_id = s.add_slice("y", variable="pressure")
    props = s.scene.get(slice_id).properties
    assert props["position"] == pytest.approx((NY - 1) / 2)
    assert (
        "session.add_item('slice', normal='y', position=2.0, variable='pressure')" in s.to_script()
    )
    s.set_property(slice_id, "position", 2.5)
    mesh = s.data.load(slice_id)
    np.testing.assert_allclose(mesh.points[:, 1], 2.5, atol=1e-5)
    # the section spans the whole length and the water depth
    x, z = mesh.points[:, 0], mesh.points[:, 2]
    assert x.min() == pytest.approx(0) and x.max() == pytest.approx(NX - 1)
    assert z.min() == pytest.approx(0, abs=1e-6)
    # pressure is linear in z in the fixture, so the interpolation is exact
    np.testing.assert_allclose(
        mesh.point_data["pressure"], 1000 * 9.81 * (DEPTH - z), rtol=1e-4, atol=1e-2
    )
    shown = s.data.display(slice_id, mesh)
    assert shown.values is not None and shown.colormap == "ocean"


def test_vertical_slice_follows_the_free_surface(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    slice_id = s.add_slice("x", position=3.5)
    for frame in (0, 2):  # the cut plan is reused, the points move with eta
        s.set_time(frame)
        mesh = s.data.load(slice_id)
        top = mesh.points[:, 2].max()
        y = np.linspace(0, NY - 1, 50)
        expected = DEPTH + eta(np.full_like(y, 3.5), y, frame)
        assert top == pytest.approx(expected.max(), abs=0.01)


def test_horizontal_slice(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    slice_id = s.add_slice("z", position=0.7)
    mesh = s.data.load(slice_id)
    np.testing.assert_allclose(mesh.points[:, 2], 0.7, atol=1e-5)
    assert area(mesh) == pytest.approx((NX - 1) * (NY - 1), rel=1e-6)


def test_slice_without_volume_is_empty(tmp_path):
    from reef3d_fixtures import make_run

    run = make_run(tmp_path / "surfaces", volume=False)
    s = Session()
    s.open_case(run)
    slice_id = s.add_slice("y")
    assert s.scene.get(slice_id).properties["position"] == 0.0
    assert s.data.load(slice_id) is None
