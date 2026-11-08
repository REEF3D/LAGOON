# LAGOON — a run of half a symmetric domain shown whole (Mirror in y)
# Architect: Hans Bihs
"""The case's *Mirror in y*: every item of the case with its mirror image about the
domain's lower or upper side, its vectors' and y components' signs turned."""

from __future__ import annotations

import numpy as np
import pytest
from reef3d_fixtures import make_run

from lagoonviz.core import Session
from lagoonviz.core.data.mesh import SurfaceMesh, mirrored_y


def test_the_mirror_image_of_a_mesh():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    mesh = SurfaceMesh(
        points,
        np.array([[0, 1, 2]], np.uint32),
        {
            "velocity": np.array([[1, 2, 3]] * 3, np.float32),
            "vorticity": np.array([[1, 2, 3]] * 3, np.float32),
            "v": np.array([2, 2, 2], np.float32),
            "eta": np.array([5, 6, 7], np.float32),
            "id": np.array([0, 1, 2]),
        },
        normals=np.array([[0, 0.6, 0.8]] * 3, np.float32),
        segments=np.array([[0, 1]], np.uint32),
    )
    both = mirrored_y(mesh, -5.0)  # far from it: no strip between them
    assert both.n_points == 6 and both.n_triangles == 2
    np.testing.assert_allclose(both.points[3:, 1], [-10, -10, -11])  # y' = 2 * -5 - y
    np.testing.assert_array_equal(both.triangles[1], [5, 4, 3])  # wound the other way
    np.testing.assert_allclose(both.point_data["velocity"][3], [1, -2, 3])
    np.testing.assert_allclose(both.point_data["vorticity"][3], [-1, 2, -3])  # pseudo
    np.testing.assert_allclose(both.point_data["v"][3:], -2)
    np.testing.assert_allclose(both.point_data["eta"][3:], [5, 6, 7])
    np.testing.assert_array_equal(both.point_data["id"][3:], [-1, -2, -3])
    np.testing.assert_allclose(both.normals[3], [0, -0.6, 0.8], atol=1e-6)
    np.testing.assert_array_equal(both.segments, [[0, 1], [3, 4]])


def test_a_case_mirrored_about_its_lower_side(tmp_path):
    run = make_run(tmp_path / "half", frames=2)
    (run / "control.txt").write_text("B 10 0.0 8.0 0.0 4.0 0.0 1.0\n")
    session = Session()
    session.open_case(run)
    surface = next(i for i in session.scene if i.kind == "free_surface")
    before = session.data.load(surface.id)
    assert session.data.mirror_plane(surface.id) is None
    case = next(i for i in session.scene if i.kind == "case")
    session.set_property(case.id, "mirror_y", "lower side")
    assert session.data.mirror_plane(surface.id) == 0.0
    after = session.data.load(surface.id)
    assert after.n_points == 2 * before.n_points
    assert after.points[:, 1].min() == pytest.approx(-before.points[:, 1].max())
    shown = session.data.display(surface.id, after)
    assert len(shown.triangles) == 2 * len(session.data.display(surface.id, before).triangles)
    # the upper side: mirrored about y = 4 (control.txt's B 10)
    session.set_property(case.id, "mirror_y", "upper side")
    assert session.data.mirror_plane(surface.id) == 4.0
    upper = session.data.load(surface.id)
    assert upper.points[:, 1].max() == pytest.approx(8.0 - before.points[:, 1].min())
    session.set_property(case.id, "mirror_y", "off")
    assert session.data.load(surface.id).n_points == before.n_points


def test_no_gap_at_the_plane():
    """A surface whose points stop short of the plane (cells' centres, half a cell
    from the wall): a strip of triangles joins it to its image; on the plane, none."""
    x, y = np.meshgrid(np.arange(0.0, 3.0), np.arange(0.5, 3.0), indexing="ij")  # y from 0.5
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)].astype(np.float32)
    ny = y.shape[1]
    triangles = []
    for i in range(x.shape[0] - 1):
        for j in range(ny - 1):
            a = i * ny + j
            triangles += [[a, a + ny, a + 1], [a + ny, a + ny + 1, a + 1]]
    mesh = SurfaceMesh(points, np.array(triangles, np.uint32), {})
    both = mirrored_y(mesh, 0.0)
    extra = both.n_triangles - 2 * mesh.n_triangles
    assert extra == 2 * (x.shape[0] - 1)  # a quad per cell along the plane
    centres = both.points[both.triangles[2 * mesh.n_triangles :]].mean(axis=1)
    assert np.allclose(centres[:, 1], 0.0, atol=0.34)  # across the plane
    # the strip faces as the surface does (up), not down
    corners = both.points[both.triangles].astype(np.float64)
    normal_z = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2]
    assert (np.sign(normal_z) == np.sign(normal_z[0])).all()
    on_plane = mirrored_y(mesh, 0.5)
    assert on_plane.n_triangles == 2 * mesh.n_triangles


def test_another_cases_control_txt_does_not_move_the_plane(tmp_path):
    """Bug: a store opened in a folder whose control.txt is another case's (the home
    folder) was mirrored about that domain's side, far from its own: a wide gap. B 10
    counts only when the results lie in it."""
    run = make_run(tmp_path / "half", frames=2)
    (run / "control.txt").write_text("B 10 0.0 8640.0 1000.0 1001.5 0.0 1.0\n")
    session = Session()
    session.open_case(run)
    surface = next(i for i in session.scene if i.kind == "free_surface")
    low = float(session.data.load(surface.id).points[:, 1].min())
    case = next(i for i in session.scene if i.kind == "case")
    session.set_property(case.id, "mirror_y", "lower side")
    assert session.data.mirror_plane(surface.id) == pytest.approx(low)
