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
    both = mirrored_y(mesh, -1.0)
    assert both.n_points == 6 and both.n_triangles == 2
    np.testing.assert_allclose(both.points[3:, 1], [-2, -2, -3])  # y' = 2 * -1 - y
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
