"""Vector arrows on surfaces."""

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core.vectors import arrow_segments, grid_spacing, pick_points


def test_points_are_thinned_evenly():
    xs, ys = np.meshgrid(np.linspace(0, 10, 101), np.linspace(0, 5, 51), indexing="ij")
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    picked = pick_points(points, np.arange(len(points)), 50)
    assert 30 <= len(picked) <= 80
    chosen = points[picked]
    assert np.ptp(chosen[:, 0]) > 8 and np.ptp(chosen[:, 1]) > 4  # over the whole surface
    # a vertical plane (y = const) is thinned in x and z, not to one column
    vertical = points[:, [0, 2, 1]]
    assert np.ptp(vertical[pick_points(vertical, np.arange(len(points)), 50)][:, 2]) > 4


def test_arrows_spaced_for_the_area_covered():
    """A narrow river across a wide domain gets the arrows wanted on its water, not a
    few spaced for the domain's bounding box."""
    xs, ys = np.meshgrid(np.linspace(0, 1000, 201), np.linspace(0, 1000, 201), indexing="ij")
    points = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    n = 201
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    base = (i * n + j).ravel()
    quads = np.stack([base, base + n, base + n + 1, base + 1], axis=1)
    # the water: a diagonal band 50 m wide
    centre = points[quads].mean(axis=1)
    wet = np.abs(centre[:, 0] - centre[:, 1]) < 25.0
    triangles = np.concatenate([quads[wet][:, [0, 1, 2]], quads[wet][:, [0, 2, 3]]])
    used = np.unique(triangles)
    box = pick_points(points, used, 400)
    assert len(box) < 80  # spaced for the whole box: few arrows on the river
    spacing = grid_spacing(points, 400, triangles)
    assert spacing == pytest.approx(np.sqrt(50 * 1000 / 400), rel=0.2)  # band: 50 x 1000 m²
    picked = pick_points(points, used, 400, spacing)
    assert 250 <= len(picked) <= 700  # cells the banks cut get one too
    assert grid_spacing(points, 400) == pytest.approx(50.0)  # no triangles: the box


def test_arrows_have_a_shaft_and_a_head():
    base = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    segments = arrow_segments(base, np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 0.5)
    assert segments.shape == (3, 2, 3)  # the zero vector draws nothing
    np.testing.assert_allclose(segments[0], [[0, 0, 0], [1, 0, 0]])  # shaft: 2 x 0.5
    np.testing.assert_allclose(segments[1:, 0], [[1, 0, 0], [1, 0, 0]])  # head at the tip
    assert np.all(segments[1:, 1, 0] < 1.0)  # strokes point back


def test_velocity_arrows_on_the_free_surface(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    surface = "free_surface-1"
    assert s.data.choices(surface, "vectors") == ["none", "velocity"]
    mesh = s.data.load(surface)
    assert s.data.vector_arrows(surface, mesh, s.data.display(surface, mesh)) is None  # off
    s.set_property(surface, "vectors_of", "velocity")
    s.set_property(surface, "vector_count", 12)
    mesh = s.data.load(surface)
    arrows = s.data.vector_arrows(surface, mesh, s.data.display(surface, mesh))
    assert arrows is not None and arrows.segments.shape[0] % 3 == 0
    assert 3 * 4 <= arrows.segments.shape[0] <= 3 * 30
    longest = np.linalg.norm(arrows.segments[:, 1] - arrows.segments[:, 0], axis=1).max()
    assert 0.1 < longest < 3.0  # about the arrow spacing
    s.set_property(surface, "vector_scale", 2.0)
    longer = s.data.vector_arrows(surface, mesh, s.data.display(surface, mesh))
    assert np.linalg.norm(longer.segments[:, 1] - longer.segments[:, 0], axis=1).max() > longest
    s.set_plan_view(True)
    flat = s.data.vector_arrows(surface, mesh, s.data.display(surface, mesh))
    assert np.ptp(flat.segments[..., 2]) < 1e-6  # flat on top in 2D
