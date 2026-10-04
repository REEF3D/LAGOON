"""Streamlines through column-structured (σ) volumes."""

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core.streamlines import ColumnGrid, ColumnPlan, NotColumnGrid, trace


def sigma_grid(field):
    """A σ-grid: 6 x 4 columns, 5 levels between a sloping bed and a wavy surface,
    listed in a jumbled order and with one MPI boundary column twice."""
    xs, ys, s = np.linspace(0, 5, 6), np.linspace(0, 3, 4), np.linspace(0, 1, 5)
    x, y, sig = np.meshgrid(xs, ys, s, indexing="ij")
    bed, top = -2.0 + 0.1 * x, 0.2 * np.sin(x + y)
    z = bed + sig * (top - bed)
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    values = field(points)
    order = np.random.default_rng(3).permutation(len(points))
    return points[order], values[order]


def two_ranks(points, values):
    """Split like two MPI ranks that share the column line x = 2 (written by both)."""
    left, right = points[:, 0] <= 2.0, points[:, 0] >= 2.0
    return [points[left], points[right]], np.concatenate([values[left], values[right]])


def test_linear_fields_are_sampled_exactly():
    linear = lambda p: 2 * p[:, 0] + 3 * p[:, 1] + 5 * p[:, 2]  # noqa: E731
    points, values = sigma_grid(linear)
    pieces, joined = two_ranks(points, values)
    plan = ColumnPlan.build(pieces)
    grid = plan.grid(np.concatenate(pieces), {"f": joined})
    assert grid.z.shape == (6, 4, 5)
    rng = np.random.default_rng(1)
    probe = np.stack([rng.uniform(0, 5, 50), rng.uniform(0, 3, 50), np.zeros(50)], axis=1)
    probe[:, 2] = -1.0 + 0.1 * probe[:, 0]  # inside the water everywhere
    sampled, inside = grid.sample("f", probe)
    assert inside.all()
    np.testing.assert_allclose(sampled, linear(probe), atol=1e-9)
    _, outside = grid.sample("f", np.array([[2.0, 1.0, 5.0], [9.0, 1.0, -1.0]]))
    assert not outside.any()  # above the surface, beyond the domain


def test_a_uniform_current_gives_straight_lines_across():
    points, velocity = sigma_grid(lambda p: np.tile([1.0, 0.0, 0.0], (len(p), 1)))
    grid = ColumnGrid.from_points(points, {"velocity": velocity})
    lines = trace(grid, "velocity", grid.seeds(0, 2.5, 6))
    assert lines
    for line in lines:
        assert np.ptp(line[:, 1]) < 1e-9 and np.ptp(line[:, 2]) < 1e-9  # straight in x
        assert np.ptp(line[:, 0]) > 4.0  # from one end of the domain to the other


def test_dry_columns_stop_lines():
    """FNPF writes a dry column with all its levels on the bed: it holds no water,
    and points over it are outside."""
    points, values = sigma_grid(lambda p: np.tile([1.0, 0.0, 0.0], (len(p), 1)))
    dry = (points[:, 0] == 5.0) & (points[:, 1] == 3.0)
    points[dry, 2] = points[dry, 2].min()  # all its levels on the bed
    grid = ColumnGrid.from_points(points, {"velocity": values})
    assert np.isnan(grid.z[5, 3]).all() and np.isfinite(grid.z[0, 0]).all()
    _, inside = grid.sample("velocity", np.array([[4.9, 2.9, -1.0], [1.0, 1.0, -1.0]]))
    assert inside.tolist() == [False, True]
    assert trace(grid, "velocity", grid.seeds(1, 3.0, 4))  # still lines elsewhere


def test_scattered_points_are_refused():
    rng = np.random.default_rng(0)
    with pytest.raises(NotColumnGrid):
        ColumnGrid.from_points(rng.uniform(size=(60, 3)), {})


def test_streamline_item(reef3d_run):
    s = Session()
    s.open_case(reef3d_run)
    item = s.add_streamlines()
    props = s.scene.get(item).properties
    assert props["seed_normal"] == "y" and props["seed_position"] == pytest.approx(2.0)
    mesh = s.data.load(item)
    assert mesh.segments is not None and len(mesh.segments) > 10
    assert mesh.triangles.size == 0 and "velocity" in mesh.point_data
    shown = s.data.display(item, mesh)
    assert shown.segments is not None and shown.values is not None  # coloured by speed
    low, high = s.data.case(item).datasets["nhflow_volume"].bounds(0)
    assert np.all(mesh.points >= low - 1e-4) and np.all(mesh.points <= high + 1e-4)


def test_lines_in_the_seed_plane_and_on_the_free_surface(reef3d_run):
    from reef3d_fixtures import DEPTH, eta

    s = Session()
    s.open_case(reef3d_run)
    s.set_time(1.0)
    plane = s.add_streamlines("y", 2.0, lines_in="seed plane", seeds=20)
    mesh = s.data.load(plane)
    assert len(mesh.segments) > 5
    np.testing.assert_allclose(mesh.points[:, 1], 2.0, atol=1e-5)  # they stay in the plane

    surface = s.add_streamlines(lines_in="free surface", seeds=20)
    assert s.data.dataset(surface).role == "free_surface"
    mesh = s.data.load(surface)
    assert len(mesh.segments) > 5
    # on the surface: the height is the surface's at each point
    x, y = mesh.points[:, 0], mesh.points[:, 1]
    np.testing.assert_allclose(mesh.points[:, 2], DEPTH + eta(x, y, 1), atol=0.05)
