"""White water: born where the surface breaks, then spray, foam and bubbles."""

import math

import numpy as np
import pytest

from lagoonviz.core.foam import FoamSettings, FoamSimulation, SurfaceGrid, potentials

DEPTH = 1.0
C = math.sqrt(9.81 * DEPTH)
XS = np.arange(0.0, 20.0, 0.05)
YS = np.arange(0.0, 2.0, 0.05)


def wave(t: float, breaking: bool = False) -> SurfaceGrid:
    """A steep, skewed wave running along x at the shallow water speed."""
    x, _ = np.meshgrid(XS, YS, indexing="ij")
    phase = x - 2.0 - C * t
    eta = 0.45 * np.exp(-((phase / 0.6) ** 2)) * (1 + 0.6 * np.tanh(phase / 0.3))
    u = C * eta / (DEPTH + eta)
    flag = (np.abs(phase) < 0.2) * 1.0 if breaking else None
    return SurfaceGrid(XS, YS, DEPTH + eta, u, 0 * u, 0 * u, flag)


def run(seconds: float, **settings) -> FoamSimulation:
    sim = FoamSimulation(DEPTH, FoamSettings(**settings))
    steps = int(round(seconds / 0.1))
    for k in range(steps):
        sim.advance(wave(0.1 * k), wave(0.1 * (k + 1)), 0.1 * k, 0.1 * (k + 1))
    return sim


def test_grid_from_points_in_any_order():
    x, y = np.meshgrid(XS[:30], YS[:10], indexing="ij")
    points = np.c_[x.ravel(), y.ravel(), np.sin(x).ravel()]
    order = np.random.default_rng(3).permutation(len(points))
    grid = SurfaceGrid.from_points(points[order], np.zeros((len(points), 3)))
    assert grid.z.shape == (30, 10)
    assert np.allclose(grid.z, np.sin(x))
    z = grid.sample(np.array([[0.025, 0.0], [99.0, 0.0]]), grid.z)[0]
    assert z[0] == pytest.approx(0.5 * (np.sin(0.0) + np.sin(0.05)), rel=1e-4)
    assert np.isnan(z[1])  # outside
    scattered = np.random.default_rng(1).random((200, 3))
    assert SurfaceGrid.from_points(scattered, np.zeros((200, 3))) is None


def test_a_surface_on_no_grid_is_put_on_one():
    """CFD's free surface (an iso-surface: points on the cells' edges, in no order) and
    AMR: put on a grid of the cell size, each grid point the highest point near it;
    a hole of a grid point or two takes its neighbours' mean, no false crest."""
    surface = wave(0.0)
    x, y = np.meshgrid(XS, YS, indexing="ij")
    nodes = np.c_[x.ravel(), y.ravel(), surface.z.ravel()]
    # the iso-surface's points between the nodes too (on the horizontal edges), lower
    half = np.c_[x.ravel() + 0.025, y.ravel(), surface.z.ravel() - 0.01][: -len(YS)]
    rng = np.random.default_rng(5)
    keep = rng.random(len(nodes)) > 0.02  # a few holes
    points = np.concatenate([nodes[keep], half])
    velocity = np.c_[np.interp(points[:, 0], XS, surface.u[:, 0]), 0 * points[:, :2]]
    order = rng.permutation(len(points))
    grid = SurfaceGrid.from_scattered(points[order], velocity[order], cell=0.05)
    assert grid.z.shape == surface.z.shape
    assert np.isfinite(grid.z).all()  # the holes filled
    assert np.abs(grid.z - surface.z).max() < 0.05  # half a cell of the steep face
    crest, _ = potentials(grid, DEPTH)
    assert crest.max() > 0.3 and crest[x > 6].max() == 0.0  # at the crest only
    # the cell size from the triangles' edges; the next frame on the same grid
    triangles = np.arange(3 * (len(nodes) // 3)).reshape(-1, 3)
    estimated = SurfaceGrid.from_scattered(nodes, 0 * nodes, triangles=triangles)
    assert estimated is not None and 0.02 < estimated.h < 0.2
    first_half = nodes[: len(nodes) // 2]
    later = SurfaceGrid.from_scattered(first_half, 0 * first_half, like=grid)
    assert later.z.shape == grid.z.shape and np.isnan(later.z).any()  # dry where none
    assert SurfaceGrid.from_scattered(nodes[:2], 0 * nodes[:2]) is None


def test_born_at_the_breaking_crest():
    crest, trapped = potentials(wave(0.0), DEPTH)
    x, _ = np.meshgrid(XS, YS, indexing="ij")
    assert crest.max() > 0.3
    assert abs(x[crest > 0.3].mean() - 2.0) < 0.6  # at the crest, not on the still water
    assert crest[x > 6].max() == 0.0
    # REEF3D's breaking flag counts in full
    flagged, _ = potentials(wave(0.0, breaking=True), DEPTH)
    assert flagged.max() == 1.0
    assert trapped.shape == crest.shape


def test_spray_foam_and_bubbles():
    sim = run(3.0)
    particles = sim.particles()
    assert set(particles) == {"spray", "foam", "bubbles"}
    surface = wave(3.0)
    crest = 2.0 + 3.0 * C
    for kind, (positions, radius) in particles.items():
        assert positions.dtype == np.float32 and len(radius) == len(positions)
        assert (radius > 0).all() and radius.max() < 0.5 * surface.h
        z = surface.sample(positions[:, :2], surface.z)[0]
        height = positions[:, 2] - z
        if kind == "spray":
            assert np.median(height) > 0  # above the water
        if kind == "bubbles":
            assert np.median(height) < 0  # under it
    foam = particles["foam"][0]
    assert foam[:, 0].min() < crest - 3.0  # a trail behind the crest
    assert foam[:, 0].max() < crest + 1.0  # none ahead of it


def test_same_settings_same_water_and_a_limit():
    a, b = run(1.0), run(1.0)
    assert np.array_equal(a.position, b.position)
    small = run(2.0, limit=500)
    assert len(small) <= 500


def test_how_much_white_water():
    from lagoonviz.core.foam import WHITE_WATER_AMOUNTS, white_water_amount

    assert white_water_amount(True) == WHITE_WATER_AMOUNTS["medium"] < 1.0
    assert white_water_amount(False) == 0.0 and white_water_amount("none") == 0.0
    assert white_water_amount("Heavy") == 1.0 and white_water_amount(0.5) == 0.5
    with pytest.raises(ValueError, match="white water"):
        white_water_amount("lots")
    less = FoamSettings().scaled(0.5)
    assert (less.rate, less.lifetime) == (10.0, 6.0)
    # less white water: fewer particles from the same breaking wave
    full = run(2.0)
    light = run(2.0, rate=20.0 * 0.35, lifetime=12.0 * 0.35)
    assert len(light) < 0.6 * len(full)


def test_coarse_and_fine_triangles_leave_no_holes():
    """AMR: a refined block's small triangles beside level 0's large ones; the grid
    (of the small cell size) has a value under the large ones too."""
    xc, yc = np.meshgrid(np.arange(0.0, 4.01), np.arange(0.0, 2.01), indexing="ij")
    coarse = np.c_[xc.ravel(), yc.ravel(), np.ones(xc.size)]  # cells of 1 m
    xf, yf = np.meshgrid(np.arange(4.0, 6.01, 0.1), np.arange(0.0, 2.01, 0.1), indexing="ij")
    fine = np.c_[xf.ravel(), yf.ravel(), np.ones(xf.size)]
    points = np.concatenate([coarse, fine])
    triangles = []
    for offset, (nx, ny) in ((0, xc.shape), (len(coarse), xf.shape)):
        for i in range(nx - 1):
            for j in range(ny - 1):
                a = offset + i * ny + j
                triangles += [[a, a + ny, a + 1], [a + ny, a + ny + 1, a + 1]]
    grid = SurfaceGrid.from_scattered(points, np.zeros_like(points), triangles=triangles, fill=0)
    assert grid.h == pytest.approx(0.1, rel=0.3)
    assert np.isfinite(grid.z).all() and np.allclose(grid.z, 1.0)
