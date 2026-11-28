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


def test_a_fine_surface_on_a_coarser_grid():
    """CFD's free surface, cut from a volume of fine cells (a river 80 m long in cells
    of 5 cm), on a grid of at most FoamSettings.grid points along its longer side: on
    its own cells the particles (a fraction of a grid cell) were millimetres, too
    small to see, and so many that the limit left little foam."""
    xs, ys = np.arange(0.0, 80.0, 0.05), np.arange(0.0, 20.0, 0.05)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    z = 2.0 + 0.2 * np.sin(x) * np.cos(y)
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]
    velocity = np.c_[np.full(len(points), 3.0), np.zeros((len(points), 2))]
    most = FoamSettings().grid
    grid = SurfaceGrid.from_scattered(points, velocity, cell=0.05, most=most)
    assert max(grid.z.shape) <= most and grid.h == pytest.approx(80.0 / (most - 1), rel=0.01)
    assert np.isfinite(grid.z).all()
    # each grid point the highest point near it
    assert np.nanmax(grid.z) == pytest.approx(z.max(), abs=0.01)
    fine = SurfaceGrid.from_scattered(points[:4000], velocity[:4000], cell=0.05, most=most)
    assert fine.h == pytest.approx(0.05)  # a small surface keeps its cells


def test_the_limit_keeps_the_foam():
    """At the limit only as many are born as have gone, not the oldest dropped: foam
    lives longest, and dropping the oldest left only spray and bubbles just born."""
    free = run(3.0).particles()
    limited = run(3.0, limit=300).particles()
    share = len(free["foam"][0]) / sum(len(p) for p, _ in free.values())
    kept = len(limited["foam"][0]) / sum(len(p) for p, _ in limited.values())
    assert sum(len(p) for p, _ in limited.values()) <= 300
    assert kept > 0.5 * share


def _cut(surface: SurfaceGrid, rough: float, seed: int = 3) -> SurfaceGrid:
    """A surface as CFD's is put on a grid: from points on no grid, the heights rough
    by ``rough`` [m] and the velocity by 5 ``rough`` [m/s] from cell to cell."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(surface.xs, surface.ys, indexing="ij")
    points = np.c_[x.ravel(), y.ravel(), surface.z.ravel() + rng.normal(0, rough, x.size)]
    velocity = np.c_[surface.u.ravel(), surface.v.ravel(), surface.w.ravel()]
    velocity = velocity + rng.normal(0, 5 * rough, velocity.shape)
    cell = float(surface.xs[1] - surface.xs[0])
    return SurfaceGrid.from_scattered(points, velocity, cell=cell)


def test_no_white_water_on_waves_that_do_not_break():
    """Hans: a CFD flume had much foam where its waves were made and on their way to
    the beach, before they broke. CFD's surface cut from the volume is rough from
    cell to cell; its curvature and the flow's convergence (derivatives) made crests
    and trapped air everywhere. Taken of it smoothed, and air trapped only near a
    breaking crest, a train of steep waves that do not break has next to none, and
    a breaking crest as much as before."""
    xs, ys = np.arange(0.0, 20.0, 0.05), np.arange(0.0, 2.0, 0.05)
    x, _ = np.meshgrid(xs, ys, indexing="ij")
    k, height = 2 * np.pi / 4.0, 0.3  # steep (H / d = 0.3), not breaking
    eta = 0.5 * height * np.cos(k * x) + 0.03 * np.cos(2 * k * x)
    u = 0.5 * height * np.sqrt(9.81 / DEPTH) * np.cos(k * x)  # u / sqrt(g d) < 0.15
    train = SurfaceGrid(xs, ys, DEPTH + eta, u, 0 * u, 0 * u)
    crest, trapped = potentials(_cut(train, 0.01), DEPTH)
    assert crest.sum() < 0.002 * crest.size and trapped.sum() < 0.001 * trapped.size
    # the breaking crest of the steep wave, cut as CFD's
    crest, trapped = potentials(_cut(wave(0.0), 0.01), DEPTH)
    xb, _ = np.meshgrid(XS, YS, indexing="ij")
    assert crest.max() > 0.5 and crest[np.abs(xb - 2.0) >= 1.0].sum() < 1.0
    assert trapped.sum() > 10.0  # air under it
