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
