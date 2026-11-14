"""A river's channel carved into geo.dat where the terrain model has only the
water's surface: Manning's mean depth for a discharge over the water's width at
each cross-section, parabolic or trapezoidal, the land of the margin untouched;
carved again from the geo.dat before, and put back."""

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core import corridor as C
from lagoonviz.core.channel import carve, manning_depth

#: a river 80 m wide in a valley 5 m above it, its water falling 1 m over 5 km
XS, YS = np.arange(0, 5000.1, 10.0), np.arange(-300, 300.1, 10.0)


def terrain():
    x, y = np.meshgrid(XS, YS)
    level = 10.0 - 2e-4 * x
    z = np.where(np.abs(y) <= 40, level, level + 5.0)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def test_manning_depth():
    # Q = W h^(5/3) S^(1/2) / n
    h = manning_depth(200.0, 80.0, 2e-4, 0.035)
    assert 80.0 * h ** (5 / 3) * np.sqrt(2e-4) / 0.035 == pytest.approx(200.0)
    assert manning_depth(200.0, 80.0, 0.0, 0.035) == manning_depth(200.0, 80.0, 1e-5, 0.035)


@pytest.mark.parametrize("shape", ["parabolic", "trapezoid"])
def test_the_channel_carries_the_discharge(shape):
    points = terrain()
    corridor = C.from_line([(0, 0), (2500, 0), (5000, 0)], 160.0, 12)  # water and margin
    carved, found = carve(points, corridor, 200.0, 0.035, shape=shape)
    assert found.slope == pytest.approx(2e-4, rel=1e-3)  # from the water's surface
    ok = np.isfinite(found.depth)
    assert ok.mean() > 0.9
    np.testing.assert_allclose(found.width[ok], 80.0, atol=10.0)  # the water, not the margin
    np.testing.assert_allclose(found.level[ok], 10.0 - 2e-4 * found.along[ok], atol=0.05)
    lowered = points[:, 2] - carved[:, 2]
    assert (lowered >= 0).all()  # only ever lower
    land = np.abs(points[:, 1]) > 40
    assert not lowered[land].any()  # the margin's land stays
    # the cross-section's area: the width times Manning's depth (for the discharge)
    x, y = points[:, 0], points[:, 1]
    for at in (1000.0, 2500.0, 4000.0):
        section = np.isclose(x, at)
        integrate = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 and 1
        area = integrate(lowered[section][np.argsort(y[section])], np.sort(y[section]))
        h = manning_depth(200.0, 80.0, 2e-4, 0.035)
        # (8 cells across: the trapezoid's banks fall between them)
        assert area == pytest.approx(80.0 * h, rel=0.1 if shape == "trapezoid" else 0.05), at
    deepest = lowered.max()
    assert deepest == pytest.approx((1.5 if shape == "parabolic" else 1.25) * h, rel=0.05)
    assert "points lowered" in found.text()


def test_a_case_carves_its_river_again_and_back(tmp_path):
    case = tmp_path / "river"
    case.mkdir()
    (case / "control.txt").write_text("C 11 1\nB 1 10.0\nB 10 0 5000 -300 300 0 20\nM 10 1\n")
    (case / "ctrl.txt").write_text("A 10 2\nF 60 10.0\n")
    np.savetxt(case / "geo.dat", terrain(), fmt="%.4f")
    before = (case / "geo.dat").read_text()
    s = Session()
    s.open_case(case)
    with pytest.raises(ValueError):
        s.carve_river(200.0)  # no corridor yet
    s.set_corridor(C.from_line([(0, 0), (2500, 0), (5000, 0)], 160.0, 12))
    found = s.carve_river(200.0)
    assert found.lowered > 1000
    first = np.loadtxt(case / "geo.dat")
    assert (case / "LAGOON_geo-uncarved.dat").read_text() == before
    # again, with more water: from the geo.dat before, not deeper on top of the first
    deeper = s.carve_river(400.0, manning=0.04)
    again = np.loadtxt(case / "geo.dat")
    h1 = manning_depth(200.0, 80.0, 2e-4, 0.035)
    h2 = manning_depth(400.0, 80.0, 2e-4, 0.04)
    assert np.nanmax(10.0 - again[:, 2] - 2e-4 * again[:, 0]) == pytest.approx(1.5 * h2, rel=0.05)
    assert np.nanmax(deeper.depth) > np.nanmax(found.depth) and h2 > h1
    assert (first[:, 2] >= again[:, 2] - 1e-9).all()
    assert "session.carve_river(400.0, manning=0.04)" in s.to_script()
    # the set-up reads the carved bed
    points = s.data.case_setup(s._case_id(None)).geo_points(limit=10**9)
    assert points[:, 2].min() == pytest.approx(again[:, 2].min(), abs=1e-3)
    # back
    assert s.uncarve_river() is True
    assert (case / "geo.dat").read_text() == before
    assert not (case / "LAGOON_geo-uncarved.dat").exists()
    assert s.uncarve_river() is False
