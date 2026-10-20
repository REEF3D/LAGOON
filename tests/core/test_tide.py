"""The tide at an open sea boundary: constituents with nodal corrections, a gauge's or
a file's levels, background-1.dat and REEF3D's B 510 - B 515, and the open edges
(B 520, B 521, B 523) of block 0's tide faces."""

import datetime as dt

import numpy as np
import pytest

from lagoonviz.core import Session
from lagoonviz.core import multiblock as M
from lagoonviz.core import tide as T
from lagoonviz.core.geo import gauges as gg
from lagoonviz.core.geo import metocean as mo

CASCAIS = "M2 1.00 61\nS2 0.35 89\nN2 0.21 44\nK2 0.10 87\nK1 0.07 52\nO1 0.06 310"


def test_constituents_and_their_sum():
    found = T.read_constants(CASCAIS + "\n# a comment\n")
    assert found["M2"] == (1.0, 61.0) and len(found) == 6
    with pytest.raises(ValueError, match="unknown"):
        T.read_constants("Z9 1 2")
    times = np.arange(0.0, 15 * 86400.0, 600.0)
    eta = T.synthesize(found, "2026-01-19T00:00", times)
    # spring and neap within a fortnight: about M2 + S2 and M2 - S2 (with N2, K2, ...)
    assert 1.4 < eta.max() < 1.9 and -1.9 < eta.min() < -1.2
    assert abs(eta.mean()) < 0.05
    # REEF3D's sum of B 511's constituents is the same tide
    parts = T.harmonic(found, "2026-01-19T00:00")
    reef3d = sum(a * np.cos(2 * np.pi * times / p - np.radians(ph)) for _n, a, p, ph in parts)
    assert np.abs(reef3d - eta).max() < 0.01
    # M2's period and the tide a lunar day later
    assert parts[0][2] == pytest.approx(12.4206012 * 3600)


def test_levels_from_a_file(tmp_path):
    seconds = tmp_path / "levels.txt"
    seconds.write_text("t level\n0 0.1\n3600 0.5\n7200 -0.2\n")
    t, z = T.read_levels(seconds)
    assert list(t) == [0, 3600, 7200] and z[1] == 0.5
    dates = tmp_path / "cascais.csv"
    dates.write_text("2026-01-19T00:00, 1.2\n2026-01-19 01:00;1.6\n2026-01-19T02:00Z,1.0\n")
    t, z = T.read_levels(dates, "2026-01-19T00:00")
    assert list(t) == [0, 3600, 7200]
    found = T.make("2026-01-19T00:00", days=1 / 12, step=1800, levels=dates)
    assert found.levels[0] == pytest.approx(1.2) and found.levels[-1] == pytest.approx(1.0)
    assert found.lines[0] == ("B 510", [1, 2, 0.0, 10800.0])
    text = T.background_text(found.times, found.levels)
    assert text.splitlines()[2] == "0.0 1.2000 0.0 0.0"
    assert (
        T.background_text(found.times, found.levels, current=True).splitlines()[2] == "0.0 1.2000"
    )


def test_a_tide_gauge(monkeypatch):
    station = gg.Station("IOC", "casc", "Cascais", 38.69, -9.42, 3.0)
    start = dt.datetime(2026, 1, 19, tzinfo=dt.UTC)
    times = start.timestamp() + np.arange(-3600.0, 2 * 86400.0, 600.0)
    level = 2.0 + np.cos(2 * np.pi * (times - times[0]) / 44714.0)

    def levels(st, begin, end, fetch=None, progress=None):
        return mo.Series(times, {"observed": level, "predicted": level - 0.1})

    monkeypatch.setattr(gg, "levels", levels)
    found = T.make(start, days=1.0, step=600.0, gauge=station)
    assert abs(found.levels.mean()) < 1e-9  # its mean taken away (the gauge's datum)
    assert found.levels.max() == pytest.approx(1.0, abs=0.05)
    assert "observed at Cascais" in found.text()
    with pytest.raises(ValueError, match="give the tide"):
        T.make(start, constants=CASCAIS, gauge=station)


@pytest.fixture
def estuary(tmp_path):
    folder = tmp_path / "estuary"
    folder.mkdir()
    (folder / "control.txt").write_text("C 11 21\nB 1 1.0\nB 10 0 1 0 1 0 5\nR 20 1\nM 10 2\n")
    (folder / "ctrl.txt").write_text(
        "A 10 5\nF 60 650.0\nB 520 7 1 0\nB 521 7 2 0.0 0.0 50.0\n"
        "B 520 1 3 0\nB 521 1 1 0.0 0.0 0.0\nB 523 1 1\nB 510 1 3 0.0 0.0\n"
    )
    return folder


def test_the_session_writes_the_tide_and_the_open_edge(estuary):
    s = Session()
    s.open_case(estuary)
    found = s.write_tide(
        "2026-01-19T00:00",
        days=1.0,
        constants=CASCAIS,
        direction=90.0,
        reference=(43761.1, 42647.5),
        depth=5800.0,
    )
    assert len(found.levels) == 289
    text = (estuary / "background-1.dat").read_text().splitlines()
    assert text[0].startswith("# constituents M2 1.000 m") and len(text) == 291
    ctrl = s.input_file("ctrl.txt")
    assert ctrl.all("B 510") == [[1, 2, 90.0, 10800.0]]  # background 1 made again
    assert ctrl.all("B 515") == [[1, 43761.1, 42647.5, 5800.0]]
    assert "B 510 1 2 90.0 10800.0" in ctrl.text()
    # REEF3D sums the constituents: B 511, no file needed
    s.write_tide("2026-01-19T00:00", constants=CASCAIS, harmonic=True)
    ctrl = s.input_file("ctrl.txt")
    assert len(ctrl.all("B 511")) == 6 and ctrl.all("B 510")[0][1] == 1 and not ctrl.all("B 515")
    # block 0's tide face: the open edge, the other zones kept
    s.start_multiblock([0, 1000, 0, 500], cells=(20, 10))
    s.set_block_face(0, "i0", "tide")
    assert M.settings(s.multiblock()).tides == [(1, 3)]
    s.make_multiblock_grid()
    ctrl = s.input_file("ctrl.txt")
    assert sorted(ctrl.all("B 520")) == [[1, 3, 0], [7, 1, 0]]
    assert [1, 1, 0.0, 0.0, 0.0] in ctrl.all("B 521") and ctrl.all("B 523") == [[1, 1]]
    assert ctrl.values("B 90") == [1] and s.input_file("control.txt").values("C 11") == [2]
    assert "session.write_tide('2026-01-19T00:00', days=1.0" in s.to_script()
    # a tide face on another block is refused
    s.extrude_block("b0_e", 100.0, cells=10)
    s.set_block_face(1, "i1", "tide")
    assert any("only on block 0" in i.text for i in M.check(s.multiblock()))
    # undo takes the file away again
    for _ in range(7):
        s.undo()
    assert not (estuary / "background-1.dat").exists()
