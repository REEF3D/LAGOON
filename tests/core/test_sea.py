"""The sea at a case for its set-up (NORA3, NorKyst from MET Norway, a stand-in for
its server here): storms, conditions as plots, the spectrum file and the boundary
hydrographs REEF3D reads, and ctrl.txt set to them."""

import datetime as dt
import json
import math

import numpy as np
import pytest
from metocean_fake import FakeThredds, nora3_month, norkyst, spectra_day

from lagoonviz.core import Session
from lagoonviz.core.data.controlfile import ControlFile
from lagoonviz.core.geo import metocean as mo
from lagoonviz.core.geo.crs import transformer

CONTROL = "B 1 50\nB 10 0.0 2000.0 0.0 1000.0 -30.0 30.0\nC 11 6\nC 14 7\n"
CTRL = "A 10 5\nF 60 30.0\n"


@pytest.fixture
def server(monkeypatch):
    fake = FakeThredds()
    for month in (1, 2):
        nora3_month(fake, 2020, month, storm_hour=100 if month == 1 else None)
    spectra_day(fake, dt.date(2020, 1, 5))
    for day in (4, 5, 6):
        t0 = dt.datetime(2020, 1, day, tzinfo=dt.UTC).timestamp()
        norkyst(fake, mo.norkyst2_path(dt.date(2020, 1, day)), t0 + 3600.0 * np.arange(24),
                packed=False)  # fmt: skip
    monkeypatch.setattr(mo, "FETCH", fake)
    return fake


@pytest.fixture
def case(tmp_path):
    """A 2 km x 1 km domain west of 4° E at 61° N (in the sea of the stand-in), its
    x-axis turned 20° north of east, waves made at x min."""
    folder = tmp_path / "fjord"
    folder.mkdir()
    (folder / "control.txt").write_text(CONTROL)
    (folder / "ctrl.txt").write_text(CTRL)
    x, y = transformer("EPSG:4326", "EPSG:25832")(3.95, 61.0)
    georef = {"version": 1, "crs": "EPSG:25832", "origin": [float(x), float(y)],
              "rotation_deg": 20.0}  # fmt: skip
    (folder / "georef.json").write_text(json.dumps(georef))
    return folder


def _beta(s):
    """Where the waves (going east) go in the model: its rotation and the grid
    convergence."""
    from lagoonviz.core import sea

    case_id = next(i.id for i in s.scene if i.kind == "case")
    return (sea.frame(s, case_id).north_deg - 90.0 - 20.0) % 360.0


def test_storms_at_the_wave_generation_side(server, case):
    s = Session()
    s.open_case(case)
    found = s.find_storms(2020, 2020, count=2)
    assert found[0].when == dt.datetime(2020, 1, 5, 4, tzinfo=dt.UTC)
    assert found[0].hs > 8.0
    # where: the middle of x min, through the georeference
    from lagoonviz.core import sea

    case_id = next(i.id for i in s.scene if i.kind == "case")
    spot = sea.place(s, case_id)
    assert "x min" in spot.label and (spot.x, spot.y) == (0.0, 500.0)
    lat, lon = spot.lat, spot.lon
    assert lat == pytest.approx(61.0, abs=0.01) and lon == pytest.approx(3.94, abs=0.01)


def test_conditions_as_plots(server, case):
    s = Session()
    s.open_case(case)
    items = s.sea_conditions("2020-01-05T00:00Z", "2020-01-06T00:00Z")
    names = [s.scene.get(i).name for i in items]
    assert names == ["NORA3 waves 2020-01-05 00:00", "NorKyst sea 2020-01-05 00:00"]
    waves = s.data.series(items[0])
    assert waves.time[0] == 0.0 and waves.time[-1] == 24 * 3600.0
    assert np.nanmax(waves.columns["Hs"]) > 8.0
    # waves going east: from the west, and -20° in a model turned 20° north of east,
    # less UTM 32's grid convergence at 4° E (4.4°: north is east of the grid's y)
    assert waves.columns["dir_from"][0] == pytest.approx(270.0)
    assert waves.columns["beta"][0] == pytest.approx(_beta(s), abs=0.01)
    assert _beta(s) == pytest.approx(360.0 - 20.0 - 4.4, abs=0.1)
    sea_state = s.data.series(items[1])
    assert set(sea_state.columns) >= {"zeta", "level", "current", "u", "v"}
    assert sea_state.columns["level"][0] == pytest.approx(30.0 + sea_state.columns["zeta"][0])
    # the current goes east: in the model mostly +x, some -y
    assert sea_state.columns["u"][0] > 0 > sea_state.columns["v"][0]
    # still a case to set up: the set-up stays
    assert s.data.case(items[0]).setup_only
    assert s.sea_conditions("2020-01-05T00:00Z", "2020-01-06T00:00Z") == items  # shown already


def test_the_spectrum_and_the_wave_generation(server, case):
    s = Session()
    s.open_case(case)
    facts = s.write_spectrum("2020-01-05T04:00Z")
    assert facts["file"] == case / "spectrum-file-2d.dat" and facts["file"].is_file()
    assert facts["Hs"] == pytest.approx(4.0, rel=1e-3)
    ctrl = s.input_file("ctrl.txt")
    assert ctrl.values("B 85") == [11.0] and ctrl.values("B 92") == [31.0]
    assert ctrl.values("B 93") == pytest.approx([4.0, facts["Tp"]], abs=0.01)
    assert ctrl.values("B 105")[0] == pytest.approx(_beta(s), abs=0.01)
    assert ctrl.values("B 105")[1:] == [0.0, 0.0] and ctrl.values("B 130") == [1.0]
    assert "set_input_text('ctrl.txt'" in s.to_script()
    s.undo()  # one step
    assert s.input_file("ctrl.txt").values("B 85") is None


def test_boundary_hydrographs(server, case):
    s = Session()
    s.open_case(case)
    sides = s.write_boundaries("2020-01-05T00:00Z", "2020-01-05T12:00Z")
    assert sorted(side.side for side in sides) == [1, 2, 3, 4]  # all in the sea
    level = np.loadtxt(case / "hydrograph_FSF_1.dat")
    assert level[0, 0] == 0.0 and level[-1, 0] == 12 * 3600.0
    t0 = dt.datetime(2020, 1, 5, tzinfo=dt.UTC).timestamp()
    tide = 0.5 * np.sin(2 * np.pi * ((t0 + level[:, 0]) / 3600.0) / 12.42)
    np.testing.assert_allclose(level[:, 1], 30.0 + tide, atol=2e-3)
    # the current goes east, the model's x-axis 20° north of it: in at x min, out at
    # x max, in at y max (the current's -y part), out at y min
    q = {n: np.loadtxt(case / f"hydrograph_Q_{n}.dat")[:, 1] for n in (1, 2, 3, 4)}
    assert q[1][0] > 0 > q[4][0] and q[2][0] > 0 > q[3][0]
    depth_mean = (0.3 * 1.5 + 0.26 * 5.0 + (0.3 - 0.2 * 10 / 15) * 6.0 + 0.1 * 7.5) / 20.0
    turned = math.radians(360.0 - _beta(s))  # the current's angle to the model's x-axis
    depth = 20.0 + np.loadtxt(case / "hydrograph_FSF_1.dat")[0, 1] - 30.0
    across = 1000.0 * depth * math.cos(turned) * depth_mean
    assert q[1][0] == pytest.approx(across, rel=0.01)
    ctrl = s.input_file("ctrl.txt")
    assert ctrl.all("B 440")[0] == [1.0, 1.0, 0.0, 0.0, 0.0, 1000.0]
    assert sorted(v[0] for v in ctrl.all("B 421")) == [1.0, 2.0, 3.0, 4.0]
    table = next(i for i in s.scene if i.kind == "series" and i.name.startswith("Boundaries"))
    assert "level_xmin" in s.data.series(table.id).columns
    # again: the patches are set again, not added twice
    s.write_boundaries("2020-01-05T00:00Z", "2020-01-05T06:00Z", sides=[1])
    assert len(s.input_file("ctrl.txt").all("B 440")) == 4


def test_without_a_georeference(server, tmp_path):
    (tmp_path / "control.txt").write_text(CONTROL)
    (tmp_path / "ctrl.txt").write_text(CTRL)
    s = Session()
    s.open_case(tmp_path)
    with pytest.raises(ValueError, match="georeference"):
        s.find_storms(2020, 2020)


def test_control_file_left_as_it_was(server, case):
    s = Session()
    s.open_case(case)
    s.write_spectrum("2020-01-05T04:00Z", apply=False)
    assert ControlFile.read(case / "ctrl.txt").text() == CTRL
    assert not s.inputs_modified()
