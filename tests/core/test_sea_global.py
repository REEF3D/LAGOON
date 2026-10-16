"""The sea anywhere (Open-Meteo: ERA5 waves, Météo-France's sea level and currents,
GloFAS rivers) and the water levels measured at tide gauges (Kartverket, NOAA, IOC),
stand-ins for the services here: storms, conditions as plots, the waves to make,
the boundary hydrographs, a river's hydrograph, gauges to compare with."""

import datetime as dt
import json

import numpy as np
import pytest
from openmeteo_fake import STORM, FakeGauges, FakeOpenMeteo, tide

from lagoonviz.core import Session
from lagoonviz.core.geo import gauges as gg
from lagoonviz.core.geo import openmeteo as om
from lagoonviz.core.geo.crs import transformer

CONTROL = "B 1 50\nB 10 0.0 2000.0 0.0 1000.0 -20.0 10.0\nC 11 6\nC 14 7\n"
CTRL = "A 10 5\nF 60 0.0\n"
LAT, LON = 40.45, -73.85  # off New York: outside MET Norway's waters


@pytest.fixture
def meteo(monkeypatch):
    fake = FakeOpenMeteo()
    monkeypatch.setattr(om, "FETCH", fake)
    return fake


@pytest.fixture
def gauges(monkeypatch):
    fake = FakeGauges(LAT, LON)
    monkeypatch.setattr(gg, "FETCH", fake)
    return fake


def _case(folder, control=CONTROL):
    folder.mkdir()
    (folder / "control.txt").write_text(control)
    (folder / "ctrl.txt").write_text(CTRL)
    x, y = transformer("EPSG:4326", "EPSG:32618")(LON, LAT)
    georef = {"version": 1, "crs": "EPSG:32618", "origin": [float(x), float(y)],
              "rotation_deg": 0.0}  # fmt: skip
    (folder / "georef.json").write_text(json.dumps(georef))
    return folder


@pytest.fixture
def case(tmp_path):
    return _case(tmp_path / "bay")


def _session(case):
    s = Session()
    s.open_case(case)
    return s, next(i.id for i in s.scene if i.kind == "case")


def test_open_meteo_away_from_norway(meteo, case):
    from lagoonviz.core import sea

    s, case_id = _session(case)
    assert sea.default_source(s, case_id) == sea.OPEN_METEO
    found = s.find_storms(2023, 2023, count=3)
    assert found[0].when == STORM and found[0].hs == pytest.approx(6.0, abs=0.01)
    assert found[0].tp == pytest.approx(14.0, abs=0.01) and found[0].direction == 300.0
    assert found[0].wind == 10.0 and found[0].wind_from == 250.0
    asked = [a for a in meteo.asked if "marine" in a]
    assert all("models=era5_ocean" in a for a in asked)
    with pytest.raises(ValueError, match="unknown source"):
        s.find_storms(2023, 2023, source="nowhere")


def test_conditions_from_open_meteo(meteo, case):
    s, _ = _session(case)
    items = s.sea_conditions("2023-03-10T00:00Z", "2023-03-11T00:00Z")
    names = [s.scene.get(i).name for i in items]
    assert names == ["ERA5 waves 2023-03-10 00:00", "Open-Meteo sea 2023-03-10 00:00"]
    waves = s.data.series(items[0])
    assert waves.time[0] == 0.0 and waves.time[-1] == 24 * 3600.0
    assert np.nanmax(waves.columns["Hs"]) == pytest.approx(6.0, abs=0.01)
    sea_state = s.data.series(items[1])
    t0 = dt.datetime(2023, 3, 10, tzinfo=dt.UTC).timestamp()
    np.testing.assert_allclose(sea_state.columns["zeta"], tide(t0 + sea_state.time), atol=1e-5)
    assert sea_state.columns["current"][0] == pytest.approx(1.0)  # km/h turned to m/s
    # going east, a model not turned: +x (the grid convergence a little off it)
    assert sea_state.columns["u"][0] == pytest.approx(1.0, abs=0.01)


def test_conditions_before_the_sea_level_starts(meteo, case):
    s, _ = _session(case)
    items = s.sea_conditions("2019-03-10T00:00Z", "2019-03-11T00:00Z")
    assert len(items) == 1  # the waves; no sea level before 2022
    assert any("2022" in note for note in s.sea_notes)


def test_waves_without_a_peak_period(monkeypatch, case):
    fake = FakeOpenMeteo(peak_period=False)
    monkeypatch.setattr(om, "FETCH", fake)
    series = om.OpenMeteo().waves(LAT, LON, STORM, STORM + dt.timedelta(hours=2))
    assert "mean period" in series.source
    assert series.columns["Tp"][0] == pytest.approx(14.0, abs=0.01)
    assert any("wave_period" in a and "wave_peak_period" not in a for a in fake.asked)


def test_the_waves_to_make(meteo, case):
    s, _ = _session(case)
    facts = s.write_waves("2023-03-10T06:20Z")
    assert facts["Hs"] == pytest.approx(6.0, abs=0.01) and facts["Tp"] == pytest.approx(14.0)
    # from 300° (west-north-west): going 120°, -30° from the model's x-axis (plus the
    # grid convergence, under a degree here)
    assert (facts["direction"] - 330.0) % 360.0 == pytest.approx(0.0, abs=1.5)
    ctrl = s.input_file("ctrl.txt")
    assert ctrl.values("B 85") == [2.0] and ctrl.values("B 92") == [31.0]
    assert ctrl.values("B 93") == pytest.approx([6.0, 14.0], abs=0.01)
    assert ctrl.values("B 105")[0] == pytest.approx(facts["direction"], abs=0.01)
    s.undo()
    assert s.input_file("ctrl.txt").values("B 85") is None


def test_boundaries_from_open_meteo(meteo, case):
    s, _ = _session(case)
    sides = s.write_boundaries("2023-03-10T00:00Z", "2023-03-10T12:00Z")
    assert sorted(side.side for side in sides) == [1, 2, 3, 4]
    level = np.loadtxt(case / "hydrograph_FSF_1.dat")
    t0 = dt.datetime(2023, 3, 10, tzinfo=dt.UTC).timestamp()
    np.testing.assert_allclose(level[:, 1], tide(t0 + level[:, 0]), atol=1e-4)
    q = {n: np.loadtxt(case / f"hydrograph_Q_{n}.dat")[:, 1] for n in (1, 2, 3, 4)}
    # 1 m/s east, the bed at the bottom of the domain (-20 m, no geo.dat): in at x min
    depth = 20.0 + level[0, 1]
    assert q[1][0] == pytest.approx(1000.0 * depth, rel=0.01)
    assert q[4][0] == pytest.approx(-q[1][0], rel=1e-6)
    assert abs(q[2][0]) < 0.05 * q[1][0]
    # asked at points about 4 km apart (Open-Meteo's resolution), not every 100 m
    marine = [a for a in meteo.asked if "sea_level_height_msl" in a]
    assert len(marine) == 1 and marine[0].count(",") < 30
    ctrl = s.input_file("ctrl.txt")
    assert sorted(v[0] for v in ctrl.all("B 422")) == [1.0, 2.0, 3.0, 4.0]


def test_boundaries_follow_the_bed(meteo, tmp_path):
    folder = _case(tmp_path / "shore", CONTROL + "G 10 1\n")
    # geo.dat: dry land beyond x 1500 (above the still water level 0)
    rows = [f"{x} {y} {(-10.0 if x < 1500 else 2.0)}" for x in range(-100, 2101, 100)
            for y in range(-100, 1101, 100)]  # fmt: skip
    (folder / "geo.dat").write_text("\n".join(rows) + "\n")
    s, _ = _session(folder)
    sides = {
        side.side: side for side in s.write_boundaries("2023-03-10T00:00Z", "2023-03-10T03:00Z")
    }
    assert 4 not in sides  # x max: on land
    assert sides[3].wet == pytest.approx(0.75, abs=0.06)  # y min three quarters in the water
    q1 = np.loadtxt(folder / "hydrograph_Q_1.dat")[0, 1]
    level = np.loadtxt(folder / "hydrograph_FSF_1.dat")[0, 1]
    assert q1 == pytest.approx(1000.0 * (10.0 + level), rel=0.01)


def test_a_rivers_discharge(meteo, case):
    s, case_id = _session(case)
    found = s.river_discharge("2023-03-10T00:00Z", "2023-03-12T00:00Z")
    rows = np.loadtxt(case / "hydrograph.dat")
    assert rows[0, 0] == 0.0 and rows[-1, 0] == 48 * 3600.0 and len(rows) == 49
    assert rows[0, 1] == pytest.approx(130.0) and rows[24, 1] == pytest.approx(140.0)
    assert rows[12, 1] == pytest.approx(135.0)  # hourly between the daily values
    assert s.input_file("ctrl.txt").values("B 60") == [2.0]
    assert found.place.label == "the centre of the domain"  # no inflow side, no corridor
    item = next(i for i in s.scene if i.kind == "series" and i.name.startswith("GloFAS river"))
    assert s.data.series(item.id).columns["Q"][0] == pytest.approx(130.0)
    # a side as a patch instead
    s.river_discharge("2023-03-10T00:00Z", "2023-03-11T00:00Z", target=1)
    assert (case / "hydrograph_Q_1.dat").is_file()
    ctrl = s.input_file("ctrl.txt")
    assert ctrl.all("B 440")[0] == [1.0, 1.0, 0.0, 0.0, 0.0, 1000.0]
    assert ctrl.all("B 421") == [[1.0, 1.0]]


def test_river_at_the_inflow_side(meteo, tmp_path):
    from lagoonviz.core import sea

    folder = _case(tmp_path / "river", CONTROL.replace("C 11 6", "C 11 1"))
    s, case_id = _session(folder)
    spot = sea.river_place(s, case_id)
    assert "inflow" in spot.label and (spot.x, spot.y) == (0.0, 500.0)


def test_gauges_near_the_case(gauges, case):
    s, case_id = _session(case)
    found = s.find_gauges()
    assert [g.source for g in found[:3]] == ["Kartverket", "NOAA", "IOC"]  # nearest first
    assert all(g.distance < 20 for g in found[:3])
    assert all(g.code != "9414290" for g in found)  # San Francisco: too far
    # NOAA: observed, predicted, the surge, as a time series
    items = s.gauge_levels("NOAA:8518750", "2023-03-10T00:00Z", "2023-03-10T06:00Z")
    item = s.scene.get(items[0])
    assert item.name == "NOAA gauge 8518750 2023-03-10 00:00"
    table = s.data.series(items[0])
    assert set(table.columns) >= {"observed", "predicted", "surge", "level"}
    assert np.nanmean(table.columns["surge"]) == pytest.approx(0.2, abs=1e-3)
    assert table.time[1] - table.time[0] == 360.0
    # Kartverket: centimetres to metres
    station = next(g for g in found if g.source == "Kartverket")
    series = gg.levels(station, "2023-03-10T00:00Z", "2023-03-10T01:00Z")
    assert np.nanmean(series.columns["surge"]) == pytest.approx(0.1, abs=1e-3)
    # IOC: no datum, about its mean; the -999 of an empty sensor left out
    station = next(g for g in found if g.source == "IOC")
    series = gg.levels(station, "2023-03-10T00:00Z", "2023-03-10T12:25Z")
    assert abs(np.nanmean(series.columns["observed"])) < 1e-9
    assert "predicted" not in series.columns and "sensor rad" in series.source


def test_a_gauge_as_the_water_level_at_a_side(gauges, case):
    s, case_id = _session(case)
    files = s.gauge_boundaries("NOAA:8518750", "2023-03-10T00:00Z", "2023-03-10T02:00Z", [4])
    rows = np.loadtxt(files[0])
    assert files[0].name == "hydrograph_FSF_4.dat" and rows[0, 0] == 0.0
    t0 = dt.datetime(2023, 3, 10, tzinfo=dt.UTC).timestamp()
    np.testing.assert_allclose(rows[:, 1], tide(t0 + rows[:, 0]) + 0.2, atol=2e-3)
    ctrl = s.input_file("ctrl.txt")
    assert ctrl.all("B 422") == [[4.0, 1.0]] and ctrl.all("B 440")[0][:2] == [4.0, 4.0]


def test_an_api_key_for_commercial_use(monkeypatch):
    monkeypatch.setattr(om, "API_KEY", "abc")
    address = om.url("marine", latitude=1.0, longitude=2.0, hourly=["wave_height"])
    assert address.startswith("https://customer-marine-api.open-meteo.com/v1/marine?")
    assert "apikey=abc" in address
    monkeypatch.setattr(om, "API_KEY", "")
    assert om.url("flood", latitude=1).startswith("https://flood-api.open-meteo.com/v1/flood?")
