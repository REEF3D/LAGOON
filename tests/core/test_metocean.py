"""Sea conditions from MET Norway (NORA3, NorKyst) for the set-up: the OPeNDAP text
answers, the nearest wet point, storms, series, spectra for REEF3D, boundary series."""

import datetime as dt
import math

import numpy as np
import pytest
from metocean_fake import FakeThredds, nora3_month, norkyst, spectra_day

from lagoonviz.core.geo import metocean as mo

#: answers as MET Norway's THREDDS gave them (October 2026), shortened
REAL_ASCII = """Dataset {
    Grid {
     ARRAY:
        Float32 hs[rlon = 1][rlat = 1][time = 3];
     MAPS:
        Float32 rlon[rlon = 1];
        Float32 rlat[rlat = 1];
        Float64 time[time = 3];
    } hs;
    Float64 time[time = 3];
} nora3_subset_wave/wave_tser/202001_NORA3wave_sub_time_unlimited.nc;
---------------------------------------------
hs.hs[1][1][3]
[0][0], 2.9429235, 2.8492146, -9999999.0

hs.rlon[1]
19.839996

hs.rlat[1]
3.4199998

hs.time[3]
1.5786144E9, 1.578618E9, 1.5786216E9


latitude.latitude[5][3]
[0], 54.479645, 62.4504, 69.63276
[1], 53.151543, 60.788498, 67.50786
[2], 51.646862, 58.946003, 65.23964
[3], 49.985645, 56.953384, 62.864586
[4], 48.186687, 54.83648, 60.408997

latitude.rlon[5]
15.399997, 18.399998, 21.399998, 24.399998, 27.399998

time[3]
1.5786144E9, 1.578618E9, 1.5786216E9

"""
REAL_DAS = """Attributes {
    zeta {
        String units "meter";
        Int16 _FillValue -32767;
        Float32 add_offset 0.0;
        Float32 scale_factor 0.001;
        String long_name "Sea surface height above geoid";
    }
    u_eastward {
        Float32 _FillValue 1.0E37;
    }
}
"""


def test_the_text_answers():
    found = mo.parse_ascii(REAL_ASCII)
    assert found["hs"].shape == (1, 1, 3) and found["hs"][0, 0, 1] == pytest.approx(2.8492146)
    assert found["latitude"].shape == (5, 3) and found["latitude"][4, 2] == pytest.approx(60.408997)
    assert found["time"].tolist() == [1.5786144e9, 1.578618e9, 1.5786216e9]
    assert found["hs.rlon"].shape == () or found["hs.rlon"].size == 1
    packing = mo.parse_das(REAL_DAS)
    assert packing["zeta"] == {"_FillValue": -32767.0, "add_offset": 0.0, "scale_factor": 0.001}
    assert packing["u_eastward"] == {"_FillValue": 1.0e37}
    dims = mo.parse_dds(REAL_ASCII)
    assert dims["hs"] == [("rlon", 1), ("rlat", 1), ("time", 3)]
    url = mo.dap_url("a/b.nc", "ascii", "hs[1][2][0:3],time[0:3]")
    assert (
        url
        == "https://thredds.met.no/thredds/dodsC/a/b.nc.ascii?hs%5B1%5D%5B2%5D%5B0:3%5D,time%5B0:3%5D"
    )


@pytest.fixture
def fake():
    fake = FakeThredds()
    nora3_month(fake, 2020, 1, storm_hour=100)
    nora3_month(fake, 2020, 2, storm_hour=500)
    nora3_month(fake, 2020, 3)
    return fake


def test_the_nearest_wet_point(fake):
    dap = mo.Dap(fake)
    # a place on land (the east of the fake grid): the nearest point in the water
    land = mo.nearest_point(dap, mo.nora3_path(2020, 1), 61.0, 4.0 + 15 * 0.055,
                            wet="model_depth", stride=4)  # fmt: skip
    assert land.i < 32 and land.distance > 0.0
    sea = mo.nearest_point(dap, mo.nora3_path(2020, 1), 61.0, 4.0, wet="model_depth", stride=4)
    assert (sea.i, sea.j) == (20, 15) and sea.distance < 0.5
    with pytest.raises(mo.MetoceanError, match="outside"):
        mo.nearest_point(dap, mo.nora3_path(2020, 1), 10.0, 100.0, stride=4)


def test_storms_and_series(fake):
    nora3 = mo.Nora3(mo.Dap(fake))
    found, point = mo.find_storms(61.0, 4.0, 2020, 2020, count=3, nora3=nora3)
    assert point.distance < 0.5
    assert [s.when for s in found[:2]] == [
        dt.datetime(2020, 1, 5, 4, tzinfo=dt.UTC),  # hour 100
        dt.datetime(2020, 2, 21, 20, tzinfo=dt.UTC),  # hour 500
    ]
    first = found[0]
    assert first.hs == pytest.approx(9.0 + 0.5 * math.sin(100 / 30.0), abs=0.01)
    assert first.direction == pytest.approx(270.0)  # going east: from the west
    assert first.wind_from == pytest.approx(250.0)
    assert "Hs 8.9 m" in first.describe()
    # (April is not there yet: the search ends with the data)
    series = nora3.series(61.0, 4.0, "2020-01-31T20:00Z", "2020-02-01T04:00Z")
    assert len(series.time) == 9 and np.all(np.diff(series.time) == 3600.0)
    assert set(series.columns) == {"Hs", "Tp", "dir_from", "wind", "wind_from"}
    one_request_a_month = [u for u in fake.asked if ".ascii?hs" in u]
    assert len(one_request_a_month) == 3 + 2


def test_norkyst_v3_and_v2(fake):
    start = mo.NORKYST3_START.timestamp()
    norkyst(fake, mo.NORKYST3, start + 3600.0 * np.arange(48), packed=True)
    days = [dt.date(2020, 1, 4), dt.date(2020, 1, 5)]
    for day in days:
        t0 = dt.datetime.combine(day, dt.time(), dt.UTC).timestamp()
        norkyst(fake, mo.norkyst2_path(day), t0 + 3600.0 * np.arange(24), packed=False)
    ocean = mo.NorKyst(mo.Dap(fake))
    lats, lons = [60.99, 61.01, 61.01], [3.98, 3.98, 4.02]
    v3 = ocean.box(lats, lons, start + 3600, start + 10 * 3600)
    assert v3.source == "NorKyst v3" and v3.zeta.shape[0] == 10
    hours = (start + 3600.0 * np.arange(1, 11)) / 3600.0
    expected = 0.5 * np.sin(2 * np.pi * hours / 12.42)
    np.testing.assert_allclose(v3.zeta[:, 0, 0], expected, atol=1e-3)
    # depth-averaged over 20 m: levels 0, 3, 10, 15 (the rest below the bed)
    assert v3.u[0, 0, 0] == pytest.approx(_depth_mean_check(), abs=2e-3)
    assert np.allclose(v3.v, 0.0)
    v2 = ocean.box(lats, lons, "2020-01-04T20:00Z", "2020-01-05T03:00Z")
    assert v2.source == "NorKyst v2" and len(v2.time) == 8
    assert np.all(np.diff(v2.time) == 3600.0)
    with pytest.raises(mo.MetoceanError, match="2017-02-20"):
        ocean.box(lats, lons, "2010-01-01", "2010-01-02")


def _depth_mean_check():
    u = {0.0: 0.3, 3.0: 0.3 - 0.2 * 3 / 15, 10.0: 0.3 - 0.2 * 10 / 15, 15.0: 0.1}
    # each level for the water half-way to its neighbours, the deepest down to 20 m
    w = {0.0: 1.5, 3.0: 5.0, 10.0: 6.0, 15.0: 7.5}
    return sum(u[d] * w[d] for d in u) / sum(w.values())


def test_the_spectrum_for_reef3d(fake, tmp_path):
    day = dt.date(2020, 1, 5)
    spectra_day(fake, day)
    spectrum = mo.nora3_spectrum("2020-01-05T04:00Z", 61.05, 4.0, mo.Dap(fake))
    assert (spectrum.lat, spectrum.lon) == (61.0, 4.0)
    assert spectrum.hs == pytest.approx(4.0, rel=1e-3)
    assert spectrum.tp == pytest.approx(10.0, rel=0.06)
    assert spectrum.mean_to == pytest.approx(90.0, abs=0.5)  # going east
    # a model turned 30° (its x-axis 30° north of east): the waves go -30° in it
    facts = mo.spectrum_file(spectrum, tmp_path / "spectrum-file-2d.dat", rotation_deg=30.0)
    assert facts["direction"] == pytest.approx(330.0, abs=0.5)
    assert facts["Hs_sector"] == pytest.approx(4.0, rel=1e-3)  # all within ±90°
    lines = (tmp_path / "spectrum-file-2d.dat").read_text().splitlines()
    head = lines[0].split()
    assert head[0] == "fq_dir" and len(head) == 1 + facts["directions"]
    theta = np.array(head[1:], float)
    assert np.all(np.diff(theta) > 0) and abs(theta.mean() - math.radians(330.0)) < 0.2
    table = np.array([line.split() for line in lines[1:]], float)
    omega, s = table[:, 0], table[:, 1:]
    dw = np.append(np.diff(omega), np.diff(omega)[-1])
    dtheta = np.diff(theta).mean()
    # REEF3D's waves: amplitude sqrt(2 S dw dtheta) each; together Hs 4 m
    assert 4 * math.sqrt(np.sum(s * dw[:, None] * dtheta)) == pytest.approx(4.0, rel=1e-3)
    with pytest.raises(mo.MetoceanError, match="30 km"):
        mo.nora3_spectrum("2020-01-05T04:00Z", 65.0, 4.0, mo.Dap(fake))


def test_directions_into_the_model():
    assert mo.model_angle(90.0, 90.0, 0.0) == pytest.approx(0.0)  # going east: +x
    assert mo.model_angle(0.0, 90.0, 0.0) == pytest.approx(90.0)  # north: +y
    assert mo.model_angle(0.0, 90.0, 90.0) == pytest.approx(0.0)  # a model turned north
    assert mo.model_angle(180.0, 92.0, 0.0) == pytest.approx(272.0)  # grid convergence


def test_tables(tmp_path):
    from lagoonviz.core.data.series import read_series

    t0 = 1.5786144e9
    path = mo.write_table(
        tmp_path / "x.dat",
        t0,
        {"Hs": np.array([1.0, np.nan, 2.0])},
        t0 + np.array([0.0, 3600.0, 7200.0]),
        {"Hs": "m"},
        "NORA3 at 61.0 N 4.0 E",
    )
    table = read_series(path)
    assert table.time.tolist() == [0.0, 3600.0, 7200.0] and table.units["Hs"] == "m"
    assert np.isnan(table.columns["Hs"][1])
    two = mo.two_column(
        tmp_path / "hydrograph_FSF_1.dat", np.array([0.0, 60.0]), np.array([1.0, 1.5])
    )
    assert two.read_text() == "0 1\n60 1.5\n"


#: NorKyst v3 as the server answered (packed Int16: scale 0.001)
REAL_NORKYST = """Dataset {
    Float64 time[time = 2];
} fou-hi/norkystv3_800m_m00_be;
---------------------------------------------
time[2]
1.7044272E9, 1.7044308E9

zeta.zeta[2][2][2]
[0][0], 280, 281
[0][1], 277, 279
[1][0], 405, 405
[1][1], 401, 402

zeta.time[2]
1.7044272E9, 1.7044308E9

zeta.Y[2]
480000, 480800


u_eastward.u_eastward[2][2][2][2]
[0][0][0], 171, 135
[0][0][1], 176, 122
[0][1][0], 167, 131
[0][1][1], 172, 118
[1][0][0], 259, 185
[1][0][1], 277, 187
[1][1][0], 254, 179
[1][1][1], -32767, 181

u_eastward.depth[2]
0.0, 1.0

"""


def test_packed_answers_are_unpacked():
    das = REAL_DAS.replace(
        "    u_eastward {\n        Float32 _FillValue 1.0E37;",
        "    u_eastward {\n        Int16 _FillValue -32767;\n        Float32 scale_factor 0.001;",
    )
    answers = {"das": das, "ascii": REAL_NORKYST}
    dap = mo.Dap(lambda url: answers[url.split("?")[0].rsplit(".", 1)[-1]].encode())
    got = dap.get(mo.NORKYST3, "zeta[0:1][0:1][0:1],u_eastward[0:1][0:1][0:1][0:1],time[0:1]")
    assert got["zeta"].shape == (2, 2, 2) and got["zeta"][1, 0, 0] == pytest.approx(0.405)
    assert got["u_eastward"].shape == (2, 2, 2, 2)
    assert got["u_eastward"][0, 0, 0, 0] == pytest.approx(0.171)
    assert np.isnan(got["u_eastward"][1, 1, 1, 0]) and got["time"][0] == 1.7044272e9
