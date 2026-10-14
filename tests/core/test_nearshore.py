"""The bed next to the coast of a download: a beach profile or the smoothest bed where
the coarse sea source cannot be trusted (no flat 1 m), files of one's own laid over
it, and ICESat-2's nearshore soundings (ATL24) guiding it."""

import json

import numpy as np
import pytest
from lagoon_format.geo import Georef

from lagoonviz.core.geo import bathy, download, icesat2, nearshore
from lagoonviz.core.geo.raster import GeoImage
from lagoonviz.core.session import Session

SPACING = 10.0
XS = np.arange(0.0, 2000.0, SPACING) + 5.0
YS = np.arange(0.0, 400.0, SPACING) + 5.0
GX, GY = np.meshgrid(XS, YS)
COAST = 1500.0  # land east of x = 1500 m
GEOREF = Georef("EPSG:25833", (270000.0, 7040000.0), 0.0)


def layers(coarse=None):
    """Copernicus: the sea 0, land rising east of the coast; ETOPO (450 m): 12 m deep
    offshore, then a flat 1 m over the last 600 m, dry over the coast."""
    cop = np.where(GX > COAST, 0.5 + 0.01 * (GX - COAST), 0.0)
    if coarse is None:
        coarse = np.where(GX < 900.0, -12.0, np.where(GX < COAST - 50.0, -1.0, 2.0))
    return {"copernicus": cop, "etopo": coarse}


def row(heights):
    return heights[len(YS) // 2]


def test_dean_profile():
    assert nearshore.dean(np.array([0.0, 1000.0]), 0.1) == pytest.approx([0.0, 10.0])
    assert nearshore.dean(-5.0) == 0.0


def test_laplace_between_two_boundaries_is_a_straight_line():
    target = np.full((3, 11), np.nan)
    target[:, 0], target[:, -1] = 0.0, -10.0
    unknown = np.zeros(target.shape, bool)
    unknown[:, 1:-1] = True
    out = nearshore.harmonic(target, unknown, (1.0, 2.0))
    assert np.allclose(out, np.linspace(0.0, -10.0, 11)[None, :])
    # cut off from every boundary value: 0
    alone = np.zeros((3, 3), bool)
    alone[1, 1] = True
    assert nearshore.harmonic(np.full((3, 3), np.nan), alone)[1, 1] == 0.0


def test_the_flat_metre_as_before():
    out, used = bathy.coastal(layers(), XS, YS, ("etopo",))  # nearshore="flat"
    near = row(out)[(XS > 900) & (XS < COAST - 100)]
    assert np.allclose(near, -1.0)  # the flat 1 m of before
    assert used == ["copernicus", "etopo"]


@pytest.mark.parametrize("mode", ["dean", "smooth"])
def test_no_flat_metre_next_to_the_coast(mode):
    out, _ = bathy.coastal(layers(), XS, YS, ("etopo",), nearshore=mode)
    line = row(out)
    sea = XS < COAST
    assert (line[sea] < 0).all() and (line[~sea] > 0).all()  # the coast stays put
    assert np.allclose(line[XS < 800], -12.0)  # trusted (beyond 675 m of ETOPO): as it is
    shore = line[(XS > 830) & sea]  # the nearshore band
    assert not np.isclose(shore, -1.0).any()
    assert (np.diff(shore) >= 0).all()  # deepening away from the coast
    assert (np.diff(shore[shore < -0.2]) > 0).all()  # steadily, where not held wet
    assert line[np.searchsorted(XS, COAST) - 1] > -1.0  # shallow at the coast
    # no step where the trusted sea begins
    assert abs(line[np.searchsorted(XS, 900.0)] - line[np.searchsorted(XS, 890.0)]) < 1.0


def test_a_shallow_coast_is_the_sea_not_a_polder():
    """Nowhere 10 m deep: Copernicus' water at 0 m is still the sea (it was taken for
    a polder behind dikes and kept at Copernicus' 0 m)."""
    coarse = np.where(GX < COAST - 300.0, -6.0, -2.0)
    out, _ = bathy.coastal(layers(coarse), XS, YS, ("etopo",), nearshore="dean")
    assert np.allclose(row(out)[XS < 800], -6.0)
    polder = layers(coarse)
    polder["copernicus"] = np.where((GX > 1700) & (GX < 1800), -2.0, polder["copernicus"])
    out, _ = bathy.coastal(polder, XS, YS, ("etopo",), nearshore="dean")
    assert np.allclose(out[(GX > 1700) & (GX < 1800)], -2.0)  # behind dikes as before


def test_dean_is_a_beach_and_smooth_a_slope():
    dean, _ = bathy.coastal(layers(), XS, YS, ("etopo",), nearshore="dean", trust=700.0)
    smooth, _ = bathy.coastal(layers(), XS, YS, ("etopo",), nearshore="smooth", trust=700.0)
    band = (XS > COAST - 690) & (XS < COAST)  # cells closer to the land than 700 m
    d, s = row(dean)[band], row(smooth)[band]
    # the smoothest bed is a straight slope from the trusted -12 m to the coast (held
    # 0.1 m deep at least); Dean's beach is deeper near the coast, concave
    assert s[0] == pytest.approx(-12.0 + 12.0 / 70, abs=1e-6)
    assert np.allclose(np.diff(s[s < -0.2], 2), 0.0, atol=1e-6)
    assert np.all(d <= s + 1e-9) and d[-1] < -0.3
    assert (np.diff(d, 2) > -1e-9).all()  # x^(2/3) from the coast: steepest there


def test_shallows_not_trusted():
    """ETOPO's flat 2 m over a flat far from the coast: trusted by default, made
    again (the smoothest bed between the coast and the deep sea) when not."""
    coarse = np.where(GX < 300.0, -12.0, -2.0)
    kept, _ = bathy.coastal(layers(coarse), XS, YS, ("etopo",), nearshore="smooth")
    again, _ = bathy.coastal(
        layers(coarse), XS, YS, ("etopo",), nearshore="smooth", trust_depth=2.5
    )
    middle = (XS > 400) & (XS < 700)
    assert np.allclose(row(kept)[middle], -2.0)
    assert not np.isclose(row(again)[middle], -2.0).any()
    assert (np.diff(row(again)[(XS > 300) & (XS < COAST)]) > 0).all()


def test_ones_own_data_and_soundings_guide_the_profile():
    known = np.where((GX > 1100) & (GX < 1200), -6.0, np.nan)  # a dredged channel
    control = np.where((GX > 1300) & (GX < 1320), -4.0, np.nan)  # ICESat-2 track
    out, _ = bathy.coastal(
        layers(), XS, YS, ("etopo",), nearshore="dean", known=known, control=control
    )
    assert np.allclose(out[np.isfinite(known)], -6.0)
    assert np.allclose(out[np.isfinite(control)], -4.0)
    # met without a step: from one to the other between them
    between = row(out)[(XS > 1200) & (XS < 1300)]
    assert (np.diff(between) > 0).all() and between.min() > -6.0 and between.max() < -4.0


def test_lay_blends_in_at_the_edges():
    heights = np.zeros((1, 41))
    values = np.full((1, 41), -10.0)
    covered = np.zeros((1, 41), bool)
    covered[0, 10:31] = True
    out = nearshore.lay(heights, values, covered, (1.0, 1.0))
    assert out[0, 20] == -10.0 and out[0, 5] == 0.0
    assert -10.0 < out[0, 10] < 0.0 and (np.diff(out[0, 10:21]) <= 0).all()


def image(values, crs="EPSG:25833"):
    """The grid's cells as a raster in the CRS, row 0 at the top."""
    top = 7040000.0 + YS[-1] + SPACING / 2
    return GeoImage(values[::-1].copy(), (270000.0, SPACING, 0.0, top, 0.0, -SPACING), crs)


@pytest.fixture
def fake_web(monkeypatch):
    lay = layers()

    def fake(key, georef_, box, spacing, fetch=None, land_only=True):
        if key == "copernicus":
            cop = lay["copernicus"]
            return [image(np.where(cop > 0, cop, np.nan) if land_only else cop)]
        if key == "etopo":
            return [image(lay["etopo"])]
        raise OSError(f"{key}: not asked for in this test")

    monkeypatch.setattr(download, "download_all", fake)
    monkeypatch.setattr(download, "covers", lambda s, g, b: s.key in ("copernicus", "etopo"))


def test_a_chart_over_the_download(fake_web, tmp_path):
    chart = tmp_path / "soundings.xyz"
    picks = [(x, y) for x in np.arange(1005.0, 1400.0, 40.0) for y in np.arange(5.0, 400.0, 40.0)]
    # depths below chart datum, which lies 2 m below MSL: 3 m below CD -> -5 m
    chart.write_text("".join(f"{270000 + x} {7040000 + y} 3.0\n" for x, y in picks))
    overlay = nearshore.load_overlay({"path": chart, "depths": True, "below_msl": 2.0})
    assert not overlay.control
    out, used = bathy.web_heights("best", GEOREF, XS, YS, SPACING, overlays=[overlay])
    assert used[-1] == "soundings.xyz"
    inside = (GX > 1100) & (GX < 1300) & (GY > 60) & (GY < 340)
    assert np.allclose(out[inside], -5.0)
    assert (out[GX < 800] == -12.0).all()  # the sea far from the chart as downloaded


def write_atl24(path, beams):
    """An ATL24 file as NSIDC writes it (the variables LAGOON reads)."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as file:
        for name, photons in beams.items():
            group = file.create_group(name)
            for key, values in photons.items():
                group[key] = np.asarray(values)


def photons(n, lon0, lat, height, cls=40, conf=0.9, low=0, exceeded=0):
    return {
        "lon_ph": lon0 + np.arange(n) * 1e-5,
        "lat_ph": np.full(n, lat),
        "ortho_h": np.full(n, height, np.float32),
        "x_atc": 1e6 + np.arange(n) * 1.0,
        "class_ph": np.full(n, cls, np.int8),
        "confidence": np.full(n, conf),
        "low_confidence_flag": np.full(n, low, np.uint8),
        "sensor_depth_exceeded": np.full(n, exceeded, np.uint8),
    }


def test_atl24_soundings(tmp_path):
    path = tmp_path / "ATL24_20250101000000_01230101_006_01_002_01.h5"
    good = photons(40, 72.80, 18.95, -3.0)
    good["ortho_h"][5] = -30.0  # a stray photon: the median keeps the bed
    write_atl24(
        path,
        {
            "gt1l": good,
            "gt1r": photons(40, 72.80, 18.96, -9.0, cls=41),  # the sea surface
            "gt2l": photons(40, 72.80, 18.97, -9.0, conf=0.3),  # not sure
            "gt2r": photons(40, 72.80, 18.98, -9.0, low=1),
            "gt3l": photons(40, 72.80, 18.99, -9.0, exceeded=1),
            "gt3r": photons(2, 72.80, 19.00, -9.0),  # too few for a sounding
        },
    )
    points = icesat2.read(path)
    assert len(points) == 4  # 40 photons, 1 m apart: four soundings of 10 m
    assert np.allclose(points[:, 2], -3.0) and np.allclose(points[:, 1], 18.95)
    assert len(icesat2.read(path, bounds=(72.8, 18.9, 72.800095, 19.0))) == 1
    assert icesat2.is_atl24(tmp_path) and icesat2.files_in(tmp_path) == [path]
    overlay = nearshore.load_overlay(tmp_path)
    assert overlay.control and overlay.bathy.crs == "EPSG:4326"


def test_atl24_files_found_and_a_login_asked_for(tmp_path, monkeypatch):
    feed = {
        "feed": {
            "entry": [
                {
                    "links": [
                        {"rel": "http://esipfed.org/ns/fedsearch/1.1/data#", "href": url}
                        for url in (
                            "s3://nsidc/ATL24_a.h5",
                            "https://data.nsidc.earthdatacloud.nasa.gov/x/ATL24_a.h5",
                        )
                    ]
                },
                {"links": [{"rel": "metadata#", "href": "https://x/ATL24_b.h5.xml"}]},
            ]
        }
    }
    asked = []

    def fetch(url):
        asked.append(url)
        return json.dumps(feed).encode()

    found = icesat2.find((72.7, 18.8, 73.0, 19.2), fetch)
    assert found == ["https://data.nsidc.earthdatacloud.nasa.gov/x/ATL24_a.h5"]
    assert "short_name=ATL24" in asked[0] and "bounding_box=72.7" in asked[0]
    monkeypatch.delenv("LAGOON_OFFLINE", raising=False)
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.netrc
    with pytest.raises(OSError, match="Earthdata login"):
        icesat2.earthdata_fetch(found[0])


def test_session_keeps_how_the_coast_was_made(fake_web, tmp_path):
    session = Session()
    case = session.new_case_from_map((63.5, 10.4), length=500.0)
    session.set_georef("EPSG:25833", origin=list(GEOREF.origin))
    session.set_domain(XS[0], XS[-1], YS[0], YS[-1])
    chart = tmp_path / "chart.xyz"
    chart.write_text("271200 7040200 -5\n271250 7040250 -5\n271200 7040250 -5\n")
    session.make_geo_dat(
        "best",
        spacing=SPACING,
        nearshore="smooth",
        trust_depth=2.5,
        overlays=[{"path": str(chart), "below_msl": 1.0}],
    )
    made = session.geo_dat_recipe()
    assert made["nearshore"] == "smooth" and made["trust_depth"] == 2.5
    assert made["overlays"][0]["below_msl"] == 1.0 and not made["icesat2"]
    assert "chart.xyz" in session.geo_sources
    job = session.remake_geo_dat_job()
    assert job.recipe["nearshore"] == "smooth" and job.recipe["overlays"] == made["overlays"]
    assert "nearshore='smooth'" in job.code and "trust_depth=2.5" in job.code
    # a geo.dat made before: made again as it was, the flat metre
    from lagoonviz.core.geo import recipe

    folder = session.data.case_folder(case)
    old = {k: v for k, v in made.items() if k not in ("nearshore", "dean_a", "overlays")}
    recipe.write(folder, old)
    assert session.remake_geo_dat_job().recipe["nearshore"] == "flat"
    with pytest.raises(ValueError, match="nearshore"):
        session.make_geo_dat("best", nearshore="steep")
    with pytest.raises(ValueError, match="over a download"):
        session.make_geo_dat(str(chart), crs="EPSG:25833", overlays=[str(chart)])
