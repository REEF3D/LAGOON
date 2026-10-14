"""The coast of a downloaded bathymetry where Copernicus' 30 m land has it, not where
a coarse sea source (EMODnet, ETOPO, GEBCO) smears it; EMODnet's depths below the
lowest astronomical tide put on mean sea level."""

import numpy as np
import pytest
from lagoon_format.geo import Georef

from lagoonviz.core.geo import bathy, download
from lagoonviz.core.geo.raster import GeoImage

SPACING = 10.0
XS = np.arange(0.0, 1000.0, SPACING) + 5.0
YS = np.arange(0.0, 600.0, SPACING) + 5.0
GX, GY = np.meshgrid(XS, YS)
COAST = 500.0  # land east of x = 500 m


def land_heights():
    """Copernicus: the sea 0, land rising east of the coast, a polder 2 m below the
    sea behind dikes (700..800 x 300..400)."""
    cop = np.where(GX > COAST, 0.5 + 0.01 * (GX - COAST), 0.0)
    polder = (GX > 700) & (GX < 800) & (GY > 300) & (GY < 400)
    return np.where(polder, -2.0, cop), polder


def coarse_sea():
    """EMODnet in 120 m blocks: 15 m deep offshore, the blocks over the coast a mix
    of land and sea (+2 m: the sea would be dry there), nothing on land."""
    block = np.floor(GX / 120.0)
    depth = np.where(GX < 360.0, -15.0, np.where(block == np.floor(COAST / 120.0), 2.0, -3.0))
    return np.where(GX > 600.0, np.nan, depth)


def test_the_coast_where_the_land_mask_has_it():
    cop, polder = land_heights()
    emodnet = coarse_sea()
    etopo = np.full(GX.shape, -1.0)  # 450 m: shallow everywhere, the polder too
    out, used = bathy.coastal({"copernicus": cop, "emodnet": emodnet, "etopo": etopo}, XS, YS)
    assert used == ["copernicus", "emodnet"]
    sea = GX < COAST
    land = (GX > COAST) & ~polder
    assert (out[sea] < 0).all() and (out[land] > 0).all()
    # the shoreline between the last sea and the first land point of each row
    row = out[len(YS) // 2]
    first_land = XS[np.argmax(row > 0)]
    assert COAST < first_land <= COAST + SPACING
    # next to the coast a shallow beach, then the coarse depths offshore
    assert row[np.searchsorted(XS, COAST) - 1] == pytest.approx(-(0.1 + 0.02 * 10.0))
    assert np.all(out[GX < 300] == -15.0)
    # the polder keeps its own height, below the sea, not flooded nor filled
    assert np.allclose(out[polder], -2.0)
    assert np.allclose(out[land], cop[land])


def test_open_sea_without_a_land_tile_and_kartverket_over_all():
    emodnet = np.full(GX.shape, -12.0)
    cop = np.full(GX.shape, np.nan)  # no tile: the sea
    fine = np.where(GY > 300, -4.0, np.nan)
    out, used = bathy.coastal({"copernicus": cop, "emodnet": emodnet, "kartverket": fine}, XS, YS)
    assert np.all(out[GY > 300] == -4.0) and np.all(out[GY < 300] == -12.0)
    assert used == ["emodnet", "kartverket"]


def test_emodnet_on_mean_sea_level_and_the_coast_from_the_web(monkeypatch):
    georef = Georef("EPSG:25833", (270000.0, 7040000.0), 0.0)
    cop, _ = land_heights()
    emodnet = coarse_sea()

    def image(values):
        # the grid's cells as an image in the CRS, row 0 at the top
        top = 7040000.0 + YS[-1] + SPACING / 2
        return GeoImage(
            values[::-1].copy(), (270000.0, SPACING, 0.0, top, 0.0, -SPACING), "EPSG:25833"
        )

    def fake(key, georef_, box, spacing, fetch=None, land_only=True):
        if key == "copernicus":
            return [image(np.where(cop > 0, cop, np.nan) if land_only else cop)]
        if key == "emodnet":
            return [image(emodnet)]
        raise OSError(f"{key}: not asked for in this test")

    monkeypatch.setattr(download, "download_all", fake)
    monkeypatch.setattr(download, "covers", lambda source, g, b: source.label[:3] != "Kar")
    out, used = bathy.web_heights("best", georef, XS, YS, SPACING, lat_below_msl=1.5)
    assert used == ["EMODnet (European seas)", "Copernicus DEM GLO-30 (land, world)"]
    assert np.all(out[GX < 300] == pytest.approx(-16.5))  # 15 m below LAT, LAT 1.5 m down
    assert (out[GX < COAST] < 0).all() and (out[(GX > COAST) & (GX < 700)] > 0).all()
    # without the coastline: EMODnet's smeared coast, as before
    old, _ = bathy.web_heights("best", georef, XS, YS, SPACING, coastline=False)
    assert (old[(GX > 480) & (GX < COAST)] > 0).any()
