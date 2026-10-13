# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Bathymetry from the web, for the area of a case: no file needed.

- **Kartverket** (Norway): the national topobathy terrain model, land and sea in
  one (the national DTM on land, depth surveys at sea; 1 m where surveyed, else
  50 m), heights in NN2000; open data. A WCS in EPSG:25832, 25833 and 25835.
- **EMODnet** (European seas): the EMODnet Digital Terrain Model, about 115 m,
  heights (negative below the sea) relative to LAT; CC BY 4.0.

:func:`download` asks the service for the area round a model rectangle, at the
spacing wanted, and returns it as a :class:`~lagoonviz.core.geo.raster.GeoImage`
in the service's CRS. Answers are kept in the disk cache (next to the map tiles).
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lagoon_format.geo import Georef

from lagoonviz.core.geo.crs import transformer
from lagoonviz.core.geo.raster import GeoImage, RasterError, read_raster
from lagoonviz.core.geo.tiles import cache_folder, http_fetch

#: the longest side of one request [pixels]
MOST_PIXELS = 4000


@dataclass(frozen=True)
class BathymetrySource:
    key: str
    label: str
    attribution: str
    licence: str
    #: what the heights are relative to
    datum: str
    #: finest cell the service has [m]
    finest: float
    #: where it has data (lon_min, lat_min, lon_max, lat_max)
    region: tuple[float, float, float, float]


SOURCES = {
    s.key: s
    for s in (
        BathymetrySource(
            "kartverket",
            "Kartverket, land and sea (Norway)",
            "© Kartverket (national topobathy terrain model)",
            "open data (NLOD / CC BY 4.0)",
            "NN2000 (EPSG:5941)",
            1.0,
            (-2.0, 57.2, 33.4, 72.2),
        ),
        BathymetrySource(
            "emodnet",
            "EMODnet (European seas)",
            "EMODnet Bathymetry Consortium, EMODnet Digital Terrain Model",
            "CC BY 4.0",
            "LAT",
            115.0,
            (-36.0, 15.0, 43.0, 90.0),
        ),
    )
}
#: the choice that takes the finest source with data, the others for the gaps
BEST = "best"

KARTVERKET_WCS = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-topobathy-{zone}"
EMODNET_WCS = "https://ows.emodnet-bathymetry.eu/wcs"


def _world_box(georef: Georef, box, crs: str, pad: float = 0.05):
    """The bounds in ``crs`` of a model rectangle (x0, x1, y0, y1), its edges
    sampled (a turned rectangle, a curved projection), grown by ``pad``."""
    x0, x1, y0, y1 = box
    t = np.linspace(0.0, 1.0, 9)
    mx = np.r_[x0 + (x1 - x0) * t, np.full(9, x1), x1 - (x1 - x0) * t, np.full(9, x0)]
    my = np.r_[np.full(9, y0), y0 + (y1 - y0) * t, np.full(9, y1), y1 - (y1 - y0) * t]
    big_x, big_y = georef.to_world(mx, my)
    if georef.crs != crs:
        big_x, big_y = transformer(georef.crs, crs)(big_x, big_y)
    lx, hx, ly, hy = big_x.min(), big_x.max(), big_y.min(), big_y.max()
    gx, gy = pad * (hx - lx), pad * (hy - ly)
    return float(lx - gx), float(ly - gy), float(hx + gx), float(hy + gy)


def kartverket_url(bounds, crs: str, spacing: float) -> str:
    """WCS 1.0.0 GetCoverage of the topobathy model over bounds (x0, y0, x1, y1) in
    one of its UTM zones (EPSG:25832, 25833, 25835)."""
    zone = crs.split(":")[-1]
    x0, y0, x1, y1 = bounds
    width, height = _size(bounds, spacing)
    query = urllib.parse.urlencode(
        {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": f"nhm_dtm_topobathy_{zone}",
            "crs": crs,
            "bbox": f"{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}",
            "width": width,
            "height": height,
            "format": "GeoTIFF",
            "interpolation": "bilinear",
        }
    )
    return KARTVERKET_WCS.format(zone=zone) + "?" + query


def emodnet_url(bounds) -> str:
    """WCS 2.0.1 GetCoverage of EMODnet's mean depth over bounds in longitude and
    latitude (x0, y0, x1, y1)."""
    lon0, lat0, lon1, lat1 = bounds
    return (
        f"{EMODNET_WCS}?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        "&CoverageId=emodnet__mean"
        f"&SUBSET=Lat({lat0:.6f},{lat1:.6f})&SUBSET=Long({lon0:.6f},{lon1:.6f})"
        "&FORMAT=image%2Ftiff"
    )


def _size(bounds, spacing: float) -> tuple[int, int]:
    x0, y0, x1, y1 = bounds
    spacing = max(float(spacing), 1.0)
    width = math.ceil((x1 - x0) / spacing)
    height = math.ceil((y1 - y0) / spacing)
    scale = max(width, height) / MOST_PIXELS
    if scale > 1:
        width, height = math.ceil(width / scale), math.ceil(height / scale)
    return max(width, 2), max(height, 2)


def kartverket_crs(georef: Georef) -> str:
    """The zone of the service to ask: the case's own when it is one of them."""
    return georef.crs if georef.crs in ("EPSG:25832", "EPSG:25833", "EPSG:25835") else "EPSG:25833"


def covers(source: BathymetrySource, georef: Georef, box) -> bool:
    """Whether a model rectangle lies (partly) where the source has data."""
    lon0, lat0, lon1, lat1 = _world_box(georef, box, "EPSG:4326", pad=0.0)
    r = source.region
    return lon1 >= r[0] and lon0 <= r[2] and lat1 >= r[1] and lat0 <= r[3]


def download(key: str, georef: Georef, box, spacing: float, fetch=None) -> GeoImage:
    """The bathymetry of a source over a model rectangle (x0, x1, y0, y1) and a
    margin, at about ``spacing`` metres (no finer than the source has), as a
    GeoImage in the service's CRS; fetched once, then from the cache."""
    fetch = fetch or http_fetch
    source = SOURCES[key]
    spacing = max(float(spacing), source.finest)
    if key == "kartverket":
        crs = kartverket_crs(georef)
        url = kartverket_url(_world_box(georef, box, crs), crs, spacing)
    elif key == "emodnet":
        url = emodnet_url(_world_box(georef, box, "EPSG:4326"))
    else:
        raise ValueError(f"unknown bathymetry source {key!r}; known: {', '.join(SOURCES)}")
    folder = cache_folder().parent / "bathymetry"
    path = folder / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".tif")
    if not path.is_file():
        try:
            data = fetch(url)
        except OSError as error:
            raise OSError(f"{source.label}: {error}") from None
        if data[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            text = data[:400].decode("utf-8", "replace")
            raise RasterError(f"{source.label} sent no GeoTIFF: {text.strip()[:300]}")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except OSError:  # no cache: read it from a temporary file
            path = Path(tempfile.mkdtemp()) / "bathymetry.tif"
            path.write_bytes(data)
    image = read_raster(path)
    if image.crs is None:
        image.crs = kartverket_crs(georef) if key == "kartverket" else "EPSG:4326"
    if image.is_image:
        raise RasterError(f"{source.label} sent a picture, not heights")
    values = np.asarray(image.data, float)
    # no data: the services' fill values (-32767, -9999, -3.4e38)
    image.data = np.where((values < -11000) | (values > 9000), np.nan, values)
    image.nodata = None
    image.name = source.label
    return image
