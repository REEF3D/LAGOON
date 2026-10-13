# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Bathymetry from the web, for the area of a case: no file needed.

- **Kartverket** (Norway): the national topobathy terrain model, land and sea in
  one (the national DTM on land, depth surveys at sea; 1 m where surveyed, else
  50 m), heights in NN2000; open data. A WCS in EPSG:25832, 25833 and 25835.
- **EMODnet** (European seas): the EMODnet Digital Terrain Model, about 115 m,
  heights (negative below the sea) relative to LAT; CC BY 4.0.
- **Copernicus DEM GLO-30** (land, the world): 30 m, from the open archive on AWS
  (1° tiles), heights to EGM2008; the sea is left to the other sources.
- **NOAA ETOPO 2022** (land and sea, the world): 15 arc-seconds (about 450 m),
  from NOAA's ERDDAP, heights to EGM2008; the last fallback.

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
        BathymetrySource(
            "copernicus",
            "Copernicus DEM GLO-30 (land, world)",
            "Copernicus DEM GLO-30: © DLR e.V. 2010-2014 and © Airbus Defence and Space "
            "GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA",
            "free (Copernicus DEM licence)",
            "EGM2008",
            30.0,
            (-180.0, -90.0, 180.0, 90.0),
        ),
        BathymetrySource(
            "etopo",
            "NOAA ETOPO 2022 (land and sea, world)",
            "NOAA National Centers for Environmental Information, ETOPO 2022 15 Arc-Second "
            "Global Relief Model (doi:10.25921/fd45-gt74)",
            "free (NOAA)",
            "EGM2008",
            450.0,
            (-180.0, -90.0, 180.0, 90.0),
        ),
    )
}
#: the choice that takes the finest source with data, the others for the gaps
BEST = "best"

KARTVERKET_WCS = "https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-topobathy-{zone}"
EMODNET_WCS = "https://ows.emodnet-bathymetry.eu/wcs"
COPERNICUS_TILE = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"  # 1° x 1° COGs
ETOPO_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.csv"
#: at most this many Copernicus tiles (1° x 1°, some 30 MB each) for one domain
MOST_TILES = 9


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


def copernicus_names(bounds) -> list[str]:
    """The Copernicus DEM tiles (named by their south-west corner) over bounds in
    longitude and latitude (x0, y0, x1, y1)."""
    lon0, lat0, lon1, lat1 = bounds
    names = []
    for lat in range(math.floor(lat0), math.floor(lat1) + 1):
        for lon in range(math.floor(lon0), math.floor(lon1) + 1):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"
            names.append(f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM")
    return names


def etopo_url(bounds, spacing: float) -> str:
    """ERDDAP query of ETOPO 2022 (15 arc-seconds) over bounds in longitude and
    latitude, every n-th value for a coarser spacing [m], as CSV."""
    lon0, lat0, lon1, lat1 = bounds
    step = 1.0 / 240.0
    cells = max((lon1 - lon0), (lat1 - lat0)) / step
    stride = max(1, int(spacing // 450.0), math.ceil(cells / MOST_PIXELS))
    query = f"z[({lat0:.6f}):{stride}:({lat1:.6f})][({lon0:.6f}):{stride}:({lon1:.6f})]"
    return ETOPO_ERDDAP + "?" + urllib.parse.quote(query, safe="():")


def _cached(url: str, suffix: str, fetch, label: str, check=None) -> Path | None:
    """The answer to a URL from the disk cache, else fetched and kept; None for a
    tile the server does not have (404: Copernicus has no tiles over the sea)."""
    folder = cache_folder().parent / "bathymetry"
    path = folder / (hashlib.sha1(url.encode()).hexdigest()[:20] + suffix)
    if path.is_file():
        return path if path.stat().st_size else None
    try:
        data = fetch(url)
    except OSError as error:
        if "404" in str(error) or "Not Found" in str(error):
            data = b""  # nothing there: remembered as an empty file
        else:
            raise OSError(f"{label}: {error}") from None
    if data and check is not None:
        check(data)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError:  # no cache: a temporary file
        path = Path(tempfile.mkdtemp()) / ("answer" + suffix)
        path.write_bytes(data)
    return path if data else None


def _tiff_check(label: str):
    def check(data: bytes) -> None:
        if data[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            text = data[:400].decode("utf-8", "replace")
            raise RasterError(f"{label} sent no GeoTIFF: {text.strip()[:300]}")

    return check


def _heights(image: GeoImage, label: str, low: float = -11000.0) -> GeoImage:
    if image.is_image:
        raise RasterError(f"{label} sent a picture, not heights")
    values = np.asarray(image.values() if image.nodata is not None else image.data, float)
    # no data: the services' fill values (-32767, -9999, -3.4e38)
    image.data = np.where((values < low) | (values > 9000), np.nan, values)
    image.nodata = None
    image.name = label
    return image


def download(key: str, georef: Georef, box, spacing: float, fetch=None) -> GeoImage:
    """The bathymetry of a source over a model rectangle (x0, x1, y0, y1) and a
    margin, at about ``spacing`` metres (no finer than the source has), as one
    GeoImage in the service's CRS; fetched once, then from the cache."""
    images = download_all(key, georef, box, spacing, fetch)
    if not images:
        raise RasterError(f"{SOURCES[key].label}: no data here")
    return images[0]


def download_all(key: str, georef: Georef, box, spacing: float, fetch=None) -> list[GeoImage]:
    """As :func:`download`, as the GeoImages the source sends (Copernicus: one per
    1° tile; none where it has no data)."""
    fetch = fetch or http_fetch
    if key not in SOURCES:
        raise ValueError(f"unknown bathymetry source {key!r}; known: {', '.join(SOURCES)}")
    source = SOURCES[key]
    label = source.label
    spacing = max(float(spacing), source.finest)
    if key == "kartverket":
        crs = kartverket_crs(georef)
        url = kartverket_url(_world_box(georef, box, crs), crs, spacing)
        path = _cached(url, ".tif", fetch, label, _tiff_check(label))
        image = read_raster(path)
        image.crs = image.crs or crs
        return [_heights(image, label)]
    if key == "emodnet":
        url = emodnet_url(_world_box(georef, box, "EPSG:4326"))
        path = _cached(url, ".tif", fetch, label, _tiff_check(label))
        image = read_raster(path)
        image.crs = image.crs or "EPSG:4326"
        return [_heights(image, label)]
    bounds = _world_box(georef, box, "EPSG:4326")
    if key == "copernicus":
        names = copernicus_names(bounds)
        if len(names) > MOST_TILES:
            raise RasterError(
                f"{label}: the domain spans {len(names)} tiles of 1°; at most {MOST_TILES}"
            )
        images = []
        for name in names:
            path = _cached(
                COPERNICUS_TILE.format(name=name), ".tif", fetch, label, _tiff_check(label)
            )
            if path is None:
                continue  # sea: no tile
            image = read_raster(path, window=bounds)
            image.crs = image.crs or "EPSG:4326"
            # land only: the sea is 0 in the model (left to the sea's sources)
            image = _heights(image, label)
            image.data = np.where(image.data <= 0.0, np.nan, image.data)
            images.append(image)
        return images
    # etopo: CSV of latitude, longitude, z from ERDDAP
    path = _cached(etopo_url(bounds, spacing), ".csv", fetch, label)
    if path is None:
        return []
    return [_heights(_grid_from_csv(path, label), label)]


def _grid_from_csv(path: Path, label: str) -> GeoImage:
    """ERDDAP's griddap CSV (latitude, longitude, z; a header and a units line) as a
    north-up raster in longitude and latitude."""
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=2, usecols=(0, 1, 2), ndmin=2)
    except ValueError:
        text = path.read_text(encoding="utf-8", errors="replace")[:300]
        raise RasterError(f"{label} sent no heights: {text.strip()}") from None
    lats = np.unique(values[:, 0])
    lons = np.unique(values[:, 1])
    if len(lats) < 2 or len(lons) < 2:
        raise RasterError(f"{label}: too few values for the area")
    grid = np.full((len(lats), len(lons)), np.nan)
    i = np.searchsorted(lats, values[:, 0])
    j = np.searchsorted(lons, values[:, 1])
    grid[i, j] = values[:, 2]
    dlat, dlon = float(lats[1] - lats[0]), float(lons[1] - lons[0])
    transform = (float(lons[0]) - dlon / 2, dlon, 0.0, float(lats[-1]) + dlat / 2, 0.0, -dlat)
    return GeoImage(grid[::-1], transform, "EPSG:4326", None, label)
