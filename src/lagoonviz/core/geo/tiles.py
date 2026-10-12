# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Map tiles: background maps and satellite images from tile servers, cached on disk.

A tile source serves 256 x 256 pixel images in Web Mercator (``EPSG:3857``), by
zoom level ``z`` and column/row ``x``, ``y`` (XYZ, the "GoogleMapsCompatible" tile
matrix set of WMTS). :func:`mosaic` puts together the tiles over an area into one
image; the background map item then warps it into the model's frame.

Every source names its licence and the attribution the map must show. Tiles are
kept in a disk cache (``LAGOON_CACHE`` or the user's cache folder), so a map is
fetched once; LAGOON asks the servers with its own name, as their usage policies
want, and only for the tiles in view.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lagoonviz.core.geo.crs import MERCATOR_HALF

TILE = 256


@dataclass(frozen=True)
class TileSource:
    """A tile server: its URL with ``{z}``, ``{x}``, ``{y}``, and its terms."""

    key: str
    label: str
    url: str
    attribution: str
    licence: str
    max_zoom: int = 19
    #: what it shows: "map", "chart" or "imagery"
    kind: str = "map"
    #: where it has tiles (lon_min, lat_min, lon_max, lat_max); None: the world
    region: tuple[float, float, float, float] | None = None


#: the tile sources of the background map item
SOURCES: dict[str, TileSource] = {
    s.key: s
    for s in (
        TileSource(
            "kartverket_topo",
            "Kartverket topographic map (Norway)",
            "https://cache.kartverket.no/v1/wmts/1.0.0/topo/default/webmercator/{z}/{y}/{x}.png",
            "© Kartverket",
            "CC BY 4.0",
            max_zoom=18,
            region=(-10.0, 56.0, 35.0, 82.0),
        ),
        TileSource(
            "kartverket_grey",
            "Kartverket topographic map, grey (Norway)",
            "https://cache.kartverket.no/v1/wmts/1.0.0/topograatone/default/webmercator/{z}/{y}/{x}.png",
            "© Kartverket",
            "CC BY 4.0",
            max_zoom=18,
            region=(-10.0, 56.0, 35.0, 82.0),
        ),
        TileSource(
            "kartverket_chart",
            "Kartverket nautical chart (Norway)",
            "https://cache.kartverket.no/v1/wmts/1.0.0/sjokartraster/default/webmercator/{z}/{y}/{x}.png",
            "© Kartverket",
            "CC BY 4.0",
            max_zoom=18,
            kind="chart",
            region=(-10.0, 56.0, 35.0, 82.0),
        ),
        TileSource(
            "sentinel2_2017",
            "Sentinel-2 cloudless 2017 (EOX, satellite)",
            "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2017_3857/default/"
            "GoogleMapsCompatible/{z}/{y}/{x}.jpg",
            "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH "
            "(Contains modified Copernicus Sentinel data 2016 & 2017)",
            "CC BY 4.0",
            max_zoom=15,
            kind="imagery",
        ),
        TileSource(
            "openstreetmap",
            "OpenStreetMap",
            "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "© OpenStreetMap contributors",
            "ODbL",
            max_zoom=19,
        ),
    )
}
#: the source a new background map starts with
DEFAULT_SOURCE = "kartverket_topo"


def custom_source(url: str, attribution: str = "") -> TileSource:
    """A tile server of the user's (an XYZ URL with {z}, {x}, {y})."""
    for key in ("{z}", "{x}", "{y}"):
        if key not in url:
            raise ValueError(f"a tile URL needs {{z}}, {{x}} and {{y}}; {url!r} has no {key}")
    return TileSource("custom", "Tile server", url, attribution or url.split("/")[2], "")


def source_for(lon: float, lat: float, wanted: str | None = None) -> TileSource:
    """The source wanted, else the default one if it covers the place, else
    OpenStreetMap."""
    if wanted in SOURCES:
        return SOURCES[wanted]
    source = SOURCES[DEFAULT_SOURCE]
    if source.region is not None:
        lon0, lat0, lon1, lat1 = source.region
        if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
            return SOURCES["openstreetmap"]
    return source


# ------------------------------------------------------------------ tile maths
def resolution(zoom: int) -> float:
    """Metres (Web Mercator) per pixel at a zoom level."""
    return 2 * MERCATOR_HALF / (TILE * 2**zoom)


def zoom_for(metres_per_pixel: float, max_zoom: int = 19) -> int:
    """The coarsest zoom whose pixels are at most this big (Web Mercator metres)."""
    if not metres_per_pixel > 0:
        return max_zoom
    zoom = math.ceil(math.log2(2 * MERCATOR_HALF / (TILE * metres_per_pixel)))
    return int(min(max(zoom, 0), max_zoom))


def tile_range(bounds: tuple[float, float, float, float], zoom: int):
    """Columns and rows of the tiles over Web Mercator bounds (x0, y0, x1, y1):
    ``(col0, col1, row0, row1)``, inclusive."""
    x0, y0, x1, y1 = bounds
    size = 2 * MERCATOR_HALF / 2**zoom
    n = 2**zoom
    col0 = int(math.floor((x0 + MERCATOR_HALF) / size))
    col1 = int(math.floor((x1 + MERCATOR_HALF) / size))
    row0 = int(math.floor((MERCATOR_HALF - y1) / size))
    row1 = int(math.floor((MERCATOR_HALF - y0) / size))
    return max(col0, 0), min(col1, n - 1), max(row0, 0), min(row1, n - 1)


def tile_bounds(zoom: int, col: int, row: int) -> tuple[float, float, float, float]:
    """Web Mercator bounds (x0, y0, x1, y1) of a tile."""
    size = 2 * MERCATOR_HALF / 2**zoom
    x0 = -MERCATOR_HALF + col * size
    y1 = MERCATOR_HALF - row * size
    return x0, y1 - size, x0 + size, y1


# ------------------------------------------------------------------ the cache
def cache_folder() -> Path:
    """Where tiles are kept: ``$LAGOON_CACHE/tiles``, else the user's cache folder."""
    given = os.environ.get("LAGOON_CACHE")
    if given:
        return Path(given).expanduser() / "tiles"
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "LAGOON"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LAGOON" / "Cache"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lagoon"
    return base / "tiles"


def _user_agent() -> str:
    from lagoonviz import __version__

    return f"LAGOON/{__version__} (REEF3D post-processor; +https://github.com/REEF3D/LAGOON)"


#: Fetch signature: ``fetch(url) -> bytes`` (raises OSError when it cannot)
Fetch = Callable[[str], bytes]


def http_fetch(url: str, timeout: float = 20.0) -> bytes:
    if os.environ.get("LAGOON_OFFLINE"):  # tests, or no network wanted: the cache only
        raise OSError("offline (LAGOON_OFFLINE)")
    request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(request, timeout=timeout) as answer:  # noqa: S310 (https tile URLs)
        return answer.read()


class TileCache:
    """Tiles on disk by source and z/x/y; missing ones are fetched (``fetch``)."""

    def __init__(self, folder: str | Path | None = None, fetch: Fetch | None = None):
        self.folder = Path(folder) if folder is not None else cache_folder()
        self.fetch = fetch or http_fetch
        self._lock = threading.Lock()
        #: tiles that could not be fetched this session (not asked for again)
        self.missing: set[str] = set()

    def _path(self, source: TileSource, zoom: int, col: int, row: int) -> Path:
        name = source.key
        if source.key == "custom":
            name = "custom-" + hashlib.sha1(source.url.encode()).hexdigest()[:12]
        suffix = ".jpg" if source.url.endswith(".jpg") or source.url.endswith(".jpeg") else ".png"
        return self.folder / name / str(zoom) / str(col) / f"{row}{suffix}"

    def tile(self, source: TileSource, zoom: int, col: int, row: int) -> np.ndarray | None:
        """A tile as (256, 256, 4) uint8 RGBA, or None when it cannot be had."""
        path = self._path(source, zoom, col, row)
        url = source.url.format(z=zoom, x=col, y=row)
        data = None
        if path.is_file():
            try:
                data = path.read_bytes()
            except OSError:
                data = None
        if data is None:
            if url in self.missing:
                return None
            try:
                data = self.fetch(url)
            except (OSError, urllib.error.URLError, ValueError):
                with self._lock:
                    self.missing.add(url)
                return None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(path)
            except OSError:
                pass  # no cache: the tile is still shown
        return decode(data)


def decode(data: bytes) -> np.ndarray | None:
    """PNG or JPEG bytes as (h, w, 4) uint8 RGBA."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGBA"), np.uint8)
    except (OSError, ValueError):
        return None


@dataclass
class Mosaic:
    """Tiles put together: an RGBA image and its Web Mercator bounds."""

    image: np.ndarray  # (h, w, 4) uint8, row 0 at the top (north)
    bounds: tuple[float, float, float, float]  # x0, y0, x1, y1
    zoom: int
    #: tiles that could not be had (shown transparent)
    missing: int = 0


#: at most this many tiles in one mosaic (the zoom is lowered to stay under it)
MAX_TILES = 256


def mosaic(
    source: TileSource,
    bounds: tuple[float, float, float, float],
    zoom: int,
    cache: TileCache | None = None,
    threads: int = 6,
) -> Mosaic:
    """The tiles of a source over Web Mercator bounds at a zoom level (lowered while
    more than :data:`MAX_TILES` would be needed), as one image."""
    cache = cache or TileCache()
    zoom = min(zoom, source.max_zoom)
    while zoom > 0:
        col0, col1, row0, row1 = tile_range(bounds, zoom)
        if (col1 - col0 + 1) * (row1 - row0 + 1) <= MAX_TILES:
            break
        zoom -= 1
    col0, col1, row0, row1 = tile_range(bounds, zoom)
    ncol, nrow = col1 - col0 + 1, row1 - row0 + 1
    image = np.zeros((nrow * TILE, ncol * TILE, 4), np.uint8)
    jobs = [(c, r) for r in range(row0, row1 + 1) for c in range(col0, col1 + 1)]
    with ThreadPoolExecutor(max(1, threads)) as pool:
        tiles = list(pool.map(lambda cr: cache.tile(source, zoom, cr[0], cr[1]), jobs))
    missing = 0
    for (c, r), tile in zip(jobs, tiles, strict=True):
        if tile is None or tile.shape[:2] != (TILE, TILE):
            missing += 1
            continue
        i, j = (r - row0) * TILE, (c - col0) * TILE
        image[i : i + TILE, j : j + TILE] = tile
    x0, _, _, y1 = tile_bounds(zoom, col0, row0)
    _, y0, x1, _ = tile_bounds(zoom, col1, row1)
    return Mosaic(image, (x0, y0, x1, y1), zoom, missing)
