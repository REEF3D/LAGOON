# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""ICESat-2's nearshore soundings: NASA's ATL24 bathymetry, the world over.

ICESat-2's green laser sees through clear water to some 20 to 40 m (a few metres
in turbid water). ATL24 (NSIDC, version 2) keeps the photons classified as the sea
floor (``class_ph`` 40), refraction corrected, with their height over the EGM2008
geoid (``ortho_h``: the datum of Copernicus' and ETOPO's heights). Along each of
the six beams of a track they are few, along lines some 90 m and 3.3 km apart:
:mod:`~lagoonviz.core.geo.nearshore` uses them to guide the profile next to the
coast, not to lie over the sea as they are.

- :func:`read` reads one ATL24 file (h5py): the sea floor photons that are kept
  (confidence, not low confidence, not deeper than the sensor sees), as medians
  over ALONG metres along each beam;
- :func:`find` asks NASA's CMR for the ATL24 files over an area (no login);
- :func:`download` fetches them into the cache (a NASA Earthdata login:
  ``EARTHDATA_TOKEN``, or ``~/.netrc`` with ``machine urs.earthdata.nasa.gov``);
- :func:`soundings` makes them one :class:`~lagoonviz.core.geo.bathy.Bathymetry`
  of points in longitude and latitude.

Files downloaded elsewhere (the NSIDC Data Access Tool, Earthdata Search,
``earthaccess``) are read as they are: a file ``ATL24_*.h5`` or a folder of them.
"""

from __future__ import annotations

import json
import netrc
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from http.cookiejar import CookieJar
from pathlib import Path

import numpy as np

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME = "ATL24"
#: the beams of a track
BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
#: class_ph of the sea floor
SEA_FLOOR = 40
#: photons kept with at least this ensemble confidence (ATL24's low confidence: 0.6)
MIN_CONFIDENCE = 0.6
#: a sounding is the median of the photons over this many metres along a beam
ALONG = 10.0
#: and needs at least this many photons there
LEAST_PHOTONS = 3
#: at most this many files for one domain (the newest; each some 5 to 50 MB)
MOST_FILES = 40
#: the Earthdata login's host
URS = "urs.earthdata.nasa.gov"
#: the attribution of the data
ATTRIBUTION = (
    "ICESat-2 ATL24 nearshore bathymetry: Parrish, C. et al., NASA NSIDC DAAC "
    "(doi:10.5067/ATLAS/ATL24.002)"
)


def is_atl24(path: str | Path) -> bool:
    """Whether a path is an ATL24 file or a folder holding some."""
    path = Path(path).expanduser()
    if path.is_dir():
        return any(path.glob("ATL24*.h5"))
    return path.suffix.lower() == ".h5"


def files_in(path: str | Path) -> list[Path]:
    """The ATL24 files of a path: itself, or those in the folder."""
    path = Path(path).expanduser()
    if path.is_dir():
        return sorted(path.glob("ATL24*.h5"))
    return [path]


def read(
    path: str | Path,
    bounds: tuple[float, float, float, float] | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    along: float = ALONG,
) -> np.ndarray:
    """The sea floor of one ATL24 file as soundings (n, 3): longitude, latitude and
    the height over the EGM2008 geoid [m] (negative under the sea). ``bounds``
    (lon0, lat0, lon1, lat1): only there. Kept: class 40 (sea floor), confidence
    at least ``min_confidence``, not flagged low confidence nor deeper than the
    sensor sees; then the median of each ``along`` metres of a beam with at least
    LEAST_PHOTONS photons (a stray photon is no sounding)."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "reading ICESat-2 files needs h5py: pip install h5py (or pip install 'lagoon-viz[geo]')"
        ) from None
    found = []
    with h5py.File(Path(path).expanduser(), "r") as file:
        for beam in BEAMS:
            if beam not in file or "class_ph" not in file[beam]:
                continue
            group = file[beam]
            keep = np.asarray(group["class_ph"][()]) == SEA_FLOOR
            if not keep.any():
                continue
            for name, good in (
                ("low_confidence_flag", 0),
                ("sensor_depth_exceeded", 0),
            ):
                if name in group:
                    keep &= np.asarray(group[name][()]) == good
            if "confidence" in group:
                keep &= np.asarray(group["confidence"][()], float) >= min_confidence
            lon = np.asarray(group["lon_ph"][()], float)[keep]
            lat = np.asarray(group["lat_ph"][()], float)[keep]
            height = np.asarray(group["ortho_h"][()], float)[keep]
            x = np.asarray(group["x_atc"][()], float)[keep]
            if bounds is not None:
                lon0, lat0, lon1, lat1 = bounds
                inside = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
                lon, lat, height, x = lon[inside], lat[inside], height[inside], x[inside]
            good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(height) & (height < 1e30)
            if good.any():
                found.append(_medians(lon[good], lat[good], height[good], x[good], along))
    if not found:
        return np.zeros((0, 3))
    return np.concatenate(found)


def _medians(lon, lat, height, x, along: float) -> np.ndarray:
    """The median sounding of each ``along`` metres of a beam with enough photons."""
    bins = np.floor(x / max(float(along), 1e-6)).astype(np.int64)
    order = np.argsort(bins, kind="stable")
    bins, lon, lat, height = bins[order], lon[order], lat[order], height[order]
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    ends = np.r_[starts[1:], len(bins)]
    out = [
        (np.median(lon[a:b]), np.median(lat[a:b]), np.median(height[a:b]))
        for a, b in zip(starts, ends, strict=True)
        if b - a >= LEAST_PHOTONS
    ]
    return np.array(out, float).reshape(-1, 3)


def soundings(paths, bounds=None, name: str = "ICESat-2 ATL24"):
    """The soundings of ATL24 files as one Bathymetry: points in EPSG:4326, heights
    over EGM2008 (``bounds``: lon0, lat0, lon1, lat1)."""
    from lagoonviz.core.geo.bathy import Bathymetry

    parts = [read(path, bounds) for path in paths]
    points = np.concatenate(parts) if parts else np.zeros((0, 3))
    return Bathymetry(name, "EPSG:4326", points=points)


# ------------------------------------------------------------------- NASA's servers
def find(
    bounds: tuple[float, float, float, float],
    fetch: Callable[[str], bytes] | None = None,
    most: int = MOST_FILES,
) -> list[str]:
    """The ATL24 files over ``bounds`` (lon0, lat0, lon1, lat1): their HTTPS
    addresses, the newest first, at most ``most`` (NASA's CMR, no login)."""
    from lagoonviz.core.geo.tiles import http_fetch

    fetch = fetch or http_fetch
    lon0, lat0, lon1, lat1 = bounds
    query = urllib.parse.urlencode(
        {
            "short_name": SHORT_NAME,
            "bounding_box": f"{lon0:.6f},{lat0:.6f},{lon1:.6f},{lat1:.6f}",
            "page_size": 2000,
            "sort_key": "-start_date",
        }
    )
    try:
        answer = json.loads(fetch(CMR + "?" + query).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise OSError(f"ICESat-2: NASA's CMR sent no list of files ({error})") from None
    urls: dict[str, str] = {}
    for entry in answer.get("feed", {}).get("entry", []):
        for link in entry.get("links", []):
            href = str(link.get("href", ""))
            if (
                href.startswith("https://")
                and href.endswith(".h5")
                and "data#" in str(link.get("rel", ""))
            ):
                urls.setdefault(href.rsplit("/", 1)[-1], href)
                break
    return list(urls.values())[: max(int(most), 0)]


def earthdata_fetch(url: str, timeout: float = 300.0) -> bytes:
    """A file from NASA's Earthdata servers, logged in with ``EARTHDATA_TOKEN`` (a
    bearer token from urs.earthdata.nasa.gov) or ``~/.netrc``."""
    from lagoonviz.core.geo.tiles import _user_agent

    if os.environ.get("LAGOON_OFFLINE"):
        raise OSError("offline (LAGOON_OFFLINE)")
    headers = {"User-Agent": _user_agent()}
    handlers: list = [urllib.request.HTTPCookieProcessor(CookieJar())]
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        try:
            login = netrc.netrc().authenticators(URS)
        except (OSError, netrc.NetrcParseError):
            login = None
        if not login:
            raise OSError(
                "ICESat-2 needs a NASA Earthdata login (free, urs.earthdata.nasa.gov): set "
                "EARTHDATA_TOKEN, or put 'machine urs.earthdata.nasa.gov login … password …' "
                "in ~/.netrc"
            )
        passwords = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        passwords.add_password(None, f"https://{URS}", login[0], login[2] or "")
        handlers.append(urllib.request.HTTPBasicAuthHandler(passwords))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=timeout) as answer:  # noqa: S310 (https from CMR)
        return answer.read()


def cache_folder() -> Path:
    """Where downloaded ATL24 files are kept (next to the map tiles)."""
    from lagoonviz.core.geo.tiles import cache_folder as tiles

    return tiles().parent / "icesat2"


def download(
    bounds: tuple[float, float, float, float],
    fetch: Callable[[str], bytes] | None = None,
    find_fetch: Callable[[str], bytes] | None = None,
    progress: Callable[[float, str], None] | None = None,
    most: int = MOST_FILES,
) -> list[Path]:
    """The ATL24 files over ``bounds`` in the cache, fetched once (``fetch``: with
    the Earthdata login, :func:`earthdata_fetch`). ``progress(fraction, text)``."""
    fetch = fetch or earthdata_fetch
    urls = find(bounds, find_fetch, most)
    folder = cache_folder()
    paths = []
    for n, url in enumerate(urls):
        path = folder / url.rsplit("/", 1)[-1]
        if not path.is_file():
            if progress is not None:
                progress(n / max(len(urls), 1), f"ICESat-2: {path.name} ({n + 1}/{len(urls)})")
            data = fetch(url)
            folder.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".h5.part")
            part.write_bytes(data)
            part.replace(path)
        paths.append(path)
    return paths
