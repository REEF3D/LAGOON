# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Coordinate reference systems: from one CRS to another.

With pyproj installed (``lagoon-viz[geo]``) any pair of CRSs works. Without it,
LAGOON transforms the ones a coastal model in Europe mostly needs itself:

- geographic WGS 84 and ETRS89 (``EPSG:4326``, ``EPSG:4258``), in degrees
- Web Mercator (``EPSG:3857``), the CRS of map tiles
- UTM on WGS 84 (``EPSG:326zz`` north, ``EPSG:327zz`` south) and on ETRS89
  (``EPSG:258zz``, e.g. 25832, 25833, 25835 in Norway)
- the Norwegian NTM zones 5 to 30 (``EPSG:5105`` - ``EPSG:5130``)

ETRS89 is taken as WGS 84: they part by well under a metre in Europe, much less
than a map tile's pixel. The transverse Mercator projection is Krüger's series to
the sixth order in n (Karney 2011), good to a few nanometres within a zone.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from lagoon_format.geo import epsg_code

#: the GRS 80 / WGS 84 ellipsoid (the two differ by 0.1 mm in the semi-minor axis)
A = 6378137.0
F = 1 / 298.257223563
N3 = F / (2 - F)

Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class CrsError(ValueError):
    pass


@dataclass(frozen=True)
class Projection:
    """A CRS LAGOON can transform without pyproj."""

    kind: str  # "geographic", "mercator", "tm"
    name: str
    lon0: float = 0.0
    k0: float = 1.0
    false_easting: float = 0.0
    false_northing: float = 0.0


#: names of common CRSs for the CRS picker: code -> name
COMMON = {
    4326: "WGS 84 (longitude, latitude)",
    4258: "ETRS89 (longitude, latitude)",
    3857: "Web Mercator",
    25831: "ETRS89 / UTM zone 31N",
    25832: "ETRS89 / UTM zone 32N",
    25833: "ETRS89 / UTM zone 33N",
    25834: "ETRS89 / UTM zone 34N",
    25835: "ETRS89 / UTM zone 35N",
    32631: "WGS 84 / UTM zone 31N",
    32632: "WGS 84 / UTM zone 32N",
    32633: "WGS 84 / UTM zone 33N",
    32634: "WGS 84 / UTM zone 34N",
    32635: "WGS 84 / UTM zone 35N",
    5110: "ETRS89 / NTM zone 10",
    5105: "ETRS89 / NTM zone 5",
}


def builtin(code: int) -> Projection | None:
    """The projection of an EPSG code LAGOON knows itself, or None."""
    if code in (4326, 4258):
        return Projection("geographic", "WGS 84" if code == 4326 else "ETRS89")
    if code in (3857, 900913):
        return Projection("mercator", "Web Mercator")
    if 32601 <= code <= 32660 or 32701 <= code <= 32760:
        zone = code % 100
        south = code >= 32701
        return Projection(
            "tm",
            f"WGS 84 / UTM zone {zone}{'S' if south else 'N'}",
            lon0=6.0 * zone - 183.0,
            k0=0.9996,
            false_easting=500000.0,
            false_northing=10000000.0 if south else 0.0,
        )
    if 25828 <= code <= 25838:
        zone = code - 25800
        return Projection(
            "tm",
            f"ETRS89 / UTM zone {zone}N",
            lon0=6.0 * zone - 183.0,
            k0=0.9996,
            false_easting=500000.0,
        )
    if 5105 <= code <= 5130:
        zone = code - 5100
        return Projection(
            "tm",
            f"ETRS89 / NTM zone {zone}",
            lon0=zone + 0.5,
            k0=1.0,
            false_easting=100000.0,
            false_northing=1000000.0,
        )
    return None


def has_pyproj() -> bool:
    try:
        import pyproj  # noqa: F401
    except ImportError:
        return False
    return True


def describe(crs: str) -> str:
    """A readable name of a CRS ("ETRS89 / UTM zone 33N"), else the text given."""
    code = epsg_code(crs)
    if code is not None:
        if code in COMMON:
            return COMMON[code]
        found = builtin(code)
        if found is not None:
            return found.name
    if has_pyproj():
        try:
            import pyproj

            return pyproj.CRS.from_user_input(crs).name
        except Exception:  # noqa: BLE001 (pyproj raises its own errors)
            pass
    return str(crs) if len(str(crs)) < 60 else str(crs)[:57] + "..."


def is_known(crs: str) -> bool:
    """Whether LAGOON can transform this CRS (itself or with pyproj)."""
    code = epsg_code(crs)
    if code is not None and builtin(code) is not None:
        return True
    if has_pyproj():
        try:
            import pyproj

            pyproj.CRS.from_user_input(crs)
        except Exception:  # noqa: BLE001
            return False
        return True
    return False


def is_geographic(crs: str) -> bool:
    code = epsg_code(crs)
    found = builtin(code) if code is not None else None
    if found is not None:
        return found.kind == "geographic"
    if has_pyproj():
        import pyproj

        try:
            return bool(pyproj.CRS.from_user_input(crs).is_geographic)
        except Exception:  # noqa: BLE001
            return False
    return False


#: where ETRS89 is used: Europe (lon_min, lat_min, lon_max, lat_max)
EUROPE = (-16.1, 32.9, 45.0, 84.2)
#: how far from its central meridian a transverse Mercator CRS is used [deg]: a UTM
#: zone and its neighbours (beyond, the map turns and stretches)
TM_REACH = 6.0


def fits(crs: str, lon: float, lat: float) -> bool:
    """Whether a CRS is meant for a place: a UTM or NTM zone near it (and ETRS89 in
    Europe), Web Mercator and geographic ones anywhere, others as pyproj's area of
    use says (True without pyproj)."""
    code = epsg_code(crs)
    found = builtin(code) if code is not None else None
    if found is not None:
        if found.kind != "tm":
            return found.kind == "geographic" or abs(lat) < 85.06
        if found.name.startswith("ETRS89"):
            lon0, lat0, lon1, lat1 = EUROPE
            if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
                return False
        if found.name.endswith("N") and lat < -1.0:  # a northern zone south of the equator
            return False
        if "UTM" in found.name and found.name.endswith("S") and lat > 1.0:
            return False
        return abs(math.remainder(lon - found.lon0, 360.0)) <= TM_REACH
    if has_pyproj():
        try:
            import pyproj

            area = pyproj.CRS.from_user_input(crs).area_of_use
        except Exception:  # noqa: BLE001 (pyproj raises its own errors)
            return True
        if area is None:
            return True
        west, south, east, north = area.bounds
        inside_lon = west <= lon <= east if west <= east else (lon >= west or lon <= east)
        return inside_lon and south - 1.0 <= lat <= north + 1.0
    return True


def utm_for(lon: float, lat: float, etrs89: bool = True) -> str:
    """The UTM CRS of a place (ETRS89 in Europe, else WGS 84): "EPSG:25833"."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    if 56 <= lat < 64 and 3 <= lon < 12:  # south-west Norway: zone 32 is wider
        zone = 32
    if etrs89 and lat >= 0 and 28 <= zone <= 38 and 34 <= lat <= 72:
        return f"EPSG:{25800 + zone}"
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


# ------------------------------------------------------------------ transforms
def transformer(source: str, target: str) -> Transform:
    """A function ``(x, y) -> (x, y)`` from one CRS to another (arrays). Geographic
    coordinates are (longitude, latitude) in degrees."""
    return _transformer(str(source).strip(), str(target).strip())


@lru_cache(maxsize=64)
def _transformer(source: str, target: str) -> Transform:
    if source == target:
        return lambda x, y: (np.asarray(x, float), np.asarray(y, float))
    codes = epsg_code(source), epsg_code(target)
    projections = [builtin(c) if c is not None else None for c in codes]
    if projections[0] is not None and projections[1] is not None:
        to_lonlat = _inverse(projections[0])
        from_lonlat = _forward(projections[1])

        def both(x, y):
            lon, lat = to_lonlat(np.asarray(x, float), np.asarray(y, float))
            return from_lonlat(lon, lat)

        return both
    if has_pyproj():
        import pyproj

        try:
            change = pyproj.Transformer.from_crs(source, target, always_xy=True)
        except Exception as error:  # noqa: BLE001
            raise CrsError(f"cannot transform {source} to {target}: {error}") from None
        return lambda x, y: change.transform(np.asarray(x, float), np.asarray(y, float))
    unknown = source if projections[0] is None else target
    raise CrsError(
        f"{unknown}: LAGOON knows WGS 84, ETRS89, Web Mercator, UTM and NTM by itself; "
        "for other CRSs install pyproj (pip install 'lagoon-viz[geo]')"
    )


def _forward(projection: Projection) -> Transform:
    """(lon, lat) in degrees to the projection."""
    if projection.kind == "geographic":
        return lambda lon, lat: (np.asarray(lon, float), np.asarray(lat, float))
    if projection.kind == "mercator":
        return mercator_forward
    return lambda lon, lat: tm_forward(lon, lat, projection)


def _inverse(projection: Projection) -> Transform:
    """The projection to (lon, lat) in degrees."""
    if projection.kind == "geographic":
        return lambda x, y: (np.asarray(x, float), np.asarray(y, float))
    if projection.kind == "mercator":
        return mercator_inverse
    return lambda x, y: tm_inverse(x, y, projection)


# ------------------------------------------------------------------ Web Mercator
#: half the width of the Web Mercator world [m]
MERCATOR_HALF = math.pi * A


def mercator_forward(lon, lat):
    lon = np.asarray(lon, float)
    lat = np.clip(np.asarray(lat, float), -85.05112878, 85.05112878)
    x = A * np.radians(lon)
    y = A * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def mercator_inverse(x, y):
    lon = np.degrees(np.asarray(x, float) / A)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y, float) / A)) - np.pi / 2)
    return lon, lat


# ------------------------------------------------------------------ transverse Mercator
def _series():
    n = N3
    n2, n3, n4, n5, n6 = n**2, n**3, n**4, n**5, n**6
    big_a = A / (1 + n) * (1 + n2 / 4 + n4 / 64 + n6 / 256)
    alpha = (
        n / 2 - 2 * n2 / 3 + 5 * n3 / 16 + 41 * n4 / 180 - 127 * n5 / 288 + 7891 * n6 / 37800,
        13 * n2 / 48 - 3 * n3 / 5 + 557 * n4 / 1440 + 281 * n5 / 630 - 1983433 * n6 / 1935360,
        61 * n3 / 240 - 103 * n4 / 140 + 15061 * n5 / 26880 + 167603 * n6 / 181440,
        49561 * n4 / 161280 - 179 * n5 / 168 + 6601661 * n6 / 7257600,
        34729 * n5 / 80640 - 3418889 * n6 / 1995840,
        212378941 * n6 / 319334400,
    )
    beta = (
        n / 2 - 2 * n2 / 3 + 37 * n3 / 96 - n4 / 360 - 81 * n5 / 512 + 96199 * n6 / 604800,
        n2 / 48 + n3 / 15 - 437 * n4 / 1440 + 46 * n5 / 105 - 1118711 * n6 / 3870720,
        17 * n3 / 480 - 37 * n4 / 840 - 209 * n5 / 4480 + 5569 * n6 / 90720,
        4397 * n4 / 161280 - 11 * n5 / 504 - 830251 * n6 / 7257600,
        4583 * n5 / 161280 - 108847 * n6 / 3991680,
        20648693 * n6 / 638668800,
    )
    return big_a, alpha, beta


_BIG_A, _ALPHA, _BETA = _series()
_E = math.sqrt(F * (2 - F))


def tm_forward(lon, lat, p: Projection):
    """Geographic (degrees) to transverse Mercator (Krüger, Karney 2011)."""
    phi = np.radians(np.asarray(lat, float))
    lam = np.radians(np.asarray(lon, float) - p.lon0)
    tau = np.tan(phi)
    sigma = np.sinh(_E * np.arctanh(_E * tau / np.sqrt(1 + tau**2)))
    tau_c = tau * np.sqrt(1 + sigma**2) - sigma * np.sqrt(1 + tau**2)  # conformal latitude
    xi_p = np.arctan2(tau_c, np.cos(lam))
    eta_p = np.arcsinh(np.sin(lam) / np.sqrt(tau_c**2 + np.cos(lam) ** 2))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_ALPHA, start=1):
        xi = xi + a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta + a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    x = p.false_easting + p.k0 * _BIG_A * eta
    y = p.false_northing + p.k0 * _BIG_A * xi
    return x, y


def tm_inverse(x, y, p: Projection):
    """Transverse Mercator to geographic (degrees)."""
    xi = (np.asarray(y, float) - p.false_northing) / (p.k0 * _BIG_A)
    eta = (np.asarray(x, float) - p.false_easting) / (p.k0 * _BIG_A)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, start=1):
        xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    tau_c = np.sin(xi_p) / np.sqrt(np.sinh(eta_p) ** 2 + np.cos(xi_p) ** 2)
    lam = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    # the latitude from the conformal one: Newton's method on tau (Karney eq. 19-21)
    tau = tau_c.copy()
    for _ in range(5):
        sigma = np.sinh(_E * np.arctanh(_E * tau / np.sqrt(1 + tau**2)))
        tau_i = tau * np.sqrt(1 + sigma**2) - sigma * np.sqrt(1 + tau**2)
        d_tau = (
            (tau_c - tau_i)
            / np.sqrt(1 + tau_i**2)
            * (1 + (1 - _E**2) * tau**2)
            / ((1 - _E**2) * np.sqrt(1 + tau**2))
        )
        tau = tau + d_tau
    lat = np.degrees(np.arctan(tau))
    lon = np.degrees(lam) + p.lon0
    return lon, lat
