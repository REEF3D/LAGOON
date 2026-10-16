# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Measured water levels at tide gauges near a case, to compare a run with (and,
where wanted, to drive its boundaries):

- **Kartverket** (Norway, ``vannstand.kartverket.no``): observations and the tide
  predicted, relative to mean sea level, every 10 minutes; NLOD 2.0.
- **NOAA CO-OPS** (USA, ``api.tidesandcurrents.noaa.gov``): observations (6
  minutes) and predictions relative to MSL; public domain.
- **IOC Sea Level Station Monitoring Facility** (the world,
  ``ioc-sealevelmonitoring.org``): the raw observations of some 1000 stations of
  many agencies, without a vertical datum (LAGOON gives them about their mean over
  the window) and not quality-controlled.

All over plain HTTPS without an account. Times in UTC; levels in metres.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lagoonviz.core.geo.metocean import MetoceanError, Series, _utc, distance_km

Fetch = Callable[[str], bytes]
Progress = Callable[[float, str], None]
#: how answers are fetched when a source is not given one (tests put a stand-in)
FETCH: Fetch | None = None

KARTVERKET = "https://vannstand.kartverket.no/tideapi.php"
NOAA_DATA = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_STATIONS = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
IOC = "https://www.ioc-sealevelmonitoring.org/service.php"
#: days asked for in one request (the services limit a request to a month or so)
CHUNK_DAYS = 30

ATTRIBUTION = {
    "Kartverket": "Kartverket (NLOD 2.0)",
    "NOAA": "NOAA CO-OPS (public domain)",
    "IOC": "IOC Sea Level Station Monitoring Facility (VLIZ/UNESCO-IOC)",
}


@dataclass(frozen=True)
class Station:
    """A tide gauge: who runs it (``source``), its code, name and place, and how far
    it is from where it was looked for [km]."""

    source: str
    code: str
    name: str
    lat: float
    lon: float
    distance: float = 0.0

    def describe(self) -> str:
        return f"{self.name} ({self.source} {self.code}, {self.distance:.0f} km)"


def _fetch(fetch: Fetch | None):
    if fetch is not None:
        return fetch
    if FETCH is not None:
        return FETCH
    from lagoonviz.core.geo.tiles import http_fetch

    return lambda address: http_fetch(address, timeout=120.0)


def _days(start: dt.datetime, end: dt.datetime, days: int = CHUNK_DAYS):
    """The window in pieces of at most ``days`` days."""
    a = start
    while a < end:
        b = min(end, a + dt.timedelta(days=days))
        yield a, b
        a = b


# ======================================================================== sources
class Kartverket:
    name = "Kartverket"

    def __init__(self, fetch: Fetch | None = None) -> None:
        self.fetch = _fetch(fetch)
        self._stations: list[Station] | None = None

    def _xml(self, **params) -> ET.Element:
        address = KARTVERKET + "?" + urllib.parse.urlencode(params, safe=":,")
        text = self.fetch(address).decode("utf-8", errors="replace")
        try:
            return ET.fromstring(text)
        except ET.ParseError:
            raise MetoceanError(f"Kartverket did not answer {address}: {text[:200]}") from None

    def stations(self) -> list[Station]:
        if self._stations is None:
            root = self._xml(tide_request="stationlist", lang="en")
            self._stations = [
                Station(
                    self.name,
                    s.get("code", ""),
                    s.get("name", ""),
                    float(s.get("latitude", "nan")),
                    float(s.get("longitude", "nan")),
                )
                for s in root.iter("station")
                if s.get("code")
            ]
        return self._stations

    def levels(self, station: Station, start, end, progress: Progress | None = None) -> Series:
        start, end = _utc(start), _utc(end)
        found: dict[str, dict[float, float]] = {"observed": {}, "predicted": {}}
        pieces = list(_days(start, end))
        for number, (a, b) in enumerate(pieces):
            if progress is not None:
                progress(number / len(pieces), f"Kartverket {station.code} {a:%Y-%m-%d}")
            root = self._xml(
                tide_request="stationdata",
                stationcode=station.code,
                fromtime=f"{a:%Y-%m-%dT%H:%M}",
                totime=f"{b:%Y-%m-%dT%H:%M}",
                datatype="all",
                refcode="msl",
                interval=10,
                tzone=0,
                dst=0,
                lang="en",
            )
            for data in root.iter("data"):
                kind = (data.get("type") or "").lower()
                scale = 0.01 if (data.get("unit") or "cm").lower() == "cm" else 1.0
                for level in data.iter("waterlevel"):
                    flag = (level.get("flag") or kind).lower()
                    what = (
                        "observed"
                        if flag.startswith("obs")
                        else "predicted"
                        if flag.startswith("pre")
                        else None
                    )
                    if what is None or "forecast" in kind:
                        continue
                    when = _utc(level.get("time", "")).timestamp()
                    found[what][when] = float(level.get("value", "nan")) * scale
        return _series(found, station, "relative to mean sea level (Kartverket, refcode msl)")


class Noaa:
    name = "NOAA"

    def __init__(self, fetch: Fetch | None = None) -> None:
        self.fetch = _fetch(fetch)
        self._stations: list[Station] | None = None

    def _json(self, address: str) -> dict:
        text = self.fetch(address).decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise MetoceanError(f"NOAA did not answer {address}: {text[:200]}") from None

    def stations(self) -> list[Station]:
        if self._stations is None:
            answer = self._json(NOAA_STATIONS + "?type=waterlevels")
            self._stations = [
                Station(
                    self.name,
                    str(s["id"]),
                    str(s.get("name", "")),
                    float(s["lat"]),
                    float(s["lng"]),
                )
                for s in answer.get("stations", [])
                if s.get("id") is not None and s.get("lat") is not None
            ]
        return self._stations

    def levels(self, station: Station, start, end, progress: Progress | None = None) -> Series:
        start, end = _utc(start), _utc(end)
        found: dict[str, dict[float, float]] = {"observed": {}, "predicted": {}}
        pieces = list(_days(start, end))
        for number, (a, b) in enumerate(pieces):
            if progress is not None:
                progress(number / len(pieces), f"NOAA {station.code} {a:%Y-%m-%d}")
            for product, what, key in (
                ("water_level", "observed", "data"),
                ("predictions", "predicted", "predictions"),
            ):
                params = {
                    "product": product,
                    "application": "LAGOON",
                    "station": station.code,
                    "begin_date": f"{a:%Y%m%d %H:%M}",
                    "end_date": f"{b:%Y%m%d %H:%M}",
                    "datum": "MSL",
                    "units": "metric",
                    "time_zone": "gmt",
                    "format": "json",
                }
                if product == "predictions":
                    params["interval"] = "6"
                answer = self._json(NOAA_DATA + "?" + urllib.parse.urlencode(params))
                for row in answer.get(key, []) or []:
                    value = row.get("v")
                    if value in (None, ""):
                        continue
                    when = dt.datetime.strptime(row["t"], "%Y-%m-%d %H:%M").replace(tzinfo=dt.UTC)
                    found[what][when.timestamp()] = float(value)
        return _series(found, station, "relative to MSL (NOAA CO-OPS)")


class Ioc:
    name = "IOC"

    def __init__(self, fetch: Fetch | None = None) -> None:
        self.fetch = _fetch(fetch)
        self._stations: list[Station] | None = None

    def _json(self, **params):
        address = IOC + "?" + urllib.parse.urlencode(params, safe=":")
        text = self.fetch(address).decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise MetoceanError(
                f"The IOC service did not answer {address}: {text[:200]} (it may now need "
                "a registration at ioc-sealevelmonitoring.org)"
            ) from None

    def stations(self) -> list[Station]:
        if self._stations is None:
            answer = self._json(query="stationlist", showall="a")
            found = []
            for s in answer if isinstance(answer, list) else []:
                try:
                    lat, lon = float(s["Lat"]), float(s["Lon"])
                except (KeyError, TypeError, ValueError):
                    continue
                name = str(s.get("Location") or s.get("Code"))
                country = s.get("countryname") or s.get("country")
                found.append(
                    Station(
                        self.name,
                        str(s["Code"]),
                        f"{name}, {country}" if country else name,
                        lat,
                        lon,
                    )
                )
            self._stations = found
        return self._stations

    def levels(self, station: Station, start, end, progress: Progress | None = None) -> Series:
        start, end = _utc(start), _utc(end)
        by_sensor: dict[str, dict[float, float]] = {}
        pieces = list(_days(start, end))
        for number, (a, b) in enumerate(pieces):
            if progress is not None:
                progress(number / len(pieces), f"IOC {station.code} {a:%Y-%m-%d}")
            rows = self._json(
                query="data",
                code=station.code,
                timestart=f"{a:%Y-%m-%dT%H:%M}",
                timestop=f"{b:%Y-%m-%dT%H:%M}",
                format="json",
            )
            for row in rows if isinstance(rows, list) else []:
                try:
                    value = float(row["slevel"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(value) or abs(value) >= 999:
                    continue
                stamp = str(row.get("stime", "")).strip()
                when = dt.datetime.strptime(stamp[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.UTC)
                by_sensor.setdefault(str(row.get("sensor", "")), {})[when.timestamp()] = value
        if not by_sensor:
            raise MetoceanError(f"IOC: no observations at {station.name} in this window")
        sensor = max(by_sensor, key=lambda s: len(by_sensor[s]))  # the fullest sensor
        values = by_sensor[sensor]
        mean = float(np.mean(list(values.values())))
        observed = {t: v - mean for t, v in values.items()}
        return _series(
            {"observed": observed, "predicted": {}},
            station,
            f"IOC sensor {sensor}, raw, about its mean over the window ({mean:.3f} m): no datum",
        )


SOURCES: dict[str, type] = {"Kartverket": Kartverket, "NOAA": Noaa, "IOC": Ioc}


def _series(found: dict[str, dict[float, float]], station: Station, datum: str) -> Series:
    times = sorted(set(found["observed"]) | set(found["predicted"]))
    if not times:
        raise MetoceanError(f"{station.source}: no water levels at {station.name} in this window")
    time = np.array(times, np.float64)
    columns, units = {}, {}
    for what in ("observed", "predicted"):
        if found[what]:
            columns[what] = np.array([found[what].get(t, np.nan) for t in times])
            units[what] = "m"
    if "observed" in columns and "predicted" in columns:
        columns["surge"] = columns["observed"] - columns["predicted"]
        units["surge"] = "m"
    return Series(time, columns, units, None, f"{station.describe()}; {datum}")


# ======================================================================== finding
def nearest(
    lat: float,
    lon: float,
    sources=("Kartverket", "NOAA", "IOC"),
    count: int = 8,
    radius_km: float = 300.0,
    fetch: Fetch | None = None,
    made: dict | None = None,
) -> list[Station]:
    """The ``count`` gauges nearest to (lat, lon) within ``radius_km`` among the
    sources' station lists (a source that cannot be reached is left out); ``made``:
    source objects to use (their station lists are kept)."""
    found: list[Station] = []
    problems = []
    for name in sources:
        source = (made or {}).get(name) or SOURCES[name](fetch)
        if made is not None:
            made[name] = source
        try:
            stations = source.stations()
        except (OSError, MetoceanError) as error:
            problems.append(f"{name}: {error}")
            continue
        for s in stations:
            d = float(distance_km(lat, lon, s.lat, s.lon))
            if d <= radius_km:
                found.append(Station(s.source, s.code, s.name, s.lat, s.lon, d))
    found.sort(key=lambda s: s.distance)
    if not found and problems:
        raise MetoceanError("; ".join(problems))
    return found[:count]


def levels(
    station: Station, start, end, fetch: Fetch | None = None, progress: Progress | None = None
) -> Series:
    """The water levels measured (and the tide predicted, where the source has it) at
    a gauge from ``start`` to ``end`` (UTC)."""
    return SOURCES[station.source](fetch).levels(station, start, end, progress)
