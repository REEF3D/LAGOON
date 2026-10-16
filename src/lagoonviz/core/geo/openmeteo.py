# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The sea and rivers anywhere on Earth from Open-Meteo, for the set-up of a case.

- **Waves**: ERA5-Ocean (ECMWF's reanalysis, 0.5°, hourly, 1940 - a few days ago)
  from Open-Meteo's marine API: significant wave height, peak period (the mean
  period when the peak period is not served) and the direction the waves come from;
  the wind at 10 m from ERA5 (Open-Meteo's archive API).
- **Water level and current**: the sea level above mean sea level with the tide
  and the surge, and the surface current, from Open-Meteo's marine API (Météo-France's
  SMOC, about 8 km, hourly, from 2022).
- **River discharge**: GloFAS (the Copernicus flood service, 5 km, daily, from 1984)
  from Open-Meteo's flood API.

Plain HTTPS with JSON answers, no account. Open-Meteo is free for non-commercial use;
commercial use needs an API key of theirs (:data:`API_KEY`, the ``customer-``
servers). Attribution: Open-Meteo.com (CC BY 4.0), with ERA5 and GloFAS by
Copernicus/ECMWF and the currents and tides by Météo-France / Copernicus Marine.

Directions as everywhere in LAGOON's sea: waves and wind where they come from,
currents where they go, clockwise from north; :mod:`lagoonviz.core.sea` turns them
into the model's frame.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

import numpy as np

from lagoonviz.core.geo.metocean import (
    GridPoint,
    MetoceanError,
    Ocean,
    Series,
    _utc,
    distance_km,
)

ATTRIBUTION = (
    "Open-Meteo.com (CC BY 4.0): ERA5 and GloFAS by Copernicus/ECMWF, sea level and "
    "currents by Météo-France / Copernicus Marine"
)
#: an Open-Meteo API key for commercial use (empty: the free servers)
API_KEY = ""

Fetch = Callable[[str], bytes]
Progress = Callable[[float, str], None]
#: how answers are fetched when an :class:`OpenMeteo` is not given one (tests)
FETCH: Fetch | None = None

#: ERA5-Ocean's waves start in 1940; the sea level and currents (SMOC) in 2022;
#: GloFAS's discharge in 1984
ERA5_START = dt.datetime(1940, 1, 1, tzinfo=dt.UTC)
OCEAN_START = dt.datetime(2022, 1, 1, tzinfo=dt.UTC)
GLOFAS_START = dt.datetime(1984, 1, 1, tzinfo=dt.UTC)
#: the sea level's and currents' resolution [m]: points along a side further apart
#: than this are asked for one by one, nearer ones interpolated
OCEAN_SPACING = 4000.0
#: places asked for in one request (Open-Meteo answers a list)
BATCH = 50

#: m/s per unit Open-Meteo may give a speed in
SPEED_UNITS = {"m/s": 1.0, "km/h": 1.0 / 3.6, "kn": 0.514444, "knots": 0.514444, "mp/h": 0.44704}


def _server(api: str) -> str:
    host = {"marine": "marine-api", "archive": "archive-api", "flood": "flood-api"}[api]
    path = {"marine": "marine", "archive": "archive", "flood": "flood"}[api]
    prefix = "customer-" if API_KEY else ""
    return f"https://{prefix}{host}.open-meteo.com/v1/{path}"


def url(api: str, **params) -> str:
    """The URL of a request to one of Open-Meteo's APIs (``marine``, ``archive``,
    ``flood``); lists become comma-separated values."""
    query = {
        k: ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v
        for k, v in params.items()
        if v is not None
    }
    if API_KEY:
        query["apikey"] = API_KEY
    return _server(api) + "?" + urllib.parse.urlencode(query, safe=",")


def _http(address: str) -> bytes:
    """GET; Open-Meteo's refusals (HTTP 400 with a JSON reason) come back as text."""
    from lagoonviz.core.geo.tiles import _user_agent

    request = urllib.request.Request(address, headers={"User-Agent": _user_agent()})
    try:
        with urllib.request.urlopen(request, timeout=120.0) as answer:  # noqa: S310
            return answer.read()
    except urllib.error.HTTPError as error:
        body = error.read()
        if body.lstrip().startswith(b"{"):
            return body
        raise OSError(f"Open-Meteo: HTTP {error.code} for {address}") from None


class OpenMeteo:
    """Asks Open-Meteo's APIs, one request at a time."""

    def __init__(self, fetch: Fetch | None = None) -> None:
        self.fetch = fetch or FETCH or _http

    def get(self, api: str, **params) -> list[dict]:
        """The answer for one place or several (always a list)."""
        address = url(api, **params)
        data = self.fetch(address)
        try:
            answer = json.loads(data.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            raise MetoceanError(f"Open-Meteo did not answer {address}") from None
        if isinstance(answer, dict) and answer.get("error"):
            raise MetoceanError(f"Open-Meteo: {answer.get('reason', 'refused')}")
        return answer if isinstance(answer, list) else [answer]

    # ------------------------------------------------------------- waves
    def waves(self, lat: float, lon: float, start, end, progress: Progress | None = None) -> Series:
        """Hourly Hs, Tp, where the waves come from, the wind speed and where it
        comes from at the ERA5 cell nearest to (lat, lon) in the sea, ``start`` to
        ``end`` (UTC), asked a year at a time."""
        start, end = _utc(start), _utc(end)
        if end < ERA5_START:
            raise MetoceanError("ERA5 starts in 1940")
        start = max(start, ERA5_START)
        years = list(range(start.year, end.year + 1))
        times, cols, point, period = [], {}, None, "wave_peak_period"
        for number, year in enumerate(years):
            if progress is not None:
                progress(number / len(years), f"ERA5 waves {year} (Open-Meteo)")
            a = max(start, dt.datetime(year, 1, 1, tzinfo=dt.UTC))
            b = min(end, dt.datetime(year, 12, 31, 23, tzinfo=dt.UTC))
            try:
                hourly, answer, period = self._waves(lat, lon, a, b, period)
            except MetoceanError:
                if times:
                    break  # ERA5 ends here (a few days behind)
                raise
            wind = self._wind(lat, lon, a, b)
            if point is None:
                point = _point(answer, lat, lon)
            t = np.asarray(hourly["time"], np.float64)
            times.append(t)
            for name, values in (
                ("Hs", hourly.get("wave_height")),
                ("Tp", hourly.get(period)),
                ("dir_from", hourly.get("wave_direction")),
                ("wind", _on(wind, "wind_speed_10m", t)),
                ("wind_from", _on(wind, "wind_direction_10m", t)),
            ):
                cols.setdefault(name, []).append(_floats(values, len(t)))
        if progress is not None:
            progress(1.0, "ERA5 waves")
        if not times:
            raise MetoceanError(f"no ERA5 waves between {start:%Y-%m-%d} and {end:%Y-%m-%d}")
        series = Series(
            np.concatenate(times),
            {k: np.concatenate(v) for k, v in cols.items()},
            {"Hs": "m", "Tp": "s", "dir_from": "deg", "wind": "m/s", "wind_from": "deg"},
            point,
            "ERA5" if period == "wave_peak_period" else "ERA5 (Tp: the mean period)",
        )
        keep = np.isfinite(series.columns["Hs"])
        if not keep.any():
            raise MetoceanError(
                f"ERA5 has no waves at ({lat:.4f}, {lon:.4f}): on land in its 0.5° grid"
            )
        return series.window(start.timestamp(), end.timestamp())

    def _waves(self, lat, lon, a, b, period):
        names = ["wave_height", period, "wave_direction"]
        try:
            answer = self.get(
                "marine",
                latitude=round(lat, 5),
                longitude=round(lon, 5),
                hourly=names,
                models="era5_ocean",
                start_date=f"{a:%Y-%m-%d}",
                end_date=f"{b:%Y-%m-%d}",
                timeformat="unixtime",
                timezone="GMT",
                cell_selection="sea",
            )[0]
        except MetoceanError as error:
            if period == "wave_peak_period" and "wave_peak_period" in str(error):
                return self._waves(lat, lon, a, b, "wave_period")  # not served: Tm
            raise
        hourly = answer.get("hourly") or {}
        if period == "wave_peak_period" and _empty(hourly.get(period)):
            return self._waves(lat, lon, a, b, "wave_period")
        return hourly, answer, period

    def _wind(self, lat, lon, a, b) -> dict:
        try:
            answer = self.get(
                "archive",
                latitude=round(lat, 5),
                longitude=round(lon, 5),
                hourly=["wind_speed_10m", "wind_direction_10m"],
                models="era5",
                wind_speed_unit="ms",
                start_date=f"{a:%Y-%m-%d}",
                end_date=f"{b:%Y-%m-%d}",
                timeformat="unixtime",
                timezone="GMT",
            )[0]
        except MetoceanError:
            return {}  # the waves without the wind
        return answer.get("hourly") or {}

    # ------------------------------------------------------------- sea level, currents
    def ocean(self, lats, lons, start, end, progress: Progress | None = None) -> Ocean:
        """The sea level (tide and surge, above mean sea level) and the surface current
        at places, ``start`` to ``end`` (UTC), as an :class:`Ocean` of one row of
        points: ``zeta``, ``u``, ``v`` (t, 1, n) east- and northward; ``h`` unknown
        (NaN: the case's own bed gives the depth)."""
        start, end = _utc(start), _utc(end)
        if end < OCEAN_START:
            raise MetoceanError(
                "Open-Meteo's sea level and currents start in 2022: no water level or current "
                f"for {start:%Y-%m-%d}"
            )
        start = max(start, OCEAN_START)
        lats, lons = np.ravel(np.asarray(lats, float)), np.ravel(np.asarray(lons, float))
        answers: list[dict] = []
        batches = math.ceil(len(lats) / BATCH)
        for k in range(batches):
            if progress is not None:
                progress(k / batches, "Sea level and currents (Open-Meteo)")
            part = slice(k * BATCH, (k + 1) * BATCH)
            answers += self.get(
                "marine",
                latitude=[round(v, 5) for v in lats[part]],
                longitude=[round(v, 5) for v in lons[part]],
                hourly=[
                    "sea_level_height_msl",
                    "ocean_current_velocity",
                    "ocean_current_direction",
                ],
                start_date=f"{start:%Y-%m-%d}",
                end_date=f"{end:%Y-%m-%d}",
                timeformat="unixtime",
                timezone="GMT",
                cell_selection="sea",
            )
        if progress is not None:
            progress(1.0, "Sea level and currents")
        if len(answers) != len(lats):
            raise MetoceanError("Open-Meteo answered for other places than asked")
        time = np.asarray(answers[0]["hourly"]["time"], np.float64)
        keep = (time >= start.timestamp()) & (time <= end.timestamp())
        n, nt = len(answers), int(keep.sum())
        zeta, u, v = (np.full((nt, 1, n), np.nan) for _ in range(3))
        cell_lat, cell_lon = np.full((1, n), np.nan), np.full((1, n), np.nan)
        for k, answer in enumerate(answers):
            hourly = answer.get("hourly") or {}
            count = len(time)
            level = _floats(hourly.get("sea_level_height_msl"), count)[keep]
            speed = _floats(hourly.get("ocean_current_velocity"), count)[keep]
            speed = speed * SPEED_UNITS.get(
                str((answer.get("hourly_units") or {}).get("ocean_current_velocity", "km/h")), 1.0
            )
            going = np.radians(_floats(hourly.get("ocean_current_direction"), count)[keep])
            zeta[:, 0, k] = level
            u[:, 0, k] = speed * np.sin(going)
            v[:, 0, k] = speed * np.cos(going)
            cell_lat[0, k] = answer.get("latitude", lats[k])
            cell_lon[0, k] = answer.get("longitude", lons[k])
        if not np.isfinite(zeta).any():
            raise MetoceanError("Open-Meteo has no sea level here (on land in its grid?)")
        return Ocean(
            time[keep],
            zeta,
            u,
            v,
            np.full((1, n), np.nan),
            cell_lat,
            cell_lon,
            "Open-Meteo (Météo-France SMOC)",
        )

    def sea(self, lat: float, lon: float, start, end) -> Series:
        """The sea level and the surface current at one place: ``zeta`` [m above mean
        sea level], ``current`` [m/s], ``current_to`` [deg], ``u_east``, ``v_north``."""
        ocean = self.ocean([lat], [lon], start, end)
        u, v = ocean.u[:, 0, 0], ocean.v[:, 0, 0]
        point = _point({"latitude": ocean.lat[0, 0], "longitude": ocean.lon[0, 0]}, lat, lon)
        return Series(
            ocean.time,
            {
                "zeta": ocean.zeta[:, 0, 0],
                "current": np.hypot(u, v),
                "current_to": np.degrees(np.arctan2(u, v)) % 360.0,
                "u_east": u,
                "v_north": v,
            },
            {"zeta": "m", "current": "m/s", "current_to": "deg", "u_east": "m/s", "v_north": "m/s"},
            point,
            ocean.source,
        )

    # ------------------------------------------------------------- rivers
    def discharge(self, lat: float, lon: float, start, end) -> Series:
        """GloFAS's daily river discharge [m3/s] at the river cell Open-Meteo takes
        for (lat, lon), ``start`` to ``end`` (UTC; the days that hold them)."""
        start, end = _utc(start), _utc(end)
        if end < GLOFAS_START:
            raise MetoceanError("GloFAS starts in 1984")
        start = max(start, GLOFAS_START)
        answer = self.get(
            "flood",
            latitude=round(lat, 5),
            longitude=round(lon, 5),
            daily="river_discharge",
            start_date=f"{start:%Y-%m-%d}",
            end_date=f"{end:%Y-%m-%d}",
            timeformat="unixtime",
        )[0]
        daily = answer.get("daily") or {}
        time = np.asarray(daily.get("time", []), np.float64)
        q = _floats(daily.get("river_discharge"), len(time))
        if not len(time) or not np.isfinite(q).any():
            raise MetoceanError(f"GloFAS has no river at ({lat:.4f}, {lon:.4f})")
        return Series(time, {"Q": q}, {"Q": "m3/s"}, _point(answer, lat, lon), "GloFAS")


# ======================================================================== helpers
def _floats(values, count: int) -> np.ndarray:
    """A JSON list of numbers (None: missing) as floats, NaN where missing."""
    if values is None:
        return np.full(count, np.nan)
    return np.array([np.nan if v is None else float(v) for v in values], np.float64)


def _empty(values) -> bool:
    return values is None or all(v is None for v in values)


def _on(hourly: dict, name: str, time: np.ndarray) -> np.ndarray:
    """An hourly column of another answer at the times ``time`` (NaN where it has none)."""
    if not hourly or name not in hourly:
        return np.full(len(time), np.nan)
    t = np.asarray(hourly["time"], np.float64)
    values = _floats(hourly[name], len(t))
    index = {float(s): k for k, s in enumerate(t)}
    return np.array([values[index[s]] if s in index else np.nan for s in time])


def _point(answer: dict, lat: float, lon: float) -> GridPoint:
    """The cell Open-Meteo took (its centre), as a grid point."""
    clat = float(answer.get("latitude", lat))
    clon = float(answer.get("longitude", lon))
    return GridPoint(0, 0, clat, clon, float(distance_km(lat, lon, clat, clon)))
