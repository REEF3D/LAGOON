# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""Sea conditions for the set-up of a case from MET Norway's hindcasts.

- **NORA3** (waves and wind, 3 km, 1959 - today): the wave hindcast's time series
  subset (``nora3_subset_wave/wave_tser``, monthly files, ``hs``, ``tp``, ``thq``
  wave direction, ``ff``/``dd`` wind) for the conditions and the storms, and its 2D
  spectra (``windsurfer/mywavewam3km_spectra``, daily files, 30 frequencies x 24
  directions at some 28 700 points) for the wave generation.
- **NorKyst** (ocean, 800 m): v3 (2024 - today, one aggregated dataset) and v2
  (2017-02-20 - 2025-10, daily files): water level ``zeta`` and currents
  ``u_eastward``, ``v_northward`` on depth levels.

All from ``thredds.met.no`` over OPeNDAP, asked one request at a time (a shared
public service: no parallel sessions) as text (``.ascii``), so no NetCDF library is
needed. Directions in the files are where the waves and wind go (oceanographic,
clockwise from north); LAGOON shows where they come from, and turns them into the
model's frame (counter-clockwise from the model's x-axis, as REEF3D's B 105 and its
2D spectrum file) with the case's georeference.

What it gives the set-up:

- :func:`find_storms`: the largest wave events at a place over some years;
- :func:`conditions`: waves, wind, water level and current at a place over a
  window, as tables (:func:`write_table`) the plots show;
- :func:`spectrum_file`: NORA3's 2D spectrum at a moment as REEF3D's
  ``spectrum-file-2d.dat`` (B 85 11);
- :func:`boundary_series`: NorKyst's water level and discharge along the sides of
  the domain over a window, as REEF3D's ``hydrograph_FSF_<n>.dat`` and
  ``hydrograph_Q_<n>.dat`` (patchBC, B 421 / B 422).

Data: NLOD / CC BY 4.0, Norwegian Meteorological Institute (MET Norway).
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
import re
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE = "https://thredds.met.no/thredds"
ATTRIBUTION = "MET Norway (NORA3, NorKyst), NLOD / CC BY 4.0"

Fetch = Callable[[str], bytes]
Progress = Callable[[float, str], None]
#: how answers are fetched when a :class:`Dap` is not given one (tests put a stand-in)
FETCH: Fetch | None = None

#: NORA3's wave time series subset: one file a month
NORA3_START = dt.datetime(1959, 1, 1, tzinfo=dt.UTC)
#: NorKyst v2 hourly files (daily), and v3's aggregation
NORKYST2_START = dt.datetime(2017, 2, 20, tzinfo=dt.UTC)
NORKYST2_END = dt.datetime(2025, 10, 6, tzinfo=dt.UTC)
NORKYST3_START = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
NORKYST3 = "fou-hi/norkystv3_800m_m00_be"
#: NORA3's wave variables read for the conditions
WAVE_VARIABLES = ("hs", "tp", "thq", "ff", "dd")


class MetoceanError(ValueError):
    """The data asked for is not there (a time or place outside a hindcast)."""


def nora3_path(year: int, month: int) -> str:
    return f"nora3_subset_wave/wave_tser/{year:04d}{month:02d}_NORA3wave_sub_time_unlimited.nc"


def spectra_path(day: dt.date) -> str:
    return f"windsurfer/mywavewam3km_spectra/{day:%Y}/{day:%m}/SPC{day:%Y%m%d}00.nc"


def norkyst2_path(day: dt.date) -> str:
    return f"fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.{day:%Y%m%d}00.nc"


# ======================================================================== OPeNDAP
def dap_url(path: str, suffix: str, constraint: str = "") -> str:
    """The URL of a dataset's ``.dds``, ``.das`` or ``.ascii`` answer; brackets in
    the constraint are escaped (the server refuses them as they are)."""
    url = f"{BASE}/dodsC/{path}.{suffix}"
    if constraint:
        url += "?" + urllib.parse.quote(constraint, safe=",:")
    return url


def parse_dds(text: str) -> dict[str, list[tuple[str, int]]]:
    """The arrays of a ``.dds`` answer: name -> [(dimension, size), ...] (the arrays
    of grids by their own name)."""
    found: dict[str, list[tuple[str, int]]] = {}
    for match in re.finditer(r"(?:Float|Int|UInt|Byte)\w*\s+(\w+)((?:\[\w+ = \d+\])+);", text):
        name = match.group(1)
        dims = [(d, int(n)) for d, n in re.findall(r"\[(\w+) = (\d+)\]", match.group(2))]
        found.setdefault(name, dims)  # a grid's array first, its maps after
    return found


def parse_das(text: str) -> dict[str, dict[str, float]]:
    """The packing of each variable of a ``.das`` answer: ``_FillValue``,
    ``scale_factor``, ``add_offset`` (those it has)."""
    found: dict[str, dict[str, float]] = {}
    name = None
    for line in text.splitlines():
        block = re.match(r"^    (\w+) \{\s*$", line)
        if block:
            name = block.group(1)
            found.setdefault(name, {})
            continue
        attr = re.match(r"^\s+\w+\s+(_FillValue|scale_factor|add_offset)\s+(\S+);", line)
        if attr and name is not None:
            try:
                found[name][attr.group(1)] = float(attr.group(2))
            except ValueError:
                pass
    return found


def parse_ascii(text: str) -> dict[str, np.ndarray]:
    """The arrays of an ``.ascii`` answer, by name (``hs`` for a grid's ``hs.hs``,
    ``hs.time`` for its maps), shaped as the header says."""
    body = text.split("-" * 45, 1)[-1]
    found: dict[str, np.ndarray] = {}
    for block in re.split(r"\n\s*\n", body):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        head = re.match(r"^([\w.]+)((?:\[\d+\])*)\s*$", lines[0].strip())
        if head is None:
            continue
        name = head.group(1)
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == parts[1]:
            name = parts[0]
        shape = [int(n) for n in re.findall(r"\[(\d+)\]", head.group(2))]
        values: list[str] = []
        for line in lines[1:]:
            line = re.sub(r"^(\[\d+\])+,?", "", line.strip())
            values += [v for v in line.split(",") if v.strip()]
        array = np.array([float(v) for v in values], np.float64)
        if shape and array.size == math.prod(shape):
            array = array.reshape(shape)
        elif not shape and array.size == 1:
            array = array.reshape(())
        found[name] = array
    return found


class Dap:
    """Asks MET Norway's THREDDS for parts of datasets, one request at a time; the
    structure and packing of each dataset are asked once."""

    def __init__(self, fetch: Fetch | None = None) -> None:
        if fetch is None and FETCH is not None:
            fetch = FETCH
        if fetch is None:
            from lagoonviz.core.geo.tiles import http_fetch

            def fetch(url: str) -> bytes:
                return http_fetch(url, timeout=180.0)

        self.fetch = fetch
        self._dds: dict[str, dict] = {}
        self._das: dict[str, dict] = {}

    def _text(self, url: str) -> str:
        try:
            data = self.fetch(url)
        except OSError as error:
            if "404" in str(error):
                raise MetoceanError(f"not on MET Norway's server: {url}") from None
            raise
        text = data.decode("utf-8", errors="replace")
        if text.lstrip().startswith(("<!doctype", "<html", "Error {")):
            raise MetoceanError(f"MET Norway's server did not answer {url}: {text[:200]}")
        return text

    def dims(self, path: str) -> dict[str, list[tuple[str, int]]]:
        if path not in self._dds:
            self._dds[path] = parse_dds(self._text(dap_url(path, "dds")))
        return self._dds[path]

    def packing(self, path: str) -> dict[str, dict[str, float]]:
        if path not in self._das:
            self._das[path] = parse_das(self._text(dap_url(path, "das")))
        return self._das[path]

    def get(self, path: str, constraint: str, unpack: bool = True) -> dict[str, np.ndarray]:
        """The arrays a constraint asks for, unpacked (scaled, fill values NaN)."""
        found = parse_ascii(self._text(dap_url(path, "ascii", constraint)))
        if not unpack:
            return found
        packing = self.packing(path)
        for name, values in found.items():
            info = packing.get(name, {})  # (a grid's maps, "hs.time", stay as they are)
            if not info:
                continue
            values = np.array(values, np.float64)
            fill = info.get("_FillValue")
            if fill is not None:
                values[values == fill] = np.nan
            values = values * info.get("scale_factor", 1.0) + info.get("add_offset", 0.0)
            found[name] = values
        return found


# ======================================================================== places
def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance [km]."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat, dlon = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class GridPoint:
    """The grid point of a dataset nearest to a place (in the water)."""

    i: int  # the first horizontal index of the dataset's arrays
    j: int  # the second
    lat: float
    lon: float
    #: from the place asked for [km]
    distance: float


def nearest_point(
    dap: Dap,
    path: str,
    lat: float,
    lon: float,
    names: tuple[str, str] = ("latitude", "longitude"),
    wet: str | None = None,
    stride: int = 16,
) -> GridPoint:
    """The wet grid point nearest to (lat, lon) of a curvilinear grid (2D latitude
    and longitude arrays), found coarse first (every ``stride``-th point), then in a
    window round it at full resolution; ``wet``: a 2D array that is positive in the
    water (a depth)."""
    lat_name, lon_name = names
    (_, n0), (_, n1) = dap.dims(path)[lat_name][:2]
    s = max(1, stride)
    coarse = dap.get(
        path,
        f"{lat_name}[0:{s}:{n0 - 1}][0:{s}:{n1 - 1}],{lon_name}[0:{s}:{n0 - 1}][0:{s}:{n1 - 1}]",
    )
    d = distance_km(lat, lon, coarse[lat_name], coarse[lon_name])
    a, b = np.unravel_index(np.nanargmin(d), d.shape)
    if float(np.nanmin(d)) > 50.0 * s:
        raise MetoceanError(f"({lat:.4f}, {lon:.4f}) lies outside {path.split('/')[0]}")
    reach = 2 * s
    i0, i1 = max(0, a * s - reach), min(n0 - 1, a * s + reach)
    j0, j1 = max(0, b * s - reach), min(n1 - 1, b * s + reach)
    window = f"{lat_name}[{i0}:{i1}][{j0}:{j1}],{lon_name}[{i0}:{i1}][{j0}:{j1}]"
    if wet:
        window += f",{wet}[{i0}:{i1}][{j0}:{j1}]"
    fine = dap.get(path, window)
    d = distance_km(lat, lon, fine[lat_name], fine[lon_name])
    if wet:
        depth = fine[wet]
        d = np.where(np.isfinite(depth) & (depth > 0), d, np.inf)
    if not np.isfinite(d).any():
        raise MetoceanError(f"no sea near ({lat:.4f}, {lon:.4f}) in {path.split('/')[0]}")
    a, b = np.unravel_index(np.argmin(d), d.shape)
    return GridPoint(
        int(i0 + a),
        int(j0 + b),
        float(fine[lat_name][a, b]),
        float(fine[lon_name][a, b]),
        float(d[a, b]),
    )


# ======================================================================== series
@dataclass
class Series:
    """Values over time at a place: ``time`` in seconds since 1970 (UTC)."""

    time: np.ndarray
    columns: dict[str, np.ndarray]
    units: dict[str, str] = field(default_factory=dict)
    point: GridPoint | None = None
    source: str = ""

    def window(self, start: float, end: float) -> Series:
        keep = (self.time >= start) & (self.time <= end)
        return Series(
            self.time[keep],
            {k: v[keep] for k, v in self.columns.items()},
            dict(self.units),
            self.point,
            self.source,
        )


def _utc(when) -> dt.datetime:
    if isinstance(when, (int, float)):
        return dt.datetime.fromtimestamp(float(when), dt.UTC)
    if isinstance(when, str):
        when = dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.UTC)
    return when.astimezone(dt.UTC)


def _months(start: dt.datetime, end: dt.datetime):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


class Nora3:
    """NORA3's wave time series at a place (the monthly subset files)."""

    def __init__(self, dap: Dap | None = None) -> None:
        self.dap = dap or Dap()
        self._points: dict[tuple, GridPoint] = {}

    def point(self, lat: float, lon: float, path: str | None = None) -> GridPoint:
        key = (round(lat, 5), round(lon, 5))
        if key not in self._points:
            path = path or nora3_path(2020, 1)  # the grid is the same every month
            self._points[key] = nearest_point(
                self.dap, path, lat, lon, ("latitude", "longitude"), wet="model_depth"
            )
        return self._points[key]

    def series(
        self, lat: float, lon: float, start, end, progress: Progress | None = None
    ) -> Series:
        """Hourly ``hs``, ``tp``, wave direction, wind speed and direction at the
        NORA3 point nearest to (lat, lon), from ``start`` to ``end`` (UTC)."""
        start, end = _utc(start), _utc(end)
        if end < NORA3_START:
            raise MetoceanError("NORA3 starts in 1959")
        start = max(start, NORA3_START)
        months = list(_months(start, end))
        point = self.point(lat, lon)
        times, values = [], {name: [] for name in WAVE_VARIABLES}
        for number, (year, month) in enumerate(months):
            if progress is not None:
                progress(number / len(months), f"NORA3 {year}-{month:02d}")
            path = nora3_path(year, month)
            try:
                count = dict(self.dap.dims(path)["hs"])["time"]
            except (MetoceanError, KeyError):
                if times:
                    break  # the hindcast ends here (it runs a month or two behind)
                raise
            first = dt.datetime(year, month, 1, tzinfo=dt.UTC)
            a = max(0, int((start - first).total_seconds() // 3600))
            b = min(count - 1, int((end - first).total_seconds() // 3600))
            if b < a:
                continue
            ask = ",".join(f"{v}[{point.i}][{point.j}][{a}:{b}]" for v in WAVE_VARIABLES)
            got = self.dap.get(path, ask + f",time[{a}:{b}]")
            times.append(np.asarray(got["time"], np.float64).ravel())
            for name in WAVE_VARIABLES:
                values[name].append(np.asarray(got[name], np.float64).ravel())
        if progress is not None:
            progress(1.0, "NORA3")
        if not times:
            raise MetoceanError(f"no NORA3 data between {start:%Y-%m-%d} and {end:%Y-%m-%d}")
        cols = {name: np.concatenate(parts) for name, parts in values.items()}
        columns = {
            "Hs": cols["hs"],
            "Tp": cols["tp"],
            "dir_from": (cols["thq"] + 180.0) % 360.0,
            "wind": cols["ff"],
            "wind_from": (cols["dd"] + 180.0) % 360.0,
        }
        units = {"Hs": "m", "Tp": "s", "dir_from": "deg", "wind": "m/s", "wind_from": "deg"}
        return Series(np.concatenate(times), columns, units, point, "NORA3")


@dataclass(frozen=True)
class Storm:
    """A wave event: its peak (time, Hs, Tp, where the waves come from, wind)."""

    peak: float  # seconds since 1970 (UTC)
    hs: float
    tp: float
    direction: float  # from, clockwise from north [deg]
    wind: float
    wind_from: float

    @property
    def when(self) -> dt.datetime:
        return _utc(self.peak)

    def describe(self) -> str:
        return (
            f"{self.when:%Y-%m-%d %H:00} UTC: Hs {self.hs:.1f} m, Tp {self.tp:.1f} s, "
            f"from {self.direction:.0f}°, wind {self.wind:.0f} m/s from {self.wind_from:.0f}°"
        )


def storms(series: Series, count: int = 10, apart_hours: float = 72.0) -> list[Storm]:
    """The ``count`` largest wave events of a series: the highest Hs, then the
    highest more than ``apart_hours`` from any chosen (independent storms)."""
    hs = np.asarray(series.columns["Hs"], np.float64)
    order = np.argsort(np.where(np.isfinite(hs), -hs, np.inf), kind="stable")
    chosen: list[int] = []
    for index in order:
        if not np.isfinite(hs[index]) or len(chosen) >= count:
            break
        if all(abs(series.time[index] - series.time[c]) > apart_hours * 3600 for c in chosen):
            chosen.append(int(index))
    c = series.columns
    return [
        Storm(
            float(series.time[k]),
            float(c["Hs"][k]),
            float(c["Tp"][k]),
            float(c["dir_from"][k]),
            float(c["wind"][k]),
            float(c["wind_from"][k]),
        )
        for k in chosen
    ]


def find_storms(
    lat: float,
    lon: float,
    first_year: int,
    last_year: int,
    count: int = 10,
    nora3: Nora3 | None = None,
    progress: Progress | None = None,
) -> tuple[list[Storm], GridPoint]:
    """The largest wave events at (lat, lon) from ``first_year`` to ``last_year``
    (NORA3, about a second a month of data)."""
    nora3 = nora3 or Nora3()
    start = dt.datetime(first_year, 1, 1, tzinfo=dt.UTC)
    end = min(dt.datetime(last_year, 12, 31, 23, tzinfo=dt.UTC), dt.datetime.now(dt.UTC))
    series = nora3.series(lat, lon, start, end, progress)
    return storms(series, count), series.point


# ======================================================================== NorKyst
@dataclass
class Ocean:
    """NorKyst over a window of grid points and a time window: ``zeta`` (t, a, b),
    depth-averaged ``u``, ``v`` east- and northward (t, a, b), the depth ``h`` and
    ``lat``, ``lon`` (a, b)."""

    time: np.ndarray
    zeta: np.ndarray
    u: np.ndarray
    v: np.ndarray
    h: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    source: str


class NorKyst:
    """NorKyst's water level and currents: v3 from 2024, v2 before (daily files)."""

    def __init__(self, dap: Dap | None = None) -> None:
        self.dap = dap or Dap()
        self._times: np.ndarray | None = None

    def version(self, start: dt.datetime) -> int:
        if start >= NORKYST3_START:
            return 3
        if NORKYST2_START <= start < NORKYST2_END:
            return 2
        raise MetoceanError(
            f"NorKyst covers 2017-02-20 onwards; {start:%Y-%m-%d} is before it (no currents "
            "and water levels for this event)"
        )

    def _v3_times(self) -> np.ndarray:
        if self._times is None:
            n = dict(self.dap.dims(NORKYST3)["zeta"])["time"]
            self._times = np.asarray(
                self.dap.get(NORKYST3, f"time[0:{n - 1}]", unpack=False)["time"], np.float64
            ).ravel()
        return self._times

    def box(
        self, lats, lons, start, end, margin: int = 1, progress: Progress | None = None
    ) -> Ocean:
        """NorKyst round some places (a domain's corners) from ``start`` to ``end``."""
        start, end = _utc(start), _utc(end)
        version = self.version(start)
        if version == 3:
            pieces = [(NORKYST3, None)]
        else:
            days = []
            day = start.date()
            while day <= end.date():
                days.append(day)
                day += dt.timedelta(days=1)
            pieces = [(norkyst2_path(day), day) for day in days]
        grid_path = pieces[0][0]
        corners = [
            nearest_point(self.dap, grid_path, la, lo, ("lat", "lon"), stride=32)
            for la, lo in zip(np.ravel(lats), np.ravel(lons), strict=True)
        ]
        (_, na), (_, nb) = self.dap.dims(grid_path)["lat"][:2]
        a0 = max(0, min(p.i for p in corners) - margin)
        a1 = min(na - 1, max(p.i for p in corners) + margin)
        b0 = max(0, min(p.j for p in corners) - margin)
        b1 = min(nb - 1, max(p.j for p in corners) + margin)
        space = f"[{a0}:{a1}][{b0}:{b1}]"
        static = self.dap.get(grid_path, f"h{space},lat{space},lon{space}")
        h = np.asarray(static["h"], np.float64)
        depths = np.asarray(self.dap.get(grid_path, "depth")["depth"], np.float64).ravel()
        deepest = float(np.nanmax(h)) if np.isfinite(h).any() else 0.0
        k1 = int(min(len(depths) - 1, np.searchsorted(depths, deepest) + 1))
        times, zetas, us, vs = [], [], [], []
        for number, (path, day) in enumerate(pieces):
            if progress is not None:
                progress(number / len(pieces), f"NorKyst v{version} {path.split('/')[-1]}")
            if version == 3:
                t = self._v3_times()
                t0 = int(np.searchsorted(t, start.timestamp()))
                t1 = int(np.searchsorted(t, end.timestamp(), side="right")) - 1
                if t1 < t0:
                    raise MetoceanError("NorKyst v3 has no output in this window")
            else:
                first = dt.datetime.combine(day, dt.time(), dt.UTC)
                t0 = max(0, int((start - first).total_seconds() // 3600))
                t1 = min(23, int((end - first).total_seconds() // 3600))
                if t1 < t0:
                    continue
            span = f"[{t0}:{t1}]"
            got = self.dap.get(
                path,
                f"zeta{span}{space},u_eastward{span}[0:{k1}]{space},"
                f"v_northward{span}[0:{k1}]{space},time{span}",
            )
            times.append(np.asarray(got["time"], np.float64).ravel())
            zetas.append(np.asarray(got["zeta"], np.float64).reshape(-1, *h.shape))
            u = np.asarray(got["u_eastward"], np.float64).reshape(-1, k1 + 1, *h.shape)
            v = np.asarray(got["v_northward"], np.float64).reshape(-1, k1 + 1, *h.shape)
            us.append(depth_mean(u, depths[: k1 + 1], h))
            vs.append(depth_mean(v, depths[: k1 + 1], h))
        if progress is not None:
            progress(1.0, "NorKyst")
        if not times:
            raise MetoceanError("NorKyst has no output in this window")
        h = np.where(h > 0, h, np.nan)
        return Ocean(
            np.concatenate(times),
            np.concatenate(zetas),
            np.concatenate(us),
            np.concatenate(vs),
            h,
            np.asarray(static["lat"], np.float64),
            np.asarray(static["lon"], np.float64),
            f"NorKyst v{version}",
        )


def depth_mean(values: np.ndarray, depths: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The mean over depth (t, k, a, b) -> (t, a, b) of values on depth levels down to
    the bed ``h`` (a, b): each level stands for the water half-way to its neighbours,
    the deepest down to the bed; levels below the bed (NaN) left out."""
    depths = np.asarray(depths, np.float64)
    edges = np.concatenate([[0.0], 0.5 * (depths[1:] + depths[:-1]), [np.inf]])
    top = edges[:-1][None, :, None, None]
    bottom = np.minimum(edges[1:][None, :, None, None], np.asarray(h)[None, None])
    weight = np.clip(bottom - top, 0.0, None) * np.isfinite(values)
    total = weight.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (np.nan_to_num(values) * weight).sum(axis=1) / total, np.nan)


# ======================================================================== spectra
@dataclass
class Spectrum:
    """A 2D wave spectrum at a moment: ``energy`` (f, theta) [m^2 / (Hz rad)] over
    ``freq`` [Hz] and ``direction`` [deg, where the waves go, clockwise from north]."""

    time: float
    freq: np.ndarray
    direction: np.ndarray
    energy: np.ndarray
    lat: float
    lon: float

    @property
    def widths(self) -> np.ndarray:
        """The frequency bins [Hz]: half-way to the neighbours."""
        f = self.freq
        edges = np.concatenate([[f[0] - 0.5 * (f[1] - f[0])], 0.5 * (f[1:] + f[:-1])])
        edges = np.append(edges, f[-1] + 0.5 * (f[-1] - f[-2]))
        return np.diff(edges)

    @property
    def dtheta(self) -> float:
        return math.radians(360.0 / len(self.direction))

    @property
    def hs(self) -> float:
        m0 = float(np.nansum(self.energy * self.widths[:, None]) * self.dtheta)
        return 4.0 * math.sqrt(max(m0, 0.0))

    @property
    def tp(self) -> float:
        per_f = np.nansum(self.energy, axis=1)
        return 1.0 / float(self.freq[int(np.argmax(per_f))])

    @property
    def mean_to(self) -> float:
        """Mean direction the waves go [deg, clockwise from north]."""
        rad = np.radians(self.direction)
        e = np.nansum(self.energy * self.widths[:, None], axis=0)
        return float(np.degrees(np.arctan2((e * np.sin(rad)).sum(), (e * np.cos(rad)).sum())) % 360)


def nora3_spectrum(when, lat: float, lon: float, dap: Dap | None = None) -> Spectrum:
    """NORA3's 2D spectrum at the spectrum point nearest to (lat, lon), at the hour
    ``when`` (UTC)."""
    dap = dap or Dap()
    when = _utc(when)
    path = spectra_path(when.date())
    n = dict(dap.dims(path)["latitude"])["x"]
    got = dap.get(path, f"latitude[0][0:{n - 1}],longitude[0][0:{n - 1}],freq,direction")
    lats, lons = np.ravel(got["latitude"]), np.ravel(got["longitude"])
    d = distance_km(lat, lon, lats, lons)
    p = int(np.argmin(d))
    if d[p] > 30.0:
        raise MetoceanError(f"no NORA3 spectrum point within 30 km of ({lat:.4f}, {lon:.4f})")
    hour = when.hour
    nf, nd = len(np.ravel(got["freq"])), len(np.ravel(got["direction"]))
    energy = dap.get(path, f"SPEC[{hour}][0][{p}][0:{nf - 1}][0:{nd - 1}]")["SPEC"]
    energy = np.asarray(energy, np.float64).reshape(nf, nd)
    energy = np.where(energy < 0, np.nan, energy)
    day = dt.datetime.combine(when.date(), dt.time(), dt.UTC)
    return Spectrum(
        (day + dt.timedelta(hours=hour)).timestamp(),
        np.ravel(got["freq"]).astype(np.float64),
        np.ravel(got["direction"]).astype(np.float64),
        energy,
        float(lats[p]),
        float(lons[p]),
    )


def model_angle(bearing_to, north_deg: float, rotation_deg: float):
    """A direction where something goes, clockwise from north [deg], in the model's
    frame: counter-clockwise from the model's x-axis [deg]. ``north_deg``: the
    direction of north in the CRS, counter-clockwise from its x-axis (90 without
    grid convergence); ``rotation_deg``: the georeference's rotation."""
    return (north_deg - np.asarray(bearing_to, np.float64) - rotation_deg) % 360.0


def spectrum_file(
    spectrum: Spectrum,
    path: str | Path,
    north_deg: float = 90.0,
    rotation_deg: float = 0.0,
    half_width: float = 90.0,
) -> dict[str, float]:
    """Write ``spectrum-file-2d.dat`` for REEF3D (B 85 11): a header ``fq_dir`` and
    the directions [rad, where the waves go, counter-clockwise from the model's
    x-axis], then per angular frequency [rad/s] the spectral values [m^2 s/rad^2].

    Only the directions within ``half_width`` degrees of the mean go in (the waves
    a boundary can make), unwrapped and rising. REEF3D makes one wave per
    (omega, theta) with amplitude sqrt(2 S domega dtheta), domega the forward
    difference (the last repeated): S is written so each carries the energy of its
    NORA3 bin. Returns the facts for the wave generation (Hs, Tp, the direction,
    Hs of the sector written)."""
    theta = model_angle(spectrum.direction, north_deg, rotation_deg)
    mean = float(model_angle(spectrum.mean_to, north_deg, rotation_deg))
    offset = (theta - mean + 180.0) % 360.0 - 180.0
    keep = np.abs(offset) <= half_width + 1e-9
    order = np.argsort(offset[keep])
    angles = np.radians(mean + offset[keep][order])
    energy = np.nan_to_num(spectrum.energy[:, keep][:, order])
    omega = 2.0 * math.pi * spectrum.freq
    dw = np.append(np.diff(omega), np.diff(omega)[-1])
    values = energy * spectrum.widths[:, None] / dw[:, None]  # E df dtheta = S dw dtheta
    lines = ["fq_dir " + " ".join(f"{a:.10g}" for a in angles)]
    for w, row in zip(omega, values, strict=True):
        lines.append(f"{w:.10g} " + " ".join(f"{v:.10g}" for v in row))
    Path(path).write_text("\n".join(lines) + "\n")
    sector = float(np.sum(energy * spectrum.widths[:, None]) * spectrum.dtheta)
    return {
        "Hs": spectrum.hs,
        "Tp": spectrum.tp,
        "direction": mean,
        "Hs_sector": 4.0 * math.sqrt(max(sector, 0.0)),
        "directions": int(keep.sum()),
    }


# ======================================================================== tables
def write_table(
    path: str | Path,
    start: float,
    series: dict[str, np.ndarray],
    time: np.ndarray,
    units: dict[str, str],
    note: str,
) -> Path:
    """A time-series table the plots read (as REEF3D's): a preamble with where and
    when, a header ``time`` and the columns, the time in seconds from ``start``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(series)
    head = ["time [s]"] + [f"{n} [{units[n]}]" if n in units else n for n in names]
    lines = [note, f"t = 0 s at {_utc(start):%Y-%m-%d %H:%M} UTC", "", "\t".join(head)]
    for k, t in enumerate(time):
        row = [f"{t - start:.0f}"] + [
            "-1e+20" if not np.isfinite(series[n][k]) else f"{series[n][k]:.6g}" for n in names
        ]
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def two_column(path: str | Path, time: np.ndarray, values: np.ndarray) -> Path:
    """A REEF3D input time series: time [s] and value, one line each."""
    path = Path(path)
    rows = [f"{t:.6g} {v:.6g}" for t, v in zip(time, values, strict=True) if np.isfinite(v)]
    path.write_text("\n".join(rows) + "\n")
    return path


def hours_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1] * 24
