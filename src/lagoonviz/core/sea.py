# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The sea at a case, for its set-up: the storms there, the conditions over a window
as plots, the waves to generate, the water level and discharge along the sides of
the domain as REEF3D's patchBC hydrographs, a river's discharge, and the water
levels measured at tide gauges nearby.

Two sources of the sea (``source``): **MET Norway** (:mod:`lagoonviz.core.geo.metocean`:
NORA3 waves with 2D spectra, NorKyst water level and currents; Norwegian waters) and
**Open-Meteo** (:mod:`lagoonviz.core.geo.openmeteo`: ERA5 waves, Météo-France's sea
level with the tide and currents; anywhere). A river's discharge from GloFAS
(Open-Meteo), measured water levels from Kartverket, NOAA or IOC
(:mod:`lagoonviz.core.geo.gauges`).

Where: the middle of the wave generation side (C 1x 6) when the case has one, else
the centre of the domain, both through the case's georeference (needed). Tables
go into ``LAGOON_Sea`` in the case folder, where they open as time series; the
files REEF3D reads go into the case folder itself. Time 0 of every file written is
the start of the window: the start of the run.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lagoonviz.core.geo import gauges as gg
from lagoonviz.core.geo import metocean as mo
from lagoonviz.core.geo import openmeteo as om
from lagoonviz.core.geo.crs import transformer

#: the folder of a case the sea's tables go into
FOLDER = "LAGOON_Sea"
#: the sides of the domain (REEF3D's faces: C 11 - C 14, B 440): name, inward normal
SIDES = {
    1: ("x min", (1.0, 0.0)),
    4: ("x max", (-1.0, 0.0)),
    3: ("y min", (0.0, 1.0)),
    2: ("y max", (0.0, -1.0)),
}
#: sample points along a side: one every this many metres (five at least)
SIDE_STEP = 100.0
#: a point of a side further than this from NorKyst's water [km] is on land
DRY_KM = 1.2
#: the sources of the sea
MET_NORWAY, OPEN_METEO = "MET Norway", "Open-Meteo"
#: Open-Meteo's ERA5 waves with CERRA's wind (5.5 km, Europe, 1985 - mid 2021)
CERRA = "Open-Meteo, CERRA wind"
SOURCES = (MET_NORWAY, OPEN_METEO, CERRA)
#: the wind of Open-Meteo's sources
WIND = {OPEN_METEO: "era5", CERRA: "cerra"}
#: where MET Norway's hindcasts reach (NORA3: the Nordic seas, the North Sea and the
#: Barents Sea; NorKyst: the Norwegian coast): lat, lon box
MET_BOX = (54.0, 82.0, -12.0, 45.0)


@dataclass
class Place:
    """Where the sea is read for a case."""

    lat: float
    lon: float
    label: str
    #: the model point (x, y)
    x: float
    y: float


@dataclass
class Frame:
    """A case's georeference, as the sea needs it."""

    georef: object
    #: north in the CRS: degrees counter-clockwise from its x-axis
    north_deg: float
    domain: tuple[float, float, float, float]

    @property
    def rotation_deg(self) -> float:
        return float(self.georef.rotation_deg)

    def lonlat(self, x, y):
        """Longitude and latitude of model points."""
        wx, wy = self.georef.to_world(np.asarray(x, float), np.asarray(y, float))
        return transformer(self.georef.crs, "EPSG:4326")(wx, wy)

    def to_model(self, east, north):
        """An east, north vector into the model's frame (x, y)."""
        a_e = math.radians(self.north_deg - 90.0 - self.rotation_deg)
        a_n = math.radians(self.north_deg - self.rotation_deg)
        east, north = np.asarray(east), np.asarray(north)
        return (
            east * math.cos(a_e) + north * math.cos(a_n),
            east * math.sin(a_e) + north * math.sin(a_n),
        )


def frame(session, case_id: str) -> Frame:
    georef = session.georef(case_id)
    if georef is None:
        raise ValueError(
            "the case has no georeference: set it in the Map tab first (the sea is read "
            "where the domain lies)"
        )
    domain = session.data.backgrounds.domain(case_id)
    if domain is None:
        raise ValueError("the domain is not known: set B 10 first")
    x0, x1, y0, y1 = (float(v) for v in domain[:4])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    to_crs = transformer("EPSG:4326", georef.crs)
    lon, lat = transformer(georef.crs, "EPSG:4326")(*georef.to_world(cx, cy))
    ax, ay = to_crs(lon, lat)
    bx, by = to_crs(lon, lat + 0.01)
    north = math.degrees(math.atan2(float(by) - float(ay), float(bx) - float(ax)))
    return Frame(georef, north, (x0, x1, y0, y1))


def place(session, case_id: str, where: Frame | None = None) -> Place:
    """The middle of the wave generation side, else the centre of the domain."""
    where = where or frame(session, case_id)
    x0, x1, y0, y1 = where.domain
    setup = session.data.case_setup(case_id)
    side = None
    if setup is not None:
        side = next((s for s, kind in setup.sides.items() if kind == 6 and s in SIDES), None)
    middles = {
        1: (x0, (y0 + y1) / 2),
        4: (x1, (y0 + y1) / 2),
        3: ((x0 + x1) / 2, y0),
        2: ((x0 + x1) / 2, y1),
    }
    if side is not None:
        x, y = middles[side]
        label = f"the wave generation side ({SIDES[side][0]})"
    else:
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        label = "the centre of the domain"
    lon, lat = where.lonlat(x, y)
    return Place(float(lat), float(lon), label, x, y)


def case_folder(session, case_id: str) -> Path:
    folder = session.data.case_folder(case_id)
    if folder is None:
        raise ValueError("the case has no folder")
    return folder


def default_source(session, case_id: str) -> str:
    """MET Norway in its waters (finer, with 2D spectra), Open-Meteo elsewhere."""
    try:
        spot = place(session, case_id)
    except (ValueError, OSError):
        return OPEN_METEO
    lat0, lat1, lon0, lon1 = MET_BOX
    return MET_NORWAY if lat0 <= spot.lat <= lat1 and lon0 <= spot.lon <= lon1 else OPEN_METEO


def _source(session, case_id: str, source: str | None) -> str:
    if source is None:
        return default_source(session, case_id)
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}: {', '.join(SOURCES)}")
    return source


def _waves(source: str, lat, lon, start, end, progress, nora3=None, openmeteo=None) -> mo.Series:
    if source == MET_NORWAY:
        return (nora3 or mo.Nora3()).series(lat, lon, start, end, progress)
    return (openmeteo or om.OpenMeteo()).waves(lat, lon, start, end, progress, WIND[source])


# ======================================================================== storms
def find_storms(
    session,
    case_id: str,
    first_year: int,
    last_year: int,
    count: int = 10,
    progress=None,
    nora3: mo.Nora3 | None = None,
    source: str | None = None,
    openmeteo: om.OpenMeteo | None = None,
):
    """The largest wave events at the case (NORA3, or ERA5 from Open-Meteo):
    (storms, the hindcast's point, place)."""
    where = place(session, case_id)
    source = _source(session, case_id, source)
    if source == MET_NORWAY:
        found, point = mo.find_storms(
            where.lat, where.lon, first_year, last_year, count, nora3=nora3, progress=progress
        )
        return found, point, where
    start = dt.datetime(first_year, 1, 1, tzinfo=dt.UTC)
    end = min(dt.datetime(last_year, 12, 31, 23, tzinfo=dt.UTC), dt.datetime.now(dt.UTC))
    series = (openmeteo or om.OpenMeteo()).waves(
        where.lat, where.lon, start, end, progress, WIND[source]
    )
    return mo.storms(series, count), series.point, where


# ======================================================================== conditions
@dataclass
class Written:
    """What was written: files, and words for the status line."""

    files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def conditions(
    session,
    case_id: str,
    start,
    end,
    progress=None,
    nora3: mo.Nora3 | None = None,
    norkyst: mo.NorKyst | None = None,
    source: str | None = None,
    openmeteo: om.OpenMeteo | None = None,
) -> Written:
    """The waves and wind and the water level and current at the case from ``start``
    to ``end``, as tables in ``LAGOON_Sea`` (time 0 at ``start``): NORA3 and NorKyst
    (MET Norway), or ERA5 and Météo-France's sea level and currents (Open-Meteo)."""
    where = frame(session, case_id)
    spot = place(session, case_id, where)
    folder = case_folder(session, case_id) / FOLDER
    start_s = mo._utc(start).timestamp()
    stamp = f"{mo._utc(start):%Y%m%d%H}"
    out = Written()
    source = _source(session, case_id, source)
    if source in WIND:
        meteo = openmeteo or om.OpenMeteo()
        return _conditions_openmeteo(
            session, case_id, where, spot, folder, start, end, progress, meteo, WIND[source]
        )
    nora3 = nora3 or mo.Nora3()

    def part(low, high):
        if progress is None:
            return None
        return lambda f, text: progress(low + (high - low) * f, text)

    waves = nora3.series(spot.lat, spot.lon, start, end, part(0.0, 0.4))
    columns = dict(waves.columns)
    to = (np.asarray(waves.columns["dir_from"]) + 180.0) % 360.0
    columns["beta"] = mo.model_angle(to, where.north_deg, where.rotation_deg)  # B 105
    units = {**waves.units, "beta": "deg"}
    p = waves.point
    note = (
        f"NORA3 waves and wind at {p.lat:.4f} N {p.lon:.4f} E ({p.distance:.1f} km from "
        f"{spot.label}); dir_from, wind_from: where from, clockwise from north; beta: "
        f"where the waves go in the model, from its x-axis (B 105)"
    )
    out.files.append(
        mo.write_table(folder / f"NORA3-{stamp}.dat", start_s, columns, waves.time, units, note)
    )
    norkyst = norkyst or mo.NorKyst(nora3.dap)
    try:
        ocean = norkyst.box([spot.lat], [spot.lon], start, end, progress=part(0.4, 1.0))
    except mo.MetoceanError as error:
        out.notes.append(str(error))
        return out
    a, b = _nearest_wet(ocean, spot.lat, spot.lon)
    setup = session.data.case_setup(case_id)
    level = setup.water_level if setup is not None and setup.water_level is not None else 0.0
    u, v = where.to_model(ocean.u[:, a, b], ocean.v[:, a, b])
    speed = np.hypot(ocean.u[:, a, b], ocean.v[:, a, b])
    going = np.degrees(np.arctan2(ocean.u[:, a, b], ocean.v[:, a, b])) % 360.0
    columns = {
        "zeta": ocean.zeta[:, a, b],
        "level": level + ocean.zeta[:, a, b],
        "current": speed,
        "current_to": going,
        "u": u,
        "v": v,
    }
    units = {
        "zeta": "m",
        "level": "m",
        "current": "m/s",
        "current_to": "deg",
        "u": "m/s",
        "v": "m/s",
    }
    note = (
        f"{ocean.source} at {ocean.lat[a, b]:.4f} N {ocean.lon[a, b]:.4f} E, depth "
        f"{ocean.h[a, b]:.1f} m; level: the still water level (F 60) + zeta; current: "
        "depth-averaged, current_to where it goes (clockwise from north), u, v in the "
        "model's frame"
    )
    out.files.append(
        mo.write_table(folder / f"NorKyst-{stamp}.dat", start_s, columns, ocean.time, units, note)
    )
    return out


def _conditions_openmeteo(
    session, case_id, where, spot, folder, start, end, progress, meteo, wind="era5"
):
    start_s = mo._utc(start).timestamp()
    stamp = f"{mo._utc(start):%Y%m%d%H}"
    out = Written()

    def part(low, high):
        if progress is None:
            return None
        return lambda f, text: progress(low + (high - low) * f, text)

    waves = meteo.waves(spot.lat, spot.lon, start, end, part(0.0, 0.6), wind)
    columns = dict(waves.columns)
    to = (np.asarray(waves.columns["dir_from"]) + 180.0) % 360.0
    columns["beta"] = mo.model_angle(to, where.north_deg, where.rotation_deg)  # B 105
    units = {**waves.units, "beta": "deg"}
    p = waves.point
    note = (
        f"{waves.source} waves and wind (Open-Meteo) at {p.lat:.3f} N {p.lon:.3f} E "
        f"({p.distance:.1f} km from {spot.label}); dir_from, wind_from: where from, "
        "clockwise from north; beta: where the waves go in the model, from its x-axis (B 105)"
    )
    out.files.append(
        mo.write_table(folder / f"ERA5-{stamp}.dat", start_s, columns, waves.time, units, note)
    )
    if progress is not None:
        progress(0.7, "Sea level and currents (Open-Meteo)")
    try:
        sea = meteo.sea(spot.lat, spot.lon, start, end)
    except mo.MetoceanError as error:
        out.notes.append(str(error))
        return out
    setup = session.data.case_setup(case_id)
    level = setup.water_level if setup is not None and setup.water_level is not None else 0.0
    u, v = where.to_model(sea.columns["u_east"], sea.columns["v_north"])
    columns = {
        "zeta": sea.columns["zeta"],
        "level": level + sea.columns["zeta"],
        "current": sea.columns["current"],
        "current_to": sea.columns["current_to"],
        "u": u,
        "v": v,
    }
    units = {
        k: ("deg" if k == "current_to" else "m/s" if k in ("current", "u", "v") else "m")
        for k in columns
    }
    p = sea.point
    note = (
        f"{sea.source} at {p.lat:.3f} N {p.lon:.3f} E ({p.distance:.1f} km away); zeta: "
        "the sea level above mean sea level with the tide and surge; level: the still water "
        "level (F 60) + zeta; current: at the surface, current_to where it goes (clockwise "
        "from north), u, v in the model's frame"
    )
    out.files.append(
        mo.write_table(folder / f"OpenMeteo-{stamp}.dat", start_s, columns, sea.time, units, note)
    )
    if progress is not None:
        progress(1.0, "Open-Meteo")
    return out


def _nearest_wet(ocean: mo.Ocean, lat: float, lon: float) -> tuple[int, int]:
    d = mo.distance_km(lat, lon, ocean.lat, ocean.lon)
    wet = np.isfinite(ocean.h) & np.isfinite(ocean.zeta[0])
    d = np.where(wet, d, np.inf)
    if not np.isfinite(d).any():
        raise mo.MetoceanError("NorKyst has no water here")
    a, b = np.unravel_index(np.argmin(d), d.shape)
    return int(a), int(b)


# ======================================================================== spectrum
def spectrum(
    session,
    case_id: str,
    when,
    apply: bool = True,
    half_width: float = 90.0,
    dap: mo.Dap | None = None,
) -> dict:
    """NORA3's 2D spectrum at the case at ``when`` as ``spectrum-file-2d.dat`` in the
    case folder; with ``apply`` the wave generation in ctrl.txt is set to it (B 85
    11, B 92 31, B 93 Hs Tp, B 105 direction, B 130 1: one undo step, saved with the
    input files). Returns the facts (Hs, Tp, direction, the file)."""
    where = frame(session, case_id)
    spot = place(session, case_id, where)
    found = mo.nora3_spectrum(when, spot.lat, spot.lon, dap)
    path = case_folder(session, case_id) / "spectrum-file-2d.dat"
    facts = mo.spectrum_file(found, path, where.north_deg, where.rotation_deg, half_width)
    facts.update(file=path, lat=found.lat, lon=found.lon, time=found.time)
    if apply:
        apply_spectrum(session, case_id, facts)
    return facts


def apply_spectrum(session, case_id: str, facts: dict) -> None:
    """ctrl.txt's wave generation from a spectrum written (:func:`spectrum`): B 85 11,
    B 92 31, B 93 Hs Tp, B 105 the direction, B 130 1; one undo step."""
    hs, tp, beta = facts["Hs"], facts["Tp"], facts["direction"]
    _edit_ctrl(
        session,
        case_id,
        [
            ("B 85", [11]),
            ("B 92", [31]),
            ("B 93", [round(hs, 3), round(tp, 3)]),
            ("B 105", [round(beta, 2), *_origin(session, case_id)]),
            ("B 130", [1]),
        ],
        "Wave generation from NORA3",
    )


def waves_at(
    session,
    case_id: str,
    when,
    apply: bool = True,
    openmeteo: om.OpenMeteo | None = None,
) -> dict:
    """The waves at the case at the hour ``when`` from ERA5 (Open-Meteo: Hs, Tp and the
    mean direction, no 2D spectrum) for a JONSWAP spectrum; with ``apply`` ctrl.txt's
    wave generation follows them (B 85 2, B 92 31, B 93 Hs Tp, B 105 the direction;
    one undo step). Returns the facts (Hs, Tp, direction in the model, the point)."""
    where = frame(session, case_id)
    spot = place(session, case_id, where)
    t = mo._utc(when)
    hour = dt.datetime(t.year, t.month, t.day, t.hour, tzinfo=dt.UTC)
    series = (openmeteo or om.OpenMeteo()).waves(
        spot.lat, spot.lon, hour - dt.timedelta(hours=3), hour + dt.timedelta(hours=3)
    )
    k = int(np.argmin(np.abs(series.time - hour.timestamp())))
    hs, tp = float(series.columns["Hs"][k]), float(series.columns["Tp"][k])
    if not (math.isfinite(hs) and math.isfinite(tp)):
        raise mo.MetoceanError(f"ERA5 has no waves at {hour:%Y-%m-%d %H:00}")
    to = (float(series.columns["dir_from"][k]) + 180.0) % 360.0
    beta = float(mo.model_angle(to, where.north_deg, where.rotation_deg))
    facts = {
        "Hs": hs,
        "Tp": tp,
        "direction": beta,
        "from": float(series.columns["dir_from"][k]),
        "time": float(series.time[k]),
        "lat": series.point.lat,
        "lon": series.point.lon,
        "source": series.source,
    }
    if apply:
        apply_waves(session, case_id, facts)
    return facts


def apply_waves(session, case_id: str, facts: dict) -> None:
    """ctrl.txt's wave generation from Hs, Tp and a direction (:func:`waves_at`): a
    JONSWAP spectrum, B 85 2, B 92 31, B 93 Hs Tp, B 105 the direction; one undo step."""
    _edit_ctrl(
        session,
        case_id,
        [
            ("B 85", [2]),
            ("B 92", [31]),
            ("B 93", [round(facts["Hs"], 3), round(facts["Tp"], 3)]),
            ("B 105", [round(facts["direction"], 2), *_origin(session, case_id)]),
        ],
        "Wave generation from ERA5",
    )


def _origin(session, case_id: str) -> list[float]:
    setup = session.data.case_setup(case_id)
    old = setup.ctrl.values("B 105") if setup is not None else None
    if old and len(old) >= 3:
        return [float(old[1]), float(old[2])]
    domain = session.data.backgrounds.domain(case_id)
    return [float(domain[0]), float(domain[2])] if domain else [0.0, 0.0]


def _edit_ctrl(session, case_id: str, lines, label: str) -> None:
    """Set commands of ctrl.txt as one undo step (``label``: what for). ``lines``:
    (code, values) for a command's first line, or (code, values, key) for the line
    whose first value is ``key`` (a patch's ID; added when there is none)."""
    control = session.input_file("ctrl.txt", case_id).copy()
    for entry in lines:
        code, values = entry[0], entry[1]
        if len(entry) == 2:
            control.set(code, values)
            continue
        key = float(entry[2])
        found = [k for k, v in enumerate(control.all(code)) if v and abs(float(v[0]) - key) < 1e-9]
        if found:
            control.set(code, values, found[0])
        else:
            control.add(code, values)
    # as a change of the whole text: one undo step, and the script repeats it offline
    session.set_input_text("ctrl.txt", control.text(), case=case_id)


# ======================================================================== boundaries
@dataclass
class SideSeries:
    """NorKyst along one side of the domain over time."""

    side: int
    name: str
    #: the side's ends in the model (x0, y0), (x1, y1)
    ends: tuple[tuple[float, float], tuple[float, float]]
    time: np.ndarray
    level: np.ndarray  # still water level + zeta, mean along the side [m]
    discharge: np.ndarray  # into the domain [m3/s]
    wet: float  # the part of the side in NorKyst's water


def side_series(
    ocean: mo.Ocean, where: Frame, side: int, swl: float, dry_km: float = DRY_KM
) -> SideSeries:
    x0, x1, y0, y1 = where.domain
    ends = {
        1: ((x0, y0), (x0, y1)),
        4: ((x1, y0), (x1, y1)),
        3: ((x0, y0), (x1, y0)),
        2: ((x0, y1), (x1, y1)),
    }[side]
    (ax, ay), (bx, by) = ends
    length = math.hypot(bx - ax, by - ay)
    n = max(5, math.ceil(length / SIDE_STEP))
    f = (np.arange(n) + 0.5) / n
    lon, lat = where.lonlat(ax + f * (bx - ax), ay + f * (by - ay))
    wet_cells = np.isfinite(ocean.h) & np.isfinite(ocean.zeta[0])
    cells = []
    for la, lo in zip(np.ravel(lat), np.ravel(lon), strict=True):
        d = np.where(wet_cells, mo.distance_km(la, lo, ocean.lat, ocean.lon), np.inf)
        k = np.unravel_index(np.argmin(d), d.shape)
        cells.append(k if d[k] <= dry_km else None)
    wet = [c for c in cells if c is not None]
    nx, ny = SIDES[side][1]
    nt = len(ocean.time)
    if not wet:
        empty = np.full(nt, np.nan)
        return SideSeries(side, SIDES[side][0], ends, ocean.time, empty, empty, 0.0)
    a = np.array([c[0] for c in wet])
    b = np.array([c[1] for c in wet])
    zeta = ocean.zeta[:, a, b]
    u, v = where.to_model(ocean.u[:, a, b], ocean.v[:, a, b])
    depth = ocean.h[a, b][None, :] + zeta
    dl = length / n
    discharge = np.nansum((u * nx + v * ny) * depth * dl, axis=1)
    level = swl + np.nanmean(zeta, axis=1)
    return SideSeries(side, SIDES[side][0], ends, ocean.time, level, discharge, len(wet) / n)


def boundaries(
    session,
    case_id: str,
    start,
    end,
    sides=None,
    apply: bool = True,
    progress=None,
    norkyst: mo.NorKyst | None = None,
    source: str | None = None,
    openmeteo: om.OpenMeteo | None = None,
) -> list[SideSeries]:
    """NorKyst's water level and discharge along the sides of the domain from
    ``start`` to ``end``: per side in the water (``sides``: the side numbers, default
    those more than half in NorKyst's water) REEF3D's ``hydrograph_FSF_<n>.dat`` (time,
    water level) and ``hydrograph_Q_<n>.dat`` (time, discharge into the domain), n the
    side's number; a table of all in ``LAGOON_Sea``. With ``apply``, ctrl.txt gets
    the patches (B 440 n side line, B 421 n 1, B 422 n 1; one undo step)."""
    where = frame(session, case_id)
    setup = session.data.case_setup(case_id)
    swl = setup.water_level if setup is not None and setup.water_level is not None else 0.0
    if _source(session, case_id, source) in WIND:
        oceans = _side_oceans(session, case_id, where, swl, start, end, progress, openmeteo)
        found = [side_series(oceans[s], where, s, swl, dry_km=0.01) for s in SIDES]
        ocean = oceans[next(iter(SIDES))]
    else:
        x0, x1, y0, y1 = where.domain
        lon, lat = where.lonlat([x0, x1, x1, x0], [y0, y0, y1, y1])
        norkyst = norkyst or mo.NorKyst()
        ocean = norkyst.box(np.ravel(lat), np.ravel(lon), start, end, progress=progress)
        found = [side_series(ocean, where, s, swl) for s in SIDES]
    chosen = [s for s in found if (s.side in sides if sides else s.wet > 0.5)]
    folder = case_folder(session, case_id)
    start_s = mo._utc(start).timestamp()
    t = ocean.time - start_s
    columns, units = {}, {}
    for s in chosen:
        mo.two_column(folder / f"hydrograph_FSF_{s.side}.dat", t, s.level)
        mo.two_column(folder / f"hydrograph_Q_{s.side}.dat", t, s.discharge)
        key = s.name.replace(" ", "")
        columns[f"level_{key}"], units[f"level_{key}"] = s.level, "m"
        columns[f"Q_{key}"], units[f"Q_{key}"] = s.discharge, "m3/s"
    if columns:
        note = (
            f"{ocean.source} along the sides of the domain: level (still water level + "
            "zeta, mean along the side), Q (discharge into the domain)"
        )
        stamp = f"{mo._utc(start):%Y%m%d%H}"
        mo.write_table(
            folder / FOLDER / f"Boundaries-{stamp}.dat", start_s, columns, ocean.time, units, note
        )
    if apply and chosen:
        apply_boundaries(session, case_id, chosen)
    return chosen


def side_ends(where: Frame, side: int) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, x1, y0, y1 = where.domain
    return {
        1: ((x0, y0), (x0, y1)),
        4: ((x1, y0), (x1, y1)),
        3: ((x0, y0), (x1, y0)),
        2: ((x0, y1), (x1, y1)),
    }[side]


def bed_at(session, case_id: str, x, y) -> tuple[np.ndarray, str]:
    """The bed's height at model points: geo.dat's points interpolated, else the
    bottom of the domain (B 10); and where it came from."""
    from lagoonviz.core.corridor import interpolate_bed

    setup = session.data.case_setup(case_id)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if setup is not None and (setup.path / "geo.dat").is_file():
        points = setup.geo_points(limit=10**9)
        if len(points):
            return interpolate_bed(points, x, y).reshape(x.shape), "geo.dat"
    if setup is not None and setup.domain is not None:
        return np.full(x.shape, float(setup.domain[4])), "the bottom of the domain (B 10)"
    raise ValueError("the bed along the sides is not known: no geo.dat, no B 10")


def _side_oceans(session, case_id, where, swl, start, end, progress, meteo) -> dict:
    """Open-Meteo's sea level and current along each side: asked at points about
    :data:`~lagoonviz.core.geo.openmeteo.OCEAN_SPACING` apart (its resolution),
    interpolated to points :data:`SIDE_STEP` apart; the depth from the case's bed."""
    meteo = meteo or om.OpenMeteo()
    coarse = {}
    lats, lons = [], []
    for side in SIDES:
        (ax, ay), (bx, by) = side_ends(where, side)
        length = math.hypot(bx - ax, by - ay)
        f = np.linspace(0.0, 1.0, max(2, math.ceil(length / om.OCEAN_SPACING) + 1))
        lon, lat = where.lonlat(ax + f * (bx - ax), ay + f * (by - ay))
        coarse[side] = (f, len(lats))
        lats += list(np.ravel(lat))
        lons += list(np.ravel(lon))
    ocean = meteo.ocean(lats, lons, start, end, progress)
    out = {}
    for side, (fc, first) in coarse.items():
        (ax, ay), (bx, by) = side_ends(where, side)
        length = math.hypot(bx - ax, by - ay)
        n = max(5, math.ceil(length / SIDE_STEP))
        f = (np.arange(n) + 0.5) / n
        x, y = ax + f * (bx - ax), ay + f * (by - ay)
        lon, lat = where.lonlat(x, y)
        part = slice(first, first + len(fc))

        def along(values, fc=fc, f=f, part=part):  # (t, 1, k) -> (t, 1, n)
            rows = values[:, 0, part]
            out = np.full((rows.shape[0], 1, len(f)), np.nan)
            for t, row in enumerate(rows):
                ok = np.isfinite(row)
                if ok.sum() >= 2:
                    out[t, 0] = np.interp(f, fc[ok], row[ok])
                elif ok.sum() == 1:
                    out[t, 0] = row[ok][0]
            return out

        zb, _ = bed_at(session, case_id, x, y)
        depth = swl - zb
        out[side] = mo.Ocean(
            ocean.time,
            along(ocean.zeta),
            along(ocean.u),
            along(ocean.v),
            np.where(depth > 0, depth, np.nan)[None, :],
            np.ravel(lat)[None, :],
            np.ravel(lon)[None, :],
            ocean.source,
        )
    return out


def apply_boundaries(session, case_id: str, chosen: list[SideSeries]) -> None:
    """ctrl.txt's patches for the hydrographs written (:func:`boundaries`): per side
    B 440 n n (the side as a line), B 421 n 1, B 422 n 1; one undo step."""
    lines = []
    for s in chosen:
        (ax, ay), (bx, by) = s.ends
        lines += [
            ("B 440", [s.side, s.side, ax, bx, ay, by], s.side),
            ("B 421", [s.side, 1], s.side),
            ("B 422", [s.side, 1], s.side),
        ]
    _edit_ctrl(session, case_id, lines, "Boundaries from the sea")


# ======================================================================== rivers
def river_place(session, case_id: str, where: Frame | None = None) -> Place:
    """Where a river's discharge is read: the end of the case's river corridor nearer
    an inflow side (C 1x = 1), else the middle of the inflow side, else the centre."""
    where = where or frame(session, case_id)
    x0, x1, y0, y1 = where.domain
    setup = session.data.case_setup(case_id)
    inflow = None
    if setup is not None:
        inflow = next((s for s, kind in setup.sides.items() if kind == 1 and s in SIDES), None)
    middles = {
        1: (x0, (y0 + y1) / 2),
        4: (x1, (y0 + y1) / 2),
        3: ((x0 + x1) / 2, y0),
        2: ((x0 + x1) / 2, y1),
    }
    corridor = session.corridor(case_id) if hasattr(session, "corridor") else None
    if corridor is not None:
        ends = [
            (
                0.5 * (corridor.left[0][0] + corridor.right[0][0]),
                0.5 * (corridor.left[0][1] + corridor.right[0][1]),
            ),
            (
                0.5 * (corridor.left[-1][0] + corridor.right[-1][0]),
                0.5 * (corridor.left[-1][1] + corridor.right[-1][1]),
            ),
        ]
        target = middles[inflow] if inflow is not None else ends[0]
        x, y = min(ends, key=lambda e: math.hypot(e[0] - target[0], e[1] - target[1]))
        label = "the river corridor's upstream end"
    elif inflow is not None:
        x, y = middles[inflow]
        label = f"the inflow side ({SIDES[inflow][0]})"
    else:
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        label = "the centre of the domain"
    lon, lat = where.lonlat(x, y)
    return Place(float(lat), float(lon), label, float(x), float(y))


@dataclass
class River:
    """A river's discharge over a window, as written."""

    time: np.ndarray  # seconds since 1970, hourly over the window
    discharge: np.ndarray  # m3/s
    lat: float  # the GloFAS cell
    lon: float
    distance: float  # from where it was asked for [km]
    place: Place
    files: list[Path] = field(default_factory=list)
    target: str = ""


def river(
    session,
    case_id: str,
    start,
    end,
    target: str | int = "B 60",
    apply: bool = True,
    at: tuple[float, float] | None = None,
    openmeteo: om.OpenMeteo | None = None,
) -> River:
    """GloFAS's discharge of the river at the case (:func:`river_place`, or ``at`` =
    (x, y) in the model) from ``start`` to ``end``, hourly (between GloFAS's daily
    values), time 0 at ``start``: for ioFlow (``target`` "B 60": ``hydrograph.dat``,
    B 60 2) or a side as a patch (``target`` its number: ``hydrograph_Q_<n>.dat``, B
    440 n, B 421 n 1). A table in ``LAGOON_Sea``. With ``apply`` ctrl.txt follows (one
    undo step)."""
    hours, q, p, spot = glofas(session, case_id, start, end, at, openmeteo)
    t0 = mo._utc(start)
    folder = case_folder(session, case_id)
    t = hours - t0.timestamp()
    if target in ("B 60", "ioflow", None):
        written = mo.two_column(folder / "hydrograph.dat", t, q)
        target = "B 60"
    else:
        side = int(target)
        if side not in SIDES:
            raise ValueError(f"no side {target}: 1 x min, 2 y max, 3 y min, 4 x max")
        written = mo.two_column(folder / f"hydrograph_Q_{side}.dat", t, q)
        target = side
    note = glofas_note(p, spot)
    table = mo.write_table(
        folder / FOLDER / f"GloFAS-{t0:%Y%m%d%H}.dat",
        t0.timestamp(),
        {"Q": q},
        hours,
        {"Q": "m3/s"},
        note,
    )
    found = River(hours, q, p.lat, p.lon, p.distance, spot, [written, table], str(target))
    if apply:
        apply_river(session, case_id, found)
    return found


def glofas(session, case_id: str, start, end, at=None, openmeteo: om.OpenMeteo | None = None):
    """GloFAS's daily discharge at the case (:func:`river_place`, or ``at`` = (x, y) in
    the model), hourly from ``start`` to ``end``: (times s since 1970, discharge
    m3/s, GloFAS's point, the :class:`Place` asked for)."""
    where = frame(session, case_id)
    if at is not None:
        lon, lat = where.lonlat(at[0], at[1])
        spot = Place(
            float(lat), float(lon), f"x {at[0]:g}, y {at[1]:g}", float(at[0]), float(at[1])
        )
    else:
        spot = river_place(session, case_id, where)
    t0, t1 = mo._utc(start), mo._utc(end)
    daily = (openmeteo or om.OpenMeteo()).discharge(
        spot.lat, spot.lon, t0 - dt.timedelta(days=1), t1 + dt.timedelta(days=1)
    )
    hours = np.arange(t0.timestamp(), t1.timestamp() + 1.0, 3600.0)
    ok = np.isfinite(daily.columns["Q"])
    if not ok.any():
        raise mo.MetoceanError(f"GloFAS has no discharge at {spot.label}")
    q = np.interp(hours, daily.time[ok], daily.columns["Q"][ok])
    return hours, q, daily.point, spot


def glofas_note(point, spot: Place) -> str:
    return (
        f"GloFAS river discharge (Open-Meteo) at {point.lat:.3f} N {point.lon:.3f} E, "
        f"{point.distance:.1f} km from {spot.label}; daily values, hourly between them"
    )


def apply_river(session, case_id: str, found: River) -> None:
    """ctrl.txt for a river's hydrograph (:func:`river`): B 60 2 for ioFlow, else the
    side as a patch with its discharge hydrograph (B 440, B 421); one undo step."""
    if found.target == "B 60":
        lines = [("B 60", [2])]
    else:
        side = int(found.target)
        (ax, ay), (bx, by) = side_ends(frame(session, case_id), side)
        lines = [("B 440", [side, side, ax, bx, ay, by], side), ("B 421", [side, 1], side)]
    _edit_ctrl(session, case_id, lines, "River discharge from GloFAS")


# ======================================================================== gauges
def find_gauges(
    session,
    case_id: str,
    sources=("Kartverket", "NOAA", "IOC"),
    count: int = 8,
    radius_km: float = 300.0,
    fetch=None,
) -> tuple[list[gg.Station], Place]:
    """The tide gauges nearest to the case (to the centre of its domain)."""
    where = frame(session, case_id)
    x0, x1, y0, y1 = where.domain
    lon, lat = where.lonlat((x0 + x1) / 2, (y0 + y1) / 2)
    spot = Place(float(lat), float(lon), "the centre of the domain", (x0 + x1) / 2, (y0 + y1) / 2)
    made = session.__dict__.setdefault("_gauge_sources", {}) if fetch is None else None
    return gg.nearest(spot.lat, spot.lon, sources, count, radius_km, fetch, made), spot


def gauge_levels(
    session, case_id: str, station: gg.Station, start, end, progress=None, fetch=None
) -> tuple[mo.Series, Path]:
    """The water levels measured at a gauge (and the tide predicted, where its source
    has it) from ``start`` to ``end``, as a table in ``LAGOON_Sea`` (time 0 at
    ``start``): ``observed``, ``predicted``, ``surge``, and ``level`` the case's still
    water level (F 60) plus the observed. Returns the series and the table."""
    series = gg.levels(station, start, end, fetch, progress)
    setup = session.data.case_setup(case_id)
    swl = setup.water_level if setup is not None and setup.water_level is not None else 0.0
    columns = dict(series.columns)
    if "observed" in columns:
        columns["level"] = swl + columns["observed"]
    units = {k: "m" for k in columns}
    t0 = mo._utc(start)
    note = (
        f"Water levels at {series.source}; level: the still water level (F 60) + observed. "
        f"Data: {gg.ATTRIBUTION[station.source]}"
    )
    code = "".join(c for c in station.code if c.isalnum())
    table = mo.write_table(
        case_folder(session, case_id) / FOLDER / f"Gauge_{station.source}_{code}-{t0:%Y%m%d%H}.dat",
        t0.timestamp(),
        columns,
        series.time,
        units,
        note,
    )
    return series, table


def gauge_boundaries(
    session,
    case_id: str,
    station: gg.Station,
    start,
    end,
    sides,
    apply: bool = True,
    fetch=None,
    series: mo.Series | None = None,
) -> list[Path]:
    """A gauge's observed water level (gaps filled linearly) as the water level at
    sides of the domain: ``hydrograph_FSF_<n>.dat`` (time 0 at ``start``, the still
    water level F 60 plus the observed); with ``apply`` the patches (B 440, B 422)."""
    if not sides:
        raise ValueError("choose the sides")
    if series is None:
        series = gg.levels(station, start, end, fetch)
    if "observed" not in series.columns:
        raise mo.MetoceanError(f"{station.describe()}: no observations")
    setup = session.data.case_setup(case_id)
    swl = setup.water_level if setup is not None and setup.water_level is not None else 0.0
    t0, t1 = mo._utc(start).timestamp(), mo._utc(end).timestamp()
    ok = np.isfinite(series.columns["observed"])
    if ok.sum() < 2:
        raise mo.MetoceanError(f"{station.describe()}: too few observations in the window")
    step = float(np.median(np.diff(series.time[ok]))) if ok.sum() > 1 else 600.0
    times = np.arange(t0, t1 + 0.5 * step, step)
    level = swl + np.interp(times, series.time[ok], series.columns["observed"][ok])
    folder = case_folder(session, case_id)
    where = frame(session, case_id)
    written, lines = [], []
    for side in sides:
        if side not in SIDES:
            raise ValueError(f"no side {side}: 1 x min, 2 y max, 3 y min, 4 x max")
        written.append(mo.two_column(folder / f"hydrograph_FSF_{side}.dat", times - t0, level))
        (ax, ay), (bx, by) = side_ends(where, side)
        lines += [("B 440", [side, side, ax, bx, ay, by], side), ("B 422", [side, 1], side)]
    if apply:
        _edit_ctrl(session, case_id, lines, f"Water level from {station.name}")
    return written


def default_years() -> tuple[int, int]:
    """The last ten years NORA3 has (it runs a month or two behind)."""
    last = dt.datetime.now(dt.UTC).year - 1
    return last - 9, last
