# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The sea at a case, from MET Norway (:mod:`lagoonviz.core.geo.metocean`), for its
set-up: the storms there, the conditions over a window as plots, NORA3's 2D spectrum
as REEF3D's ``spectrum-file-2d.dat``, NorKyst's water level and discharge along the
sides of the domain as REEF3D's patchBC hydrographs.

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

from lagoonviz.core.geo import metocean as mo
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


# ======================================================================== storms
def find_storms(
    session,
    case_id: str,
    first_year: int,
    last_year: int,
    count: int = 10,
    progress=None,
    nora3: mo.Nora3 | None = None,
):
    """The largest wave events at the case (NORA3): (storms, the NORA3 point, place)."""
    where = place(session, case_id)
    found, point = mo.find_storms(
        where.lat, where.lon, first_year, last_year, count, nora3=nora3, progress=progress
    )
    return found, point, where


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
) -> Written:
    """NORA3's waves and wind and NorKyst's water level and current at the case from
    ``start`` to ``end``, as tables in ``LAGOON_Sea`` (time 0 at ``start``)."""
    where = frame(session, case_id)
    spot = place(session, case_id, where)
    folder = case_folder(session, case_id) / FOLDER
    start_s = mo._utc(start).timestamp()
    stamp = f"{mo._utc(start):%Y%m%d%H}"
    out = Written()
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


def side_series(ocean: mo.Ocean, where: Frame, side: int, swl: float) -> SideSeries:
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
        cells.append(k if d[k] <= DRY_KM else None)
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
) -> list[SideSeries]:
    """NorKyst's water level and discharge along the sides of the domain from
    ``start`` to ``end``: per side in the water (``sides``: the side numbers, default
    those more than half in NorKyst's water) REEF3D's ``hydrograph_FSF_<n>.dat`` (time,
    water level) and ``hydrograph_Q_<n>.dat`` (time, discharge into the domain), n the
    side's number; a table of all in ``LAGOON_Sea``. With ``apply``, ctrl.txt gets
    the patches (B 440 n side line, B 421 n 1, B 422 n 1; one undo step)."""
    where = frame(session, case_id)
    x0, x1, y0, y1 = where.domain
    lon, lat = where.lonlat([x0, x1, x1, x0], [y0, y0, y1, y1])
    norkyst = norkyst or mo.NorKyst()
    ocean = norkyst.box(np.ravel(lat), np.ravel(lon), start, end, progress=progress)
    setup = session.data.case_setup(case_id)
    swl = setup.water_level if setup is not None and setup.water_level is not None else 0.0
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
    _edit_ctrl(session, case_id, lines, "Boundaries from NorKyst")


def default_years() -> tuple[int, int]:
    """The last ten years NORA3 has (it runs a month or two behind)."""
    last = dt.datetime.now(dt.UTC).year - 1
    return last - 9, last
