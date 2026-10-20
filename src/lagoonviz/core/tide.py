# SPDX-License-Identifier: GPL-3.0-or-later
# Architect: Hans Bihs
"""The tide at an open sea boundary of REEF3D, as its *background* (B 510 - B 515).

REEF3D drives an open edge (B 520 method 3 Riemann, 4 Flather, 5 clamped level; B 521
the edge; B 523 the background) with a background water level (``background_state``):

- ``B 510 id mode dir t_ramp``: mode 1 harmonic (constituents ``B 511 id a T phase``:
  a cos(2 pi t / T - phase), phase in degrees), 2 a time series (``background-<id>.dat``:
  "t eta" or "t eta U V" a line), 3 constant (``B 514 id eta0 U V``); ``dir`` where it
  travels (degrees from x), ``t_ramp`` the ramp at the start (s);
- ``B 515 id x0 y0 h_ref``: it travels along ``dir`` from (x0, y0), where the series
  holds, at sqrt(g h_ref) (a tide running along a coast).

A series of two columns makes REEF3D add the long-wave current eta sqrt(g / h)
along ``dir``; four columns "t eta 0 0" give none, the right state for a standing
coastal tide at a Riemann edge (the Tagus case).

The water level comes from tidal constituents (amplitude and Greenwich phase, as
tide tables, FES2014 or TPXO give them) with the nodal corrections and the
astronomical arguments of the start date (Pugh 1987, Table 4.3), from a tide gauge
(:mod:`lagoonviz.core.geo.gauges`: observed or predicted, the mean of the window
taken away) or from a file of times and levels.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: the constituents known: period [h]
PERIODS = {
    "M2": 12.4206012,
    "S2": 12.0,
    "N2": 12.65834751,
    "K2": 11.96723606,
    "K1": 23.93446966,
    "O1": 25.81934166,
    "P1": 24.06589016,
    "Q1": 26.86835667,
    "M4": 6.210300601,
    "MS4": 6.103339275,
}
#: B 520 methods for an open edge with a background
METHODS = {3: "Riemann", 4: "Flather", 5: "clamped level"}
#: block 0's faces as REEF3D's edges (B 521): x-, x+, y-, y+
EDGES = {"i0": 1, "i1": 2, "j0": 3, "j1": 4}
#: the background the tide is
BACKGROUND = 1


def _utc(when) -> dt.datetime:
    if isinstance(when, dt.datetime):
        moment = when
    else:
        moment = dt.datetime.fromisoformat(str(when).strip().replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.UTC)
    return moment.astimezone(dt.UTC)


def astro(seconds: np.ndarray) -> tuple:
    """The mean longitudes [deg] of the moon (s), the sun (h), the lunar perigee (p)
    and the moon's node (N), and the solar hour angle T, at times in seconds since
    1970 (UTC)."""
    seconds = np.asarray(seconds, float)
    days = (seconds - 946_728_000.0) / 86400.0  # since 2000-01-01 12:00
    c = days / 36525.0
    s = 218.3165 + 481267.8813 * c
    h = 280.4665 + 36000.7698 * c
    p = 83.3532 + 4069.0137 * c
    n = 125.0445 - 1934.1363 * c
    hours = np.mod(seconds, 86400.0) / 3600.0
    return s, h, p, n, 180.0 + 15.0 * hours


def nodal(name: str, s, h, p, n, hour) -> tuple:
    """A constituent's nodal factor f and its argument V + u [deg]."""
    r = np.radians(n)
    if name in ("M2", "N2"):
        f = 1.0004 - 0.0373 * np.cos(r) + 0.0002 * np.cos(2 * r)
        u = -2.14 * np.sin(r)
    elif name == "K2":
        f = 1.0241 + 0.2863 * np.cos(r) + 0.0083 * np.cos(2 * r) - 0.0015 * np.cos(3 * r)
        u = -17.74 * np.sin(r) + 0.68 * np.sin(2 * r) - 0.04 * np.sin(3 * r)
    elif name == "K1":
        f = 1.0060 + 0.1150 * np.cos(r) - 0.0088 * np.cos(2 * r) + 0.0006 * np.cos(3 * r)
        u = -8.86 * np.sin(r) + 0.68 * np.sin(2 * r) - 0.07 * np.sin(3 * r)
    elif name in ("O1", "Q1"):
        f = 1.0089 + 0.1871 * np.cos(r) - 0.0147 * np.cos(2 * r) + 0.0014 * np.cos(3 * r)
        u = 10.80 * np.sin(r) - 1.34 * np.sin(2 * r) + 0.19 * np.sin(3 * r)
    elif name == "M4":
        f = (1.0004 - 0.0373 * np.cos(r)) ** 2
        u = -4.28 * np.sin(r)
    elif name == "MS4":
        f = 1.0004 - 0.0373 * np.cos(r)
        u = -2.14 * np.sin(r)
    else:
        f, u = np.ones_like(r), np.zeros_like(r)
    tau = hour + h - s
    v = {
        "M2": 2 * tau,
        "S2": 2 * hour,
        "N2": 2 * tau - s + p,
        "K2": 2 * hour + 2 * h,
        "K1": tau + s + 90.0,
        "O1": tau - s - 90.0,
        "Q1": tau - 2 * s + p - 90.0,
        "P1": hour - h + 270.0,
        "M4": 4 * tau,
        "MS4": 2 * tau + 2 * hour,
    }[name]
    return f, v + u


def read_constants(text) -> dict[str, tuple[float, float]]:
    """Constituents from text (or a file): "name amplitude phase" a line (m, degrees
    Greenwich), ``#`` comments. Names as in PERIODS."""
    if isinstance(text, Path) or (
        isinstance(text, str) and "\n" not in text and Path(text).is_file()
    ):
        text = Path(text).read_text(encoding="utf-8", errors="replace")
    if isinstance(text, dict):
        found = {str(k).upper(): (float(v[0]), float(v[1])) for k, v in text.items()}
    else:
        found = {}
        for row in str(text).splitlines():
            words = row.split("#")[0].replace(",", " ").replace(";", " ").split()
            if len(words) >= 3:
                found[words[0].upper()] = (float(words[1]), float(words[2]))
    unknown = [k for k in found if k not in PERIODS]
    if unknown:
        raise ValueError(
            f"unknown constituents {', '.join(unknown)}: LAGOON knows {', '.join(PERIODS)}"
        )
    if not found:
        raise ValueError('no constituents: "name amplitude phase" a line, e.g. "M2 1.00 61"')
    return found


def synthesize(constants: dict, start, times: np.ndarray) -> np.ndarray:
    """The water level [m] at ``times`` seconds after ``start`` (UTC) from
    constituents (name -> (amplitude m, Greenwich phase deg))."""
    times = np.asarray(times, float)
    seconds = _utc(start).timestamp() + times
    s, h, p, n, hour = astro(seconds)
    eta = np.zeros_like(times)
    for name, (amplitude, phase) in constants.items():
        f, argument = nodal(name, s, h, p, n, hour)
        eta += f * amplitude * np.cos(np.radians(argument - phase))
    return eta


def harmonic(constants: dict, start) -> list[tuple[str, float, float, float]]:
    """B 511's constituents for a run starting at ``start``: (name, amplitude, period
    s, phase deg) with REEF3D's a cos(2 pi t / T - phase) the tide from t = 0 on (the
    nodal corrections of the start kept over the run)."""
    seconds = np.array([_utc(start).timestamp()])
    s, h, p, n, hour = astro(seconds)
    out = []
    for name, (amplitude, phase) in constants.items():
        f, argument = nodal(name, s, h, p, n, hour)
        out.append(
            (
                name,
                float(f[0] * amplitude),
                PERIODS[name] * 3600.0,
                float(np.mod(phase - argument[0], 360.0)),
            )
        )
    return out


def read_levels(path, start=None) -> tuple[np.ndarray, np.ndarray]:
    """A file of water levels: "time level" a line, the time as seconds from the
    start of the run, or a date and time (ISO, UTC) taken from ``start``; spaces,
    tabs, commas or semicolons; a header line and ``#`` comments are skipped.
    Returns (seconds, levels)."""
    times, levels = [], []
    origin = _utc(start).timestamp() if start is not None else None
    for row in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        words = row.split("#")[0].replace(";", " ").replace(",", " ").replace("\t", " ").split()
        if len(words) < 2:
            continue
        try:
            level = float(words[-1])
        except ValueError:
            continue
        stamp = " ".join(words[:-1]) if len(words) > 2 else words[0]
        try:
            times.append(float(stamp))
        except ValueError:
            try:
                moment = _utc(stamp.replace(" ", "T", 1) if "T" not in stamp else stamp)
            except ValueError:
                continue
            if origin is None:
                raise ValueError(f"{path}: dates need the start of the run") from None
            times.append(moment.timestamp() - origin)
        levels.append(level)
    if len(times) < 2:
        raise ValueError(f"{path}: fewer than two times with a level")
    order = np.argsort(times)
    return np.asarray(times, float)[order], np.asarray(levels, float)[order]


@dataclass(frozen=True)
class Tide:
    """The tide written for a run: what it came from, its times and levels (the
    series) or its constituents (harmonic), and the lines of ctrl.txt."""

    source: str
    times: np.ndarray
    levels: np.ndarray
    constituents: list
    lines: list

    def text(self) -> str:
        if len(self.levels):
            return (
                f"{self.source}: {len(self.levels):,} levels over "
                f"{self.times[-1] / 86400.0:.3g} days, {self.levels.min():.2f} to "
                f"{self.levels.max():.2f} m"
            )
        return f"{self.source}: {len(self.constituents)} constituents (B 511)"


def background_text(times, levels, current: bool = False, source: str = "") -> str:
    """background-1.dat: "t eta U V" with U = V = 0 (no background current; with
    ``current`` "t eta": REEF3D adds the long-wave current)."""
    rows = [f"# {source}".rstrip(), "# t [s]  eta [m]" + ("" if current else "  U V [m/s]")]
    for t, e in zip(times, levels, strict=True):
        if np.isfinite(e):
            rows.append(f"{t:.1f} {e:.4f}" + ("" if current else " 0.0 0.0"))
    return "\n".join(rows) + "\n"


#: the kinds of the values of the background's commands
KINDS = {
    "B 510": ["int", "int", "double", "double"],
    "B 511": ["int", "double", "double", "double"],
    "B 515": ["int", "double", "double", "double"],
}


def ctrl_lines(
    mode: int,
    ramp: float,
    direction: float = 0.0,
    constituents=(),
    reference=None,
    depth: float | None = None,
) -> list[tuple[str, list]]:
    """The background's commands (code, values; their kinds in KINDS): B 510, B 511
    (harmonic), B 515 (travelling from ``reference`` (x0, y0) at sqrt(g depth))."""
    b = BACKGROUND
    lines = [("B 510", [b, int(mode), float(direction), float(ramp)])]
    for _name, amplitude, period, phase in constituents:
        lines.append(("B 511", [b, round(amplitude, 4), round(period, 3), round(phase, 2)]))
    if reference is not None:
        lines.append(("B 515", [b, float(reference[0]), float(reference[1]), float(depth or 0.0)]))
    return lines


def make(
    start,
    days: float = 3.0,
    step: float = 300.0,
    constants=None,
    levels=None,
    gauge=None,
    predicted: bool = False,
    harmonic_in_reef3d: bool = False,
    ramp: float = 10800.0,
    direction: float = 0.0,
    reference=None,
    depth: float | None = None,
    fetch=None,
) -> Tide:
    """The tide for a run from ``start`` (UTC) over ``days``, every ``step`` s, from
    ``constants`` (text, a file or a dict), a file of ``levels`` or a ``gauge``
    (:class:`~lagoonviz.core.geo.gauges.Station`; its observations, or its
    prediction with ``predicted``; the mean of the window taken away). With
    ``harmonic_in_reef3d`` the constituents go to B 511 (REEF3D sums them) instead
    of a series."""
    given = [v is not None for v in (constants, levels, gauge)]
    if sum(given) != 1:
        raise ValueError("give the tide's constituents, a file of levels or a tide gauge")
    times = np.arange(0.0, float(days) * 86400.0 + 0.5 * float(step), float(step))
    if constants is not None:
        found = read_constants(constants)
        names = " ".join(f"{k} {a:.3f} m {g:.1f}°" for k, (a, g) in found.items())
        if harmonic_in_reef3d:
            parts = harmonic(found, start)
            lines = ctrl_lines(1, ramp, direction, parts, reference, depth)
            return Tide(f"constituents {names}", np.zeros(0), np.zeros(0), parts, lines)
        eta = synthesize(found, start, times)
        source = f"constituents {names}, from {_utc(start):%Y-%m-%d %H:%M} UTC"
    elif levels is not None:
        t, z = read_levels(levels, start)
        eta = np.interp(times, t, z, left=np.nan, right=np.nan)
        if np.isnan(eta).all():
            raise ValueError(f"{levels}: no levels between the start and the end of the run")
        eta = _filled(times, eta)
        source = f"levels of {Path(str(levels)).name}"
    else:
        from lagoonviz.core.geo import gauges as gg

        begin = _utc(start)
        end = begin + dt.timedelta(days=float(days))
        series = gg.levels(gauge, begin, end, fetch)
        column = "predicted" if predicted else "observed"
        if column not in series.columns:
            raise ValueError(f"{gauge.describe()}: no {column} levels")
        values = np.asarray(series.columns[column], float)
        ok = np.isfinite(values)
        if ok.sum() < 2:
            raise ValueError(f"{gauge.describe()}: too few {column} levels in the window")
        eta = np.interp(begin.timestamp() + times, series.time[ok], values[ok])
        eta = eta - float(np.mean(eta))
        source = f"{column} at {gauge.describe()}, the mean taken away"
    lines = ctrl_lines(2, ramp, direction, (), reference, depth)
    return Tide(source, times, eta, [], lines)


def _filled(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    ok = np.isfinite(values)
    return np.interp(times, times[ok], values[ok])


def end_of(tide: Tide) -> float:
    """The last time the tide has [s] (inf for constituents)."""
    return float(tide.times[-1]) if len(tide.times) else math.inf
